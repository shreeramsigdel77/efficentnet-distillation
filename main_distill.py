import os
import sys
import time
import random
import datetime

import copy
import torch
import torch.nn.functional as F
import numpy as np

import timm
from timm import create_model
from tqdm import tqdm


from utils.yaml_utils import load_yaml
from utils.general_utils import create_directory, create_unique_directory

from utils.log_files import log_to_csv_distillation
from utils.compute_metrices import compute_metrics, plot_metrics


from model.data_plotting import plot_confusion_matrix ,plot_learning_rate
from model.data_loader import ImageTrainTransform,ImageTestTransform, TestDatasetLoader
from model.trainner_custefficent import TrainingPlotter, CustomEfficientNet
from model.data_loader import DatasetLoader, ModelValidator


# List out all available  models
print(timm.list_models("efficientnet*"))


# Function to Set any fixed seed value for reproducibility
def set_seed(seed_value):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility in experiments.

    This function sets a fixed random seed for the random number generators used in Python, NumPy,
    and PyTorch (both CPU and CUDA) to guarantee that the sequence of random numbers generated 
    in each run is the same, allowing for consistent results in machine learning experiments.

    Parameters:
    ----------
    seed_value : int
        The seed value to set for the random number generators. Any integer can be used.
        
    Notes:
    ------
    - It is essential to call this function before starting the training loop or any random operations.
    - For multi-GPU setups, `torch.cuda.manual_seed_all()` ensures reproducibility across all GPUs.
    - Setting `torch.backends.cudnn.deterministic = True` forces deterministic algorithms for GPU operations.
    - Setting `torch.backends.cudnn.benchmark = False` ensures reproducibility by avoiding non-deterministic algorithms for performance.

    Example:
    --------
    To set the seed to `42` for reproducibility:
    
    set_seed(42)
    
    This will ensure that all random number generators used in the experiment (Python, NumPy, and PyTorch)
    will produce the same sequence of numbers, leading to identical results on each run.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Optional for reproducibility

set_seed(42)  # Set any fixed seed value for reproducibility





# Load Configuration
yaml_path = "./config_files/efficentNetB7_B0_kana.yaml"
config = load_yaml(file_path=yaml_path)

# Check if config is loaded
if config:
    print("DATASET_PATH:", config.get("dataset_path"))
    print("CATEGORIES:", config.get("categories"))
    print("STUDENT_NETWORK_ARCHITECTURE:", config.get("student_network_architecture"))
    print("TEACHER_NETWORK_ARCHITECTURE:", config.get("teacher_network_architecture"))
    print("INPUT_IMG_SIZE:", config.get("input_img_size"))
    print("LEARNING_RATE:", config.get("lr"))
    print("MIN_LR:", config.get("end_lr"))
    print("BATCH_SIZE:", config.get("batch_size"))
    print("NUM_WORKERS:", config.get("num_workers"))
    print("EPOCHS:", config.get("epochs"))
    print("EARLY_STOPPING_PATIENCE:", config.get("early_stopping_patience"))
    print("PROJECT_BASE_DIR:", config.get("project_base_dir"))
    print("PROJECT_NAME:", config.get("project_name"))
    if config.get("teacher_student_distillation"):
        print("TEMPERATURE:", config.get("temperature"))
        print("ALPHA:", config.get("alpha"))
        print("TEACHER_PRE_TRAINED_MODEL:", config.get("teacher_pre_trained_model"))
    
    

else:
    print("Problem loading cofigfiles")
    sys.exit()


# 🚀 **CONFIGURATIONS**
DATASET_PATH = config.get("dataset_path")
CATEGORIES = config.get("categories")
STUDENT_NETWORK_ARCHITECTURE = config.get("student_network_architecture")
TEACHER_NETWORK_ARCHITECTURE = config.get("teacher_network_architecture")
TEACHER_PRE_TRAINED_MODEL = config.get("teacher_pre_trained_model")

INPUT_IMG_SIZE = config.get("input_img_size")
LEARNING_RATE = config.get("lr")
MIN_LEARNING_RATE = config.get("end_lr")
BATCH_SIZE = config.get("batch_size")
NUM_WORKERS = config.get("num_workers")
EPOCHS = config.get("epochs")
EARLY_STOPPING_PATIENCE = config.get("early_stopping_patience")
PROJECT_BASE_DIR_NAME = config.get("project_base_dir")
PROJECT_NAME = config.get("project_name")
if config.get("teacher_student_distillation"):
    TEMPERATURE = config.get("temperature")
    ALPHA = config.get("alpha")


# Create Project directory and get absolute path
PJ_BASE_DIR_PATH = os.path.abspath(create_directory(PROJECT_BASE_DIR_NAME))
CURRENT_PJ_DIR_PATH = create_unique_directory(os.path.join(PJ_BASE_DIR_PATH,PROJECT_NAME))
WEIGHTS_DIR = create_directory(os.path.join(CURRENT_PJ_DIR_PATH,"weights"))



# 🚀 **Knowledge Distillation Loss**
def knowledge_distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    soft_targets = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean"
    ) * (temperature ** 2)

    hard_targets = F.cross_entropy(student_logits, labels)

    return alpha * soft_targets + (1 - alpha) * hard_targets

# 🚀 **AUTO DISTILLATION TO B0 WITH KD LOSS**
def distill_to_b0(dataset_path, category, batch_size, min_learning_rate,train_transform, test_transform, current_pj_dir_path):
    print(f"🔹 Distilling Started...")

    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Initialize the data loader
    train_loader, val_loader, all_labels = DatasetLoader(
        dataset_path=dataset_path,
        category=category,
        batch_size=batch_size,
        train_transform=train_transform,
        test_transform=test_transform,
        current_pj_path=current_pj_dir_path
    ).load()

    num_classes = len(train_loader.dataset.classes)


    # Load trained B7 teacher model
    teacher_model = create_model(
        model_name= TEACHER_NETWORK_ARCHITECTURE, 
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.3  # Applies 30% dropout to reduce overfitting

    )
    
    # Customize the network
    teacher_model = CustomEfficientNet(teacher_model)
    teacher_model.load_state_dict(
        torch.load(TEACHER_PRE_TRAINED_MODEL) 
    )
    teacher_model.to(device)
    teacher_model.eval()

    
    # Initialize B0 student model
    student_model = create_model(
        model_name=STUDENT_NETWORK_ARCHITECTURE, 
        pretrained=True, 
        num_classes=num_classes, 
        drop_rate=0.3
    )
    student_model = CustomEfficientNet(student_model)
    student_model.to(device)

    optimizer = torch.optim.Adam(
        student_model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=1e-4
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,   # Reduce by 50% instead of 90% (0.1) to slow down reduction
        patience=3,    # Increase patience to allow more epochs before reducing
        threshold=0.01, # Set a threshold to ensure reduction happens only when necessary 
        min_lr= min_learning_rate,
        verbose=True,
    )

     # Initialize the plotter
    plotter = TrainingPlotter()
    # Initilize best model
    best_model = copy.deepcopy(student_model)
    best_loss = np.inf
    train_losses, val_losses, val_accuracies = [], [], []
    epochs_metrics, precisions_metrics, recalls_metrics, f1_scores_metrics = [], [], [], []
    precision_teachers, recall_teachers, f1_teachers = [], [], [] 
    lrs = []

    print("Training started....")
    patience_counter = 0

    for epoch in range(1,EPOCHS+1):
        start_time = time.time()  # Track epoch start time
        student_model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_teacher_preds = []

        progress_bar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch}/{EPOCHS}", leave=True)
        # for images, labels in train_loader:
        for _, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Resets gradients before backprop

            with torch.no_grad():
                teacher_outputs = teacher_model(images)
                _, preds_teacher = torch.max(teacher_outputs,1)

            student_outputs = student_model(images)
            _, preds = torch.max(student_outputs,1)

            loss = knowledge_distillation_loss(
                student_outputs, 
                teacher_outputs, 
                labels,
                temperature=TEMPERATURE, 
                alpha=ALPHA
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_teacher_preds.extend(preds_teacher.cpu().numpy())

            progress_bar.set_postfix(loss=f"{loss.item():.5f}")


        # Compute metrics
        precision, recall, f1 = compute_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            num_classes=num_classes
        )

        precision_teacher, recall_teacher, f1_teacher = compute_metrics(
            y_true=all_labels,
            y_pred=all_teacher_preds,
            num_classes=num_classes
        )

        # Store metrics
        epochs_metrics.append(epoch)
        precisions_metrics.append(precision)
        recalls_metrics.append(recall)
        f1_scores_metrics.append(f1)

        precision_teachers.append(precision_teacher)
        recall_teachers.append(recall_teacher)
        f1_teachers.append(f1_teacher)


        avg_train_loss = total_loss / len(train_loader)
        val_accuracy, val_loss = ModelValidator(
            model=student_model,
            criterion=criterion,
            device=device).validate(val_loader=val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)


        scheduler.step(val_loss)  # Adjust learning rate based on validation loss

        for param_group in optimizer.param_groups:
            print(f"Learning Rate: {param_group['lr']}")
            current_lr = param_group['lr']
            lrs.append(current_lr)

        # Calculate elapsed time & ETA
        elapsed_time = time.time() - start_time
        remaining_epochs = EPOCHS- (epoch)
        eta_seconds = elapsed_time * remaining_epochs
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

        print(f"Epoch {epoch}/{EPOCHS} - Lr: {current_lr} - Train Loss: {avg_train_loss:.5f} - Val Loss: {val_loss:.5f} - Val Acc: {val_accuracy:.2f}% - f1: {f1:.4f} - precision: {precision:.4f} - recall: {recall:.4f} - ETA: {eta_str}")


        # print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.2f}%")
        log_to_csv_distillation(
            output_path=current_pj_dir_path,
            epoch= epoch,
            lr = current_lr,
            train_loss= avg_train_loss,
            val_loss= val_loss,
            val_acc= val_accuracy,
            precision_teacher=precision_teacher,
            recall_teacher=recall_teacher,
            f1_teacher=f1_teacher,
            precision_student=precision,
            recall_student=recall,
            f1_student=f1)  # Save to CSV
        
        # Save last model
        torch.save(student_model.state_dict(), os.path.join(WEIGHTS_DIR,"last.pth"))


        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model = copy.deepcopy(student_model)
            torch.save(student_model.state_dict(), os.path.join(WEIGHTS_DIR,"best.pth"))
            # torch.save(student_model.state_dict(), f"B0_Distillation_{category}.pth")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered for {category} at epoch {epoch}.")
            break
    
    print(f"✅ Saved {STUDENT_NETWORK_ARCHITECTURE} model for {category}")

    # Update the plot at the end of training
    plotter.update_plot(train_losses=train_losses,
                        val_losses=val_losses)
    
    # Save the plot to a file (e.g., "training_plot.png")
    plotter.save_plot(os.path.join(current_pj_dir_path,"results.png"))
    plot_learning_rate(save_path=current_pj_dir_path, lr_values=lrs)

    plot_metrics(
            output_path=current_pj_dir_path,
            epochs_metrics=epochs_metrics,
            precisions_metrics=precisions_metrics,
            recalls_metrics=recalls_metrics,
            f1_scores_metrics=f1_scores_metrics
    )

    plot_metrics(
            output_path=current_pj_dir_path,
            epochs_metrics=epochs_metrics,
            precisions_metrics=precision_teachers,
            recalls_metrics=recall_teachers,
            f1_scores_metrics=f1_teachers,
            teacher_model=True
    )
    return best_model





# 🚀 **RUN DISTILLATION**
for category in CATEGORIES:
    # student_best_model = distill_to_b0(category)
    student_best_model = distill_to_b0(
        dataset_path= DATASET_PATH, 
        category= category, 
        batch_size = BATCH_SIZE, 
        min_learning_rate = MIN_LEARNING_RATE,
        train_transform = ImageTrainTransform(INPUT_IMG_SIZE), 
        test_transform= ImageTestTransform(INPUT_IMG_SIZE), 
        current_pj_dir_path= CURRENT_PJ_DIR_PATH
    )

    criterion = torch.nn.CrossEntropyLoss()

    data_type = "valid"
    valid_loader = TestDatasetLoader(
        dataset_path=os.path.join(DATASET_PATH,category),
        data_category=data_type,
        batch_size=BATCH_SIZE,
        transform=ImageTestTransform(INPUT_IMG_SIZE)).load()


    plot_confusion_matrix(
        best_model=student_best_model, 
        test_loader= valid_loader, 
        criterion = criterion, 
        output_dir=CURRENT_PJ_DIR_PATH, 
        data_type=data_type
    )
    
    data_type = "test"
    test_loader = TestDatasetLoader(
        dataset_path=os.path.join(DATASET_PATH,category),
        data_category=data_type,
        batch_size=BATCH_SIZE,
        transform=ImageTestTransform(INPUT_IMG_SIZE)).load()


    plot_confusion_matrix(
        best_model=student_best_model, 
        test_loader= test_loader, 
        criterion = criterion, 
        output_dir=CURRENT_PJ_DIR_PATH, 
        data_type=data_type
    )


    data_type = "test_real"
    if not os.path.exists(os.path.join(DATASET_PATH,category,data_type)):
        print("Path does not exist: ",os.path.join(DATASET_PATH,category,data_type))
        exit()

    test_real_loader = TestDatasetLoader(
        dataset_path=os.path.join(DATASET_PATH,category),
        data_category=data_type,
        batch_size=BATCH_SIZE,
        transform=ImageTestTransform(INPUT_IMG_SIZE)).load()  # Call load() to get the DataLoader


    plot_confusion_matrix(
        best_model=student_best_model, 
        test_loader= test_real_loader, 
        criterion = criterion, 
        output_dir=CURRENT_PJ_DIR_PATH, 
        data_type=data_type
    )
    
    
    
