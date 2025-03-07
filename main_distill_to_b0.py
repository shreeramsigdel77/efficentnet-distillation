import torch
import torchvision.transforms as transforms
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from timm import create_model
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F

import sys
import datetime
import time
from tqdm import tqdm
import seaborn as sns


from utils.graph_plotting import init_plot, update_plot,generate_eda_plot
from utils.yaml_utils import load_yaml
from utils.general_utils import create_directory, create_unique_directory
from utils.log_files import log_to_csv
from sklearn.metrics import confusion_matrix
import copy


# # 🚀 **CONFIGURATIONS**
# DATASET_PATH = "/home/shreeram/workspace/ambl/custom_efficent_autodistillation/Dataset"
# # CATEGORIES = ["Rock Paper Scissors.v1-raw-300x300.folder"]
# # CATEGORIES = ["dataset_2_kana_update2"] 
# CATEGORIES = ["dataset_0_region_update2"] 
# # BATCH_SIZE_B7 = 8 customize incase of required
# BATCH_SIZE_B0 = 16
# EPOCHS_B0 = 50
# EARLY_STOPPING_PATIENCE = 10


# LEARNING_RATE = 1e-4
# TEMPERATURE = 3.0  # Temperature for distillation 3.0 Default (try higher temperature 5.0)
# ALPHA = 0.7  # Weighting for soft vs. hard target loss  Default 0.7 (try 0.5)




# Load Configuration
yaml_path = "./config_files/efficentNetB3_B0.yaml"
config = load_yaml(file_path=yaml_path)

# Check if config is loaded
if config:
    print("DATASET_PATH:", config.get("dataset_path"))
    print("CATEGORIES:", config.get("categories"))
    print("STUDENT_NETWORK_ARCHITECTURE:", config.get("student_network_architecture"))
    print("TEACHER_NETWORK_ARCHITECTURE:", config.get("teacher_network_architecture"))
    print("INPUT_IMG_SIZE:", config.get("input_img_size"))
    print("LEARNING_RATE:", config.get("lr"))
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




# 🚀 **DATA TRANSFORMS DEFAULT**
transform = transforms.Compose([
    transforms.Resize((INPUT_IMG_SIZE, INPUT_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Searched for later used 
transform2 = transforms.Compose([
    transforms.Resize((128, 128)),  # Keep a consistent size
    transforms.RandomRotation(degrees=10),  # Allow small rotation variations
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random shifts & scaling
    transforms.RandomInvert(p=0.2),  # Occasionally invert colors (helps with OCR variations)
    transforms.ColorJitter(contrast=0.2, brightness=0.2),  # Adjust brightness & contrast
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),  # Add slight blur for robustness
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale images
])


# 🚀 **LOAD TEST DATASET**
def load_test_dataset(dataset_path, data_category="test", batch_size="8", transform=transform):
    test_path = os.path.join(dataset_path, data_category)  # Use test path for loading dataset
    test_dataset = ImageFolder(test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# 🚀 **LOAD DATASET**
def load_dataset(category, batch_size):
    train_path = os.path.join(DATASET_PATH, category, "train")
    val_path = os.path.join(DATASET_PATH, category, "valid")

    train_dataset = ImageFolder(train_path, transform=transform)
    val_dataset = ImageFolder(val_path, transform=transform)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,  #Loads data in parallel using 4 CPU threads (adjust based on system)
        pin_memory=True, #Speeds up data transfer to GPU.
        persistent_workers=True # Avoids worker restarts after each epoch.
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )


    #  Generate and Save EDA Plot
    generate_eda_plot(
        output_path=CURRENT_PJ_DIR_PATH,
        train_dataset=train_dataset,
        val_dataset= val_dataset,
        category= category
    )


    return train_loader, val_loader



# 🚀 **VALIDATE MODEL_cm**
def validate_model_cm(model, test_loader, criterion):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)  # Move model to appropriate device
    model.eval()

    correct, total, total_loss = 0, 0, 0
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    return accuracy, avg_loss, cm

# 🚀 **VALIDATE MODEL**
def validate_model(model, val_loader, criterion):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader)
    return accuracy, avg_loss

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
def distill_to_b0(category):
    print(f"🔹 Distilling from B3 to B0...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    train_loader, val_loader = load_dataset(category, BATCH_SIZE)


    # Load trained B7 teacher model
    teacher_model = create_model(
        model_name= TEACHER_NETWORK_ARCHITECTURE, 
        num_classes=len(train_loader.dataset.classes),
        drop_rate=0.3  # Applies 30% dropout to reduce overfitting

    )
    
    teacher_model.load_state_dict(
        torch.load(TEACHER_PRE_TRAINED_MODEL) 
    )
    teacher_model.to(device)
    teacher_model.eval()

    
    # Initialize B0 student model
    student_model = create_model(
        model_name=STUDENT_NETWORK_ARCHITECTURE, 
        pretrained=True, 
        num_classes=len(train_loader.dataset.classes), 
        drop_rate=0.3
    )
    student_model.to(device)

    optimizer = torch.optim.Adam(
        student_model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=1e-4
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = np.inf
    patience_counter = 0
    train_losses, val_losses, val_accuracies = [], [], []

    # Initilize best model
    best_model = copy.deepcopy(student_model)
    fig, ax = init_plot()

    for epoch in range(EPOCHS):

        start_time = time.time()  # Track epoch start time
        student_model.train()
        total_loss = 0

        # 🟢 FIX: Use enumerate(train_loader, 1) so tqdm updates correctly
        progress_bar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)



        # for images, labels in train_loader:
        for _, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(images)

            student_outputs = student_model(images)

            loss = knowledge_distillation_loss(
                student_outputs, 
                teacher_outputs, 
                labels,
                temperature=TEMPERATURE, 
                alpha=ALPHA
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_accuracy, val_loss = validate_model(student_model, val_loader, criterion)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Calculate elapsed time & ETA
        elapsed_time = time.time() - start_time
        remaining_epochs = EPOCHS- (epoch + 1)
        eta_seconds = elapsed_time * remaining_epochs
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.5f} - Val Loss: {val_loss:.5f} - Val Acc: {val_accuracy:.2f}% - ETA: {eta_str}")


        # print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.2f}%")
        log_to_csv(
            output_path=CURRENT_PJ_DIR_PATH,
            category=category,
            epoch= epoch + 1,
            train_loss= avg_train_loss,
            val_loss= val_loss,
            val_acc= val_accuracy,
            model_type= STUDENT_NETWORK_ARCHITECTURE)  # 📌 Save to CSV
        

        # update_plot(train_losses, val_losses, val_accuracies, category)

        update_plot(fig, ax, train_losses, val_losses, val_accuracies, category)  # 📊 Update live plot

        # Save last model
        torch.save(student_model.state_dict(), os.path.join(WEIGHTS_DIR,"last.pth"))


        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model = copy.deepcopy(student_model)
            torch.save(student_model.state_dict(), os.path.join(WEIGHTS_DIR,"best_distillation.pth"))
            # torch.save(student_model.state_dict(), f"B0_Distillation_{category}.pth")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"⚠️ Early stopping triggered for {category} at epoch {epoch+1}.")
            break
    # Set a font that supports Japanese characters
    plt.rcParams["font.family"] = "Noto Sans CJK JP"  # Alternative: "IPAexGothic" or "Yu Gothic"

    print(f"✅ Saved {STUDENT_NETWORK_ARCHITECTURE} model for {category}")

    # Set a font that supports Japanese characters
    plt.rcParams["font.family"] = "Noto Sans CJK JP"  # Alternative: "IPAexGothic" or "Yu Gothic"
    plt.savefig(os.path.join(CURRENT_PJ_DIR_PATH,f"result.png"), dpi=300)  # Save plot as PNG
    # plt.savefig(f"training_plot_{category}_Distillation_B0.png")
    plt.close()

    return best_model



def plot_confusion_matrix(best_model, test_loader, criterion, output_dir, data_type):
    test_accuracy, test_loss, test_cm = validate_model_cm(best_model, test_loader, criterion)

    
    # Output results
    print(f"{data_type} Loss: {test_loss:.4f}")
    print(f"{data_type} Accuracy: {test_accuracy:.2f}%")

    # Plot the confusion matrix
    # Set a font that supports Japanese characters
    plt.rcParams["font.family"] = "Noto Sans CJK JP"  # Alternative: "IPAexGothic" or "Yu Gothic"

    plt.figure(figsize=(10, 7))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_loader.dataset.classes, yticklabels=test_loader.dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix \n {data_type} Accuracy: {test_accuracy:.2f}%')
    # plt.savefig("confusion_matrix_b0_only")
    plt.savefig(os.path.join(output_dir,f"confusion_matrix_{data_type}.png"))
    plt.close()


# 🚀 **RUN DISTILLATION**
for category in CATEGORIES:
    student_best_model = distill_to_b0(category)

    criterion = torch.nn.CrossEntropyLoss()

    data_type = "valid"
    test_loader = load_test_dataset(
        dataset_path=os.path.join(DATASET_PATH,category),
        data_category=data_type,
        batch_size=BATCH_SIZE,
        transform=transform)


    plot_confusion_matrix(best_model=student_best_model, test_loader= test_loader, criterion = criterion, output_dir=CURRENT_PJ_DIR_PATH, data_type=data_type)
    
    data_type = "test"
    test_loader = load_test_dataset(
        dataset_path=os.path.join(DATASET_PATH,category),
        data_category=data_type,
        batch_size=BATCH_SIZE,
        transform=transform)


    plot_confusion_matrix(best_model=student_best_model, test_loader= test_loader, criterion = criterion, output_dir=CURRENT_PJ_DIR_PATH, data_type=data_type)
    
    
