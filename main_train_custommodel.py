import os
import sys
import random

import torch
import timm
import numpy as np

from utils.yaml_utils import load_yaml
from utils.general_utils import create_directory, create_unique_directory

from model.data_plotting import plot_confusion_matrix
from model.data_loader import ImageTrainTransform,ImageTestTransform, TestDatasetLoader
from model.trainner_custefficent import CustomModelTrainer

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
yaml_path = "./config_files/efficentNetB0_Test.yaml"
# yaml_path = "./config_files/efficentNetB0.yaml"
# yaml_path="./config_files/efficentNetB0_kana_params.yaml"
config = load_yaml(file_path=yaml_path)

# Check if config is loaded
if config:
    print("DATASET_PATH:", config.get("dataset_path"))
    print("CATEGORIES:", config.get("categories"))
    print("NETWORK_ARCHITECTURE:", config.get("network_architecture"))
    print("INPUT_IMG_SIZE:", config.get("input_img_size"))
    print("LR:", config.get("lr"))
    print("MIN_LR:", config.get("lr_end"))
    print("BATCH_SIZE:", config.get("batch_size"))
    print("NUM_WORKERS:", config.get("num_workers"))
    print("EPOCHS:", config.get("epochs"))
    print("EARLY_STOPPING_PATIENCE:", config.get("early_stopping_patience"))
    print("PROJECT_BASE_DIR:", config.get("project_base_dir"))
    print("PROJECT_NAME:", config.get("project_name"))
    if config.get("teacher_student_distillation"):
        print("TEMPERATURE:", config.get("temperature"))
        print("ALPHA:", config.get("alpha"))

else:
    print("Problem loading cofigfiles")
    sys.exit()


# 🚀 **CONFIGURATIONS**
DATASET_PATH = config.get("dataset_path")
CATEGORIES = config.get("categories")
NETWORK_ARCHITECTURE = config.get("network_architecture")
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





# 🚀 **TRAIN & DISTILL ALL CATEGORIES**
for category in CATEGORIES:
    customModelTrainer = CustomModelTrainer(
        dataset_path=DATASET_PATH,
        category=category,
        batch_size=BATCH_SIZE,
        train_transform=ImageTrainTransform(INPUT_IMG_SIZE),
        test_transform=ImageTestTransform(INPUT_IMG_SIZE),
        network_architecture=NETWORK_ARCHITECTURE,
        learning_rate=LEARNING_RATE,
        min_learning_rate= MIN_LEARNING_RATE,
        epochs=EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        current_pj_dir_path=CURRENT_PJ_DIR_PATH,
        weights_dir= WEIGHTS_DIR,
    )
    best_model= customModelTrainer.train()

    criterion = torch.nn.CrossEntropyLoss()

    data_type = "valid"
    test_loader = TestDatasetLoader(
        dataset_path=os.path.join(DATASET_PATH,category),
        data_category=data_type,
        batch_size=BATCH_SIZE,
        transform=ImageTestTransform(INPUT_IMG_SIZE)).load()  # Call load() to get the DataLoader


    plot_confusion_matrix(
        best_model=best_model, 
        test_loader= test_loader, 
        criterion = criterion, 
        output_dir=CURRENT_PJ_DIR_PATH, 
        data_type=data_type
    )
    
    data_type = "test"
    test_loader = TestDatasetLoader(
        dataset_path=os.path.join(DATASET_PATH,category),
        data_category=data_type,
        batch_size=BATCH_SIZE,
        transform=ImageTestTransform(INPUT_IMG_SIZE)).load()  # Call load() to get the DataLoader


    plot_confusion_matrix(
        best_model=best_model, 
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
        best_model=best_model, 
        test_loader= test_real_loader, 
        criterion = criterion, 
        output_dir=CURRENT_PJ_DIR_PATH, 
        data_type=data_type
    )
    
    