import os
import csv

def log_to_csv(output_path, category, epoch, lr, train_loss, val_loss, val_acc, precision, recall, f1, model_type="B0"):
    """
    Logs training and validation metrics to a CSV file.

    Args:
        output_path (str): Directory path where the CSV file will be saved.
        category (str): Category for the log file name (e.g., 'train' or 'test').
        epoch (int): Current epoch number.
        lr (float): Current lr.
        train_loss (float): Training loss for the current epoch.
        val_loss (float): Validation loss for the current epoch.
        val_acc (float): Validation accuracy for the current epoch.
        precision (float): Precision score for the validation set.
        recall (float): Recall score for the validation set.
        f1 (float): F1 score for the validation set.
        model_type (str, optional): Model type identifier (default is "B0"), used in the CSV file name.

    Returns:
        None
    """
    # Create the CSV file path
    csv_file = os.path.join(output_path, f"metrics_{category}_{model_type}.csv")

    # Check if the file already exists to avoid overwriting the header
    file_exists = os.path.isfile(csv_file)

    # Open the CSV file in append mode
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)

        # If the file doesn't exist, write the header row
        if not file_exists:
            writer.writerow(["Epoch","Learning Rate", "Train Loss", "Validation Loss", "Validation Accuracy", "Precision", "Recall", "F1"])

        # Write the current epoch's metrics
        writer.writerow([epoch, lr, train_loss, val_loss, val_acc, precision, recall, f1])

    # print(f"Logged epoch {epoch} to {csv_file}")





def log_to_csv_distillation(output_path, epoch, lr, train_loss, val_loss, val_acc, precision_teacher, recall_teacher, f1_teacher, precision_student, recall_student, f1_student):
    """
    Logs training and validation metrics to a CSV file.

    """
    # Create the CSV file path
    csv_file = os.path.join(output_path, f"metrics.csv")

    # Check if the file already exists to avoid overwriting the header
    file_exists = os.path.isfile(csv_file)

    # Open the CSV file in append mode
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)

        # If the file doesn't exist, write the header row
        if not file_exists:
            writer.writerow(["Epoch","Learning Rate", "Distillaiton Train Loss", "Distillaiton Validation Loss", "Distillaiton Validation Accuracy", "Student P", "Student R", "Student F1", "Teacher P", "Teacher R", "Teacher F1"])

        # Write the current epoch's metrics
        writer.writerow([epoch, lr, train_loss, val_loss, val_acc, precision_student, recall_student, f1_student, precision_teacher, recall_teacher, f1_teacher])

    # print(f"Logged epoch {epoch} to {csv_file}")
