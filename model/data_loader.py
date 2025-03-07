
import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.custom_augmentation import HSVTransform
from model.data_plotting import generate_eda_plot

class ImageTransform:
    def __init__(self, input_size):
        """
        Initializes the image transformation pipeline.

        Args:
            input_size (int): The size to which the image should be resized.
        """
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),  # Keep a consistent size
            transforms.RandomRotation(degrees=10),  # Allow small rotation variations
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random shifts & scaling
            transforms.RandomInvert(p=0.2),  # Occasionally invert colors (helps with OCR variations)
            transforms.ColorJitter(contrast=0.2, brightness=0.2),  # Adjust brightness & contrast
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),  # Add slight blur for robustness
            HSVTransform(h_gain=0.015, s_gain=0.7, v_gain=0.4),  # Apply HSV augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.Normalize(mean=[0.5], std=[0.5])  # Uncomment for grayscale images
        ])

    def __call__(self, image):
        """
        Applies the transformation pipeline to an image.

        Args:
            image (PIL.Image or Tensor): The input image.

        Returns:
            Tensor: The transformed image.
        """
        return self.transform(image)




class DatasetLoader:
    """
    Class to load training and validation datasets.
    """
    def __init__(self, dataset_path, category, batch_size, transform, current_pj_path):
        self.dataset_path = dataset_path
        self.category = category
        self.batch_size = batch_size
        self.transform = transform
        self.current_pj_path = current_pj_path

    def load(self):
        train_path = os.path.join(self.dataset_path, self.category, "train")
        val_path = os.path.join(self.dataset_path, self.category, "valid")

        train_dataset = ImageFolder(train_path, transform=self.transform)
        val_dataset = ImageFolder(val_path, transform=self.transform)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

        # Get the class names from the dataset
        all_labels = train_dataset.classes
        print("Train Classes:", all_labels)

        # Generate and Save EDA Plot
        self.generate_eda_plot(train_dataset, val_dataset)

        return train_loader, val_loader, all_labels

    def generate_eda_plot(self, train_dataset, val_dataset):
        """
        Generates and saves an EDA (Exploratory Data Analysis) plot.
        """
        generate_eda_plot(
            output_path=self.current_pj_path,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            category=self.category
        )


class ModelValidator:
    """
    Class to validate a trained model on a validation dataset.
    """
    def __init__(self, model, criterion, device=None):
        self.model = model
        self.criterion = criterion
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def validate(self, val_loader):
        """
        Validates the model on the validation dataset.

        Args:
            val_loader (DataLoader): DataLoader for validation data.

        Returns:
            Tuple: (accuracy, avg_loss)
        """
        self.model.eval()
        correct, total, total_loss = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(val_loader)
        return accuracy, avg_loss







class TestDatasetLoader:
    def __init__(self, dataset_path, data_category="test", batch_size=8, transform=None):
        """
        Initializes the test dataset loader.

        Args:
            dataset_path (str): Path to the dataset.
            data_category (str): Category folder name (default: "test").
            batch_size (int): Number of samples per batch (default: 8).
            transform (callable, optional): Transformation to apply to test images.
        """
        self.dataset_path = dataset_path
        self.data_category = data_category
        self.batch_size = batch_size
        self.transform = transform if transform else transforms.ToTensor()  # Default to tensor transform

    def load(self):
        """
        Loads the test dataset as a DataLoader.

        Returns:
            DataLoader: The test dataset loader.
        """
        test_path = os.path.join(self.dataset_path, self.data_category)
        test_dataset = ImageFolder(test_path, transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return test_loader
