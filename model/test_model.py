import torch
import torchvision.transforms as transforms
import os
import numpy as np
from timm import create_model

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_test_dataset

# 🚀 **CONFIGURATIONS**
DATASET_PATH = "/home/shreeram/workspace/ambl/custom_efficent_autodistillation/Dataset"
CATEGORY = "Rock Paper Scissors.v1-raw-300x300.folder"  # Change to the test category
CATEGORY = "dataset_2_kana_update2"
BATCH_SIZE = 8

# 🚀 **DATA TRANSFORMS**
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_loader = load_test_dataset(dataset_path=os.path.join(DATASET_PATH,CATEGORY),
                                data_category="test",
                                batch_size=16,
                                transform=transform
                                )

# 🚀 **VALIDATE MODEL**
def validate_model(model, test_loader, criterion):
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


# Load the trained model weights
model = create_model(
    "efficientnet_b0", 
    pretrained=False,  # Do not load pretrained weights since we have our own
    num_classes=len(os.listdir(os.path.join(DATASET_PATH, CATEGORY, "test"))),  # Match the number of classes
    drop_rate=0.3
)
model.load_state_dict(torch.load(f"./B0_dataset_2_kana_update2resources/B0_{CATEGORY}.pth"))

# model.load_state_dict(torch.load(f"./B0_distillation_dataset_2_kana_update2resources/B0_Distillation_{CATEGORY}.pth"))
model.eval()  # Set the model to evaluation mode

# Load the test dataset
test_loader = load_test_dataset(CATEGORY, BATCH_SIZE)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

# Validate the model on test data
test_accuracy, test_loss, test_cm = validate_model(model, test_loader, criterion)

# Output results
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Plot the confusion matrix
# Set a font that supports Japanese characters
plt.rcParams["font.family"] = "Noto Sans CJK JP"  # Alternative: "IPAexGothic" or "Yu Gothic"

plt.figure(figsize=(10, 7))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_loader.dataset.classes, yticklabels=test_loader.dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
# plt.savefig("confusion_matrix_b0_only")
plt.savefig("confusion_matrix_b0_distilled")
plt.close()
# plt.show()
