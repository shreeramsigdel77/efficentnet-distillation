

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import torch
from collections import Counter

def save_batch_collage_with_labels(data_loader, save_path, all_labels, font_path="NotoSansCJK-Regular.ttc", nrow=8):
    # Get a batch of images from the DataLoader
    batch_temp = next(iter(data_loader))  
    images_temp, labels_temp = batch_temp 

    # Define font (use a Japanese-compatible font)
    try:
        font = ImageFont.truetype(font_path, 20)
    except:
        font = ImageFont.load_default()

    # Convert images to PIL directly from tensors (avoiding ToPILImage if not necessary)
    to_pil = transforms.ToPILImage()

    # If images are normalized, reverse the normalization
    mean = [0.485, 0.456, 0.406]  
    std = [0.229, 0.224, 0.225]    
    denormalize = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])

    # Create individual row images with labels
    rows = []  # To store the rows of images
    for row_start in range(0, len(images_temp), nrow):
        # Get the images and labels for this row
        row_images = images_temp[row_start: row_start + nrow]
        row_labels = labels_temp[row_start: row_start + nrow]

        # Create a new image for the row (width is sum of images, height is image + label)
        row_width = sum([to_pil(denormalize(img)).size[0] for img in row_images])
        row_height = max([to_pil(denormalize(img)).size[1] for img in row_images]) + 40  # 40 pixels for the label space
        row_img = Image.new("RGB", (row_width, row_height), (255, 255, 255))

        # Paste the images in the row and add labels
        x_offset = 0
        for img, label in zip(row_images, row_labels):
            # Denormalize the image
            img = denormalize(img)
            pil_img = to_pil(img).convert("RGB")  # Explicitly convert to RGB
            row_img.paste(pil_img, (x_offset, 0))  # Paste image

            # Ensure 'label_str' holds the correct class name as a string
            label_str = str(all_labels[label.item()]) if all_labels else str(label.item())
            draw = ImageDraw.Draw(row_img)
            
            # Get the bounding box of the text
            text_bbox = draw.textbbox((0, 0), label_str, font=font)
            label_width = text_bbox[2] - text_bbox[0]
            label_height = text_bbox[3] - text_bbox[1]
            
            # Draw label centered below the image
            draw.text(
                (x_offset + (pil_img.size[0] - label_width) // 2, pil_img.size[1]),
                label_str, fill="black", font=font
            )
            x_offset += pil_img.size[0]  # Update x_offset for the next image

        rows.append(row_img)  # Add this row image to the list

    # Combine all rows vertically to create the final collage
    final_width = max([row.size[0] for row in rows])
    final_height = sum([row.size[1] for row in rows])
    final_img = Image.new("RGB", (final_width, final_height), (255, 255, 255))

    # Paste each row into the final image
    y_offset = 0
    for row in rows:
        final_img.paste(row, (0, y_offset))
        y_offset += row.size[1]

    # Save final image with labels
    final_img.save(save_path)

    print(f"Batch collage with labels saved at: {save_path}")


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


def plot_confusion_matrix(best_model, test_loader, criterion, data_type, output_dir):
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

def plot_learning_rate(lr_values, save_path):
    """
    Plots and saves the learning rate schedule over training epochs.

    Args:
        lr_values (list): A list of learning rate values recorded at each epoch.
        save_path (str): The file path to save the plot image (default: 'learning_rate_plot.png').
    """

    if not save_path:
        save_path="learning_rate_plot.png"
    else:
        save_path=os.path.join(save_path,"learning_rate_curve.png")

    epochs = range(1, len(lr_values) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lr_values, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    # plt.grid(True)
    
    # Save the plot as an image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory








def generate_eda_plot(output_path, train_dataset, val_dataset, category):
    """Generates a bar plot showing class distribution in train and validation datasets."""
    
    # Count occurrences of each class index in train & val datasets
    train_counts = Counter(train_dataset.targets)
    val_counts = Counter(val_dataset.targets)

    # Get class names
    class_labels = train_dataset.classes  # List of class names
    
    # Create lists for plotting
    train_class_counts = [train_counts[i] if i in train_counts else 0 for i in range(len(class_labels))]
    val_class_counts = [val_counts[i] if i in val_counts else 0 for i in range(len(class_labels))]

    # Define bar width and positions
    bar_width = 0.4
    x = range(len(class_labels))

    # Plot bar chart

    # Set a font that supports Japanese characters
    plt.rcParams["font.family"] = "Noto Sans CJK JP"  # Alternative: "IPAexGothic" or "Yu Gothic"


    plt.figure(figsize=(10, 5))
    plt.bar(x, train_class_counts, width=bar_width, alpha=0.7, label="Train", color="blue")
    plt.bar([pos + bar_width for pos in x], val_class_counts, width=bar_width, alpha=0.7, label="Validation", color="orange")

    # Formatting
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.title(f"Class Distribution in {category} Dataset")
    plt.xticks([pos + bar_width / 2 for pos in x], class_labels, rotation=45, ha="right")  # Adjust for clarity
    plt.legend()
    plt.tight_layout()

    # Save plot
    save_path = os.path.join(output_path, "class_distribution.png")
    os.makedirs("EDA_Plots", exist_ok=True)  # Ensure directory exists
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Class distribution plot saved at: {save_path}")