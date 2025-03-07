import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from matplotlib.ticker import FormatStrFormatter

# Function to compute metrics
def compute_metrics(y_true, y_pred, num_classes=142):
    # Average = macro for balanced data set but weighted is best for imballance dataset
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return precision, recall, f1



# Function to save metrics plots
def plot_metrics(output_path, epochs_metrics, precisions_metrics, recalls_metrics, f1_scores_metrics):
    """
    Saves the Precision, Recall, and F1 Score plot as an image file (both combined and separate graphs).

    Args:
        output_path (str): Directory where the plots will be saved.
        epochs_metrics (list): List of epoch numbers.
        precisions_metrics (list): List of precision values.
        recalls_metrics (list): List of recall values.
        f1_scores_metrics (list): List of F1 score values.

    Returns:
        None
    """
    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists

    # Combined Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_metrics, precisions_metrics, label="Precision", marker="o", linestyle="-")
    plt.plot(epochs_metrics, recalls_metrics, label="Recall", marker="s", linestyle="-")
    plt.plot(epochs_metrics, f1_scores_metrics, label="F1 Score", marker="d", linestyle="-")

    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("F1 Score, Precision, and Recall Over Epochs")
    plt.legend()
    # plt.grid()
    # Format y-axis ticks to show 6 decimal places
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
    plt.savefig(os.path.join(output_path, "metrics_combined.png"), bbox_inches="tight")
    plt.close()  # Close figure

    # Individual plots
    metrics = {
        "precision": precisions_metrics,
        "recall": recalls_metrics,
        "f1_score": f1_scores_metrics
    }

    for metric_name, values in metrics.items():
        plt.figure(figsize=(8, 4))
        plt.plot(epochs_metrics, values, marker="o", linestyle="-", label=metric_name.capitalize())

        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.title(f"{metric_name.capitalize()} Over Epochs")
        plt.legend()
        # plt.grid()
        # Format y-axis ticks for individual plots
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
        plt.savefig(os.path.join(output_path, f"{metric_name}.png"), bbox_inches="tight")
        plt.close()  # Close figure

    print(f"Plots saved in: {output_path}")