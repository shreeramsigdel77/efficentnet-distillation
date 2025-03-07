import os
import matplotlib.pyplot as plt

from collections import Counter


# 📊 **Initialize Live Plotting**
def init_plot():
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # Two subplots (Loss, Accuracy)
    return fig, ax



# 📊 **Update Live Plot**
def update_plot(fig, ax, train_losses, val_losses, val_accuracies, category):
    ax[0].clear()
    ax[1].clear()

    epochs = range(1, len(train_losses) + 1)

    ax[0].plot(epochs, train_losses, label="Train Loss", marker="o")
    ax[0].plot(epochs, val_losses, label="Validation Loss", marker="o")
    ax[0].set_title(f"Loss for {category}")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(epochs, val_accuracies, label="Validation Accuracy", marker="o", color="green")
    ax[1].set_title(f"Accuracy for {category}")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].legend()

    # plt.pause(0.1)  # Pause to update
    # plt.draw()
    # plt.show(block=False)

