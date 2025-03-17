import torch
import timm
import time
import copy
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.log_files import log_to_csv
from utils.compute_metrices import compute_metrics, plot_metrics
from model.data_plotting import save_batch_collage_with_labels, plot_learning_rate
from model.data_loader import DatasetLoader, ModelValidator

import torch.nn as nn


class TrainingPlotter:
    def __init__(self):
        # Create a figure with two subplots (1 row, 2 columns)
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Configure Training Loss plot (left)
        self.ax1.set_title("Training Loss")
        self.ax1.set_xlabel("Epochs")
        self.ax1.set_ylabel("Loss")
        self.ax1.grid(True)

        # Configure Validation Loss plot (right)
        self.ax2.set_title("Validation Loss")
        self.ax2.set_xlabel("Epochs")
        self.ax2.set_ylabel("Loss")
        self.ax2.grid(True)

    def update_plot(self, train_losses, val_losses):
        """
        Updates the two subplots separately:
        - Left: Training Loss
        - Right: Validation Loss
        """
        self.ax1.clear()
        self.ax2.clear()

        # Reapply titles and labels after clearing
        self.ax1.set_title("Training Loss")
        self.ax1.set_xlabel("Epochs")
        self.ax1.set_ylabel("Loss")
        self.ax1.grid(True)

        self.ax2.set_title("Validation Loss")
        self.ax2.set_xlabel("Epochs")
        self.ax2.set_ylabel("Loss")
        self.ax2.grid(True)

        # Plot the losses separately
        self.ax1.plot(train_losses, label="Train Loss", color='blue', linestyle='dashed')
        self.ax2.plot(val_losses, label="Validation Loss", color='red')

        self.ax1.legend()
        self.ax2.legend()

        # Adjust layout and update plot
        self.fig.tight_layout()
        self.fig.canvas.draw()

    def save_plot(self, filename):
        """Save the plot to a file."""
        self.fig.savefig(filename)
        

class CustomEfficientNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        # Correct attribute for timm models
        num_features = base_model.classifier.in_features
        # self.bn = nn.BatchNorm1d(num_features=base_model.fc5.in_features)  
        self.bn = nn.BatchNorm1d(num_features=num_features)  
        # self.fc5 = base_model.fc5
        self.fc = nn.Linear(num_features, base_model.classifier.out_features)  # Replacing fc5
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.base_model.extract_features(x)
        x = self.base_model.forward_features(x)  # timm uses forward_features instead of extract_features
        x = x.mean([2, 3])  # Global Average Pooling for final feature map
        x = self.bn(x)
        x = self.fc(x)
        x = self.relu(x)
        return x




class CustomModelTrainer:
    def __init__(self, dataset_path, category, batch_size, train_transform, test_transform, network_architecture,
                 learning_rate, min_learning_rate, epochs, early_stopping_patience, weights_dir, current_pj_dir_path):
        self.dataset_path = dataset_path
        self.category = category
        self.batch_size = batch_size
        self.train_transform= train_transform
        self.test_transform = test_transform
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.weights_dir = weights_dir
        self.current_pj_dir_path = current_pj_dir_path


    def train(self):
        print(f"🔹 Training {self.network_architecture} for {self.category}...")

        # Initialize the data loader
        train_loader, val_loader, all_labels = DatasetLoader(
            dataset_path=self.dataset_path,
            category=self.category,
            batch_size=self.batch_size,
            train_transform=self.train_transform,
            test_transform=self.test_transform,
            current_pj_path=self.current_pj_dir_path
        ).load()

        save_batch_collage_with_labels(
            data_loader=train_loader,
            save_path=os.path.join(self.current_pj_dir_path, "batch_preview.png"),
            all_labels = all_labels,
        )

        num_classes = len(train_loader.dataset.classes)
        print(f"🔹 Train Loader and Val Loader are completed...")
        
        bool_value = False if self.network_architecture == "efficientnet_b7" else True

        # Initialize model
        load_model = timm.create_model(
            model_name=self.network_architecture,
            pretrained=bool_value,
            num_classes=num_classes,
            drop_rate=0.3
        )

        model = CustomEfficientNet(load_model)

        print("Network output_features",model.base_model.classifier.out_features)

        print(f"🔹 Training Model initialization completed...")
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Initialize the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,   # Reduce by 50% instead of 90% (0.1) to slow down reduction
            patience=3,    # Increase patience to allow more epochs before reducing
            threshold=0.01, # Set a threshold to ensure reduction happens only when necessary 
            min_lr= self.min_learning_rate,
            verbose=True,
        )

        # Initialize the plotter
        plotter = TrainingPlotter()
        # Initilize best model
        best_model = copy.deepcopy(model)
        best_loss = np.inf
        train_losses, val_losses, val_accuracies = [], [], []
        epochs_metrics, precisions_metrics, recalls_metrics, f1_scores_metrics = [], [], [], []
        lrs = []

        


        print("Training started....")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()  # Track epoch start time
            model.train()
            total_loss = 0
            all_preds = []
            all_labels = []

            progress_bar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch}/{self.epochs}", leave=True)
            for _, (images, labels) in progress_bar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix(loss=f"{loss.item():.5f}")

            # Compute metrics
            precision, recall, f1 = compute_metrics(
                y_true=all_labels,
                y_pred=all_preds,
                num_classes=num_classes
            )

            # Store metrics
            epochs_metrics.append(epoch)
            precisions_metrics.append(precision)
            recalls_metrics.append(recall)
            f1_scores_metrics.append(f1)

            avg_train_loss = total_loss / len(train_loader)
            val_accuracy, val_loss = ModelValidator(
                model=model,
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

            elapsed_time = time.time() - start_time
            remaining_epochs = self.epochs - epoch
            eta_seconds = elapsed_time * remaining_epochs
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

            print(f"Epoch {epoch}/{self.epochs} - Lr: {current_lr} - Train Loss: {avg_train_loss:.5f} - Val Loss: {val_loss:.5f} - Val Acc: {val_accuracy:.2f}% - f1: {f1:.4f} - precision: {precision:.4f} - recall: {recall:.4f} - ETA: {eta_str}")

            log_to_csv(
                output_path=self.current_pj_dir_path,
                category=self.category,
                epoch=epoch,
                lr=current_lr,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                val_acc=val_accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                model_type=self.network_architecture
            )  # Save to CSV

            torch.save(model.state_dict(), os.path.join(self.weights_dir, "last.pth"))

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(), os.path.join(self.weights_dir, "best.pth"))
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered for {self.category} at epoch {epoch}.")
                break

        print(f"✅ Saved {self.network_architecture} model for {self.category}")


        # Update the plot at the end of training
        plotter.update_plot(train_losses, val_losses)
        
        # Save the plot to a file (e.g., "training_plot.png")
        plotter.save_plot(os.path.join(self.current_pj_dir_path,"results.png"))
        plot_learning_rate(save_path=self.current_pj_dir_path, lr_values=lrs)

        plot_metrics(
            output_path=self.current_pj_dir_path,
            epochs_metrics=epochs_metrics,
            precisions_metrics=precisions_metrics,
            recalls_metrics=recalls_metrics,
            f1_scores_metrics=f1_scores_metrics
        )

        return best_model
