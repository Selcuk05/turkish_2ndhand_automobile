import torch
import mlflow
from tqdm import tqdm
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CyclicLR
from torch.optim import AdamW

from src.classifier.data import create_dataloaders
from src.classifier.model import HierarchicalCNN, HierarchicalLoss


def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    loss_components = {"brand_loss": 0, "series_loss": 0, "model_loss": 0}

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Move data to device
        images = batch["image"].to(device)
        targets = {
            "brand": batch["brand"].to(device),
            "series": batch["series"].to(device),
            "model": batch["model"].to(device),
        }

        # Forward pass with mixed precision
        with autocast():
            outputs = model(images)
            loss, component_losses = criterion(outputs, targets)

        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        total_loss += loss.item()
        for k, v in component_losses.items():
            loss_components[k] += v

        pbar.set_postfix({"loss": loss.item()})

    # Calculate average losses
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}

    return avg_loss, avg_components


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    loss_components = {"brand_loss": 0, "series_loss": 0, "model_loss": 0}
    correct = {"brand": 0, "series": 0, "model": 0}
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch["image"].to(device)
            targets = {
                "brand": batch["brand"].to(device),
                "series": batch["series"].to(device),
                "model": batch["model"].to(device),
            }

            outputs = model(images)
            loss, component_losses = criterion(outputs, targets)

            # Update metrics
            total_loss += loss.item()
            for k, v in component_losses.items():
                loss_components[k] += v

            # Calculate accuracies
            for k in ["brand", "series", "model"]:
                pred = outputs[k].argmax(dim=1)
                correct[k] += (pred == targets[k]).sum().item()
            total += images.size(0)

    # Calculate averages
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    accuracies = {k: v / total for k, v in correct.items()}

    return avg_loss, avg_components, accuracies


def train(
    data_df,
    image_dir,
    num_epochs=100,
    batch_size=32,
    learning_rate=3e-4,
    weight_decay=1e-4,
    device="cuda",
    num_workers=4,
    experiment_name="car_classifier",
):
    # Initialize MLflow
    mlflow.set_experiment(experiment_name)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_df, image_dir, batch_size, num_workers
    )

    # Initialize model and move to device
    num_brands = len(train_loader.dataset.brand_encoder.classes_)
    num_series = len(train_loader.dataset.series_encoder.classes_)
    num_models = len(train_loader.dataset.model_encoder.classes_)

    model = HierarchicalCNN(num_brands, num_series, num_models)
    model = model.to(device)

    # Initialize optimizer and loss
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = HierarchicalLoss()
    scaler = GradScaler()

    # Initialize learning rate scheduler
    scheduler = CyclicLR(
        optimizer, base_lr=1e-5, max_lr=5e-4, step_size_up=2000, mode="triangular2"
    )

    # Initialize early stopping
    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(
            {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            }
        )

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Training phase
            train_loss, train_components = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device
            )

            # Validation phase
            val_loss, val_components, accuracies = validate(
                model, val_loader, criterion, device
            )

            # Update learning rate
            scheduler.step()

            # Log metrics
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"train_{k}": v for k, v in train_components.items()},
                **{f"val_{k}": v for k, v in val_components.items()},
                **{f"accuracy_{k}": v for k, v in accuracies.items()},
            }
            mlflow.log_metrics(metrics, step=epoch)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                mlflow.pytorch.log_model(model, "model")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

    return model


if __name__ == "__main__":
    # Example usage
    data_df = pd.read_csv("data/turkish_2ndhand_automobile_processed.csv")
    image_dir = "data/processed_images"

    model = train(
        data_df=data_df,
        image_dir=image_dir,
        num_epochs=100,
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
