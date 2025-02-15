import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import logging
import torch.nn as nn
from torchvision import transforms
import numpy as np
from torch.amp import GradScaler, autocast
import wandb

from dataset import HierarchicalVehicleDataset
from model import DynamicHierarchicalClassifier


class HierarchicalLoss(nn.Module):
    def __init__(self, dataset, label_smoothing=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.dataset = dataset

    def forward(self, outputs, targets, weights=None):
        if weights is None:
            weights = {"brand": 1.0, "model": 0.5, "trim": 0.25}

        batch_size = outputs["brand"].size(0)
        total_loss = 0.0

        # Brand loss
        brand_loss = self.ce(outputs["brand"], targets["brand"])
        total_loss += weights["brand"] * brand_loss

        # Only compute model and trim losses if brand prediction is good enough
        brand_probs = torch.softmax(outputs["brand"], dim=1)
        brand_confidence = brand_probs.max(dim=1)[0]

        # Model loss with brand confidence weighting
        model_loss = 0.0
        valid_model_count = 0
        for i in range(batch_size):
            if (
                brand_confidence[i] > 0.5
            ):  # Only consider if brand prediction is confident
                brand_idx = targets["brand"][i].item()
                brand_name = self.dataset.get_brand_name(brand_idx)
                model_logits = outputs["model"][brand_name][i].unsqueeze(0)
                model_loss += self.ce(model_logits, targets["model"][i].unsqueeze(0))
                valid_model_count += 1

        if valid_model_count > 0:
            model_loss = model_loss / valid_model_count
            total_loss += weights["model"] * model_loss

        # Trim loss with model confidence weighting
        trim_loss = 0.0
        valid_trim_count = 0
        for i in range(batch_size):
            if brand_confidence[i] > 0.5:
                brand_idx = targets["brand"][i].item()
                model_idx = targets["model"][i].item()
                brand_name = self.dataset.get_brand_name(brand_idx)
                model_name = self.dataset.get_model_name(brand_name, model_idx)
                key = f"{brand_name}_{model_name}"

                # Get model confidence
                model_logits = outputs["model"][brand_name][i]
                model_probs = torch.softmax(model_logits, dim=0)
                model_confidence = model_probs.max()

                if (
                    model_confidence > 0.5
                ):  # Only consider if model prediction is confident
                    trim_logits = outputs["trim"][key][i].unsqueeze(0)
                    trim_loss += self.ce(trim_logits, targets["trim"][i].unsqueeze(0))
                    valid_trim_count += 1

        if valid_trim_count > 0:
            trim_loss = trim_loss / valid_trim_count
            total_loss += weights["trim"] * trim_loss

        return total_loss, {
            "brand_loss": brand_loss.item(),
            "model_loss": (
                model_loss if isinstance(model_loss, float) else model_loss.item()
            ),
            "trim_loss": (
                trim_loss if isinstance(trim_loss, float) else trim_loss.item()
            ),
        }


def train_model(
    csv_path,
    batch_size=32,
    num_epochs=50,
    learning_rate=1e-4,
    device="cuda",
    use_wandb=True,
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if use_wandb:
        wandb.init(
            project="vehicle-classifier",
            config={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
            },
        )

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = HierarchicalVehicleDataset(csv_path, transform=transform)
    val_dataset = HierarchicalVehicleDataset(csv_path, transform=val_transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, _ = random_split(dataset, [train_size, val_size])
    _, val_dataset = random_split(val_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = DynamicHierarchicalClassifier(dataset.hierarchy_config).to(device)

    criterion = HierarchicalLoss(dataset).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    scaler = GradScaler()

    best_val_loss = float("inf")
    patience = 5
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        metrics = {
            "train": {
                "correct": {"brand": 0, "model": 0, "trim": 0},
                "total": {"brand": 0, "model": 0, "trim": 0},
            }
        }

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
        ):
            images = batch["image"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "image"}

            # Mixed precision training
            with autocast(device_type="cuda", dtype=torch.float16):
                # Apply mixup
                mixed_images, mixed_targets, lam = mixup_data(images, targets)

                # Forward pass
                outputs = model(mixed_images)

                # Calculate loss
                loss, loss_components = criterion(
                    outputs,
                    {
                        "brand": mixed_targets["brand"][0],
                        "model": mixed_targets["model"][0],
                        "trim": mixed_targets["trim"][0],
                    },
                )

                # Add mixup loss
                loss2, _ = criterion(
                    outputs,
                    {
                        "brand": mixed_targets["brand"][1],
                        "model": mixed_targets["model"][1],
                        "trim": mixed_targets["trim"][1],
                    },
                )
                loss = lam * loss + (1 - lam) * loss2

            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            # Calculate accuracies
            with torch.no_grad():
                update_metrics(metrics["train"], outputs, targets, dataset)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracies = {
            k: metrics["train"]["correct"][k] / metrics["train"]["total"][k]
            for k in ["brand", "model", "trim"]
        }

        model.eval()
        val_loss = 0.0
        metrics["val"] = {
            "correct": {"brand": 0, "model": 0, "trim": 0},
            "total": {"brand": 0, "model": 0, "trim": 0},
        }

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch["image"].to(device)
                targets = {k: v.to(device) for k, v in batch.items() if k != "image"}

                outputs = model(images)

                loss, val_loss_components = criterion(outputs, targets)
                val_loss += loss.item()

                update_metrics(metrics["val"], outputs, targets, dataset)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracies = {
            k: metrics["val"]["correct"][k] / metrics["val"]["total"][k]
            for k in ["brand", "model", "trim"]
        }

        scheduler.step(avg_val_loss)

        log_dict = {
            "train/loss": avg_train_loss,
            "train/brand_acc": train_accuracies["brand"],
            "train/model_acc": train_accuracies["model"],
            "train/trim_acc": train_accuracies["trim"],
            "val/loss": avg_val_loss,
            "val/brand_acc": val_accuracies["brand"],
            "val/model_acc": val_accuracies["model"],
            "val/trim_acc": val_accuracies["trim"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        if use_wandb:
            wandb.log(log_dict)

        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(
            f"Train Loss: {avg_train_loss:.4f}, Acc: B {train_accuracies['brand']:.2f}, "
            f"M {train_accuracies['model']:.2f}, T {train_accuracies['trim']:.2f}"
        )
        logger.info(
            f"Val Loss: {avg_val_loss:.4f}, Acc: B {val_accuracies['brand']:.2f}, "
            f"M {val_accuracies['model']:.2f}, T {val_accuracies['trim']:.2f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            model_save_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "hierarchy": dataset.hierarchy_config,
                },
                model_save_path,
            )
            logger.info(f"Saved best model to {model_save_path}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    if use_wandb:
        wandb.finish()


def update_metrics(metrics, outputs, targets, dataset):
    _, brand_preds = torch.max(outputs["brand"], 1)
    metrics["correct"]["brand"] += (brand_preds == targets["brand"]).sum().item()
    metrics["total"]["brand"] += targets["brand"].size(0)

    for i in range(targets["brand"].size(0)):
        brand_idx = targets["brand"][i].item()
        brand_name = dataset.get_brand_name(brand_idx)
        model_logits = outputs["model"][brand_name][i]
        _, model_pred = torch.max(model_logits, 0)
        metrics["correct"]["model"] += (model_pred == targets["model"][i]).item()
        metrics["total"]["model"] += 1

    for i in range(targets["brand"].size(0)):
        brand_idx = targets["brand"][i].item()
        model_idx = targets["model"][i].item()
        brand_name = dataset.get_brand_name(brand_idx)
        model_name = dataset.get_model_name(brand_name, model_idx)
        key = f"{brand_name}_{model_name}"
        trim_logits = outputs["trim"][key][i]
        _, trim_pred = torch.max(trim_logits, 0)
        metrics["correct"]["trim"] += (trim_pred == targets["trim"][i]).item()
        metrics["total"]["trim"] += 1


def mixup_data(x, targets, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_targets = {
        "brand": (targets["brand"], targets["brand"][index]),
        "model": (targets["model"], targets["model"][index]),
        "trim": (targets["trim"], targets["trim"][index]),
    }

    return mixed_x, mixed_targets, lam


if __name__ == "__main__":
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data/turkish_2ndhand_automobile_processed.csv",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(csv_path, device=device)
