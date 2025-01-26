import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from collections import Counter


class CarDataset(Dataset):
    def __init__(self, image_paths, brands, series, models, transform=None):
        self.image_paths = image_paths
        self.brands = brands
        self.series = series
        self.models = models
        self.transform = transform

        # Initialize label encoders
        self.brand_encoder = LabelEncoder()
        self.series_encoder = LabelEncoder()
        self.model_encoder = LabelEncoder()

        # Fit encoders
        self.brand_labels = self.brand_encoder.fit_transform(brands)
        self.series_labels = self.series_encoder.fit_transform(series)
        self.model_labels = self.model_encoder.fit_transform(models)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # Load and preprocess image
            image_path = self.image_paths[idx]
            if not os.path.isabs(image_path):
                image_path = os.path.join("data/processed_images", image_path)
            image_path = os.path.normpath(image_path)

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image at path: {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]

            return {
                "image": image,
                "brand": torch.tensor(self.brand_labels[idx], dtype=torch.long),
                "series": torch.tensor(self.series_labels[idx], dtype=torch.long),
                "model": torch.tensor(self.model_labels[idx], dtype=torch.long),
            }
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            raise


def get_transforms(is_training=True):
    if is_training:
        return A.Compose(
            [
                A.Resize(height=256, width=256),
                A.CenterCrop(224, 224),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=256, width=256),
                A.CenterCrop(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


def create_dataloaders(
    df, image_dir, batch_size=32, num_workers=4, min_examples_per_class=2
):
    """
    Create train and validation dataloaders with stratification

    Args:
        df: DataFrame containing the dataset
        image_dir: Directory containing the images
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        min_examples_per_class: Minimum number of examples required for a class to be included
    """
    from sklearn.model_selection import train_test_split

    # Make a copy of the DataFrame and handle missing values
    df = df.copy()
    initial_len = len(df)
    df = df.dropna(subset=["image_path", "marka", "seri", "model"])
    if len(df) < initial_len:
        print(f"Removed {initial_len - len(df)} rows with missing values")
        print(f"Remaining rows: {len(df)}")

    # Ensure image_path is string type and remove any duplicate base paths
    df["image_path"] = df["image_path"].astype(str).apply(lambda x: os.path.basename(x))

    # Create stratification based on brand-series combinations
    df["strat_label"] = df["marka"] + "_" + df["seri"]

    # Count examples per class
    class_counts = Counter(df["strat_label"])

    # Filter out classes with too few examples
    valid_classes = [
        cls for cls, count in class_counts.items() if count >= min_examples_per_class
    ]
    print(
        f"Filtered out {len(class_counts) - len(valid_classes)} classes with fewer than {min_examples_per_class} examples"
    )
    print(f"Remaining classes: {len(valid_classes)}")

    # Filter DataFrame
    df_filtered = df[df["strat_label"].isin(valid_classes)].copy()

    if len(df_filtered) < len(df):
        print(f"Filtered out {len(df) - len(df_filtered)} examples")
        print(f"Remaining examples: {len(df_filtered)}")

    # Split the filtered dataset
    train_df, val_df = train_test_split(
        df_filtered, test_size=0.2, stratify=df_filtered["strat_label"], random_state=42
    )

    # Create datasets
    train_dataset = CarDataset(
        image_paths=train_df["image_path"].values,
        brands=train_df["marka"].values,
        series=train_df["seri"].values,
        models=train_df["model"].values,
        transform=get_transforms(is_training=True),
    )

    val_dataset = CarDataset(
        image_paths=val_df["image_path"].values,
        brands=val_df["marka"].values,
        series=val_df["seri"].values,
        models=val_df["model"].values,
        transform=get_transforms(is_training=False),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
