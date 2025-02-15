import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class HierarchicalVehicleDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.hierarchy = self._build_hierarchy()

        # Create mappings
        self.brand_to_idx = {
            brand: idx
            for idx, brand in enumerate(self.hierarchy["levels"][0]["classes"])
        }
        self.model_to_idx = self._create_model_mapping()
        self.trim_to_idx = self._create_trim_mapping()

    def _build_hierarchy(self):
        # Build hierarchy directly from data
        hierarchy = {
            "levels": [
                {"name": "brand", "classes": []},
                {"name": "model", "classes": {}},
                {"name": "trim", "classes": {}},
            ]
        }

        # Collect unique brands (marka)
        brands = self.df["marka"].unique().tolist()
        hierarchy["levels"][0]["classes"] = sorted(brands)

        # Collect models per brand (seri)
        for brand in hierarchy["levels"][0]["classes"]:
            models = self.df[self.df["marka"] == brand]["seri"].unique().tolist()
            hierarchy["levels"][1]["classes"][brand] = sorted(models)

            # Collect trims per model (model)
            for model in models:
                key = f"{brand}_{model}"
                trims = (
                    self.df[(self.df["marka"] == brand) & (self.df["seri"] == model)][
                        "model"
                    ]
                    .unique()
                    .tolist()
                )
                hierarchy["levels"][2]["classes"][key] = sorted(trims)

        return hierarchy

    def _create_model_mapping(self):
        return {
            brand: {model: idx for idx, model in enumerate(models)}
            for brand, models in self.hierarchy["levels"][1]["classes"].items()
        }

    def _create_trim_mapping(self):
        return {
            (brand, model): {trim: idx for idx, trim in enumerate(trims)}
            for brand_model, trims in self.hierarchy["levels"][2]["classes"].items()
            for brand, model in [brand_model.split("_")]
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        brand = self.brand_to_idx[row["marka"]]
        model = self.model_to_idx[row["marka"]][row["seri"]]
        trim = self.trim_to_idx[(row["marka"], row["seri"])][row["model"]]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "brand": brand, "model": model, "trim": trim}

    @property
    def hierarchy_config(self):
        return self.hierarchy

    def get_num_brands(self):
        return len(self.brand_to_idx)

    def get_num_models_per_brand(self):
        return [len(model_dict) for model_dict in self.model_to_idx.values()]

    def get_brand_name(self, idx):
        for brand, brand_idx in self.brand_to_idx.items():
            if brand_idx == idx:
                return brand
        return None

    def get_model_name(self, brand, idx):
        for model, model_idx in self.model_to_idx[brand].items():
            if model_idx == idx:
                return model
        return None
