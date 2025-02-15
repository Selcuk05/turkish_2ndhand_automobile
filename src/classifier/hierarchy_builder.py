import pandas as pd
from collections import defaultdict


class VehicleHierarchyBuilder:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self._validate_data()

    def _validate_data(self):
        required_columns = ["marka", "seri", "model", "image_path"]
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(
                "CSV missing required columns. Needed: marka, seri, model, image_path"
            )

    def build_hierarchy(self):
        hierarchy = {
            "brands": defaultdict(lambda: defaultdict(list)),
            "brand_list": [],
            "model_map": defaultdict(list),
            "trim_map": defaultdict(list),
        }

        brands = self.df["marka"].unique().tolist()
        hierarchy["brand_list"] = sorted(brands)

        for _, row in self.df.iterrows():
            brand = row["marka"]
            model = row["seri"]
            trim = row["model"]

            if model not in hierarchy["brands"][brand]:
                hierarchy["brands"][brand][model] = []

            if trim not in hierarchy["brands"][brand][model]:
                hierarchy["brands"][brand][model].append(trim)

            if model not in hierarchy["model_map"][brand]:
                hierarchy["model_map"][brand].append(model)

            full_model_name = f"{brand}_{model}"
            if trim not in hierarchy["trim_map"][full_model_name]:
                hierarchy["trim_map"][full_model_name].append(trim)

        return {
            "levels": [
                {"name": "brand", "classes": hierarchy["brand_list"]},
                {
                    "name": "model",
                    "classes": {
                        brand: sorted(models)
                        for brand, models in hierarchy["model_map"].items()
                    },
                },
                {
                    "name": "trim",
                    "classes": {
                        model: sorted(trims)
                        for model, trims in hierarchy["trim_map"].items()
                    },
                },
            ]
        }
