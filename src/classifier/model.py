import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class SharedFeatureExtractor(nn.Module):
    def __init__(self, backbone_name="resnet50"):
        super().__init__()
        if backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Backbone {backbone_name} not supported yet")

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.shared_layer = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.out_features = 256

    def forward(self, x):
        features = self.backbone(x)
        shared_features = self.shared_layer(features)
        return shared_features


class HierarchicalAttention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

        self.brand_attention = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Tanh(),
            nn.Linear(in_features, in_features),
            nn.Sigmoid(),
        )

        self.model_attention = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Tanh(),
            nn.Linear(in_features, in_features),
            nn.Sigmoid(),
        )

    def forward(self, features, brand_logits=None, model_logits=None):
        batch_size = features.size(0)
        attention_features = features.clone()

        if brand_logits is not None:
            # Brand-level attention
            brand_probs = F.softmax(brand_logits, dim=1)  # [B, num_brands]
            brand_attention = self.brand_attention(attention_features)  # [B, F]
            features = features * brand_attention

        if model_logits is not None and isinstance(model_logits, dict):
            # Get attention weights for each brand's models
            attention_weights = []
            valid_counts = []

            for brand_logits in model_logits.values():
                # Calculate attention for this brand's models
                model_probs = F.softmax(brand_logits, dim=1)  # [B, num_models]
                valid_counts.append(brand_logits.size(1))
                attention_weights.append(
                    model_probs.mean(dim=1, keepdim=True)
                )  # [B, 1]

            # Combine attention weights
            combined_attention = torch.cat(attention_weights, dim=1)  # [B, num_brands]
            combined_attention = combined_attention.mean(dim=1, keepdim=True)  # [B, 1]

            # Apply model-level attention
            model_attention = self.model_attention(attention_features)  # [B, F]
            model_attention = (
                model_attention * combined_attention
            )  # Adjust attention weights
            features = features * model_attention

        return features


class DynamicHierarchicalClassifier(nn.Module):
    def __init__(self, hierarchy):
        super().__init__()

        # Shared feature extractor
        self.feature_extractor = SharedFeatureExtractor()
        in_features = self.feature_extractor.out_features

        # Store hierarchy for reference
        self.hierarchy = hierarchy

        # Hierarchical attention
        self.attention = HierarchicalAttention(in_features)

        # Dynamic heads based on hierarchy with reduced output dimensions
        self.brand_classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features // 2, len(hierarchy["levels"][0]["classes"])),
        )

        # Model classifiers per brand with shared features
        self.model_classifiers = nn.ModuleDict(
            {
                brand: nn.Sequential(
                    nn.Linear(in_features, in_features // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(in_features // 2, len(models)),
                )
                for brand, models in hierarchy["levels"][1]["classes"].items()
            }
        )

        # Trim classifiers per model with shared features
        self.trim_classifiers = nn.ModuleDict(
            {
                f"{brand}_{model}": nn.Sequential(
                    nn.Linear(in_features, in_features // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(in_features // 2, len(trims)),
                )
                for brand_model, trims in hierarchy["levels"][2]["classes"].items()
                for brand, model in [brand_model.split("_")]
            }
        )

        # Uncertainty estimation
        self.enable_dropout = False

    def enable_uncertainty(self, enable=True):
        """Enable dropout at inference time for uncertainty estimation"""
        self.enable_dropout = enable

    def get_features(self, x):
        """Extract features from the input image"""
        if self.training or self.enable_dropout:
            features = self.feature_extractor(x)
        else:
            with torch.no_grad():
                features = self.feature_extractor(x)
        return features

    def forward_brand(self, x):
        """Forward pass for brand prediction"""
        features = self.get_features(x)
        brand_features = features
        return self.brand_classifier(brand_features)

    def forward_model(self, x, brand_name):
        """Forward pass for model prediction"""
        features = self.get_features(x)
        brand_logits = self.forward_brand(x)
        model_features = self.attention(features, brand_logits=brand_logits)
        return self.model_classifiers[brand_name](model_features)

    def forward_trim(self, x, brand_model_key):
        """Forward pass for trim prediction"""
        features = self.get_features(x)
        brand_name = brand_model_key.split("_")[0]
        brand_logits = self.forward_brand(x)
        model_logits = self.forward_model(x, brand_name)
        model_dict = {brand_name: model_logits}
        trim_features = self.attention(
            features, brand_logits=brand_logits, model_logits=model_dict
        )
        return self.trim_classifiers[brand_model_key](trim_features)

    def forward(self, x):
        if self.training or self.enable_dropout:
            features = self.feature_extractor(x)
        else:
            with torch.no_grad():
                features = self.feature_extractor(x)

        # Brand predictions with base features
        brand_features = features  # Keep original features for brand prediction
        brand_out = self.brand_classifier(brand_features)

        # Apply attention for model predictions using brand information
        model_features = self.attention(features, brand_logits=brand_out)
        model_out = {
            brand: classifier(model_features)
            for brand, classifier in self.model_classifiers.items()
        }

        # Apply attention for trim predictions using both brand and model information
        trim_features = self.attention(model_features, model_logits=model_out)
        trim_out = {
            brand_model: classifier(trim_features)
            for brand_model, classifier in self.trim_classifiers.items()
        }

        return {
            "brand": brand_out,
            "model": model_out,
            "trim": trim_out,
            "features": features,  # Return features for potential auxiliary tasks
        }

    def predict_with_uncertainty(self, x, num_samples=10):
        """Perform Monte Carlo Dropout for uncertainty estimation"""
        self.enable_uncertainty(True)
        predictions = []

        for _ in range(num_samples):
            with torch.no_grad():
                output = self(x)
                predictions.append(
                    {
                        "brand": F.softmax(output["brand"], dim=1),
                        "model": {
                            k: F.softmax(v, dim=1) for k, v in output["model"].items()
                        },
                        "trim": {
                            k: F.softmax(v, dim=1) for k, v in output["trim"].items()
                        },
                    }
                )

        self.enable_uncertainty(False)

        # Calculate mean and variance of predictions
        brand_preds = torch.stack([p["brand"] for p in predictions])
        brand_mean = brand_preds.mean(dim=0)
        brand_var = brand_preds.var(dim=0)

        model_means = {}
        model_vars = {}
        trim_means = {}
        trim_vars = {}

        for brand in predictions[0]["model"].keys():
            model_preds = torch.stack([p["model"][brand] for p in predictions])
            model_means[brand] = model_preds.mean(dim=0)
            model_vars[brand] = model_preds.var(dim=0)

        for brand_model in predictions[0]["trim"].keys():
            trim_preds = torch.stack([p["trim"][brand_model] for p in predictions])
            trim_means[brand_model] = trim_preds.mean(dim=0)
            trim_vars[brand_model] = trim_preds.var(dim=0)

        return {
            "brand": {"mean": brand_mean, "variance": brand_var},
            "model": {"mean": model_means, "variance": model_vars},
            "trim": {"mean": trim_means, "variance": trim_vars},
        }


class HierarchicalLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.25):
        super(HierarchicalLoss, self).__init__()
        self.alpha = alpha  # Brand loss weight
        self.beta = beta  # Series loss weight
        self.gamma = gamma  # Model loss weight
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore invalid indices

    def forward(
        self,
        brand_logits,
        series_logits,
        model_logits,
        brand_labels,
        series_labels,
        model_labels,
    ):
        loss = 0

        # Brand loss
        if brand_logits is not None and brand_labels is not None:
            valid_brand_mask = (brand_labels >= 0) & (
                brand_labels < brand_logits.size(1)
            )
            if valid_brand_mask.any():
                loss += self.alpha * self.ce(brand_logits, brand_labels)

        # Series loss
        if series_logits is not None and series_labels is not None:
            # Only compute loss for valid series predictions
            valid_series_mask = series_labels >= 0
            for i in range(len(series_labels)):
                if valid_series_mask[i]:
                    brand = str(brand_labels[i].item())
                    if brand in self.series_classifiers:
                        valid_series_mask[i] &= (
                            series_labels[i] < self.num_series_per_brand[int(brand)]
                        )

            if valid_series_mask.any():
                loss += self.beta * self.ce(
                    series_logits[valid_series_mask], series_labels[valid_series_mask]
                )

        # Model loss
        if model_logits is not None and model_labels is not None:
            # Only compute loss for valid model predictions
            valid_model_mask = model_labels >= 0
            for i in range(len(model_labels)):
                if valid_model_mask[i]:
                    series = str(series_labels[i].item())
                    if series in self.model_classifiers:
                        valid_model_mask[i] &= (
                            model_labels[i] < self.num_models_per_series[int(series)]
                        )

            if valid_model_mask.any():
                loss += self.gamma * self.ce(
                    model_logits[valid_model_mask], model_labels[valid_model_mask]
                )

        return loss
