import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class HierarchicalCNN(nn.Module):
    def __init__(self, num_brands, num_series, num_models):
        super().__init__()

        # Load pretrained EfficientNet-B4
        weights = EfficientNet_B4_Weights.DEFAULT
        self.backbone = efficientnet_b4(weights=weights)
        backbone_out = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # Brand classifier
        self.brand_pool = nn.AdaptiveAvgPool2d(1)
        self.brand_dropout = nn.Dropout(0.5)
        self.brand_fc = nn.Linear(backbone_out, 128)
        self.brand_out = nn.Linear(128, num_brands)

        # Series classifier with SE block
        self.series_se = SEBlock(backbone_out)
        self.series_conv = nn.Conv2d(backbone_out, 512, 1)
        self.series_pool = nn.AdaptiveAvgPool2d(1)
        self.series_out = nn.Linear(512, num_series)

        # Model classifier
        self.model_fc1 = nn.Linear(backbone_out + 128, 256)
        self.model_dropout = nn.Dropout(0.5)
        self.model_out = nn.Linear(256, num_models)

    def forward(self, x):
        # Backbone features
        x = self.backbone.features(x)

        # Brand prediction
        brand_pooled = self.brand_pool(x).flatten(1)
        brand_features = self.brand_dropout(brand_pooled)
        brand_embedding = self.brand_fc(brand_features)
        brand_out = self.brand_out(brand_embedding)

        # Series prediction with SE attention
        series_features = self.series_se(x)
        series_features = self.series_conv(series_features)
        series_pooled = self.series_pool(series_features).flatten(1)
        series_out = self.series_out(series_pooled)

        # Model prediction with concatenated features
        model_features = torch.cat([brand_embedding, brand_pooled], dim=1)
        model_features = F.relu(self.model_fc1(model_features))
        model_features = self.model_dropout(model_features)
        model_out = self.model_out(model_features)

        return {"brand": brand_out, "series": series_out, "model": model_out}


class HierarchicalLoss:
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.3):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def __call__(self, outputs, targets):
        brand_loss = self.ce(outputs["brand"], targets["brand"])
        series_loss = self.ce(outputs["series"], targets["series"])
        model_loss = self.ce(outputs["model"], targets["model"])

        total_loss = (
            self.alpha * brand_loss + self.beta * series_loss + self.gamma * model_loss
        )

        return total_loss, {
            "brand_loss": brand_loss.item(),
            "series_loss": series_loss.item(),
            "model_loss": model_loss.item(),
        }
