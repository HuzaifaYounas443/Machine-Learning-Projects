import torch.nn as nn
from torchvision import models


class ConvNeXtModel(nn.Module):
    """
    Same architecture used to train both the Stage 1 (tumor / no tumor) and
    Stage 2 (benign / malignant) checkpoints. Must match exactly or
    load_state_dict will fail.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.convnext_base(weights=None)
        num_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Dropout(0.6),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
