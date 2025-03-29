import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


class CLR(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, hidden_dim=2048, num_classes=10):
        super(CLR, self).__init__()

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),

        #     nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),

        #     nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d(1),

        #     nn.Flatten()
        # )

        self.encoder = resnet18(weights=None)
        self.feature_dim = self.encoder.fc.in_features

        self.encoder.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes)
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(latent_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.3),
        #
        #     nn.Linear(hidden_dim, hidden_dim//2),
        #     nn.BatchNorm1d(hidden_dim//2),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.3),
        #
        #     nn.Linear(hidden_dim // 2, hidden_dim // 4),
        #     nn.BatchNorm1d(hidden_dim // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.3),
        #
        #     nn.Linear(hidden_dim // 4, hidden_dim // 8),
        #     nn.BatchNorm1d(hidden_dim // 8),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.3),
        #
        #     nn.Linear(hidden_dim//8, num_classes)
        # )

    def forward(self, x):
        h = self.encoder(x)
        out = self.projection(h)
        return F.normalize(out, dim=-1)

    def classify(self, x):
        h = self.encoder(x)
        out = self.classifier(h)
        return out
