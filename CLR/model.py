import torch
import torch.nn as nn
import torch.nn.functional as F


class CLR(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256, num_classes=10):
        super(CLR, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.projection = nn.Sequential(
            nn.Linear(latent_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim, bias=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        h = self.encoder(x)
        feature = torch.flatten(h, start_dim=1)
        out = self.projection(feature)
        return F.normalize(out, dim=-1)

    def classify(self, x):
        h = self.encoder(x)
        feature = torch.flatten(h, start_dim=1)
        out = self.fc(feature)
        return out
