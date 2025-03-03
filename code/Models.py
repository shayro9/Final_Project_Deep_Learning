import torch
import torch.nn as nn


class SelfSupervisedCIFAR10(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten()
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (8, 4, 4)),
            nn.ConvTranspose2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, num_classes)
        )
        self.to_classify = False

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.to_classify = True

    def forward(self, x):
        encoded = self.encoder(x)
        if not self.to_classify:
            return self.decoder(encoded)
        else:
            return self.classifier(encoded)
