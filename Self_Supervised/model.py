import torch
import torch.nn as nn
import torch.nn.functional as F


class Self_Supervised(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, hidden_dim=256, num_classes=10):
        super(Self_Supervised, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, latent_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 4 * 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded.view(encoded.size(0), -1)

    def reconstruct(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def classify(self, x):
        encoded = self.encoder(x)
        feature = torch.flatten(encoded, start_dim=1)
        out = self.classifier(feature)
        return out
