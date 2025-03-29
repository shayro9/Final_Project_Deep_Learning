import torch
import torch.nn as nn
import torch.nn.functional as F

class Self_Supervised(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, hidden_dim=512, num_classes=10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim),
            nn.LayerNorm(latent_dim)
        )


        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8)),

            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(),

            nn.ConvTranspose2d(32, in_channels, 3, stride=1, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.BatchNorm1d(hidden_dim // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(hidden_dim // 8, num_classes)
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
        out = self.classifier(encoded)
        return out


# class Self_Supervised(nn.Module):
#     def __init__(self, in_channels=3, latent_dim=128, hidden_dim=256, num_classes=10):
#         super(Self_Supervised, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(32, 64, 4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, latent_dim, 4, stride=2, padding=1),
#             nn.BatchNorm2d(latent_dim),
#             nn.LeakyReLU(0.2),
#             nn.Flatten(),
#             nn.Linear(2048, latent_dim)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 2048),
#             nn.LeakyReLU(0.2),
#             nn.Unflatten(1, (latent_dim, 4, 4)),
#             nn.ConvTranspose2d(latent_dim, 64, 4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
#             nn.Sigmoid(),
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.3),
#
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.BatchNorm1d(hidden_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.3),
#
#             nn.Linear(hidden_dim // 2, hidden_dim // 4),
#             nn.BatchNorm1d(hidden_dim // 4),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.3),
#
#             nn.Linear(hidden_dim // 4, hidden_dim // 8),
#             nn.BatchNorm1d(hidden_dim // 8),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.3),
#
#             nn.Linear(hidden_dim // 8, num_classes)
#         )
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         return encoded.view(encoded.size(0), -1)
#
#     def reconstruct(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
#
#     def classify(self, x):
#         encoded = self.encoder(x)
#         out = self.classifier(encoded)
#         return out
