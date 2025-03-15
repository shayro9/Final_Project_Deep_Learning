import torch
import torch.nn as nn


class SelfSupervised_MNIST(nn.Module):
    def __init__(self, latent_dim=128):
        super(SelfSupervised_MNIST, self).__init__()

        # Encoder: (1, 28, 28) → (128)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )

        # Decoder: (128) → (1, 28, 28)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0,1]
        )
        
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


class SelfSupervised_CIFAR10(nn.Module):
    def __init__(self, latent_dim=128):
        super(SelfSupervised_CIFAR10, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (128, 4, 4)
            nn.ReLU(),
            nn.Flatten(),  # Flatten to pass into linear layer
            nn.Linear(128 * 4 * 4, latent_dim),  # Latent space dimension
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),  # Flattened latent space back to original size
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),  # Unflatten to 3D (batch, 128, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (3, 32, 32)
            nn.Tanh(),  # To match the original image pixel range [-1, 1]
        )

    def forward(self, x):
        # Pass through the encoder and decoder
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

class ClassificationGuided_MNIST(nn.Module):
    def __init__(self, num_classes=10, latent_dim=128):
        super(ClassificationGuided_MNIST, self).__init__()
        
        # Encoder: This maps images to a latent space
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim),  # Map to latent space
            nn.ReLU()  # Non-linearity in the latent space
        )

        # Classifier: This maps the latent vector to class predictions
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Output layer for classification
        )

    def forward(self, x):
        latent = self.encoder(x)  # Get latent representation
        output = self.classifier(latent)  # Classify based on latent representation
        return output

class ClassificationGuided_CIFAR10(nn.Module):
    def __init__(self, num_classes=10, latent_dim=128):
        super(ClassificationGuided_CIFAR10, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (128, 4, 4)
            nn.ReLU(),
            nn.Flatten(),  # Flatten to pass into linear layer
            nn.Linear(128 * 4 * 4, latent_dim),  # Latent space dimension
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Output layer for classification
        )

    def forward(self, x):
        latent = self.encoder(x)  # Get latent representation
        output = self.classifier(latent)  # Classify based on latent representation
        return output

    


        