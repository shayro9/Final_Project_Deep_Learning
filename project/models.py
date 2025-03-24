import torch
import torch.nn as nn
import torch.nn.functional as F


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
'''

class SelfSupervised_CIFAR10(nn.Module):
    def __init__(self, latent_dim=128):
        super(SelfSupervised_CIFAR10, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5, stride=1, padding=0),  # Output: (20, 28, 28)
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size=5, stride=1, padding=0),  # Output: (40, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Output: (40, 12, 12)
            nn.Conv2d(40, 60, kernel_size=3, stride=1, padding=0),  # Output: (60, 10, 10)
            nn.ReLU(),
            nn.Conv2d(60, 80, kernel_size=3, stride=1, padding=0),  # Output: (80, 8, 8)
            nn.ReLU(),
            nn.Conv2d(80, 128, kernel_size=3, stride=1, padding=0),  # Output: (latent_dim, 6, 6)
            nn.ReLU(),
            nn.Flatten()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 6, 6)),  # Reshape the latent vector into feature map
            nn.ConvTranspose2d(128, 80, kernel_size=3, stride=1, padding=0),  # Output: (80, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(80, 60, kernel_size=3, stride=1, padding=0),  # Output: (60, 10, 10)
            nn.ReLU(),
            nn.ConvTranspose2d(60, 40, kernel_size=3, stride=1, padding=0),  # Output: (40, 12, 12)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Output: (40, 24, 24)
            nn.ConvTranspose2d(40, 20, kernel_size=5, stride=1, padding=0),  # Output: (20, 28, 28)
            nn.ReLU(),
            nn.ConvTranspose2d(20, 3, kernel_size=5, stride=1, padding=0),  # Output: (3, 32, 32)
            nn.Tanh()  # Normalize output to [-1, 1] range
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

'''
class SelfSupervised_CIFAR10(nn.Module):
    def __init__(self, latent_dim=128):
        super(SelfSupervised_CIFAR10, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.LeakyReLU(0.2),
           # nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            nn.MaxPool2d(1, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 8, 8)
            nn.LeakyReLU(0.2),
           # nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.MaxPool2d(1, stride=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (128, 4, 4)
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.MaxPool2d(1, stride=1),
            nn.Flatten(),  # Flatten to pass into linear layer
            nn.Linear(128 * 4 * 4, latent_dim),  # Latent space dimension
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),  # Flattened latent space back to original size
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Unflatten(1, (128, 4, 4)),  # Unflatten to 3D (batch, 128, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (64, 8, 8)
            nn.LeakyReLU(0.2),
          #  nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (32, 16, 16)
            nn.LeakyReLU(0.2),
          #  nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (3, 32, 32)
            nn.Tanh(),  # To match the original image pixel range [-1, 1]
        )

    def forward(self, x):
        # Pass through the encoder and decoder
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

'''
class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),  # Increased size of the first hidden layer
            nn.LeakyReLU(0.2),  # Leaky ReLU activation instead of ReLU
            nn.BatchNorm1d(256),  # Batch normalization
            nn.Dropout(0.2),  # Dropout for regularization

            nn.Linear(256, 128),  # Added another hidden layer
            nn.LeakyReLU(0.2),  # Leaky ReLU activation
            nn.BatchNorm1d(128),  # Batch normalization
            nn.Dropout(0.2),  # Dropout for regularization

            nn.Linear(128, num_classes)  # Output layer
        )

    def forward(self, x):
        return self.fc(x)
'''
class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(Classifier, self).__init__()

        # Define layers of the network
        self.fc = nn.Sequential(
            # First hidden layer with more units
            nn.Linear(latent_dim, 512),  # Increased units in the first hidden layer
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),  # Batch normalization
            nn.Dropout(0.3),  # Dropout to prevent overfitting

            # Second hidden layer
            nn.Linear(512, 256),  # Increased units in the second hidden layer
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            # Third hidden layer
            nn.Linear(256, 128),  # Increased depth by adding another hidden layer
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            # Fourth hidden layer
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            # Output layer
            nn.Linear(64, num_classes)  # Output layer for classification
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
        '''
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
        '''
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.LeakyReLU(0.2),
           # nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            nn.MaxPool2d(1, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 8, 8)
            nn.LeakyReLU(0.2),
           # nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.MaxPool2d(1, stride=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (128, 4, 4)
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.MaxPool2d(1, stride=1),
            nn.Flatten(),  # Flatten to pass into linear layer
            nn.Linear(128 * 4 * 4, latent_dim),  # Latent space dimension
        )

        self.classifier = nn.Sequential(
            # First hidden layer with more units
            nn.Linear(latent_dim, 512),  # Increased units in the first hidden layer
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),  # Batch normalization
            nn.Dropout(0.3),  # Dropout to prevent overfitting

            # Second hidden layer
            nn.Linear(512, 256),  # Increased units in the second hidden layer
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            # Third hidden layer
            nn.Linear(256, 128),  # Increased depth by adding another hidden layer
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            # Fourth hidden layer
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            # Output layer
            nn.Linear(64, num_classes)  # Output layer for classification
        )
        
        '''
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Output layer for classification
        )
        '''



    def forward(self, x):
        latent = self.encoder(x)  # Get latent representation
        output = self.classifier(latent)  # Classify based on latent representation
        return output
