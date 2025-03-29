from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class CIFAR10Pair(CIFAR10):
    def __getitem__(self, index):
        img, y = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        x1 = self.transform(img)
        x2 = self.transform(img)

        return x1, x2, y


class MNISTPair(MNIST):
    def __getitem__(self, index):
        img, y = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode='L')

        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2, y


def NTXentLoss(out1, out2, temperature=0.1, device='cpu'):
    out1 = F.normalize(out1, p=2, dim=-1)  # Explicit normalization
    out2 = F.normalize(out2, p=2, dim=-1)

    # Compute similarity matrix more efficiently
    sim_matrix = torch.einsum('nc,mc->nm', out1, out2) / temperature
    labels = torch.arange(out1.size(0), device=device)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss


CIFAR10_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

MNIST_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(28, scale=(0.2, 1.0)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST stats
])

CIFAR10_test_transform = transforms.Compose([
    transforms.ToTensor(),
])

MNIST_test_transform = transforms.Compose([
    transforms.ToTensor(),
])


def plot_tsne(model, dataloader, device):
    '''
    model - torch.nn.Module subclass. This is your encoder model
    dataloader - test dataloader to over data for which you wish to compute projections
    device - cuda or cpu (as a string)
    '''
    model.eval()

    images_list = []
    labels_list = []
    latent_list = []

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # approximate the latent space from data
            latent_vector = model(images)

            images_list.append(images.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            latent_list.append(latent_vector.cpu().numpy())

    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    latent_vectors = np.concatenate(latent_list, axis=0)

    # Plot TSNE for latent space
    tsne_latent = TSNE(n_components=2, random_state=0)
    latent_tsne = tsne_latent.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab10', s=10)  # Smaller points
    plt.colorbar(scatter)
    plt.title('t-SNE of Latent Space')
    plt.savefig('latent_tsne.png')
    plt.close()

    # plot image domain tsne
    tsne_image = TSNE(n_components=2, random_state=42)
    images_flattened = images.reshape(images.shape[0], -1)
    image_tsne = tsne_image.fit_transform(images_flattened)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(image_tsne[:, 0], image_tsne[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE of Image Space')
    plt.savefig('image_tsne.png')
    plt.close()