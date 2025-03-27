from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from train import *
import utils
from model import Self_Supervised
from torchvision.datasets import CIFAR10, MNIST

feature_dim = 128
batch_size = 128
epochs = 30

# Reconstruction train
Rec_lr = 0.0003
betas = (0.9, 0.999)

# Classify train
Class_lr = 0.001
Class_wd = 1e-6

dataset = "CIFAR10"

print(f"Set, {dataset}, Parameters: epochs-{epochs} | batch-{batch_size} | lr-{Rec_lr} | class lr-{Class_lr}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dataset == "CIFAR10":
    in_ch = 3
    train_data = CIFAR10(root='/datasets/cv_datasets/data', train=True, transform=utils.CIFAR10_train_transform,
                         download=True)
    Class_train_data = CIFAR10(root='/datasets/cv_datasets/data', train=True, transform=utils.CIFAR10_transform,
                         download=True)
    test_data = CIFAR10(root='/datasets/cv_datasets/data', train=False, transform=utils.CIFAR10_transform,
                        download=True)
else:
    in_ch = 1
    train_data = MNIST(root='./data', train=True, transform=utils.MNIST_train_transform, download=True)
    Class_train_data = MNIST(root='./data', train=True, transform=utils.MNIST_transform, download=True)
    test_data = MNIST(root='./data', train=False, transform=utils.MNIST_transform, download=True)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
Class_train_loader = DataLoader(Class_train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

model = Self_Supervised(in_ch, feature_dim).to(device)
Rec_optimizer = optim.Adam(model.parameters(), lr=Rec_lr, betas=betas)
Rec_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Rec_optimizer, 'min', patience=3)
Rec_loss_fn = nn.MSELoss()

Class_optimizer = torch.optim.Adam(model.classifier.parameters(), lr=Class_lr, betas=betas, weight_decay=Class_wd)
Class_loss_fn = nn.CrossEntropyLoss()

# Train encoder
print("----------Train------------")
for epoch in range(1, epochs + 1):
    train_loss = train_epoch(model, train_loader, Rec_optimizer, Rec_loss_fn, device)
    print('Epoch: {}, Loss: {}'.format(epoch, train_loss))
    Rec_scheduler.step(train_loss)

utils.plot_tsne(model, test_loader, device)

for param in model.encoder.parameters():
    param.requires_grad = False

# Train classifier
print("------Train Classifier------------")
for epoch in range(1, epochs + 1):
    class_loss, class_acc = train_classifier(model, Class_train_loader, Class_optimizer, Class_loss_fn, device)
    print('Classifier - Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, class_loss, class_acc))
    test_loss, test_acc = test_epoch(model, test_loader, Class_loss_fn, device)
    print('Test - Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, test_loss, test_acc))
