from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from train import *
import utils
from model import CLR

feature_dim = 128
batch_size = 512
epochs = 50

# CLR train
CLR_lr = 1e-3
CLR_wd = 1e-6
temperature = 0.05

#Classify train
Class_lr = 1e-3
Class_wd = 1e-4

print(f"Parameters: epochs-{epochs} | batch-{batch_size} | lr-{CLR_lr} | temp-{temperature} | class lr-{Class_lr}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLR_train_data = utils.CIFAR10Pair(root='/datasets/cv_datasets/data', train=True, transform=utils.train_transform, download=True)
CLR_train_loader = DataLoader(CLR_train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
Class_train_data = utils.CIFAR10(root='/datasets/cv_datasets/data', train=True, transform=utils.test_transform, download=True)
Class_train_loader = DataLoader(Class_train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_data = utils.CIFAR10(root='/datasets/cv_datasets/data', train=False, transform=utils.test_transform, download=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

model = CLR(feature_dim).to(device)
CLR_optimizer = optim.Adam(model.parameters(), lr=CLR_lr, weight_decay=CLR_wd)
CLR_loss_fn = utils.NTXentLoss
Class_optimizer = torch.optim.SGD(model.fc.parameters(), lr=Class_lr, momentum=0.9, weight_decay=Class_wd)
Class_loss_fn = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.CosineAnnealingLR(CLR_optimizer, T_max=epochs)

# Train encoder
print("----------Train------------")
for epoch in range(1, epochs + 1):
    train_loss = train_epoch(model, CLR_train_loader, CLR_optimizer, CLR_loss_fn, temperature, device)
    print('Epoch: {}, Loss: {}'.format(epoch, train_loss))
    scheduler.step()

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