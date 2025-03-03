import torch
import torch.nn as nn
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from utils import plot_tsne
import numpy as np
import random
import argparse

from utils import *
import Models as models
import Hyperparameters as hyper
import Trainning as trainning

NUM_CLASSES = 10


def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, help='Size of each batch')
    parser.add_argument('--valid_ratio', default=0.2, type=float, help='Ratio of validation set from training set')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--debug', default=True, help='Print debug info')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    return parser.parse_args()


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # one possible convenient normalization. You don't have to use it.
    ])

    args = get_args()
    print(args.device)
    freeze_seeds(args.seed)

    hypers = hyper.self_supervised_CIFAR10_hyperparams()

    if args.mnist:
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=False, transform=transform)
    else:
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)

    dl_train, dl_valid, dl_test = create_data_sets(train_dataset, test_dataset, args.valid_ratio, args.batch_size, transform, args.debug)

    model = models.SelfSupervisedCIFAR10()

    if args.debug:
        print(model.encoder)
        print(model.decoder)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hypers['learning_rate'], betas=hypers["betas"])
    encoder_trainer = trainning.SelfSupervisedTrainer(model, loss_fn, optimizer, args.device)
    encoder_trainer.fit(dl_train, dl_valid, epochs=hypers["epochs"], verbose=args.debug, early_stopping=0)

    model.freeze_encoder()

    class_loss_fn = nn.CrossEntropyLoss()
    class_optimizer = torch.optim.Adam(model.parameters(), lr=hypers['learning_rate'], betas=hypers["betas"])
    classifier_trainer = trainning.SelfSupervisedTrainer(model, class_loss_fn, class_optimizer, args.device)
    classifier_trainer.fit(dl_train, dl_valid, epochs=hypers["epochs"], verbose=args.debug, early_stopping=5)

