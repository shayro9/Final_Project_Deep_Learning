import torch
import numpy as np
from sklearn.manifold import TSNE


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


def create_data_sets(train_ds, test_ds, validation_ratio, batch_size=100, transform=None, debug=False):
    """
    Splitting the train into validation by given ratio and creating DataLoaders
    :param train_ds: training dataset
    :param test_ds: test dataset
    :param validation_ratio: in range 0,1
    :param batch_size: batch size
    :param transform: transform function (optional)
    :param debug: whether to print data
    :return: A tuple of train, validation and test DataLoader
    """

    n = len(train_ds)
    validation_size = int(n * validation_ratio)
    indices = list(range(n))
    np.random.shuffle(indices)
    val_indices = indices[:validation_size]

    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[validation_size:])

    dl_valid = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=False, sampler=val_sampler)
    dl_train = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=False, sampler=train_sampler)
    dl_test = torch.utils.data.DataLoader(test_ds, batch_size)

    if debug:
        train_idx = set(train_sampler.indices)
        valid_idx = set(val_sampler.indices)
        train_size = len(train_idx)
        valid_size = len(valid_idx)
        print('Training set size: ', train_size)
        print('Validation set size: ', valid_size)
        print('Test set size: ', len(test_ds))

    return dl_train, dl_valid, dl_test
