import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image



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

def display_tsne():
    image_path1 = "image_tsne.png"
    image_path2 = "latent_tsne.png"
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)
    
    # Create a figure and display both images side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # 1 row, 2 columns
    
    axes[0].imshow(img1)
    axes[0].axis('off')  # Hide axes
    axes[0].set_title("Image TSNE")
    
    axes[1].imshow(img2)
    axes[1].axis('off')  # Hide axes
    axes[1].set_title("Latent TSNE")
    
    plt.show()


def create_data_sets(train_ds, test_ds, validation_ratio, batch_size=256, debug=False):
    """
    Splits `train_ds` into train/validation and creates DataLoaders.
    Assumes:
    - `train_ds` returns (x1, x2, y) (wrapped with SimCLRDataset)
    - `test_ds` returns (x, y) (wrapped with TestDataset)
    """
    n = len(train_ds)
    validation_size = int(n * validation_ratio)
    indices = list(range(n))
    np.random.shuffle(indices)

    # Split indices
    val_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    # Create DataLoaders
    dl_train = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=0
    )
    dl_valid = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=val_sampler, pin_memory=True, num_workers=0
    )
    dl_test = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
    )

    if debug:
        print(f"Training set size: {len(train_indices)}")
        print(f"Validation set size: {len(val_indices)}")
        print(f"Test set size: {len(test_ds)}")

    return dl_train, dl_valid, dl_test


def show(img, ax):
    """Utility function to display an image without per-image title"""
    img = img.cpu().detach().numpy()
    if img.shape[0] == 1:  # Grayscale image
        img = img[0]
        ax.imshow(img, cmap='gray')
    else:  # RGB image
        img = img.transpose((1, 2, 0))  # Convert to HWC format
        ax.imshow(img)
    ax.axis('off')

def showReconstructions(model, test_data, device, indices):
    num_samples = len(indices)
    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))  
    #fig.suptitle("Input Images and Their Reconstructions", fontsize=14, weight='bold')

    for i, ix in enumerate(indices):
        im, _ = test_data[ix]
        _im = model(im.unsqueeze(0).to(device))[0]

        show(im, axes[0, i])  # Show original image in top row
        show(_im, axes[1, i])  # Show reconstructed image in bottom row

    axes[0, 0].set_title("Original", fontsize=12)
    axes[1, 0].set_title("Reconstruction", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Adjust layout for title
    plt.show()


def displayInterpolation(model, test_data, device, indice1, indice2):

    
    fig, axes = plt.subplots(1, 2, figsize=(3 * 2, 6))  
    #fig.suptitle("Input Images and Their Reconstructions", fontsize=14, weight='bold')


    i1 = indice1
    im1, _ = test_data[i1]
    im1_latent = model.encoder(im1.unsqueeze(0).to(device))
    
    i2 = indice2
    im2, _ = test_data[i2]
    im2_latent = model.encoder(im2.unsqueeze(0).to(device))
    show(im1, axes[0])  # Show original image in top row
    show(im2, axes[1])

    axes[0].set_title("Original Images", fontsize=12)

        
    fig, axes = plt.subplots(1, 10, figsize=(3 * 10, 6))  
    axes[0].set_title("Linear Interpolations", fontsize=20)
    
    i = 0
    for alpha in np.linspace(0, 1, 10):
        interpolated_latent = (1 - alpha) * im1_latent + alpha * im2_latent
        x = model.decoder(interpolated_latent).squeeze(0)
        show(x, axes[i])
        i += 1





    
    