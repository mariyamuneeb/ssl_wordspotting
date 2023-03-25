import matplotlib.pyplot as plt
import wandb
import numpy as np
import torch


## Plotting Few Samples
def plot_samples(dataset, num_samples):
    random_imgs = dataset.get_random_samples(num_samples)
    _, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(random_imgs, axs):
        ax.imshow(img)
        ax.title.set_text(f'Image Shape {img.size},{img.mode}')
    plt.show()


## Plotting Samples During Training
def plot_reconstructions(encoder, decoder, test_dataset, device, dataset_name, n=10, ):
    wandb_imgs = list()
    wandb_rec_imgs = list()
    my_table = wandb.Table(columns=["Original", "Reconstruction"])
    plt.figure(figsize=(16, 4.5))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = test_dataset[i][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img = decoder(encoder(img))
        plt.imshow(img.cpu().squeeze().numpy().T, cmap='gist_gray')  # for MNIST remove the transpose
        #   plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray') # for MNIST remove the transpose
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy().T, cmap='gist_gray')  # for MNIST remove the transpose
        #   plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  #for MNIST remove the transpose
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed images')
        my_table.add_data(wandb.Image(img.cpu()), wandb.Image(rec_img.cpu()))
        wandb_imgs.append(img.cpu())
        wandb_rec_imgs.append(rec_img.cpu())
    plt.show()
    wandb.log({dataset_name: my_table})


## Plotting Samples During Training for Standard Datasets like CIFAR and MNIST
def plot_reconstructions_sd(encoder, decoder, test_dataset, device, dataset_name,n=10):
    wandb_imgs = list()
    wandb_rec_imgs = list()
    my_table = wandb.Table(columns=["Original", "Reconstruction"])
    plt.figure(figsize=(16, 4.5))
    targets = np.array(test_dataset.targets)  # for MNIST change this to test_dataset.targets.numpy()
    # targets =  test_dataset.targets.numpy()
    t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img = decoder(encoder(img))
        plt.imshow(img.cpu().squeeze().numpy().T, cmap='gist_gray')  # for MNIST remove the transpose
        #   plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray') # for MNIST remove the transpose
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy().T, cmap='gist_gray')  # for MNIST remove the transpose
        #   plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  #for MNIST remove the transpose
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed images')
        my_table.add_data(wandb.Image(img.cpu()), wandb.Image(rec_img.cpu()))
        wandb_imgs.append(img.cpu())
        wandb_rec_imgs.append(rec_img.cpu())
    plt.show()
    wandb.log({dataset_name: my_table})