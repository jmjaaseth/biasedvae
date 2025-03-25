import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from constants import LATENT_DIM, DEVICE

def plot_histogram(latents, title="Latent Space Histogram"):
    plt.figure(figsize=(10, 6))
    sns.histplot(latents, bins=30, kde=True)
    plt.title(title)
    plt.xlabel("Latent Value")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_qq(latents, title="Q-Q Plot of Latent Variables"):
    plt.figure(figsize=(8, 8))
    stats.probplot(latents.flatten(), dist="norm", plot=plt)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_latent_space_2d(latents, labels=None, title="2D Latent Space Visualization", dim1 = 0, dim2 = 1):
        
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents[:, dim1], latents[:, dim2], c=labels, cmap='tab10', alpha=0.6)
    if labels is not None:
        plt.colorbar(scatter, label='Digit Class')
    
    plt.title(title)
    plt.xlabel(f"Latent Dimension {dim1}")
    plt.ylabel(f"Latent Dimension {dim2}")
    plt.grid(True, alpha=0.3)
    plt.show()

def sample_and_visualize(model, num_samples=16, temperature=1.0):
    model.eval()
    with torch.no_grad():
        # Sample from latent space with temperature
        sampled_latents = torch.randn((num_samples, LATENT_DIM), device=DEVICE) * temperature
        decoded_samples = model.decode(sampled_latents)
        
        # Create interpolation between two random points
        z1 = torch.randn((1, LATENT_DIM), device=DEVICE)
        z2 = torch.randn((1, LATENT_DIM), device=DEVICE)
        interpolation = torch.linspace(0, 1, 8, device=DEVICE).view(-1, 1)
        z_interp = z1 * (1 - interpolation) + z2 * interpolation
        decoded_interp = model.decode(z_interp)
        
    # Plot random samples
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    plt.suptitle("Generated Samples from Latent Space")
    for i, ax in enumerate(axes.flat):
        ax.imshow(decoded_samples[i].squeeze().cpu().numpy(), cmap='gray')
        ax.axis('off')
    plt.show()
    
    # Plot interpolation
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    plt.suptitle("Latent Space Interpolation")
    for i, ax in enumerate(axes):
        ax.imshow(decoded_interp[i].squeeze().cpu().numpy(), cmap='gray')
        ax.axis('off')
    plt.show()