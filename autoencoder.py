import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import LATENT_DIM, DEVICE

class Autoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) implementation with convolutional layers.
    
    The architecture consists of:
    - Encoder: Convolutional layers followed by fully connected layers
    - Decoder: Fully connected layers followed by transposed convolutional layers
    
    The latent space is parameterized by mean (mu) and log variance (logvar)
    for the reparameterization trick.
    """
    
    def __init__(self, latent_dim=LATENT_DIM):
        """
        Initialize the VAE architecture.
        
        Args:
            latent_dim (int): Dimension of the latent space
        """
        super(Autoencoder, self).__init__()

        # Encoder architecture
        # Input: 1x28x28 -> Output: latent_dim*2 (mu and logvar)
        self.encoder = nn.Sequential(
            # First conv layer: 1x28x28 -> 32x14x14
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Second conv layer: 32x14x14 -> 64x7x7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Flatten and dense layers: 64x7x7 -> 256 -> latent_dim*2
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim * 2)  # 2x for mean and logvar
        ).to(DEVICE)
        
        # Decoder architecture
        # Input: latent_dim -> Output: 1x28x28
        self.decoder = nn.Sequential(
            # Dense layers: latent_dim -> 256 -> 64x7x7
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64 * 7 * 7),
            nn.BatchNorm1d(64 * 7 * 7),
            nn.LeakyReLU(0.2),
            
            # Transposed conv layers: 64x7x7 -> 32x14x14 -> 1x28x28
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Ensure output is in [0,1] range
        ).to(DEVICE)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming initialization for better training.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        """
        Encode input data into latent space parameters.
        
        Args:
            x (torch.Tensor): Input data tensor
            
        Returns:
            tuple: (mu, logvar) - Mean and log variance of latent distribution
        """
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        # Clamp values for numerical stability
        logvar = torch.clamp(logvar, -20, 2)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Perform the reparameterization trick to sample from the latent distribution.
        
        Args:
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(self, z):
        """
        Decode latent vector back to data space.
        
        Args:
            z (torch.Tensor): Latent vector
            
        Returns:
            torch.Tensor: Reconstructed data
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass of the VAE.
        
        Args:
            x (torch.Tensor): Input data tensor
            
        Returns:
            tuple: (x_recon, z, mu, logvar)
                - x_recon: Reconstructed data
                - z: Latent vector
                - mu: Mean of latent distribution
                - logvar: Log variance of latent distribution
        """
        x = x.to(DEVICE)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, z, mu, logvar

    def save_model(self, path):
        """
        Save model state to disk.
        
        Args:
            path (str): Path to save the model
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """
        Load model state from disk.
        
        Args:
            path (str): Path to load the model from
        """
        self.load_state_dict(torch.load(path, map_location=DEVICE))
        self.to(DEVICE)