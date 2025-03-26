# Biased VAE

This project implements Variational Autoencoders (VAE) with two different approaches to shift the latent space based on expert-provided labels.

## Project Overview

The project trains a base VAE on the MNIST dataset and explores two different methods to incorporate expert knowledge into the latent space:

1. Goal VAE: Shifts the latent space towards expert-provided labels based on mean and variance of the target digit class
2. GMM VAE: Uses a Gaussian Mixture Model approach to structure the latent space according to expert labels

## Usage
To reproduce images found here, use:
```bash
python alltests.py
```

For detailed configuration options, refer to `constants.py`.

## Dependencies
- Python 3.x
- PyTorch
- NumPy
- Matplotlib

## Authors
Developed by Jørgen Mjaaseth.