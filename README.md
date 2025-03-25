# Biased VAE

This project implements a Variational Autoencoder (VAE) with two different approaches to shift the latent space based on expert-provided labels.

## Project Overview

The project trains a base VAE on the MNIST dataset and explores two different methods to incorporate expert knowledge into the latent space:

1. Goal VAE: Shifts the latent space towards expert-provided labels using a goal-oriented loss function
2. GMM VAE: Uses a Gaussian Mixture Model approach to structure the latent space according to expert labels

## Loss Functions

### Goal VAE
The Goal VAE modifies the standard VAE loss function by adding a goal-oriented term that encourages the latent space to align with expert-provided labels. The loss function consists of:
- Reconstruction loss (standard VAE)
- KL divergence term
- Goal-oriented term that minimizes the distance between latent representations and their corresponding expert labels

### GMM VAE
The GMM VAE approach structures the latent space using a Gaussian Mixture Model, where:
- Each class is represented by a Gaussian component
- The latent space is organized to cluster similar digits together
- The loss function includes:
  - Reconstruction loss
  - KL divergence
  - GMM clustering loss that enforces class separation

## Usage

The project includes visualization tools to compare the latent spaces of different VAE approaches:
- Standard VAE
- Goal VAE
- GMM VAE

Images can be saved to disk or displayed interactively based on the `SAVE_IMAGES` configuration. 