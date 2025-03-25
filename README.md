# Biased VAE

This project implements Variational Autoencoders (VAE) with two different approaches to shift the latent space based on expert-provided labels.

## Project Overview

The project trains a base VAE on the MNIST dataset and explores two different methods to incorporate expert knowledge into the latent space:

1. Goal VAE: Shifts the latent space towards expert-provided labels based on mean and variance of the target digit class
2. GMM VAE: Uses a Gaussian Mixture Model approach to structure the latent space according to expert labels


## Loss Functions

### Base VAE Loss
The base VAE uses the standard ELBO loss:

$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL}}
$$

where:

$$
\mathcal{L}_{\text{recon}} = \frac{1}{N}\sum_{i=1}^N \|x_i - \hat{x}_i\|^2
$$

$$
\mathcal{L}_{\text{KL}} = -\frac{1}{2}\sum_{j=1}^d (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)
$$

### Goal-Conditioned VAE Loss
The GA-VAE modifies the KL divergence to match a target distribution:

$$
\mathcal{L}_{\text{GA}} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL-GA}}
$$

where:

$$
\mathcal{L}_{\text{KL-GA}} = \frac{1}{2}\sum_{j=1}^d \left(\frac{\sigma_j^2}{\sigma_{p,j}^2} + \frac{(\mu_j - \mu_{p,j})^2}{\sigma_{p,j}^2} - 1 - \log(\sigma_j^2) + \log(\sigma_{p,j}^2)\right)
$$

Here, $$\mu_p$$ and $$\sigma_p^2$$ are the mean and variance of the target digit class.

### GMM-Based VAE Loss
The GMM-VAE uses a mixture of Gaussians as the prior:

$$
\mathcal{L}_{\text{GMM}} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL-GMM}}
$$

where:

$$
\mathcal{L}_{\text{KL-GMM}} = \mathbb{E}_{q(z|x)}[\log q(z|x) - \log p(z)]
$$

with 

$$p(z) = \sum_{k=1}^K \pi_k \mathcal{N}(z|\mu_k, \Sigma_k)$$ being the GMM prior.

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