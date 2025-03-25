import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.mixture import GaussianMixture
import numpy as np
from autoencoder import Autoencoder
from stats import plot_histogram, plot_qq, plot_latent_space_2d, sample_and_visualize, ensure_image_path
from constants import *
import os
from tqdm import tqdm

def get_mnist_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    dataset = datasets.MNIST(root=MNIST_PATH, train=True, transform=transform, download=True)
    
    # Split into train and validation
    train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

def evaluate_model(model, dataloader, criterion, beta=BETA_BASE):
    """
    Evaluate the model on the given dataloader.
    Computes both reconstruction and KL loss components.
    
    Args:
        model: The VAE model to evaluate
        dataloader: DataLoader containing validation data
        criterion: Loss function for reconstruction
        beta: Weight for KL divergence term
    
    Returns:
        tuple: (total_loss, recon_loss, kl_loss)
    """
    model.eval()
    total_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(DEVICE)
            recon, _, mu, logvar = model(images)
            
            # Compute losses separately for monitoring
            recon_loss = criterion(recon, images) / images.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)
            
            # Total loss with beta weighting
            loss = recon_loss + beta * kl_loss
            
            total_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
    
    return (total_loss / len(dataloader),
            recon_loss_total / len(dataloader),
            kl_loss_total / len(dataloader))

def collect_latents(model, dataloader):
    """Collect latents from the model for EXPERT_GOALS"""
    model.eval()
    latents = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Collecting latents"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            mask = torch.isin(labels, torch.tensor(EXPERT_GOALS, device=DEVICE))
            if not mask.any():
                continue            
            images = images[mask]            
            mu, _ = model.encode(images.to(DEVICE))
            latents.append(mu.cpu().numpy())
    
    if not latents:
        raise ValueError("No valid latents collected! Check if EXPERT_GOALS exist in dataset.")
    
    all_latents = np.concatenate(latents, axis=0)
    # Remove any remaining invalid values
    valid_mask = ~(np.isnan(all_latents).any(axis=1) | np.isinf(all_latents).any(axis=1))
    if not valid_mask.any():
        raise ValueError("All collected latents contain invalid values!")
    
    return all_latents[valid_mask]

def getModelLatents(model, dataloader):
    model.eval()
    latents = []
    labels_list = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            _, z, _, _ = model(images)
            latents.append(z.cpu().numpy())
            labels_list.append(labels.numpy())
    
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels_list, axis=0)   

    return latents, labels

def analyze_latents(model, dataloader, model_name="base"):
    model.eval()
    latents = []
    labels_list = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            _, z, _, _ = model(images)
            latents.append(z.cpu().numpy())
            labels_list.append(labels.numpy())
    
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    plot_histogram(latents[:, 0], "Latent Space Distribution", model_name)
    plot_qq(latents[:, 0], "Q-Q Plot", model_name)
    return latents, labels

def train_base(model, train_loader, val_loader, epochs=EPOCHS):
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0
        
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(DEVICE)
            optimizer.zero_grad()
            recon, _, mu, logvar = model(images)
            
            # Compute losses separately for monitoring
            recon_loss = criterion(recon, images) / images.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)
            
            # Total loss with beta weighting
            loss = recon_loss + BETA_BASE * kl_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
        
        train_loss /= len(train_loader)
        recon_loss_avg = recon_loss_total / len(train_loader)
        kl_loss_avg = kl_loss_total / len(train_loader)
        
        # Evaluate validation loss with same components
        val_loss, val_recon_loss, val_kl_loss = evaluate_model(model, val_loader, criterion, BETA_BASE)
        
        # Print training progress
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Training Loss: {train_loss:.4f} (Recon: {recon_loss_avg:.4f}, KL: {kl_loss_avg:.4f})")
        print(f"Validation Loss: {val_loss:.4f} (Recon: {val_recon_loss:.4f}, KL: {val_kl_loss:.4f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_model(BASEPATH)
            print(f"New best validation loss!")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered")
                break
        
        scheduler.step(val_loss)
    
    return model

def train_goal(base_model, train_loader, val_loader, goal_latents):
    vae_GA = Autoencoder().to(DEVICE)
    vae_GA.load_state_dict(base_model.state_dict())
    
    # Compute prior parameters with numerical stability
    mu_prior = torch.tensor(np.mean(goal_latents, axis=0), device=DEVICE)
    var_prior = torch.tensor(np.var(goal_latents, axis=0) + 1e-6, device=DEVICE)

    def kl_divergence(mu, logvar):
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(
            var/var_prior + (mu - mu_prior).pow(2)/var_prior - 1 - logvar + torch.log(var_prior + 1e-8), 
            dim=1
        )
        return kl

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(vae_GA.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        vae_GA.train()
        train_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0
        
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(DEVICE)
            optimizer.zero_grad()
            recon, _, mu, logvar = vae_GA(images)
            
            # Compute losses separately for monitoring
            recon_loss = criterion(recon, images) / images.size(0)
            kl_loss = kl_divergence(mu, logvar).mean()
            
            # Total loss with beta weighting
            loss = recon_loss + BETA_GOAL * kl_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_GA.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
        
        train_loss /= len(train_loader)
        recon_loss_avg = recon_loss_total / len(train_loader)
        kl_loss_avg = kl_loss_total / len(train_loader)
        
        # Evaluate validation loss
        val_loss, val_recon_loss, val_kl_loss = evaluate_model(vae_GA, val_loader, criterion, BETA_GOAL)
        
        # Print training progress
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Training Loss: {train_loss:.4f} (Recon: {recon_loss_avg:.4f}, KL: {kl_loss_avg:.4f})")
        print(f"Validation Loss: {val_loss:.4f} (Recon: {val_recon_loss:.4f}, KL: {val_kl_loss:.4f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            vae_GA.save_model(GOALPATH)
            print(f"New best validation loss!")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered")
                break
        
        scheduler.step(val_loss)
    
    return vae_GA

def fit_gmm_to_latents(latents, n_components=5):
    """Fit a GMM to the latent space and return the model"""
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(latents)
    return gmm


def train_gmm_vae(base_model, train_loader, val_loader, goal_latents):
    # Check if we have valid latents
    if goal_latents is None or len(goal_latents) == 0:
        raise ValueError("No valid latents provided for GMM VAE training")
    
    vae_GMM = Autoencoder().to(DEVICE)
    vae_GMM.load_state_dict(base_model.state_dict())
    
    # Fit GMM to goal latents to get the prior distribution
    gmm = fit_gmm_to_latents(goal_latents, n_components=GMM_N_COMPONENTS)
    
    def gmm_kl_divergence(mu, logvar):
        """
        Compute KL divergence between variational posterior q(z|x) and GMM prior p(z)
        """
        # Convert logvar to variance
        var = logvar.exp()
        
        # Compute q(z|x), which is a normal distribution with diagonal covariance
        q_dist = MultivariateNormal(mu, torch.diag_embed(var))
        
        # Compute the log likelihood of q(z|x)
        log_qzx = q_dist.log_prob(mu)
        
        # Compute the log likelihood of p(z) (GMM prior)
        log_pz = []
        for pi_k, mu_k, sigma_k in zip(gmm.weights_, gmm.means_, gmm.covariances_):
            pi_k = torch.tensor(pi_k, dtype=torch.float32, device=mu.device)
            mu_k = torch.tensor(mu_k, dtype=torch.float32, device=mu.device)
            sigma_k = torch.tensor(sigma_k, dtype=torch.float32, device=mu.device)
            p_dist = MultivariateNormal(mu_k, sigma_k)
            log_pz.append(torch.log(pi_k) + p_dist.log_prob(mu))
        
        log_pz = torch.stack(log_pz, dim=-1)
        log_pz = torch.logsumexp(log_pz, dim=-1)
        
        # Compute KL divergence
        kl = log_qzx - log_pz
        return kl

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(vae_GMM.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        vae_GMM.train()
        train_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0
        
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(DEVICE)
            optimizer.zero_grad()
            recon, _, mu, logvar = vae_GMM(images)
            
            # Compute losses separately for monitoring
            recon_loss = criterion(recon, images) / images.size(0)
            kl_loss = gmm_kl_divergence(mu, logvar).mean()
            
            # Total loss with beta weighting
            loss = recon_loss + BETA_GMM * kl_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_GMM.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
        
        train_loss /= len(train_loader)
        recon_loss_avg = recon_loss_total / len(train_loader)
        kl_loss_avg = kl_loss_total / len(train_loader)
        
        # Evaluate validation loss
        val_loss, val_recon_loss, val_kl_loss = evaluate_model(vae_GMM, val_loader, criterion, BETA_GMM)
        
        # Print training progress
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Training Loss: {train_loss:.4f} (Recon: {recon_loss_avg:.4f}, KL: {kl_loss_avg:.4f})")
        print(f"Validation Loss: {val_loss:.4f} (Recon: {val_recon_loss:.4f}, KL: {val_kl_loss:.4f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            vae_GMM.save_model(GMMPATH)
            print(f"New best validation loss!")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered")
                break
        
        scheduler.step(val_loss)
    
    return vae_GMM

def load_models():
    # Load data once
    train_loader, val_loader = get_mnist_loader()
    
    # Initialize models
    vae_base = Autoencoder().to(DEVICE)
    vae_goal = Autoencoder().to(DEVICE)
    vae_gmm = Autoencoder().to(DEVICE)
    
    # Load or train base model
    if os.path.exists(BASEPATH) and not FORCE_RETRAIN_BASE:
        print("Loading base model...")
        vae_base.load_state_dict(torch.load(BASEPATH))
    else:
        print("Training base model...")
        vae_base = train_base(vae_base, train_loader, val_loader)
    
    # Collect latents for goal and GMM models
    print("Collecting latents for goal and GMM models...")
    goal_latents = collect_latents(vae_base, train_loader)
    
    # Load or train goal model
    if os.path.exists(GOALPATH) and not FORCE_RETRAIN_GOAL:
        print("Loading goal model...")
        vae_goal.load_state_dict(torch.load(GOALPATH))
    else:
        print("Training goal model...")
        vae_goal = train_goal(vae_base, train_loader, val_loader, goal_latents)
    
    # Load or train GMM model
    if os.path.exists(GMMPATH) and not FORCE_RETRAIN_GMM:
        print("Loading GMM model...")
        vae_gmm.load_state_dict(torch.load(GMMPATH))
    else:
        print("Training GMM model...")
        vae_gmm = train_gmm_vae(vae_base, train_loader, val_loader, goal_latents)
    
    return vae_base, vae_goal, vae_gmm, train_loader


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # Ensure image path exists if saving is enabled
    if SAVE_IMAGES:
        ensure_image_path()

    vae_Base, vae_GA, vae_GMM, train_loader = load_models()
    
    print("\nAnalyzing latent space of VAE Base...")
    base_latents, base_labels = analyze_latents(vae_Base, train_loader, "base")
    plot_latent_space_2d(base_latents, base_labels, title="VAE Base Latent Space", model_name="base")
    sample_and_visualize(vae_Base, model_name="base")

    print("\nAnalyzing latent space of VAE GA...")
    ga_latents, ga_labels = analyze_latents(vae_GA, train_loader, "goal")
    plot_latent_space_2d(ga_latents, ga_labels, title="VAE GA Latent Space", model_name="goal")
    sample_and_visualize(vae_GA, model_name="goal")
    
    print("\nAnalyzing latent space of VAE GMM...")
    gmm_latents, gmm_labels = analyze_latents(vae_GMM, train_loader, "gmm")
    plot_latent_space_2d(gmm_latents, gmm_labels, title="VAE GMM Latent Space", model_name="gmm")
    sample_and_visualize(vae_GMM, model_name="gmm")


    '''

    base_latents, base_labels = getModelLatents(vae_Base, train_loader)  
    gmm_latents, gmm_labels = getModelLatents(vae_GMM, train_loader)

    plot_latent_space_2d(base_latents, base_labels, title="VAE Base Latent Space", dim1=0, dim2=1)
    plot_latent_space_2d(gmm_latents, gmm_labels, title="VAE GMM Latent Space", dim1=0, dim2=1)

    plot_latent_space_2d(base_latents, base_labels, title="VAE Base Latent Space", dim1=0, dim2=2)
    plot_latent_space_2d(gmm_latents, gmm_labels, title="VAE GMM Latent Space", dim1=0, dim2=2)

    plot_latent_space_2d(base_latents, base_labels, title="VAE Base Latent Space", dim1=0, dim2=3)
    plot_latent_space_2d(gmm_latents, gmm_labels, title="VAE GMM Latent Space", dim1=0, dim2=3)

    plot_latent_space_2d(base_latents, base_labels, title="VAE Base Latent Space", dim1=1, dim2=2)
    plot_latent_space_2d(gmm_latents, gmm_labels, title="VAE GMM Latent Space", dim1=1, dim2=2)

    plot_latent_space_2d(base_latents, base_labels, title="VAE Base Latent Space", dim1=1, dim2=3)
    plot_latent_space_2d(gmm_latents, gmm_labels, title="VAE GMM Latent Space", dim1=1, dim2=3)

    plot_latent_space_2d(base_latents, base_labels, title="VAE Base Latent Space", dim1=2, dim2=3)
    plot_latent_space_2d(gmm_latents, gmm_labels, title="VAE GMM Latent Space", dim1=2, dim2=3)  
    '''   

