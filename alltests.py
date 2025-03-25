import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.mixture import GaussianMixture
import numpy as np
from autoencoder import Autoencoder
from stats import plot_histogram, plot_qq, plot_latent_space_2d, sample_and_visualize
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

def analyze_latents(model, dataloader):
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
    
    plot_histogram(latents[:, 0], "Latent Space Distribution")
    plot_qq(latents[:, 0], "Q-Q Plot")
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
            print(f"New best validation loss! Patience counter: {patience_counter}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered")
                break
        
        scheduler.step(val_loss)

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
        return torch.mean(torch.clamp(kl, min=0, max=100))
    
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(vae_GA.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    vae_GA.train()
    for epoch in range(EPOCHS):
        train_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0
        
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(DEVICE)
            optimizer.zero_grad()
            recon, _, mu, logvar = vae_GA(images)
            
            recon_loss = criterion(recon, images) / images.size(0)
            kl_loss = kl_divergence(mu, logvar)
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
        
        # Evaluate validation loss with same components
        val_loss, val_recon_loss, val_kl_loss = evaluate_model(vae_GA, val_loader, criterion, BETA_GOAL)
        
        # Print training progress
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Training Loss: {train_loss:.4f} (Recon: {recon_loss_avg:.4f}, KL: {kl_loss_avg:.4f})")
        print(f"Validation Loss: {val_loss:.4f} (Recon: {val_recon_loss:.4f}, KL: {val_kl_loss:.4f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            vae_GA.save_model(GOALPATH)
            print(f"New best validation loss! Patience counter: {patience_counter}")
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

def gmm_log_likelihood(z, gmm):
    """Compute log likelihood of samples under the GMM"""
    z_np = z.detach().cpu().numpy()
    # Check for invalid values
    if np.isnan(z_np).any() or np.isinf(z_np).any():
        return torch.tensor(float('-inf'), device=z.device)
    # Clip values to prevent overflow
    z_np = np.clip(z_np, -1e6, 1e6)
    log_prob = torch.from_numpy(gmm.score_samples(z_np)).to(z.device)
    return torch.clamp(log_prob, min=-1e6, max=1e6)  # Clip log probabilities

def train_gmm_vae(base_model, train_loader, val_loader, goal_latents):
    # Check if we have valid latents
    if len(goal_latents) == 0:
        raise ValueError("No valid goal latents provided!")
    
    # Clip goal latents to prevent overflow
    goal_latents = np.clip(goal_latents, -1e6, 1e6)
    
    # Fit GMM to the latents
    print("Fitting GMM to latent space...")
    print(f"Using {len(goal_latents)} samples for GMM fitting")
    gmm = fit_gmm_to_latents(goal_latents, n_components=GMM_N_COMPONENTS)
    
    # Initialize GMM VAE
    vae_GMM = Autoencoder().to(DEVICE)
    vae_GMM.load_state_dict(base_model.state_dict())
    
    def gmm_kl_divergence(mu, logvar):
        # Clip values for numerical stability
        mu = torch.clamp(mu, -1e6, 1e6)
        logvar = torch.clamp(logvar, -1e6, 1e6)
        
        # Sample points from the encoder distribution
        z = vae_GMM.reparameterize(mu, logvar)
        
        # Ensure no invalid values
        if torch.isnan(z).any() or torch.isinf(z).any():
            print("Warning: Invalid values in sampled latents")
            return torch.tensor(1e3, device=DEVICE)  # Return large but finite value
        
        # Compute log likelihood under GMM prior
        log_p_z = gmm_log_likelihood(z, gmm)
        
        # Compute log likelihood under standard normal (encoder distribution)
        log_2pi = 1.8378770664093453  # Precomputed log(2π) for stability
        log_q_z = -0.5 * torch.sum(1 + logvar + log_2pi, dim=1)
        
        # Clip values for stability
        log_q_z = torch.clamp(log_q_z, -1e6, 1e6)
        
        # Check for invalid values
        if torch.isnan(log_p_z).any() or torch.isnan(log_q_z).any() or \
           torch.isinf(log_p_z).any() or torch.isinf(log_q_z).any():
            print("Warning: Invalid values in KL divergence computation")
            return torch.tensor(1e3, device=DEVICE)  # Return large but finite value
        
        kl_div = (log_q_z - log_p_z).mean()
        return torch.clamp(kl_div, -1e6, 1e6)  # Final clipping for stability
    
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(vae_GMM.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Training GMM VAE...")
    vae_GMM.train()
    for epoch in range(EPOCHS):
        train_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0
        num_valid_batches = 0
        
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(DEVICE)
            optimizer.zero_grad()
            recon, _, mu, logvar = vae_GMM(images)
            
            # Check for invalid values
            if torch.isnan(mu).any() or torch.isnan(logvar).any() or \
               torch.isinf(mu).any() or torch.isinf(logvar).any():
                print("Warning: Invalid values in encoder output, skipping batch")
                continue
            
            recon_loss = criterion(recon, images) / images.size(0)
            kl_loss = gmm_kl_divergence(mu, logvar)
            
            # Skip batch if loss is invalid or too large
            if torch.isinf(kl_loss) or torch.isnan(kl_loss) or kl_loss > 1e5:
                print("Warning: Invalid or too large KL loss, skipping batch")
                continue
                
            loss = recon_loss + BETA_GMM * kl_loss            
            
            # Skip if total loss is invalid
            if torch.isinf(loss) or torch.isnan(loss):
                print("Warning: Invalid total loss, skipping batch")
                continue
                
            loss.backward()
            
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(vae_GMM.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
            num_valid_batches += 1
        
        if num_valid_batches == 0:
            print("Warning: No valid batches in epoch")
            continue
            
        train_loss /= num_valid_batches
        recon_loss_avg = recon_loss_total / num_valid_batches
        kl_loss_avg = kl_loss_total / num_valid_batches
        
        # Evaluate validation loss with same components
        val_loss, val_recon_loss, val_kl_loss = evaluate_model(vae_GMM, val_loader, criterion, BETA_GMM)
        
        # Print training progress
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Training Loss: {train_loss:.4f} (Recon: {recon_loss_avg:.4f}, KL: {kl_loss_avg:.4f})")
        print(f"Validation Loss: {val_loss:.4f} (Recon: {val_recon_loss:.4f}, KL: {val_kl_loss:.4f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            vae_GMM.save_model(GMMPATH)
            print(f"New best validation loss! Patience counter: {patience_counter}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered")
                break
        
        scheduler.step(val_loss)
    
    return vae_GMM, gmm

def load_models():
    # Load data once
    train_loader, val_loader = get_mnist_loader()
    
    # Initialize models
    vae_Base = Autoencoder().to(DEVICE)
    vae_GA = Autoencoder().to(DEVICE)
    vae_GMM = Autoencoder().to(DEVICE)

    # Train or load base VAE
    if os.path.exists(BASEPATH) and not FORCE_RETRAIN_BASE:
        vae_Base.load_model(BASEPATH)
        print("Loaded pre-trained VAE Base.")
    else:
        print("Training VAE Base...")
        train_base(vae_Base, train_loader, val_loader)
    
    # Collect latents once for both GA and GMM
    print("Collecting latents from base model...")
    goal_latents = collect_latents(vae_Base, train_loader)
    
    # Train or load goal-conditioned VAE
    if os.path.exists(GOALPATH) and not FORCE_RETRAIN_GOAL:
        vae_GA.load_model(GOALPATH)
        print("Loaded pre-trained VAE GA.")
    else:
        print("Training VAE GA...")
        vae_GA = train_goal(vae_Base, train_loader, val_loader, goal_latents)
    
    # Train or load GMM VAE
    if os.path.exists(GMMPATH) and not FORCE_RETRAIN_GMM:
        vae_GMM.load_model(GMMPATH)
        print("Loaded pre-trained VAE GMM.")
    else:
        print("Training VAE GMM...")
        vae_GMM, gmm = train_gmm_vae(vae_Base, train_loader, val_loader, goal_latents)

    return vae_Base, vae_GA, vae_GMM, train_loader



if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    vae_Base, vae_GA, vae_GMM, train_loader = load_models()
    
    print("\nAnalyzing latent space of VAE Base...")
    base_latents, base_labels = analyze_latents(vae_Base, train_loader)
    plot_latent_space_2d(base_latents, base_labels, title="VAE Base Latent Space")
    sample_and_visualize(vae_Base)

    print("\nAnalyzing latent space of VAE GA...")
    ga_latents, ga_labels = analyze_latents(vae_GA, train_loader)
    plot_latent_space_2d(ga_latents, ga_labels, title="VAE GA Latent Space")
    sample_and_visualize(vae_GA)
    
    print("\nAnalyzing latent space of VAE GMM...")
    gmm_latents, gmm_labels = analyze_latents(vae_GMM, train_loader)
    plot_latent_space_2d(gmm_latents, gmm_labels, title="VAE GMM Latent Space")
    sample_and_visualize(vae_GMM)


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

