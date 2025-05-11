import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from models.vae import VAE

def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    
    return test_loader

def plot_reconstructions(model, test_loader, device, save_path):
    model.eval()
    with torch.no_grad():
        # Get a batch of test images
        data, _ = next(iter(test_loader))
        data = data.to(device)
        
        # Generate reconstructions
        recon_batch, _, _ = model(data)
        
        # Move tensors to CPU and convert to numpy
        data = data.cpu().numpy()
        recon_batch = recon_batch.cpu().numpy()
        
        # Create figure with subplots
        n = min(8, len(data))
        fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
        
        for i in range(n):
            # Original image
            axes[0, i].imshow(data[i].squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original')
            
            # Reconstructed image
            axes[1, i].imshow(recon_batch[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def plot_latent_space(model, test_loader, device, save_path):
    model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            latents.append(mu.cpu().numpy())
            labels.append(target.numpy())
    
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.5)
    plt.colorbar(scatter)
    plt.title('Latent Space Visualization')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.savefig(save_path)
    plt.close()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = VAE(latent_dim=2).to(device)
    
    # Find the best checkpoint
    checkpoint_dir = 'vae/pytorch/checkpoints'
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('vae_epoch_')]
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in the checkpoints directory")
    
    # Load the best checkpoint
    best_checkpoint = sorted(checkpoints)[-1]  # Get the latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load test data
    test_loader = load_mnist()
    
    # Create results directory
    os.makedirs('vae/pytorch/results', exist_ok=True)
    
    # Generate and save reconstructions
    plot_reconstructions(model, test_loader, device, 'vae/pytorch/results/reconstructions.png')
    print("Saved reconstructions to vae/pytorch/results/reconstructions.png")
    
    # Generate and save latent space visualization
    plot_latent_space(model, test_loader, device, 'vae/pytorch/results/latent_space.png')
    print("Saved latent space visualization to vae/pytorch/results/latent_space.png")

if __name__ == "__main__":
    main() 