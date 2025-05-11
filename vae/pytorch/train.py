import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import os
from models.vae import VAE

def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    return train_loader, test_loader

def train_epoch(model, train_loader, optimizer, device, epoch, writer):
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        loss, recon_loss, kl_loss = model.loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
        
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    # Log average losses
    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon_loss = train_recon_loss / len(train_loader.dataset)
    avg_kl_loss = train_kl_loss / len(train_loader.dataset)
    
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Reconstruction_Loss/train', avg_recon_loss, epoch)
    writer.add_scalar('KL_Loss/train', avg_kl_loss, epoch)
    
    return avg_loss

def test(model, test_loader, device, epoch, writer):
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            loss, recon_loss, kl_loss = model.loss_function(recon_batch, data, mu, log_var)
            
            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kl_loss += kl_loss.item()
    
    # Log average losses
    avg_loss = test_loss / len(test_loader.dataset)
    avg_recon_loss = test_recon_loss / len(test_loader.dataset)
    avg_kl_loss = test_kl_loss / len(test_loader.dataset)
    
    writer.add_scalar('Loss/test', avg_loss, epoch)
    writer.add_scalar('Reconstruction_Loss/test', avg_recon_loss, epoch)
    writer.add_scalar('KL_Loss/test', avg_kl_loss, epoch)
    
    print(f'====> Test set loss: {avg_loss:.4f}')
    return avg_loss

class CustomCheckpoint:
    def __init__(self, filepath, save_freq=10):
        self.filepath = filepath
        self.save_freq = save_freq

    def __call__(self, model, optimizer, epoch, loss):
        if (epoch + 1) % self.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, self.filepath.format(epoch=epoch + 1, loss=loss))

def main():
    # Create directories
    os.makedirs('vae/pytorch/logs', exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_mnist()
    
    # Create model
    model = VAE(latent_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Setup TensorBoard
    writer = SummaryWriter('vae/pytorch/logs')
    
    # Setup checkpoint
    checkpoint_path = "vae/pytorch/checkpoints/vae_epoch_{epoch:02d}_loss_{loss:.4f}.pt"
    checkpoint_callback = CustomCheckpoint(checkpoint_path, save_freq=10)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(1, 101):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, writer)
        test_loss = test(model, test_loader, device, epoch, writer)
        
        # Save checkpoint
        checkpoint_callback(model, optimizer, epoch, test_loss)
    
    writer.close()

if __name__ == "__main__":
    main() 