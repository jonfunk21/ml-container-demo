import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from models.vae import VAE

def load_mnist():
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((len(x_test), 28, 28, 1))
    return x_test, y_test

def plot_reconstructions(model, x_test, n_samples=10, save_path="results"):
    os.makedirs(save_path, exist_ok=True)
    
    # Get reconstructions
    reconstructions = model.predict(x_test[:n_samples])
    
    # Plot original and reconstructed images
    plt.figure(figsize=(20, 4))
    for i in range(n_samples):
        # Original
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        
        # Reconstruction
        plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(reconstructions[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    
    plt.savefig(os.path.join(save_path, 'reconstructions.png'))
    plt.close()

def plot_latent_space(model, x_test, y_test, save_path="results"):
    os.makedirs(save_path, exist_ok=True)
    
    # Get latent space representations
    z_mean, z_log_var, _ = model.encode(x_test)
    
    # Plot latent space with color-coded labels
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap='tab10', alpha=0.5)
    plt.colorbar(scatter)
    plt.title('Latent Space Visualization')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.savefig(os.path.join(save_path, 'latent_space.png'))
    plt.close()

def main():
    # Load test data
    x_test, y_test = load_mnist()
    
    # Create and load model
    vae = VAE(latent_dim=2)
    # Build the model by calling it on a sample input
    vae.build((None, 28, 28, 1))
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    
    # Load weights (use the best checkpoint)
    checkpoint_dir = "vae/keras/checkpoints"
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
    if checkpoints:
        best_checkpoint = sorted(checkpoints)[-1]  # Get the latest checkpoint
        vae.load_weights(os.path.join(checkpoint_dir, best_checkpoint))
        print(f"Loaded weights from {best_checkpoint}")
    else:
        print("No checkpoints found!")
        return
    
    # Generate and save reconstructions
    plot_reconstructions(vae, x_test, save_path="vae/keras/results")
    
    # Plot latent space
    plot_latent_space(vae, x_test, y_test, save_path="vae/keras/results")

if __name__ == "__main__":
    main() 