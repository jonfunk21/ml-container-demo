import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import datetime
from models.vae import VAE

def load_mnist():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), 28, 28, 1))
    x_test = x_test.reshape((len(x_test), 28, 28, 1))
    return x_train, x_test

class CustomModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath, save_freq=10):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            self.model.save_weights(self.filepath.format(epoch=epoch + 1, **logs))

def main():
    # Create directories
    os.makedirs('vae/keras/logs', exist_ok=True)
    
    # Load data
    x_train, x_test = load_mnist()
    
    # Create model
    vae = VAE(latent_dim=2)
    vae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=lambda y_true, y_pred: 0.0  # Dummy loss, real loss is handled in train_step
    )
    
    # Setup callbacks
    checkpoint_path = "vae/keras/checkpoints/vae_epoch_{epoch:02d}_loss_{loss:.4f}.weights.h5"
    checkpoint_callback = CustomModelCheckpoint(
        checkpoint_path,
        save_freq=10
    )
    
    # TensorBoard callback
    log_dir = os.path.join("vae/keras/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    # Train model
    vae.fit(
        x_train,
        epochs=100,
        batch_size=128,
        validation_data=(x_test, x_test),
        callbacks=[checkpoint_callback, tensorboard_callback]
    )

if __name__ == "__main__":
    main() 