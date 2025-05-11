import tensorflow as tf
from tensorflow import keras
import numpy as np

class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, latent_dim=2, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = keras.Sequential([
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, 3, activation='relu', strides=2, padding='same'),
            keras.layers.Conv2D(64, 3, activation='relu', strides=2, padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dense(16, activation='relu')
        ])
        
        # Latent space
        self.z_mean = keras.layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")
        self.sampling = Sampling()
        
        # Decoder
        self.decoder = keras.Sequential([
            keras.layers.Input(shape=(latent_dim,)),
            keras.layers.Dense(7 * 7 * 64, activation='relu'),
            keras.layers.Reshape((7, 7, 64)),
            keras.layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same'),
            keras.layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same'),
            keras.layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')
        ])
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def encode(self, x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        z_mean, z_log_var, z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encode(data)
            reconstruction = self.decode(z)
            
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            
            # KL divergence loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            # Total loss
            total_loss = reconstruction_loss + kl_loss
            
        # Compute gradients and update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        } 