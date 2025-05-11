# Machine Learning with Apptainers: A Practical Example

This repository demonstrates how to use Apptainers (formerly Singularity) to manage different machine learning environments. We'll implement and train a Variational Autoencoder (VAE) using both Keras (TensorFlow) and PyTorch frameworks, showcasing how Apptainers can help manage dependencies and ensure reproducibility.

## Background

### What are Containers?
Containers are like pre-configured, self-contained environments that package up code and all its dependencies. They ensure that your application runs the same way regardless of where it's deployed. Think of them as shipping containers for software - they keep everything needed to run your application together and isolated from the host system.

### Why Apptainer (formerly Singularity)?
Apptainer is a container platform specifically designed for high-performance computing and scientific workloads. It has several advantages over other container solutions:

1. **Security**: Unlike Docker, Apptainer doesn't require root privileges to run, making it ideal for shared computing environments like clusters and supercomputers.
2. **Simplicity**: Apptainer containers are single files (`.sif`), making them easy to share and move around.
3. **Performance**: Apptainer has minimal overhead and can run applications at near-native speed.
4. **Compatibility**: It can run Docker containers and is compatible with most HPC environments.

### How Do Apptainers Help in Machine Learning?
Machine learning projects often require different versions of:
- Python
- Deep learning frameworks (TensorFlow, PyTorch, etc.)
- CUDA for GPU acceleration
- Various Python packages

Managing these dependencies can be challenging, especially when:
- Working with multiple projects
- Collaborating with others
- Running on different systems
- Using shared computing resources

Apptainers solve these problems by:
1. **Isolating Environments**: Each project can have its own container with specific dependencies
2. **Ensuring Reproducibility**: The same container will run identically on any system
3. **Simplifying Deployment**: No need to install or configure dependencies on the host system
4. **Enabling Portability**: Containers can be easily shared and run on different machines

### What You'll Learn
This project demonstrates how to:
1. Create Apptainer containers for different machine learning frameworks
2. Run code inside containers without worrying about dependencies
3. Share and reproduce results across different systems
4. Manage multiple projects with different requirements

## Why Apptainers?

Apptainers are particularly useful in machine learning for several reasons:

1. **Environment Isolation**: Each framework (Keras, PyTorch) can have its own container with specific dependencies, avoiding conflicts.
2. **Reproducibility**: The exact same environment can be recreated on any system that supports Apptainers.
3. **Portability**: Containers can be easily shared and run on different systems without worrying about dependency issues.
4. **Clean Workspace**: No need to install multiple versions of Python, CUDA, or other dependencies on your system.

## Project Structure

```
aptainer_test/
├── build_containers.py     # Script to build Keras and PyTorch containers
├── train.py               # Unified training script for both frameworks
├── inference.py           # Unified inference script for both frameworks
├── vae/
│   ├── keras/            # Keras implementation
│   │   ├── models/       # VAE model definition
│   │   ├── train.py      # Training script
│   │   ├── inference.py  # Inference script
│   │   ├── checkpoints/  # Saved model weights
│   │   ├── logs/         # Training logs
│   │   └── results/      # Generated visualizations
│   └── pytorch/          # PyTorch implementation
│       ├── models/       # VAE model definition
│       ├── train.py      # Training script
│       ├── inference.py  # Inference script
│       ├── checkpoints/  # Saved model weights
│       ├── logs/         # Training logs
│       └── results/      # Generated visualizations
└── README.md
```

## Getting Started

### Prerequisites

- Apptainer (formerly Singularity) installed on your system
- Python 3.x
- Git
- You must be on a Linux machine/VM

### Building the Containers

First, build the containers for both frameworks:

```bash
python build_containers.py
```

This will create two container files:
- `keras-mnist.sif`: Container with Keras (TensorFlow) and its dependencies
- `pytorch-mnist.sif`: Container with PyTorch and its dependencies

### Training the Models

You can train either the Keras or PyTorch VAE model using the unified training script:

```bash
# Train Keras model
python train.py --framework keras

# Train PyTorch model
python train.py --framework pytorch
```

The training script will:
1. Check if the required container exists
2. Run the training process
3. Display training duration
4. Save model checkpoints every 10 epochs
5. Save training logs for TensorBoard

### Monitoring Training

You can monitor the training progress using TensorBoard:

```bash
# For Keras
tensorboard --logdir vae/keras/logs

# For PyTorch
tensorboard --logdir vae/pytorch/logs
```

### Generating Reconstructions and Visualizations

After training, you can generate reconstructions and latent space visualizations:

```bash
# Generate visualizations for Keras model
python inference.py --framework keras

# Generate visualizations for PyTorch model
python inference.py --framework pytorch
```

This will create:
- Reconstructions of test images
- Latent space visualization with color-coded digit clusters

The results will be saved in the respective `results` directories.

## How It Works

1. **Container Management**: Each framework runs in its own container, ensuring isolated environments.
2. **Unified Interface**: The `train.py` and `inference.py` scripts provide a consistent interface for both frameworks.
3. **Data Persistence**: The containers are bound to your local directory, so all data, checkpoints, and results are saved on your system.
4. **Framework-Specific Code**: Each framework's implementation is kept separate but follows the same interface.

## Benefits of This Approach

1. **Easy Framework Switching**: Train or run inference with either framework using the same commands.
2. **No Environment Conflicts**: Each framework runs in its own container with its specific dependencies.
3. **Reproducible Results**: The same container will produce the same results on any system.
4. **Clean Development**: No need to manage multiple Python environments or worry about dependency conflicts.

## Contributing

Feel free to contribute to this project by:
1. Adding support for more frameworks
2. Improving the model architectures
3. Adding more visualization options
4. Enhancing the documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details. 