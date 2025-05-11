import subprocess
import argparse
import os
import time

def train_model(framework):
    """
    Train a VAE model using the specified framework's Apptainer container.
    
    Args:
        framework (str): Either 'keras' or 'pytorch'
    """
    if framework not in ['keras', 'pytorch']:
        raise ValueError("Framework must be either 'keras' or 'pytorch'")
    
    # Get the absolute path of the current directory
    current_dir = os.path.abspath(os.path.dirname(__file__))
    
    container = os.path.join(current_dir, f"{framework}-mnist.sif")
    if not os.path.exists(container):
        raise FileNotFoundError(f"Container {container} not found. Please build it first using build_containers.py")
    
    print(f"\nTraining {framework.upper()} VAE model...")
    print("=" * 50)
    
    # Start time
    start_time = time.time()
    
    # Run training with current directory mounted and set as working directory
    train_cmd = f"apptainer run --bind {current_dir} --pwd {current_dir} {container} vae/{framework}/train.py"
    try:
        subprocess.run(train_cmd, shell=True, check=True)
        
        # Calculate and print training duration
        duration = time.time() - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Print instructions for viewing results
        print("\nTo view training progress:")
        print(f"apptainer run --bind {current_dir} --pwd {current_dir} {container} tensorboard --logdir=vae/{framework}/logs")
        
        print("\nTo generate reconstructions:")
        print(f"apptainer run --bind {current_dir} --pwd {current_dir} {container} vae/{framework}/inference.py")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during training: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Train VAE models using Apptainer containers')
    parser.add_argument('--framework', type=str, choices=['keras', 'pytorch'], required=True,
                      help='Framework to use (keras or pytorch)')
    
    args = parser.parse_args()
    
    # Get the absolute path of the current directory
    current_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Check if containers exist
    container_path = os.path.join(current_dir, f"{args.framework}-mnist.sif")
    if not os.path.exists(container_path):
        print(f"Error: {args.framework}-mnist.sif container not found.")
        print("Please build the containers first using build_containers.py")
        return
    
    # Train the model
    success = train_model(args.framework)
    
    if success:
        print("\nTraining completed successfully!")
    else:
        print("\nTraining failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 