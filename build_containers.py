import subprocess
import os
import tempfile
import shutil

def build_container(def_file, container_name):
    print(f"\nBuilding {container_name} container...")
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build in the temporary directory
        build_cmd = f"sudo apptainer build {tmpdir}/{container_name}.sif {def_file}"
        subprocess.run(build_cmd, shell=True, check=True)
        
        # Move the built container to the current directory
        shutil.move(f"{tmpdir}/{container_name}.sif", f"{container_name}.sif")
        print(f"Successfully built {container_name}.sif")

def create_directories():
    # Create VAE directory structure
    directories = [
        'vae/keras/models',
        'vae/keras/checkpoints',
        'vae/keras/logs',
        'vae/keras/results',
        'vae/pytorch/models',
        'vae/pytorch/checkpoints',
        'vae/pytorch/logs',
        'vae/pytorch/results',
        'data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    # Create necessary directories
    create_directories()
    
    # Build Keras container
    build_container("keras.def", "keras-mnist")
    
    # Build PyTorch container
    build_container("pytorch.def", "pytorch-mnist")

if __name__ == "__main__":
    main() 