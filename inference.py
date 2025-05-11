import argparse
import os
import subprocess

def run_inference(framework, checkpoint=None):
    # Determine the container to use
    container = os.path.join(os.getcwd(), f"{framework}-mnist.sif")
    if not os.path.exists(container):
        raise FileNotFoundError(f"Container {container} not found. Please build it first.")

    # Construct the command to run the inference script
    cmd = [
        "apptainer", "exec",
        "--bind", os.getcwd(),
        container,
        "python3",
        f"vae/{framework}/inference.py"
    ]
    
    if checkpoint:
        cmd.append("--checkpoint")
        cmd.append(checkpoint)

    # Run the inference command
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description='Run inference using a specified framework.')
    parser.add_argument('--framework', choices=['keras', 'pytorch'], required=True, help='Framework to use for inference.')
    parser.add_argument('--checkpoint', help='Path to the checkpoint file. If not provided, the latest checkpoint will be used.')
    args = parser.parse_args()

    run_inference(args.framework, args.checkpoint)

if __name__ == "__main__":
    main() 