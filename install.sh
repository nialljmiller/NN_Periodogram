#!/bin/bash

# NN_Periodogram Installation Script
# This script sets up NN_Periodogram with all dependencies and model files

echo "Installing NN_Periodogram and dependencies..."

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python3; then
  echo "Python 3 is not installed. Please install Python 3.6 or higher."
  exit 1
fi

# Check Python version
python_version=$(python3 --version | awk '{print $2}')
echo "Found Python $python_version"

# Compare versions to ensure 3.6+
version_check=$(python3 -c "import sys; print(sys.version_info >= (3, 6))")
if [ "$version_check" != "True" ]; then
  echo "Error: Python 3.6 or higher is required. Found $python_version"
  exit 1
fi

# Check if pip is installed
if ! command_exists pip3; then
  echo "pip3 is not installed. Installing pip..."
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python3 get-pip.py
  rm get-pip.py
fi

# Set up virtual environment (optional)
if command_exists python3 -m venv; then
  echo "Would you like to use a virtual environment? (recommended) [y/N]"
  read -r use_venv
  
  if [[ $use_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
    
    # Activate based on OS
    if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
      source venv/bin/activate
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
      source venv/Scripts/activate
    else
      echo "Warning: Unsupported OS. Activate the virtual environment manually."
    fi
    
    echo "Virtual environment activated."
  fi
fi

# Download NN_FAP model files
MODEL_URL="https://nialljmiller.com/projects/FAP/model.tar.gz"
MODEL_TAR="model.tar.gz"
NN_FAP_DIR="NN_FAP"
MODEL_DIR="$NN_FAP_DIR/model"

echo "Downloading NN_FAP model files..."

# Make sure NN_FAP directory exists
mkdir -p "$NN_FAP_DIR"

# Copy NN_FAP Python files if they exist
if [ -d "$NN_FAP" ] && [ "$(ls -A $NN_FAP)" ]; then
  echo "Using existing NN_FAP files"
else
  echo "Checking for NN_FAP files in subfolder..."
  if [ -d "NN_FAP" ] && [ -f "NN_FAP/NN_FAP.py" ]; then
    echo "Found existing NN_FAP module"
  else
    echo "Warning: NN_FAP module files not found. The package may not work correctly."
    echo "Please ensure NN_FAP.py and related files are in the NN_FAP directory."
  fi
fi

# Download model tar.gz file
if command_exists curl; then
  curl -L "$MODEL_URL" -o "$MODEL_TAR"
elif command_exists wget; then
  wget "$MODEL_URL" -O "$MODEL_TAR"
else
  echo "Error: Neither curl nor wget is installed. Please install one of them to continue."
  exit 1
fi

# Extract model files
echo "Extracting model files..."
if [ -f "$MODEL_TAR" ]; then
  tar -xzf "$MODEL_TAR"
  
  # Get the name of the extracted directory
  EXTRACTED_DIR=$(tar -tzf "$MODEL_TAR" | head -1 | cut -f1 -d"/")
  
  # Remove existing model directory if present
  if [ -d "$MODEL_DIR" ]; then
    rm -rf "$MODEL_DIR"
  fi
  
  # Move extracted directory to NN_FAP/model
  if [ -d "$EXTRACTED_DIR" ]; then
    mv "$EXTRACTED_DIR" "$MODEL_DIR"
    echo "Model files extracted to $MODEL_DIR"
  else
    echo "Error: Failed to extract model directory."
    exit 1
  fi
  
  # Clean up
  rm "$MODEL_TAR"
else
  echo "Error: Failed to download model files."
  exit 1
fi

# Create a new setup.py file that doesn't rely on external GitHub packages
cat > setup.py << 'EOL'
from setuptools import setup, find_packages, Command
import os
import tarfile
import shutil
import urllib.request
import sys
from pathlib import Path

class DownloadModelCommand(Command):
    """Custom command to download NN_FAP model files."""
    description = 'Download NN_FAP model files'
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        # URL for model download
        model_url = "https://nialljmiller.com/projects/FAP/model.tar.gz"
        # Temporary download path
        model_tar_path = "model.tar.gz"
        # Destination directory
        nn_fap_dir = "NN_FAP"
        model_dir = os.path.join(nn_fap_dir, "model")
        
        # Create NN_FAP directory if it doesn't exist
        if not os.path.exists(nn_fap_dir):
            os.makedirs(nn_fap_dir)
            print(f"Created directory: {nn_fap_dir}")
        
        # Remove existing model directory if it exists
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            print(f"Removed existing directory: {model_dir}")
        
        print(f"Downloading model files from {model_url}...")
        # Download the model tarball
        try:
            urllib.request.urlretrieve(model_url, model_tar_path)
            print(f"Downloaded model files to {model_tar_path}")
            
            # Extract the tarball
            with tarfile.open(model_tar_path) as tar:
                # Get the name of the top-level directory in the archive
                archive_root = tar.getnames()[0].split('/')[0]
                # Extract all files
                tar.extractall()
                print(f"Extracted model files to {archive_root}")
                
                # If extraction created a different directory name, rename it to "model"
                if archive_root != "model":
                    # Move the extracted directory to NN_FAP/model
                    if os.path.exists(archive_root):
                        shutil.move(archive_root, model_dir)
                        print(f"Renamed directory: {archive_root} → {model_dir}")
                else:
                    # Just move the model directory to NN_FAP/model
                    shutil.move(archive_root, nn_fap_dir)
                    print(f"Moved directory: {archive_root} → {model_dir}")
            
            # Remove the tarball
            os.remove(model_tar_path)
            print(f"Removed temporary file: {model_tar_path}")
            
        except Exception as e:
            print(f"Error downloading or extracting model files: {e}")
            print("Please download the model files manually from:")
            print("https://nialljmiller.com/projects/FAP/model.tar.gz")
            print("and extract them to NN_FAP/model directory.")
            sys.exit(1)

# Call the download model command before installing
class CustomInstallCommand(Command):
    description = 'Install package and download model files'
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        self.run_command('download_model')
        self.run_command('install')

setup(
    name="NN_Periodogram",
    version="0.1.0",
    packages=find_packages() + ['NN_FAP'],
    package_data={
        'NN_FAP': ['*.py', 'model/*'],
    },
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "astropy",
        "tqdm",
        "scipy",
    ],
    author="Niall Miller",
    author_email="niall.j.miller@gmail.com",
    description="Flexible Two-Stage NN_FAP Periodogram Analyzer",
    keywords="astronomy, periodogram, time series, NN_FAP",
    url="https://github.com/nialljmiller/NN_Periodogram",
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'nnp=NNP:main',
        ],
    },
    cmdclass={
        'download_model': DownloadModelCommand,
        'custom_install': CustomInstallCommand,
    },
)
EOL

# Install package and dependencies
echo "Installing NN_Periodogram and dependencies..."
pip3 install -e .

echo "Installation completed successfully!"
echo "You can now use NN_Periodogram with the 'nnp' command or by running 'python NNP.py'"