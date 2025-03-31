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

# Install package and dependencies
echo "Installing NN_Periodogram and dependencies..."
pip3 install -e .

echo "Installation completed successfully!"
echo "You can now use NN_Periodogram with the 'nnp' command or by running 'python NNP.py'"