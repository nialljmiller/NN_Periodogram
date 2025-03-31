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
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "astropy",
        "tqdm",
        "scipy",
        # Either include direct install from GitHub repo
        "git+https://github.com/username/NN_FAP.git",
        # Or specify it as a regular dependency if NN_FAP is on PyPI
        # "NN_FAP",
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