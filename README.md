# NN_Periodogram: Flexible Two-Stage Neural Network Periodogram Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

A powerful time series analysis tool optimized for detecting short-period signals using a novel two-stage periodogram approach with machine learning capabilities.

## Overview

NN_Periodogram implements a generalized two-stage periodogram approach for time series analysis using Neural Network-based False Alarm Probability (NN_FAP) estimation. The tool is specifically optimized for detecting short-period signals in astronomical data, but is applicable to various types of time series analysis problems.

### Key Features

- **Two-Stage Periodogram Approach**: Enhances detection by using complementary period ranges
- **Neural Network Integration**: Leverages machine learning for improved sensitivity over traditional methods
- **Intelligent Period Handling**: Mathematical justification for period density, subsampling, and complementary ranges
- **Multiple Analysis Methods**:
  - Chunk method
  - Sliding window method 
  - Subtraction method for enhanced signal detection
- **Versatile Input Support**: Automatically detects and processes various file formats (CSV, FITS, TXT)
- **Parallel Processing**: Utilizes multi-core processing for faster analysis
- **Rich Visualization**: Generates comprehensive plots including phase-folded light curves

## Installation

### Prerequisites

- Python 3.6 or higher
- Pip package manager

### Basic Installation

The easiest way to install is using the provided installation script:

```bash
# Clone the repository
git clone https://github.com/nialljmiller/NN_Periodogram.git
cd NN_Periodogram

# Run the installation script (downloads model files automatically)
bash install.sh
```

### Manual Installation

If you prefer to install manually:

```bash
# Clone the repository
git clone https://github.com/nialljmiller/NN_Periodogram.git
cd NN_Periodogram

# Install the package with model download
python setup.py download_model
python setup.py install

# Or in development mode
pip install -e .
python setup.py download_model
```

### Alternative Installation Methods

If you prefer to install directly from GitHub:

```bash
pip install git+https://github.com/nialljmiller/NN_Periodogram.git
```

**Note:** When installing from GitHub, you'll need to download the model files manually:

```bash
# Download and extract model files
wget https://nialljmiller.com/projects/FAP/model.tar.gz
tar -xzf model.tar.gz
mv model NN_FAP/model
```

## Quick Start

1. Create an `inlist.txt` configuration file or use the default one provided
2. Run the tool

```bash
python NNP.py
```

For custom configuration:

```bash
# Edit the inlist.txt file
nano inlist.txt

# Run the tool
python NNP.py
```

## Configuration

NN_Periodogram uses an `inlist.txt` file for configuration. Below is a sample configuration with explanations:

```
# Input/output parameters
input_file=detections.csv  # Path to input file (required)
output_dir=./results       # Directory for output files
output_prefix=periodogram  # Prefix for output filenames

# Data parameters
time_col=None              # Column name for time values (None for auto-detection)
flux_col=None              # Column name for flux values (None for auto-detection)
error_col=None             # Column name for error values (None for auto-detection)
file_format=None           # File format: csv, fits, txt, or None for auto-detection
normalize=True             # Normalize flux by median
remove_outliers=True       # Remove outliers using sigma clipping
sigma=5.0                  # Sigma threshold for outlier removal

# Period search parameters
period_min=0.01            # Minimum period to search (days)
period_max=10.0            # Maximum period to search (days)
n_periods=1000000          # Number of periods to sample
oversample_factor=10       # Oversampling factor for period grid
use_complementary=True     # Use complementary period range for method comparison

# NN_FAP parameters
nn_fap_model_path=NN_FAP/model/  # Path to NN_FAP model directory (required)
window_size=200            # Size of each window for sliding window method
chunk_size=200             # Size of each chunk for chunk method
step_size=50               # Step size between windows
n_workers=None             # Number of worker processes (None for auto)

# Plotting parameters
plot_log_scale=True        # Use logarithmic scale for period axes in plots
```

## Available Configuration Options

### Input/Output Parameters
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `input_file` | Path to input file (required) | None |
| `output_dir` | Directory for output files | ./results |
| `output_prefix` | Prefix for output filenames | periodogram |

### Data Parameters
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `time_col` | Column name for time values | None (auto-detect) |
| `flux_col` | Column name for flux values | None (auto-detect) |
| `error_col` | Column name for error values | None (auto-detect) |
| `file_format` | File format: csv, fits, txt | None (auto-detect) |
| `normalize` | Normalize flux by median | True |
| `remove_outliers` | Remove outliers using sigma clipping | True |
| `sigma` | Sigma threshold for outlier removal | 5.0 |

### Period Search Parameters
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `period_min` | Minimum period to search (days) | 0.01 |
| `period_max` | Maximum period to search (days) | 1.0 |
| `n_periods` | Number of periods to sample | 1000 |
| `oversample_factor` | Oversampling factor for period grid | 10 |
| `use_complementary` | Use complementary period range | True |

### NN_FAP Parameters
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `nn_fap_model_path` | Path to NN_FAP model directory | None (required) |
| `window_size` | Size of each window for sliding window method | 200 |
| `chunk_size` | Size of each chunk for chunk method | 200 |
| `step_size` | Step size between windows | 50 |
| `n_workers` | Number of worker processes | None (auto) |

### Plotting Parameters
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `plot_log_scale` | Use logarithmic scale for period axes | True |
| `plot_title` | Custom title for plots | None |

## Understanding the Methods

### Chunk Method
This method divides the time series into non-overlapping chunks and analyzes each chunk independently. The results are then averaged to produce a robust periodogram. This approach is effective for handling systematic errors and long-term trends.

### Sliding Window Method
Unlike the chunk method, the sliding window method uses overlapping windows that slide across the time series. This provides better sensitivity to signals that might fall between chunk boundaries.

### Subtraction Method
This novel approach enhances the detection of signals in specific period ranges by subtracting the complementary periodogram from the primary periodogram. This helps isolate signals of interest from the background noise and systematic effects.

## NN_FAP Models

NN_Periodogram requires trained neural network models from the NN_FAP package to function properly. These models are used to estimate False Alarm Probabilities (FAP) for detected signals.

### Model Files

The required model files will be downloaded automatically during installation. They include:
- Neural network model definition files (`.json`)
- Pre-trained weights (`.h5`)
- Model history and performance metrics

### Manual Model Download

If the automatic download fails, you can manually download the model files:

```bash
wget https://nialljmiller.com/projects/FAP/model.tar.gz
tar -xzf model.tar.gz
mkdir -p NN_FAP
mv model NN_FAP/
```

The model files should be placed in the `NN_FAP/model/` directory relative to your installation.

## Output Files

For each analysis, the following files are generated in the output directory:

- **periodogram_[filename].png**: Main periodogram plot showing all methods
- **periodogram_[filename]_folded_chunk_method.png**: Phase-folded light curve for the best period from the chunk method
- **periodogram_[filename]_folded_sliding_window_method.png**: Phase-folded light curve for the best period from the sliding window method
- **periodogram_[filename]_folded_subtraction_method.png**: Phase-folded light curve for the best period from the subtraction method
- **periodogram_[filename]_results.json**: JSON file containing the analysis results and best periods

## Technical Details

### Two-Stage Approach

The two-stage approach works as follows:

1. **Primary Periodogram**: Focuses on the user-specified period range of interest, using the sliding window method
2. **Complementary Periodogram**: Samples the complementary period range to capture harmonics and aliases, using the chunk method
3. **Subtraction Method**: Enhances signal detection by subtracting the complementary periodogram from the primary periodogram

This approach is particularly effective for:
- Detecting weak signals in noisy data
- Distinguishing true periods from aliases
- Handling data with systematic errors or gaps

### Period Grid Optimization

The tool automatically calculates an optimal period grid based on:
- The time span of the data
- Nyquist frequency considerations
- Oversampling factor for increased resolution around expected signals
- Mathematical justification for period density based on signal characteristics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This tool builds upon and extends the functionality of the NN_FAP package
- Thanks to all contributors and users who provide feedback and suggestions

## Citation

If you use this software in your research, please cite:

```
@software{NN_Periodogram,
  author       = {Niall Miller},
  title        = {NN\_Periodogram: Flexible Two-Stage Neural Network Periodogram Analyzer},
  year         = {2025},
  url          = {https://github.com/username/NN_Periodogram}
}
```