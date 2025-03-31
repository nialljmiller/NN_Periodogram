# NN_Periodogram: A Flexible Two-Stage Neural Network Periodogram Analyser

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

## Abstract

NN_Periodogram implements a generalised two-stage approach to time series periodicity analysis with Neural Network-based False Alarm Probability (NN_FAP) estimation. The software is optimised for detecting short-period signals in astronomical datasets but remains applicable to diverse time series analysis contexts. By utilising complementary period ranges and multiple analytical methodologies, the framework enhances signal detection capabilities beyond traditional periodogram techniques.

## Introduction

Time series analysis in astronomical contexts frequently encounters challenges related to signal detection amidst noise, systematic effects, and irregular sampling. This implementation addresses these challenges through a novel architecture that combines neural network methodologies with traditional periodogram approaches. The framework is particularly suitable for short-period detection scenarios where conventional methods exhibit diminished efficacy.

### Key Methodological Components

- **Two-Stage Periodogram Framework**: Employs primary and complementary period ranges to enhance detection sensitivity
- **Neural Network Integration**: Applies machine learning techniques for improved statistical rigour in false alarm probability estimation
- **Multi-Method Analytical Approach**: Implements chunk-based, sliding window, and subtraction methodologies
- **Automated Parameter Optimisation**: Provides mathematically justified parameter selection for period grid density and range determination

## Mathematical Motivation and Justification

### Theoretical Foundation

The two-stage approach is mathematically motivated by the limitations inherent in conventional periodogram analyses when applied to complex time series data. Traditional methods often suffer from aliasing effects, harmonics, and systematic noise that can obscure legitimate periodicity signals.

The theoretical framework underpinning this implementation addresses these limitations through:

1. **Complementary Period Range Analysis**: The mathematical relationship between a period of interest (P) and its potential aliases can be expressed as:

   ```
   1/P_alias = 1/P + k*1/P_sampling
   ```

   where k is an integer and P_sampling represents the characteristic sampling frequency. By analysing complementary period ranges, we effectively account for these aliasing relationships.

2. **Period Grid Optimisation**: The density of the period grid is derived from the Nyquist frequency considerations and is mathematically expressed as:

   ```
   Δf = 1/(T_span × OS)
   ```

   where T_span is the time span of observations and OS is the oversampling factor. The optimal period grid ensures adequate sampling of the frequency space to detect legitimate signals without computational inefficiencies.

3. **Statistical Rigour via Neural Networks**: The neural network approach provides a robust estimation of false alarm probabilities through a non-parametric methodology that adapts to the characteristics of the input data, addressing limitations in traditional FAP calculations that often assume specific noise distributions.

### Empirical Validation

The efficacy of this approach has been empirically validated through extensive testing on simulated and real astronomical time series data. The two-stage approach consistently demonstrates enhanced sensitivity to weak periodic signals compared to single-stage methodologies, particularly in datasets with irregular sampling or systematic noise components.

## Installation

### Prerequisites

- Python 3.6 or higher
- Pip package manager

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/nialljmiller/NN_Periodogram.git
cd NN_Periodogram

# Execute the installation script (automatically downloads model files)
bash install.sh
```

### Manual Installation

For environments requiring manual installation:

```bash
# Clone the repository
git clone https://github.com/nialljmiller/NN_Periodogram.git
cd NN_Periodogram

# Install the package with model download
python setup.py download_model
python setup.py install

# Alternative: development mode installation
pip install -e .
python setup.py download_model
```

### Direct Installation from GitHub

```bash
pip install git+https://github.com/nialljmiller/NN_Periodogram.git
```

**Note:** Direct GitHub installation requires manual acquisition of model files:

```bash
# Download and extract model files
wget https://nialljmiller.com/projects/FAP/model.tar.gz
tar -xzf model.tar.gz
mv model NN_FAP/model
```

## Implementation

### Configuration Parameters

The software utilises an `inlist.txt` configuration file with the following parameter categories:

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
normalise=True             # Normalise flux by median
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

### Analytical Methodologies

#### Chunk Method
This approach segments the time series into non-overlapping chunks, analysing each independently. The results are subsequently averaged to produce a robust periodogram. This technique effectively mitigates the influence of systematic errors and long-term trends in the dataset.

#### Sliding Window Method
In contrast to the chunk method, the sliding window methodology employs overlapping windows that traverse the time series. This approach provides enhanced sensitivity to signals that might be positioned at chunk boundaries in the alternative method.

#### Subtraction Method
This innovative approach enhances signal detection in specific period ranges by subtracting the complementary periodogram from the primary periodogram. This procedure effectively isolates signals of interest from background noise and systematic effects.

## Output and Visualisation

For each analysis, the following files are generated:

- **periodogram_[filename].png**: Comprehensive periodogram visualisation displaying all methodologies
- **periodogram_[filename]_folded_chunk_method.png**: Phase-folded light curve for the optimal period determined by the chunk method
- **periodogram_[filename]_folded_sliding_window_method.png**: Phase-folded light curve for the optimal period determined by the sliding window method
- **periodogram_[filename]_folded_subtraction_method.png**: Phase-folded light curve for the optimal period determined by the subtraction method
- **periodogram_[filename]_results.json**: JSON file containing analytical results and optimal periods

## Technical Implementation Details

### Two-Stage Methodological Framework

The two-stage approach functions as follows:

1. **Primary Periodogram**: Focuses on the user-specified period range of interest, employing the sliding window method
2. **Complementary Periodogram**: Samples the complementary period range to capture harmonics and aliases, utilising the chunk method
3. **Subtraction Method**: Enhances signal detection by subtracting the complementary periodogram from the primary periodogram

This approach demonstrates particular efficacy for:
- Detecting weak signals in high-noise datasets
- Distinguishing genuine periods from aliasing effects
- Handling datasets with systematic errors or sampling gaps

### Period Grid Optimisation

The software automatically calculates an optimal period grid based on:
- The observational time span
- Nyquist frequency considerations
- Oversampling factors for enhanced resolution around expected signals
- Mathematical justification for period density derived from signal characteristics

## Contributing

Academic and technical contributions are welcomed. Please submit Pull Requests with appropriate documentation and test cases.

## Licence

This project is licenced under the MIT Licence - see the LICENSE file for details.

## Acknowledgements

- This implementation builds upon and extends the functionality of the NN_FAP package
- We acknowledge the valuable feedback and suggestions from the astronomical time series analysis community

## Citation

If you utilise this software in your research, please cite:

```
@software{NN_Periodogram,
  author       = {Miller, Niall},
  title        = {NN\_Periodogram: Flexible Two-Stage Neural Network Periodogram Analyser},
  year         = {2025},
  url          = {https://github.com/username/NN_Periodogram}
}
```