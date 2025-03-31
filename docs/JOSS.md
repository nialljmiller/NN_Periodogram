---
title: 'NN_Periodogram: A Flexible Two-Stage Neural Network Periodogram Analyser for Astronomical Time Series'
tags:
  - Python
  - astronomy
  - time series
  - periodogram
  - machine learning
  - neural networks
  - false alarm probability
authors:
  - name: Niall J. Miller
    orcid: 0000-0002-8556-3694
    affiliation: 1
affiliations:
 - name: Centre for Astrophysics Research, University of Hertfordshire, College Lane, Hatfield, Hertfordshire AL10 9AB, UK
   index: 1
date: 31 March 2025
bibliography: paper.bib
---

# Summary

Time series analysis in astronomical contexts frequently requires the detection of weak periodic signals embedded within noisy observations. `NN_Periodogram` implements a novel two-stage approach to periodicity detection that builds upon our recently published Neural Network-based False Alarm Probability (NN_FAP) methodology [@Miller2024]. The package extends the capabilities of NN_FAP by employing complementary period ranges and multiple analytical approaches to enhance detection sensitivity, particularly for short-period signals that are often challenging to identify with conventional methods. By integrating recurrent neural networks with traditional periodogram techniques, `NN_Periodogram` provides researchers with a robust framework for periodicity analysis in astronomical datasets with irregular sampling, systematic noise, and other common observational challenges.

# Statement of Need

Astronomical time series analysis presents unique challenges due to irregular sampling, heteroscedastic noise, and systematic effects that can obscure genuine periodic signals [@VanderPlas2018]. Traditional periodogram techniques, such as Lomb-Scargle [@Lomb1976; @Scargle1982], exhibit decreased sensitivity when applied to complex astronomical datasets, particularly for short-period signals and in the presence of aliasing effects [@Graham2013].

Previous efforts in generating false alarm probabilities (FAPs) to verify periodicity have primarily focused on analyzing constructed periodograms, as with the method proposed by Baluev [@Baluev2008; @Baluev2009]. However, these approaches often feature correlations with characteristics unrelated to periodicity, such as light-curve shape, slow trends, and stochastic variability [@Miller2024]. The common assumption that photometric errors are Gaussian and well-determined is also a limitation of such analytic methods.

While our previous work on NN_FAP introduced a neural network approach that directly analyzes phase-folded light curves to estimate FAPs [@Miller2024], it lacked a comprehensive framework for periodicity detection across multiple period ranges. `NN_Periodogram` extends NN_FAP by providing:

1. A two-stage periodogram approach that utilizes complementary period ranges to enhance detection sensitivity
2. Multiple analytical methodologies (chunk, sliding window, and subtraction) that collectively improve signal detection
3. Mathematical justification for period grid density, subsampling, and complementary range determination
4. Automated parameter optimization based on dataset characteristics
5. Comprehensive visualization tools for periodogram analysis and phase-folded light curves

The package thus fulfills a significant need within the astronomical community for robust, computationally efficient periodicity detection tools that maintain statistical rigor while leveraging modern machine learning techniques.

# Mathematical Framework

The core innovation of `NN_Periodogram` lies in its integration of the NN_FAP methodology with a two-stage analytical approach, which addresses the fundamental limitations of single-stage periodogram analyses.

## NN_FAP Foundation

The foundational NN_FAP methodology utilizes a recurrent neural network with gated recurrent units (GRUs) to directly analyze phase-folded light curves [@Miller2024]. The network architecture consists of 13 GRU layers with 1024 nodes per layer, trained with a binary cross-entropy loss model. The input features include magnitude values, phase information, and change in phase (Δφ). The network was trained on a combination of real and synthetic light curves, with synthetic data generated to represent various light curve morphologies, including:

Type 1: $m(t) = 0.5\sin(\frac{2\pi t}{P}) - B_1\sin(\frac{4\pi t}{P}) - B_2\sin(\frac{6\pi t}{P})$

Types 2 & 5: $m(t) = 1 \pm \left(A_1\sin(\frac{2\pi t}{P})^2 + A_2\sin(\frac{\pi t}{P})^2\right)$

Type 3: $m(t) = \left|\sin(\frac{2\pi t}{P})\right|$

Type 4: $m(t) = \sin(\frac{2\pi t}{P})$

These formulations represent common types of pulsators and binary light curves, allowing the network to learn periodicity characteristics independent of specific light curve morphologies.

## Two-Stage Periodogram Analysis

Building upon the NN_FAP foundation, `NN_Periodogram` implements a two-stage approach:

1. The first stage (Primary Periodogram) focuses on the user-specified period range of interest utilising a sliding window methodology
2. The second stage (Complementary Periodogram) samples the complementary period range to capture potential harmonics and aliases through a chunk-based approach

The mathematical relationship between a period of interest ($P$) and its potential aliases can be expressed as:

$\frac{1}{P_{alias}} = \frac{1}{P} + k\frac{1}{P_{sampling}}$

where $k$ is an integer and $P_{sampling}$ represents the characteristic sampling frequency. By analysing complementary period ranges, we effectively account for these aliasing relationships.

The subtraction method subsequently enhances signal detection by subtracting the complementary periodogram from the primary periodogram, effectively isolating signals of interest from background noise and systematic effects. This approach has demonstrated particular efficacy for:

- Detecting weak signals in high-noise datasets
- Distinguishing genuine periods from aliasing effects
- Handling datasets with systematic errors or sampling gaps

## Period Grid Optimisation

The density of the period grid is derived from Nyquist frequency considerations and is mathematically expressed as:

$\Delta f = \frac{1}{T_{span} \times OS}$

where $T_{span}$ is the time span of observations and $OS$ is the oversampling factor. The optimal period grid ensures adequate sampling of the frequency space to detect legitimate signals without computational inefficiencies.

Empirical testing has established that this implementation remains reliable where:
- Number of data points $N > 50$ with amplitude-to-noise ratio $A/\bar{\sigma} > 10$, or 
- $A/\bar{\sigma} > 1.5$ with $N \geq 200$

This represents a significant improvement over traditional methods when dealing with sparse or noisy datasets.

# Software Architecture

`NN_Periodogram` is implemented in Python, leveraging standard scientific computing libraries (NumPy, Pandas, Matplotlib, Astropy) along with the KERAS framework for neural network operations. The package architecture comprises several key components:

1. **Data Handling Module**: Provides robust ingestion capabilities for various file formats (CSV, FITS, TXT) with automatic column detection and data preprocessing. The module implements sophisticated error handling and can automatically detect time, flux, and error columns in various astronomical data formats.

2. **Period Search Module**: Implements the core two-stage periodogram methodology with optimised period grid calculation and complementary range determination. This includes:
   - Chunk-based periodogram generation that divides time series into non-overlapping segments
   - Sliding window methodology that uses overlapping windows traversing the time series
   - Subtraction method that enhances signal detection by isolating signals of interest

3. **NN_FAP Module**: Contains the pre-trained recurrent neural network models and inference capabilities for false alarm probability estimation. This module encapsulates the methodology described in [@Miller2024], with additional functionality for integration with the two-stage periodogram approach.

4. **Visualisation Module**: Generates comprehensive visualisations including periodograms and phase-folded light curves, with specialized plotting functions for each analytical method.

The software employs a configuration-based approach through an `inlist.txt` file, allowing users to specify parameters for each analysis without modifying code. Parallel processing capabilities are included to efficiently handle large datasets, with automatic CPU core allocation.

# Example Usage

The following example demonstrates the basic usage pattern for `NN_Periodogram`:

```python
# Create or modify inlist.txt configuration file
with open('inlist.txt', 'w') as f:
    f.write("""
    # Input/output parameters
    input_file=detections.csv
    output_dir=./results
    output_prefix=periodogram
    
    # Data parameters
    time_col=None              # Auto-detect time column
    flux_col=None              # Auto-detect flux column
    error_col=None             # Auto-detect error column
    file_format=None           # Auto-detect file format
    normalize=True             # Normalize flux by median
    remove_outliers=True       # Remove outliers using sigma clipping
    
    # Period search parameters
    period_min=0.01
    period_max=10.0
    n_periods=1000000
    oversample_factor=10
    use_complementary=True
    
    # NN_FAP parameters
    nn_fap_model_path=NN_FAP/model/
    window_size=200
    chunk_size=200
    step_size=50
    n_workers=None             # Auto-detect optimal worker count
    
    # Plotting parameters
    plot_log_scale=True
    """)

# Execute the analysis
import subprocess
subprocess.run(['python', 'NNP.py'])
```

This will produce a comprehensive analysis including:

1. Primary periodogram using the sliding window method
2. Complementary periodogram using the chunk method
3. Enhanced detection using the subtraction method
4. Phase-folded light curves for the best periods from each method
5. JSON results file containing:
   - Best period from each method with uncertainty estimates
   - False alarm probability values
   - Signal-to-noise metrics
   - Additional analytical metadata

The software automatically handles detection of appropriate file formats and column identification, making it readily applicable to diverse astronomical datasets. For VVV, ZTF, Kepler, TESS, CRTS, and OGLE data, the tool has been specifically optimized to work with minimal configuration.

Additional usage options include:
- Batch processing of multiple input files
- Direct API access for integration into astronomical pipelines
- Custom preprocessing of irregular or specialized time series data

# Impact

`NN_Periodogram` addresses significant challenges in astronomical time series analysis that have become increasingly important as datasets rapidly grow in size and complexity. As demonstrated in [@Miller2024], neural network-based FAP estimation offers substantial improvements over conventional analytical methods when dealing with real astronomical data.

The package has immediate applications across multiple astronomical domains:

1. **Variable Star Classification**: Enhancing the detection and verification of periodic variables in large-scale surveys such as ZTF [@Bellm2019], TESS [@Ricker2015], VVV [@Minniti2010], and the upcoming Rubin Observatory LSST [@Ivezic2019].

2. **Asymptotic Giant Branch Stars**: Providing improved differentiation between long-term periodicity and secular variability, which is particularly challenging with conventional periodogram methods [@Templeton2005].

3. **Exoplanet Detection**: Improving sensitivity to short-period exoplanets in transit and radial velocity datasets, particularly for cases with sparse sampling or systematic noise.

4. **Sparse Time Series Analysis**: Enabling reliable periodicity detection in surveys with relatively few observations per source, such as NEOWISE [@Wright2010; @Mainzer2014] and Gaia [@GaiaCollaboration2021].

The implementation has been empirically validated on data from multiple surveys (CRTS, ZTF, Kepler, OGLE, and VVV) and demonstrated superior performance compared to the widely used Baluev method, particularly for light curves with non-sinusoidal shapes and those with fewer than 100 observations.

Beyond astronomy, the package's methodology has potential applications in any field requiring robust periodicity detection in noisy, irregularly sampled time series data, including geophysics, climatology, and financial analysis.

# Acknowledgements

The author acknowledges support from a University of Hertfordshire studentship and the computing infrastructure provided via STFC grant ST/R000905/1 at the University of Hertfordshire. I thank P. W. Lucas, Y. Sun, Z. Guo, W. J. Cooper, and C. Morris for their collaborative work on the underlying NN_FAP methodology. Special thanks are extended to Mike Kuhn for providing ZTF light curves for testing. This work builds upon and extends the functionality of the NN_FAP package developed as part of our previous research.

# References