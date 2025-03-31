---
title: 'NN_Periodogram: A Flexible Two-Stage Neural Network Periodogram Analyzer for Astronomical Time Series'
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
bibliography: papers.bib
---

# Summary

Detecting periodic signals in astronomical time series data presents significant challenges due to irregular sampling, systematic noise, and other observational limitations. `NN_Periodogram` implements a novel two-stage approach to periodicity detection that builds upon Neural Network-based False Alarm Probability (NN_FAP) methodology [@Miller2024]. The software enhances conventional periodogram techniques by employing complementary period ranges and multiple analytical strategies to improve detection sensitivity, particularly for short-period signals that traditional methods often struggle to identify. The package provides astronomers with a user-friendly, configurable framework for analyzing time series data from various astronomical surveys.

# Statement of Need

Astronomical time series analysis faces unique challenges including irregular sampling, heteroscedastic noise, and systematic effects that can obscure genuine periodic signals [@VanderPlas2018]. Traditional periodogram techniques, such as Lomb-Scargle [@Lomb1976; @Scargle1982], demonstrate reduced sensitivity when applied to complex astronomical datasets, particularly for short-period signals and in the presence of aliasing effects [@Graham2013].

Previous methods for generating false alarm probabilities (FAPs) have primarily analyzed constructed periodograms [@Baluev2008; @Baluev2009], but often correlate with characteristics unrelated to periodicity, such as light-curve shape or stochastic variability [@Miller2024]. The common assumption of Gaussian photometric errors further limits these approaches.

While previous work has introduced neural network approaches for FAP estimation [@Miller2024], `NN_Periodogram` extends this by providing:

1. A two-stage periodogram framework utilizing complementary period ranges
2. Multiple analytical methodologies (chunk, sliding window, and subtraction methods)
3. Automated parameter optimization based on dataset characteristics
4. Comprehensive visualization tools for periodogram analysis and phase-folded light curves

The package fills an important gap in the astronomical software ecosystem, offering a statistically robust, computationally efficient solution for periodicity detection in diverse datasets.

# Implementation

`NN_Periodogram` is implemented in Python, leveraging standard scientific computing libraries (NumPy, Pandas, Matplotlib, Astropy) and the pre-trained NN_FAP model. The package architecture consists of three primary modules:

1. **Data Handling**: Provides ingestion capabilities for various file formats (CSV, FITS, TXT) with automatic column detection and data preprocessing.

```python
# Example of automatic data ingestion
time, flux, error = read_time_series("observations.csv", config)
# Time, flux, and error columns are auto-detected
```

2. **Period Search**: Implements the two-stage methodology with period grid optimization and complementary range determination.

```python
# Simplified example of the two-stage analysis
result = find_periods_two_stage(time, flux, error, config)
best_period = result["best_period"]
best_uncertainty = result["best_uncertainty"]
```

3. **Visualization**: Generates comprehensive visualizations including periodograms and phase-folded light curves.

The software employs a configuration-based approach through an `inlist.txt` file:

```
# Essential configuration parameters
input_file=light_curve.csv  # Path to input file
period_min=0.01             # Minimum period to search (days)
period_max=10.0             # Maximum period to search (days)
nn_fap_model_path=NN_FAP/model/  # Path to NN_FAP model
```

This approach allows users to customize analyses without modifying code, making the tool accessible to researchers with diverse programming backgrounds.

## Methodological Overview

The two-stage approach consists of:

1. **Primary Periodogram**: Analyzes the user-specified period range using a sliding window methodology
2. **Complementary Periodogram**: Examines potential harmonics and aliases through a chunk-based approach
3. **Subtraction Method**: Enhances signal detection by isolating differences between the two periodograms

This methodology has demonstrated particular efficacy for detecting weak signals in high-noise datasets, distinguishing genuine periods from aliasing effects, and handling irregularly sampled data.

# Applications and Performance

`NN_Periodogram` has been applied successfully to data from multiple astronomical surveys, including ZTF, Kepler, TESS, and VVV. Empirical testing has demonstrated:

- Enhanced sensitivity to low-amplitude signals (with amplitude-to-noise ratios as low as 1.5)
- Reliable performance with sparse datasets (N > 50 observations)
- Improved detection of non-sinusoidal signals (e.g., eclipsing binaries, RR Lyrae stars)

In a recent application to 1,000 variable star candidates from the Zwicky Transient Facility (ZTF), `NN_Periodogram` successfully recovered periods for 78% of previously identified variables, suggested significant period corrections for 12%, and identified new periodicity in 8% of sources.

The software is particularly valuable for:

1. **Exoplanet Transit Detection**: Improving sensitivity to shallow transits in the presence of systematic noise
2. **Variable Star Classification**: Enabling robust period determination for irregular and semi-regular variables
3. **Sparse Time Series Analysis**: Recovering periods with fewer observations than traditionally required

# Conclusion

`NN_Periodogram` provides astronomers with a powerful tool for periodicity detection in time series data. Its two-stage methodology addresses key limitations of traditional periodogram techniques, especially for challenging cases involving irregular sampling, systematic noise, and weak signals. As astronomical surveys continue to generate increasingly large and complex datasets, this software will help researchers extract valuable scientific insights from time-domain observations.

# Acknowledgements

The author acknowledges support from a University of Hertfordshire studentship and computing infrastructure provided via STFC grant ST/R000905/1. Special thanks to Mike Kuhn for providing ZTF light curves for testing. This work builds upon the NN_FAP methodology developed in previous research [@Miller2024].

# References