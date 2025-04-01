#!/usr/bin/env python3
"""
NN_Periodogram package initialization file.

This package implements a flexible two-stage neural network periodogram analyzer
for astronomical time series data.
"""

from .NNPeriodogram import NNPeriodogram, main

__version__ = "0.2.0"
__author__ = "Niall Miller"
__email__ = "niall.j.miller@gmail.com"
__license__ = "MIT"
__description__ = "A Flexible Two-Stage Neural Network Periodogram Analyzer for Astronomical Time Series"

# Export the main classes and functions
__all__ = ['NNPeriodogram', 'main']