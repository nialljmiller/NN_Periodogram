#!/usr/bin/env python3
"""
Flexible Two-Stage NN_FAP Periodogram Analyzer

This tool implements a generalized two-stage periodogram approach using NN_FAP for
time series analysis, optimized for detecting short-period signals.

Features:
- Uses a two-stage periodogram approach to enhance detection of specific period ranges
- Primary periodogram focuses on user-specified period range of interest
- Secondary periodogram samples the complementary period range
- Subtraction method to enhance signal detection in period range of interest
- Mathematical justification for period density, subsampling range, and complementary range
- Configurable via inlist.txt file instead of command-line arguments

Author: Derived from nn_tess-cv-processor.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip
from astropy.io import fits
from tqdm import tqdm
from scipy.signal import savgol_filter
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import warnings
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import NN_FAP with different import styles
try:
    from NN_FAP import NN_FAP
    logger.info("Imported NN_FAP.NN_FAP successfully")
except ImportError:
    try:
        import NN_FAP
        logger.info("Imported NN_FAP successfully")
    except ImportError:
        logger.error("Could not import NN_FAP. Make sure it's installed correctly.")

warnings.filterwarnings("ignore", category=UserWarning)

class NNPeriodogram:
    """
    A class for performing two-stage periodogram analysis with Neural Network-based
    False Alarm Probability (NN_FAP) estimation.
    
    The two-stage approach uses:
    1. A primary periodogram focusing on the user-specified period range
    2. A complementary periodogram examining potential harmonics/aliases
    3. A subtraction method to enhance signal detection
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary. If None, default configuration is used.
        
    Attributes
    ----------
    config : dict
        Configuration dictionary with analysis parameters
    nn_fap_model : object
        Loaded NN_FAP model, if available
    """
    
    def __init__(self, config=None):
        # Default configuration
        self.default_config = {
            # Input/output parameters
            "input_file": None,
            "output_dir": "./results",
            "output_prefix": "periodogram",
            
            # Data parameters
            "time_col": None,
            "flux_col": None,
            "error_col": None,
            "file_format": None,
            "normalize": True,
            "remove_outliers": True,
            "sigma": 5.0,
            
            # Period search parameters
            "period_min": 0.01,
            "period_max": 1.0,
            "n_periods": 1000,
            "oversample_factor": 10,
            "use_complementary": True,
            
            # NN_FAP parameters
            "nn_fap_model_path": None,
            "window_size": 200,
            "chunk_size": 200,
            "step_size": 50,
            "n_workers": None,
            
            # Plotting parameters
            "plot_log_scale": True,
            "plot_title": None,
            
            # Additional parameters
            "use_lombscargle_fallback": True,   # Fallback to LombScargle if NN_FAP fails
            "enforce_period_range": True,       # Enforce at least some periods in the range
            "min_periods": 50,                  # Minimum number of periods to ensure
        }
        
        # Set config to default if not provided
        self.config = self.default_config.copy()
        if config is not None:
            self.config.update(config)
            
        # Handle n_workers=None by setting to a reasonable default
        if self.config["n_workers"] is None:
            import multiprocessing
            self.config["n_workers"] = max(1, multiprocessing.cpu_count() // 2)
            
        # Initialize the NN_FAP model to None (will be loaded on first use)
        self.nn_fap_model = None
    
    @classmethod
    def from_inlist(cls, inlist_file="inlist.txt"):
        """
        Create a NNPeriodogram instance using configuration from an inlist file.
        
        Parameters
        ----------
        inlist_file : str, optional
            Path to the inlist file
            
        Returns
        -------
        NNPeriodogram
            Instance with configuration from the inlist file
        """
        config = cls.read_inlist(inlist_file)
        return cls(config)
    
    @staticmethod
    def read_inlist(file_path):
        """
        Read a configuration file in inlist format (key=value pairs).
        Lines starting with # are treated as comments.
        
        Parameters
        ----------
        file_path : str
            Path to the inlist file
            
        Returns
        -------
        dict
            Dictionary of configuration parameters
        """
        # Default configuration (empty, will be filled by the class constructor)
        config = {}
        
        # Try to read the inlist file
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    # Skip comments and empty lines
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key=value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        
                        # Remove trailing comments if any
                        if '#' in value:
                            value = value.split('#', 1)[0]
                        
                        value = value.strip()
                        
                        # Convert value to appropriate type
                        if value.lower() == 'none':
                            value = None
                        elif value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        else:
                            try:
                                # Try to convert to number if possible
                                if '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            except ValueError:
                                # Keep as string if not a number
                                pass
                        
                        # Update config
                        config[key] = value
            
            logger.info(f"Successfully loaded configuration from {file_path}")
        except FileNotFoundError:
            logger.warning(f"Inlist file {file_path} not found. Using default configuration.")
        except Exception as e:
            logger.error(f"Error reading inlist file: {e}")
            logger.info("Using default configuration")
        
        return config
    
    def get_nn_fap_model(self):
        """
        Load the NN_FAP model if not already loaded.
        
        Returns
        -------
        tuple or None
            (knn, model) tuple if successful, None otherwise
        """
        model_path = self.config["nn_fap_model_path"]
        
        # Only load if not already loaded
        if self.nn_fap_model is None and model_path is not None:
            try:
                if hasattr(NN_FAP, 'get_model'):
                    logger.info(f"Loading NN_FAP model from {model_path} using get_model function")
                    self.nn_fap_model = NN_FAP.get_model(model_path)
                else:
                    logger.warning("NN_FAP.get_model not found. Using LombScargle fallback instead.")
                    self.nn_fap_model = None
            except Exception as e:
                logger.error(f"Error loading NN_FAP model: {e}")
                logger.warning(f"Unable to load model from '{model_path}'. Using LombScargle fallback.")
                self.nn_fap_model = None
                
        return self.nn_fap_model
    
    def read_time_series(self, file_path=None):
        """
        Read time series data from various file formats.
        
        Parameters
        ----------
        file_path : str, optional
            Path to the input file. If None, uses config["input_file"]
            
        Returns
        -------
        tuple
            (time, flux, error) arrays
        """
        # Use the provided file path or the one from config
        if file_path is None:
            file_path = self.config["input_file"]
            
        if file_path is None:
            raise ValueError("No input file specified")
            
        # Extract parameters from config
        time_col = self.config["time_col"]
        flux_col = self.config["flux_col"]
        error_col = self.config["error_col"]
        format = self.config["file_format"]
        normalize = self.config["normalize"]
        remove_outliers = self.config["remove_outliers"]
        sigma = self.config["sigma"]
        
        # Auto-detect format if not specified
        if format is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv':
                format = 'csv'
            elif ext in ['.fits', '.fit']:
                format = 'fits'
            elif ext in ['.txt', '.dat']:
                format = 'txt'
            else:
                raise ValueError(f"Unable to auto-detect format for file: {file_path}")
        
        # Read data based on format
        if format == 'csv':
            df = pd.read_csv(file_path)
            
            # Auto-detect columns if not specified
            if time_col is None:
                time_candidates = ['time', 'Time', 'TIME', 'bjd', 'BJD', 'jd', 'JD', 'mjd', 'MJD']
                for col in time_candidates:
                    if col in df.columns:
                        time_col = col
                        logger.info(f"Auto-detected time column: {time_col}")
                        break
                if time_col is None:
                    time_col = df.columns[0]
                    logger.info(f"Using first column as time: {time_col}")
            
            if flux_col is None:
                flux_candidates = ['flux', 'Flux', 'FLUX', 'brightness', 'mag', 'magnitude']
                for col in flux_candidates:
                    if col in df.columns:
                        flux_col = col
                        logger.info(f"Auto-detected flux column: {flux_col}")
                        break
                if flux_col is None:
                    flux_col = df.columns[1]
                    logger.info(f"Using second column as flux: {flux_col}")
            
            if error_col is None:
                error_candidates = ['error', 'err', 'Error', 'ERROR', 'flux_err', 'e_flux', 'e_magnitude', 'e_mag']
                for col in error_candidates:
                    if col in df.columns:
                        error_col = col
                        logger.info(f"Auto-detected error column: {error_col}")
                        break
            
            time = df[time_col].values
            flux = df[flux_col].values
            error = df[error_col].values if error_col in df.columns else np.ones_like(flux) * 0.001

        elif format == 'fits':
            # Use astropy.io.fits directly
            hdul = fits.open(file_path)
            
            # Try to find table data
            table_hdus = [hdu for hdu in hdul if isinstance(hdu, fits.BinTableHDU)]
            if table_hdus:
                data = table_hdus[0].data
                time_col = time_col if time_col is not None else 'TIME'
                flux_col = flux_col if flux_col is not None else 'FLUX'
                error_col = error_col if error_col is not None else 'FLUX_ERR'
                
                time = data[time_col]
                flux = data[flux_col]
                error = data[error_col] if error_col in data.names else np.ones_like(flux) * 0.001
            else:
                # As a last resort, try the primary HDU
                data = hdul[0].data
                time = np.arange(len(data))
                flux = data
                error = np.ones_like(flux) * 0.001
            
            hdul.close()
        
        elif format == 'txt':
            # Try to load as a simple text file with columns
            try:
                data = np.loadtxt(file_path)
                time = data[:, 0]
                flux = data[:, 1]
                error = data[:, 2] if data.shape[1] > 2 else np.ones_like(flux) * 0.001
            except Exception:
                # If that fails, try with delimiter auto-detection
                data = pd.read_csv(file_path, sep=None, engine='python')
                time = data.iloc[:, 0].values
                flux = data.iloc[:, 1].values
                error = data.iloc[:, 2].values if data.shape[1] > 2 else np.ones_like(flux) * 0.001
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Clean data
        # Ensure data is sorted by time
        sort_idx = np.argsort(time)
        time = time[sort_idx]
        flux = flux[sort_idx]
        error = error[sort_idx]
        
        # Remove NaN values
        mask = ~np.isnan(time) & ~np.isnan(flux)
        time = time[mask]
        flux = flux[mask]
        error = error[mask]
        
        # Remove outliers if requested
        if remove_outliers:
            clean_flux = sigma_clip(flux, sigma=sigma)
            mask = ~clean_flux.mask
            time = time[mask]
            flux = flux[mask]
            error = error[mask]
        
        # Normalize flux if requested
        if normalize:
            median_flux = np.median(flux)
            flux = flux / median_flux
            error = error / median_flux
        
        return time, flux, error
        
    def calculate_optimal_period_grid(self, time):
        """
        Calculate an optimal period grid based on the time span of the data.
        
        Parameters
        ----------
        time : array
            Time array (days)
            
        Returns
        -------
        array
            Array of periods to test (in ascending order)
        """
        min_period = self.config["period_min"]
        max_period = self.config["period_max"]
        oversample_factor = self.config["oversample_factor"]
        min_periods_count = self.config["min_periods"]
        enforce_period_range = self.config["enforce_period_range"]
        
        # Calculate the time span for determining frequency resolution
        time_span = time.max() - time.min()
        
        # Calculate the resolution in frequency space based on time span and oversampling
        delta_f = 1.0 / (oversample_factor * time_span)
        
        # Convert to period space
        f_min = 1.0 / max_period
        f_max = 1.0 / min_period
        
        # Check if we have a valid range
        if f_min >= f_max:
            logger.warning(f"Invalid frequency range: f_min ({f_min}) >= f_max ({f_max})")
            if enforce_period_range:
                logger.info("Adjusting period range to ensure valid grid")
                # Adjust min_period to create a valid range
                min_period = max_period * 0.5
                f_max = 1.0 / min_period
        
        # Create a linear frequency grid (using np.arange with sorted bounds)
        f_start = min(f_min, f_max)
        f_end = max(f_min, f_max)
        frequencies = np.arange(f_start, f_end + delta_f, delta_f)
        
        # Ensure frequencies are strictly increasing
        frequencies = np.sort(frequencies)
        
        # If no frequencies were generated or very few, use a fallback
        if len(frequencies) < min_periods_count and enforce_period_range:
            logger.warning(f"Too few periods in range ({len(frequencies)}). Using logarithmic fallback grid.")
            periods = np.logspace(np.log10(min_period), np.log10(max_period), min_periods_count)
            return periods
        
        # Convert back to periods
        periods = 1.0 / frequencies
        
        # Sort in ascending period order
        periods = np.sort(periods)
        
        return periods
    


    def run_lombscargle_periodogram(self, time, flux, error, periods):
        """
        Run a LombScargle periodogram as a fallback method.
        
        Parameters
        ----------
        time : array
            Time array (days)
        flux : array
            Flux array (normalized)
        error : array
            Error array
        periods : array
            Periods to test
            
        Returns
        -------
        array
            Power array (normalized to 0-1 range)
        """
        # Convert periods to frequencies (ensuring they're in ascending order)
        frequencies = 1.0 / periods
        
        # Check if frequencies are in ascending order
        if not np.all(np.diff(frequencies) > 0):
            # If not, sort them
            frequencies = np.sort(frequencies)
        
        # Create LombScargle object
        ls = LombScargle(time, flux, error)
        
        # Calculate power
        power = ls.power(frequencies)
        
        # Normalize to 0-1 range
        if np.max(power) > 0:
            power = power / np.max(power)
            
        # Make sure the power array is in the same order as the input periods
        if not np.all(np.diff(periods) > 0):
            # If periods weren't in ascending order, we need to remap the power values
            # Convert frequencies back to periods
            sorted_periods = 1.0 / frequencies
            # Interpolate the power values to match the original period ordering
            power = np.interp(periods, sorted_periods, power)
        
        return power

    
    def create_complementary_period_range(self, time):
        """
        Create a complementary period range based on mathematical justification.
        
        Parameters
        ----------
        time : array
            Time array (days)
            
        Returns
        -------
        tuple
            (comp_min, comp_max) complementary period range
        """
        period_min = self.config["period_min"]
        period_max = self.config["period_max"]
        min_periods_count = self.config["min_periods"]
        
        # Time span
        time_span = time.max() - time.min()
        
        # The complementary range should cover periods that could potentially leak power
        # into our period range of interest, as well as periods that could be harmonics
        
        # Lower bound: half the minimum period of interest (to catch harmonics)
        comp_min = period_min / 2.0
        
        # Upper bound: twice the maximum period of interest (to catch subharmonics)
        # but capped at a reasonable fraction of the time span
        comp_max = min(period_max * 2.0, time_span / 2.0)
        
        # Ensure the complementary range doesn't overlap with the range of interest
        if comp_min >= period_min:
            comp_min = period_min / 3.0
        
        if comp_max <= period_max:
            comp_max = period_max * 3.0

        # Ensure we have enough periods in the range
        if (comp_max / comp_min) < 1.1:
            # If range is too small, expand it
            comp_max = comp_min * 10
        
        return comp_min, comp_max



    def sliding_window_worker(self, args):
        """
        Worker function for the sliding window periodogram method.
        
        Parameters
        ----------
        args : tuple
            (window_time, window_flux, periods, model_path, use_lombscargle)
            
        Returns
        -------
        array
            Power array for this window
        """
        window_time, window_flux, periods, model_path, use_lombscargle = args
        
        if len(window_time) < 50:
            # Skip if too few points
            return np.zeros(len(periods))
        
        try:
            if use_lombscargle:
                # Use LombScargle as a fallback
                window_error = np.ones_like(window_flux) * 0.001  # Default error
                power = self.run_lombscargle_periodogram(window_time, window_flux, window_error, periods)
            else:
                # Try NN_FAP first
                try:
                    knn, model = self.get_nn_fap_model()
                    if knn is None or model is None:  # If model loading failed
                        raise ValueError("NN_FAP model loading failed")
                        
                    power = np.array([
                        1.0 - NN_FAP.inference(period, window_flux, window_time, knn, model)
                        for period in periods
                    ])
                except Exception as e:
                    logger.warning(f"NN_FAP failed in sliding window worker: {e}. Using LombScargle as fallback.")
                    window_error = np.ones_like(window_flux) * 0.001  # Default error
                    power = self.run_lombscargle_periodogram(window_time, window_flux, window_error, periods)
            
            return power
        except Exception as e:
            logger.error(f"Error in sliding window worker: {e}")
            return np.zeros(len(periods))

    def chunk_periodogram_worker(self, args):
        """
        Worker function for the chunk periodogram method.
        
        Parameters
        ----------
        args : tuple
            (chunk_time, chunk_flux, periods, model_path, use_lombscargle)
            
        Returns
        -------
        array
            Power array for this chunk
        """
        chunk_time, chunk_flux, periods, model_path, use_lombscargle = args
        
        if len(chunk_time) < 50:
            # Skip if too few points
            return np.zeros(len(periods))
        
        try:
            if use_lombscargle:
                # Use LombScargle as a fallback
                chunk_error = np.ones_like(chunk_flux) * 0.001  # Default error
                power = self.run_lombscargle_periodogram(chunk_time, chunk_flux, chunk_error, periods)
            else:
                # Try NN_FAP first
                try:
                    knn, model = self.get_nn_fap_model()
                    if knn is None or model is None:  # If model loading failed
                        raise ValueError("NN_FAP model loading failed")
                        
                    power = np.array([
                        1.0 - NN_FAP.inference(period, chunk_flux, chunk_time, knn, model)
                        for period in periods
                    ])
                except Exception as e:
                    logger.warning(f"NN_FAP failed in chunk worker: {e}. Using LombScargle as fallback.")
                    chunk_error = np.ones_like(chunk_flux) * 0.001  # Default error
                    power = self.run_lombscargle_periodogram(chunk_time, chunk_flux, chunk_error, periods)
            
            return power
        except Exception as e:
            logger.error(f"Error in chunk worker: {e}")
            return np.zeros(len(periods))



    
    def create_nn_fap_chunk_periodogram(self, time, flux, periods):
        """
        Create a periodogram using the chunk method.
        
        Parameters
        ----------
        time : array
            Time array (days)
        flux : array
            Flux array (normalized)
        periods : array
            Periods to test
            
        Returns
        -------
        array
            Power array (1-FAP values for each period)
        """
        model_path = self.config["nn_fap_model_path"]
        chunk_size = self.config["chunk_size"]
        n_workers = self.config["n_workers"]
        use_lombscargle = self.config["use_lombscargle_fallback"]
        
        # Create chunks
        chunks = []
        for i in range(0, len(time), chunk_size):
            chunk_time = time[i:i+chunk_size]
            chunk_flux = flux[i:i+chunk_size]
            if len(chunk_time) >= 50:  # Minimum size for a valid chunk
                chunks.append((chunk_time, chunk_flux, periods, model_path, use_lombscargle))
        
        if not chunks:
            logger.warning("No valid chunks found. Data may be too sparse. Using whole light curve.")
            # Use the whole light curve as a single chunk
            if len(time) >= 50:
                chunks = [(time, flux, periods, model_path, use_lombscargle)]
            else:
                logger.error("Insufficient data points for analysis (< 50 points)")
                return np.zeros(len(periods))
        
        # Process in parallel
        avg_power = np.zeros(len(periods))
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(executor.map(self.chunk_periodogram_worker, chunks), 
                              total=len(chunks), desc="Chunk Periodogram"))
        
        # Average the results
        valid_results = [r for r in results if np.sum(r) > 0]
        if valid_results:
            avg_power = np.mean(valid_results, axis=0)
        else:
            logger.warning("No valid results from any chunks. Using LombScargle on whole light curve.")
            error = np.ones_like(flux) * 0.001  # Default error
            avg_power = self.run_lombscargle_periodogram(time, flux, error, periods)
        
        return avg_power
    
    def create_nn_fap_sliding_window_periodogram(self, time, flux, periods):
        """
        Create a periodogram using a true sliding window method that slides one point at a time
        (one in, one out), instead of jumping by steps.
        
        Parameters
        ----------
        time : array
            Time array (days)
        flux : array
            Flux array (normalized)
        periods : array
            Periods to test
            
        Returns
        -------
        array
            Power array (1-FAP values for each period)
        """
        model_path = self.config["nn_fap_model_path"]
        window_size = self.config["window_size"]
        n_workers = self.config["n_workers"]
        use_lombscargle = self.config["use_lombscargle_fallback"]
        
        # Create windows with true one-in, one-out sliding (step=1)
        windows = []
        for i in range(0, len(time) - window_size + 1, 1):  # Step size is now 1
            window_time = time[i:i+window_size]
            window_flux = flux[i:i+window_size]
            if len(window_time) >= 50:  # Minimum size for a valid window
                windows.append((window_time, window_flux, periods, model_path, use_lombscargle))
        
        # If data volume is too large, use subsampling to reduce computational load
        if len(windows) > 1000:  # Arbitrary threshold to manage computational load
            subsample_size = min(1000, len(windows) // 10)  # Take at most 1000 windows, or 10% of total
            logger.info(f"Large number of windows ({len(windows)}). Subsampling to {subsample_size} windows.")
            
            # Use evenly spaced subsampling
            indices = np.linspace(0, len(windows)-1, subsample_size, dtype=int)
            windows = [windows[i] for i in indices]
        
        if not windows:
            logger.warning("No valid windows found. Data may be too sparse or window too large.")
            logger.info("Using smaller windows or whole light curve")
            
            # Try using smaller windows
            smaller_window_size = min(window_size // 2, len(time))
            if smaller_window_size >= 50:
                logger.info(f"Trying smaller window size: {smaller_window_size}")
                for i in range(0, len(time) - smaller_window_size + 1, 1):  # Also use step=1 here
                    window_time = time[i:i+smaller_window_size]
                    window_flux = flux[i:i+smaller_window_size]
                    windows.append((window_time, window_flux, periods, model_path, use_lombscargle))
            
            # If still no windows, use the whole light curve
            if not windows and len(time) >= 50:
                logger.info("Using whole light curve as a single window")
                windows = [(time, flux, periods, model_path, use_lombscargle)]
            
            if not windows:
                logger.error("Insufficient data points for analysis (< 50 points)")
                return np.zeros(len(periods))
        
        # Process in parallel
        avg_power = np.zeros(len(periods))
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(executor.map(self.sliding_window_worker, windows), 
                              total=len(windows), desc="Sliding Window Periodogram"))
        
        # Average the results
        valid_results = [r for r in results if np.sum(r) > 0]
        if valid_results:
            avg_power = np.mean(valid_results, axis=0)
        else:
            logger.warning("No valid results from any windows. Using LombScargle on whole light curve.")
            error = np.ones_like(flux) * 0.001  # Default error
            avg_power = self.run_lombscargle_periodogram(time, flux, error, periods)
        
        return avg_power
    
    def find_periods(self, time, flux, error=None):
        """
        Find periods using the two-stage NN_FAP periodogram method.
        
        Parameters
        ----------
        time : array
            Time array (days)
        flux : array
            Flux array (normalized)
        error : array, optional
            Error array. If None, uses 0.001 for all points
            
        Returns
        -------
        dict
            Dictionary of results including best periods and periodograms
        """
        # Create error array if not provided
        if error is None:
            error = np.ones_like(flux) * 0.001
            
        logger.info("Running two-stage periodogram analysis...")
        
        # Extract parameters from config
        period_min = self.config["period_min"]
        period_max = self.config["period_max"]
        model_path = self.config["nn_fap_model_path"]
        n_periods = self.config["n_periods"]
        use_complementary = self.config["use_complementary"]
        use_lombscargle = self.config["use_lombscargle_fallback"]
        
        # Check if model path is provided (only needed if not using LombScargle)
        if model_path is None and not use_lombscargle:
            logger.warning("NN_FAP model path not specified. Using LombScargle as fallback.")
            self.config["use_lombscargle_fallback"] = True
            use_lombscargle = True
        
        # Validate period range
        time_span = time.max() - time.min()
        if period_max > time_span / 2:
            logger.warning(f"Maximum period ({period_max}) exceeds half the time span ({time_span/2}).")
            logger.info("Limiting maximum period to half the time span.")
            period_max = time_span / 2
            self.config["period_max"] = period_max
        

        # Calculate optimal period grid for the primary range
        periods_primary = self.calculate_optimal_period_grid(time)
        
        # Ensure we have some periods
        if len(periods_primary) == 0:
            logger.warning("No periods in primary range. Creating a logarithmic grid.")
            periods_primary = np.logspace(
                np.log10(period_min), 
                np.log10(period_max), 
                self.config["min_periods"]
            )
        
        # Subset to n_periods if needed
        if len(periods_primary) > n_periods:
            logger.info(f"Subsampling period grid from {len(periods_primary)} to {n_periods} periods")
            periods_primary = np.logspace(
                np.log10(period_min), 
                np.log10(period_max), 
                n_periods
            )
        
        logger.info(f"Primary period range: {period_min} to {period_max} ({len(periods_primary)} periods)")
        
        # Calculate the complementary period range if enabled
        if use_complementary:
            comp_min, comp_max = self.create_complementary_period_range(time)
            
            # Temporarily update config for complementary range
            original_period_min = self.config["period_min"]
            original_period_max = self.config["period_max"]
            self.config["period_min"] = comp_min
            self.config["period_max"] = comp_max
            
            # Calculate optimal period grid for the complementary range
            self.config["oversample_factor"] = self.config["oversample_factor"] / 2  # Less dense sampling
            periods_comp = self.calculate_optimal_period_grid(time)
            
            # Restore original values
            self.config["period_min"] = original_period_min
            self.config["period_max"] = original_period_max
            self.config["oversample_factor"] = self.config["oversample_factor"] * 2
            
            # Ensure we have some periods in the complementary range
            if len(periods_comp) == 0:
                logger.warning("No periods in complementary range. Creating a logarithmic grid.")
                periods_comp = np.logspace(
                    np.log10(comp_min), 
                    np.log10(comp_max), 
                    self.config["min_periods"]
                )
            
            # Subset to n_periods if needed
            if len(periods_comp) > n_periods:
                periods_comp = np.logspace(
                    np.log10(comp_min), 
                    np.log10(comp_max), 
                    n_periods
                )
            
            logger.info(f"Complementary period range: {comp_min} to {comp_max} ({len(periods_comp)} periods)")
        else:
            periods_comp = periods_primary
        
        # Run the primary periodogram (sliding window)
        sliding_power = self.create_nn_fap_sliding_window_periodogram(
            time, flux, periods_primary
        )
        

        # Run the complementary periodogram (chunk method)
        chunk_power = self.create_nn_fap_chunk_periodogram(
            time, flux, periods_comp
        )
        
        # If complementary range is different, interpolate to match the primary range
        if use_complementary and not np.array_equal(periods_primary, periods_comp):
            chunk_power_interp = np.interp(periods_primary, periods_comp, chunk_power)
        else:
            chunk_power_interp = chunk_power
        
        # Create the subtraction periodogram
        subtraction_power = np.clip(sliding_power - chunk_power_interp, 0, None)
        
        # Find the best period from each method
        if len(chunk_power_interp) > 0 and np.any(chunk_power_interp > 0):
            chunk_best_idx = np.argmax(chunk_power_interp)
            chunk_best_period = periods_primary[chunk_best_idx]
        else:
            logger.warning("No valid power in chunk periodogram. Using median period.")
            chunk_best_idx = len(periods_primary) // 2
            chunk_best_period = periods_primary[chunk_best_idx]
        
        if len(sliding_power) > 0 and np.any(sliding_power > 0):
            sliding_best_idx = np.argmax(sliding_power)
            sliding_best_period = periods_primary[sliding_best_idx]
        else:
            logger.warning("No valid power in sliding window periodogram. Using median period.")
            sliding_best_idx = len(periods_primary) // 2
            sliding_best_period = periods_primary[sliding_best_idx]
        
        if len(subtraction_power) > 0 and np.any(subtraction_power > 0):
            subtraction_best_idx = np.argmax(subtraction_power)
            subtraction_best_period = periods_primary[subtraction_best_idx]
        else:
            logger.warning("No valid power in subtraction periodogram. Using sliding window result.")
            subtraction_best_idx = sliding_best_idx
            subtraction_best_period = sliding_best_period
        
        # Calculate uncertainty for the best period
        try:
            # Find indices where subtraction power is greater than half the max power
            max_power = np.max(subtraction_power)
            if max_power > 0:
                high_power_idx = np.where(subtraction_power > 0.5 * max_power)[0]
                if len(high_power_idx) > 1:
                    # Use the width of the peak as the uncertainty
                    period_uncertainty = 0.5 * (
                        periods_primary[high_power_idx[-1]] - periods_primary[high_power_idx[0]]
                    )
                else:
                    # If we can't determine the width, use 1% of the period as a default
                    period_uncertainty = 0.01 * subtraction_best_period
            else:
                period_uncertainty = 0.01 * subtraction_best_period
        except Exception as e:
            logger.warning(f"Error calculating period uncertainty: {e}")
            period_uncertainty = 0.01 * subtraction_best_period
        
        logger.info(f"Method 1 (Chunk): Best period = {chunk_best_period:.6f}")
        logger.info(f"Method 2 (Sliding): Best period = {sliding_best_period:.6f}")
        logger.info(f"Method 3 (Subtraction): Best period = {subtraction_best_period:.6f} ± {period_uncertainty:.6f}")
        
        # Determine the final best period based on the subtraction method
        best_period = subtraction_best_period
        best_uncertainty = period_uncertainty
        
        return {
            "primary_periods": periods_primary,
            "complementary_periods": periods_comp if use_complementary else None,
            "chunk_power": chunk_power,
            "sliding_power": sliding_power,
            "subtraction_power": subtraction_power,
            "chunk_best_period": chunk_best_period,
            "sliding_best_period": sliding_best_period,
            "subtraction_best_period": subtraction_best_period,
            "best_period": best_period,
            "best_uncertainty": best_uncertainty
        }
        


    def find_periods(self, time, flux, error=None):
        """
        Find periods using the two-stage NN_FAP periodogram method.
        
        Parameters
        ----------
        time : array
            Time array (days)
        flux : array
            Flux array (normalized)
        error : array, optional
            Error array. If None, uses 0.001 for all points
            
        Returns
        -------
        dict
            Dictionary of results including best periods and periodograms
        """
        # Create error array if not provided
        if error is None:
            error = np.ones_like(flux) * 0.001
                
        logger.info("Running two-stage periodogram analysis...")
        
        # Extract parameters from config
        period_min = self.config["period_min"]
        period_max = self.config["period_max"]
        model_path = self.config["nn_fap_model_path"]
        n_periods = self.config["n_periods"]
        use_complementary = self.config["use_complementary"]
        use_lombscargle = self.config["use_lombscargle_fallback"]
        
        # Check if model path is provided (only needed if not using LombScargle)
        if model_path is None and not use_lombscargle:
            logger.warning("NN_FAP model path not specified. Using LombScargle as fallback.")
            self.config["use_lombscargle_fallback"] = True
            use_lombscargle = True
        
        # Validate period range
        time_span = time.max() - time.min()
        if period_max > time_span / 2:
            logger.warning(f"Maximum period ({period_max}) exceeds half the time span ({time_span/2}).")
            logger.info("Limiting maximum period to half the time span.")
            period_max = time_span / 2
            self.config["period_max"] = period_max
        

        # Calculate optimal period grid for the primary range
        periods_primary = self.calculate_optimal_period_grid(time)
        
        # Ensure we have some periods
        if len(periods_primary) == 0:
            logger.warning("No periods in primary range. Creating a logarithmic grid.")
            periods_primary = np.logspace(
                np.log10(period_min), 
                np.log10(period_max), 
                self.config["min_periods"]
            )
        
        # Subset to n_periods if needed
        if len(periods_primary) > n_periods:
            logger.info(f"Subsampling period grid from {len(periods_primary)} to {n_periods} periods")
            periods_primary = np.logspace(
                np.log10(period_min), 
                np.log10(period_max), 
                n_periods
            )
        
        logger.info(f"Primary period range: {period_min} to {period_max} ({len(periods_primary)} periods)")
        
        # Calculate the complementary period range if enabled
        if use_complementary:
            comp_min, comp_max = self.create_complementary_period_range(time)
            
            # Temporarily update config for complementary range
            original_period_min = self.config["period_min"]
            original_period_max = self.config["period_max"]
            self.config["period_min"] = comp_min
            self.config["period_max"] = comp_max
            
            # Calculate optimal period grid for the complementary range
            self.config["oversample_factor"] = self.config["oversample_factor"] / 2  # Less dense sampling
            periods_comp = self.calculate_optimal_period_grid(time)
            
            # Restore original values
            self.config["period_min"] = original_period_min
            self.config["period_max"] = original_period_max
            self.config["oversample_factor"] = self.config["oversample_factor"] * 2
            
            # Ensure we have some periods in the complementary range
            if len(periods_comp) == 0:
                logger.warning("No periods in complementary range. Creating a logarithmic grid.")
                periods_comp = np.logspace(
                    np.log10(comp_min), 
                    np.log10(comp_max), 
                    self.config["min_periods"]
                )
            
            # Subset to n_periods if needed
            if len(periods_comp) > n_periods:
                periods_comp = np.logspace(
                    np.log10(comp_min), 
                    np.log10(comp_max), 
                    n_periods
                )
            
            logger.info(f"Complementary period range: {comp_min} to {comp_max} ({len(periods_comp)} periods)")
        else:
            periods_comp = periods_primary
        
        # Run the primary periodogram (sliding window)
        sliding_power = self.create_nn_fap_sliding_window_periodogram(
            time, flux, periods_primary
        )
        

        # Run the complementary periodogram (chunk method)
        chunk_power = self.create_nn_fap_chunk_periodogram(
            time, flux, periods_comp
        )
        
        # Create a clean copy of the original chunk power for returning
        original_chunk_power = chunk_power.copy()
        
        # If complementary range is different, interpolate to match the primary range
        if use_complementary and not np.array_equal(periods_primary, periods_comp):
            # Make sure periods are in ascending order for interpolation
            comp_sorted_indices = np.argsort(periods_comp)
            primary_sorted_indices = np.argsort(periods_primary)
            
            # Sort periods and powers for interpolation
            sorted_periods_comp = periods_comp[comp_sorted_indices]
            sorted_chunk_power = chunk_power[comp_sorted_indices]
            
            # Interpolate to match primary periods
            chunk_power_interp = np.interp(
                periods_primary[primary_sorted_indices], 
                sorted_periods_comp, 
                sorted_chunk_power
            )
            
            # Restore original order
            temp_power = np.zeros_like(chunk_power_interp)
            temp_power[primary_sorted_indices] = chunk_power_interp
            chunk_power_interp = temp_power
        else:
            chunk_power_interp = chunk_power
        
        # Create the subtraction periodogram
        subtraction_power = np.clip(sliding_power - chunk_power_interp, 0, None)
        
        # Find the best period from each method
        if len(chunk_power_interp) > 0 and np.any(chunk_power_interp > 0):
            chunk_best_idx = np.argmax(chunk_power_interp)
            chunk_best_period = periods_primary[chunk_best_idx]
        else:
            logger.warning("No valid power in chunk periodogram. Using median period.")
            chunk_best_idx = len(periods_primary) // 2
            chunk_best_period = periods_primary[chunk_best_idx]
        
        if len(sliding_power) > 0 and np.any(sliding_power > 0):
            sliding_best_idx = np.argmax(sliding_power)
            sliding_best_period = periods_primary[sliding_best_idx]
        else:
            logger.warning("No valid power in sliding window periodogram. Using median period.")
            sliding_best_idx = len(periods_primary) // 2
            sliding_best_period = periods_primary[sliding_best_idx]
        
        if len(subtraction_power) > 0 and np.any(subtraction_power > 0):
            subtraction_best_idx = np.argmax(subtraction_power)
            subtraction_best_period = periods_primary[subtraction_best_idx]
        else:
            logger.warning("No valid power in subtraction periodogram. Using sliding window result.")
            subtraction_best_idx = sliding_best_idx
            subtraction_best_period = sliding_best_period
        
        # Calculate uncertainty for the best period
        try:
            # Find indices where subtraction power is greater than half the max power
            max_power = np.max(subtraction_power)
            if max_power > 0:
                high_power_idx = np.where(subtraction_power > 0.5 * max_power)[0]
                if len(high_power_idx) > 1:
                    # Use the width of the peak as the uncertainty
                    period_uncertainty = 0.5 * (
                        periods_primary[high_power_idx[-1]] - periods_primary[high_power_idx[0]]
                    )
                else:
                    # If we can't determine the width, use 1% of the period as a default
                    period_uncertainty = 0.01 * subtraction_best_period
            else:
                period_uncertainty = 0.01 * subtraction_best_period
        except Exception as e:
            logger.warning(f"Error calculating period uncertainty: {e}")
            period_uncertainty = 0.01 * subtraction_best_period
        
        logger.info(f"Method 1 (Chunk): Best period = {chunk_best_period:.6f}")
        logger.info(f"Method 2 (Sliding): Best period = {sliding_best_period:.6f}")
        logger.info(f"Method 3 (Subtraction): Best period = {subtraction_best_period:.6f} ± {period_uncertainty:.6f}")
        
        # Determine the final best period based on the subtraction method
        best_period = subtraction_best_period
        best_uncertainty = period_uncertainty
        
        return {
            "primary_periods": periods_primary,
            "complementary_periods": periods_comp if use_complementary else None,
            "chunk_power": original_chunk_power,  # Return the original chunk power for the complementary range
            "chunk_power_interp": chunk_power_interp,  # Also return the interpolated version
            "sliding_power": sliding_power,
            "subtraction_power": subtraction_power,
            "chunk_best_period": chunk_best_period,
            "sliding_best_period": sliding_best_period,
            "subtraction_best_period": subtraction_best_period,
            "best_period": best_period,
            "best_uncertainty": best_uncertainty
        }






        
    def phase_fold_lightcurve(self, time, flux, error, period):
        """
        Phase fold a light curve.
        
        Parameters
        ----------
        time : array
            Time array (days)
        flux : array
            Flux array (normalized)
        error : array
            Error array
        period : float
            Period to fold at (days)
            
        Returns
        -------
        tuple
            - Phase array (0-1)
            - Flux array
            - Error array
        """
        # Calculate phase
        phase = (time / period) % 1.0
        
        # Sort by phase
        sort_idx = np.argsort(phase)
        phase = phase[sort_idx]
        flux = flux[sort_idx]
        error = error[sort_idx]
        
        return phase, flux, error
    
    def bin_phased_lightcurve(self, phase, flux, error, bins=50):
        """
        Bin a phase-folded light curve.
        
        Parameters
        ----------
        phase : array
            Phase array (0-1)
        flux : array
            Flux array
        error : array
            Error array
        bins : int
            Number of bins
            
        Returns
        -------
        tuple
            - Binned phase array
            - Binned flux array
            - Binned error array
        """
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        binned_flux = np.zeros(bins)
        binned_error = np.zeros(bins)
        
        for i in range(bins):
            mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
            if np.sum(mask) > 0:
                binned_flux[i] = np.mean(flux[mask])
                # Error propagation for bin (standard error of the mean)
                if np.sum(mask) > 1:  # More than one point in bin
                    binned_error[i] = np.sqrt(np.sum(error[mask]**2)) / np.sum(mask)
                else:  # Only one point in bin
                    binned_error[i] = error[mask][0]
            else:
                binned_flux[i] = np.nan
                binned_error[i] = np.nan
        
        return bin_centers, binned_flux, binned_error
    
    def create_phase_folded_plot(self, time, flux, error, period, method_name, output_file=None):
        """
        Create a phase-folded plot for a specific period.
        Shows data from phase 0 to 2 and includes both raw and binned/fitted data.
        
        Parameters
        ----------
        time : array
            Time array
        flux : array
            Flux array
        error : array
            Error array
        period : float
            Period to fold at
        method_name : str
            Name of the method that produced this period
        output_file : str
            Base output file path
        """
        plt.figure(figsize=(12, 6))
        
        # Phase fold the light curve
        phase, folded_flux, folded_error = self.phase_fold_lightcurve(time, flux, error, period)
        
        # Plot the raw phase-folded light curve with small markers
        plt.errorbar(phase, folded_flux, yerr=folded_error, fmt='.', color='gray', 
                    alpha=0.2, ecolor='lightgray', markersize=2, label='Raw data')
        
        # Add a second cycle of raw data
        plt.errorbar(phase + 1, folded_flux, yerr=folded_error, fmt='.', color='gray', 
                    alpha=0.2, ecolor='lightgray', markersize=2)
        
        # Bin the phase-folded light curve for better visualization
        bin_phase, bin_flux, bin_error = self.bin_phased_lightcurve(phase, folded_flux, folded_error, 50)
        
        # Plot the binned data
        plt.errorbar(bin_phase, bin_flux, yerr=bin_error, fmt='o', color='blue', 
                    alpha=0.8, markersize=5, label='Binned data')
        
        # Add a second cycle of binned data
        plt.errorbar(bin_phase + 1, bin_flux, yerr=bin_error, fmt='o', color='blue', 
                    alpha=0.8, markersize=5)
        
        # Try to fit a smoothing curve to the binned data
        try:
            # Remove NaN values for fitting
            mask = ~np.isnan(bin_flux)
            if np.sum(mask) > 5:  # Need enough points for smoothing
                x_valid = bin_phase[mask]
                y_valid = bin_flux[mask]
                
                # Sort by phase for proper smoothing
                sort_idx = np.argsort(x_valid)
                x_valid = x_valid[sort_idx]
                y_valid = y_valid[sort_idx]
                
                # Apply Savitzky-Golay filter if we have enough points
                if len(y_valid) > 10:
                    window_length = min(15, len(y_valid) // 4 * 2 + 1)  # Ensure odd window length
                    if window_length >= 3:  # Minimum window length for savgol
                        y_smooth = savgol_filter(y_valid, window_length, 3)
                        
                        # Plot the smoothed curve through both cycles
                        plt.plot(x_valid, y_smooth, 'r-', linewidth=2, label='Fitted curve')
                        plt.plot(x_valid + 1, y_smooth, 'r-', linewidth=2)
                
                # For cases where savgol isn't applicable, use a simple spline interpolation
                else:
                    from scipy.interpolate import UnivariateSpline
                    try:
                        spl = UnivariateSpline(x_valid, y_valid, s=0.01)
                        x_dense = np.linspace(0, 1, 100)
                        y_smooth = spl(x_dense)
                        plt.plot(x_dense, y_smooth, 'r-', linewidth=2, label='Fitted curve')
                        plt.plot(x_dense + 1, y_smooth, 'r-', linewidth=2)
                    except:
                        pass  # Skip if spline fitting fails
        except Exception as e:
            logger.warning(f"Error fitting smooth curve: {e}")
        
        # Set labels and title
        plt.xlabel('Phase')
        plt.ylabel('Normalized Flux')
        plt.title(f'Phase-folded Light Curve - {method_name}\nPeriod = {period:.6f} days ({period*24:.4f} hours)')
        
        # Set x-axis limits to show exactly two cycles
        plt.xlim(0, 2)
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Add annotations showing the period in different units
        plt.annotate(f'Period: {period:.6f} days = {period*24:.4f} hours', 
                    xy=(0.02, 0.02), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot if output file is specified
        if output_file:
            base, ext = os.path.splitext(output_file)
            method_suffix = method_name.replace(" ", "_").lower()
            folded_output = f"{base}_folded_{method_suffix}{ext}"
            plt.savefig(folded_output, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
        return f'Phase-folded plot created for period {period:.6f} days'
    




    def plot_results(self, time, flux, error, result, output_file=None, title=None):
        """
        Create summary plots for the periodogram analysis with enhanced features:
        1. Show the full range in all periodograms
        2. Create separate phase folded plots for each method's best period
        3. Show real and fitted data from 0-2φ in phase folded plots
        
        Parameters
        ----------
        time : array
            Time array (days)
        flux : array
            Flux array (normalized)
        error : array
            Error array
        result : dict
            Result dictionary from find_periods
        output_file : str, optional
            Output file path. If None, plots are displayed instead
        title : str, optional
            Title for the plot. If None, default title is used
            
        Returns
        -------
        str
            Success message
        """
        # Use parameters from config if not specified
        if title is None:
            title = self.config.get("plot_title")
        
        plot_log_scale = self.config["plot_log_scale"]
        
        # Extract data from result
        periods = result["primary_periods"]
        
        # Check if we have the interpolated chunk power
        if "chunk_power_interp" in result:
            chunk_power = result["chunk_power_interp"]
        elif result["complementary_periods"] is not None and len(result["complementary_periods"]) != len(periods):
            # If not present and dimensions differ, we need to interpolate
            # (this is for backward compatibility with older result dictionaries)
            comp_periods = result["complementary_periods"]
            chunk_power = np.interp(periods, comp_periods, result["chunk_power"])
        else:
            # If dimensions match, use the chunk_power directly
            chunk_power = result["chunk_power"]
                
        # The sliding power and subtraction power should already match primary_periods dimensions
        sliding_power = result["sliding_power"]
        subtraction_power = result["subtraction_power"]
        
        # Get best periods from each method
        chunk_best_period = result["chunk_best_period"]
        sliding_best_period = result["sliding_best_period"]
        subtraction_best_period = result["subtraction_best_period"]
        best_uncertainty = result["best_uncertainty"]
        
        # Create a figure with grid layout for periodograms
        plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=plt.gcf())
        
        # Plot 1: Original Light Curve
        ax1 = plt.subplot(gs[0, :])
        ax1.errorbar(time, flux, yerr=error, fmt='.', color='black', alpha=0.3, ecolor='lightgray', markersize=3)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Normalized Flux')
        ax1.set_title('Original Light Curve')
        ax1.grid(True, alpha=0.3)
        
        # Find the best indices for marking the best periods on the plot
        # We need to find the closest index in the periods array for each best period
        chunk_best_idx = np.argmin(np.abs(periods - chunk_best_period))
        sliding_best_idx = np.argmin(np.abs(periods - sliding_best_period))
        subtraction_best_idx = np.argmin(np.abs(periods - subtraction_best_period))
        
        # Plot 2: Chunk Method Periodogram
        ax2 = plt.subplot(gs[1, 0])
        ax2.plot(periods, chunk_power, 'b-', linewidth=1.5)
        # Mark the best period
        ax2.axvline(chunk_best_period, color='r', linestyle='--', alpha=0.7)
        ax2.scatter([chunk_best_period], [chunk_power[chunk_best_idx]], color='red', s=50, marker='o', zorder=5)
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Power (1-FAP)')
        ax2.set_title(f'Method 1: Chunk Method\nBest Period: {chunk_best_period:.6f}')
        if plot_log_scale:
            ax2.set_xscale('log')
        # Ensure full range is visible
        ax2.set_xlim(min(periods), max(periods))
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Sliding Window Periodogram
        ax3 = plt.subplot(gs[1, 1])
        ax3.plot(periods, sliding_power, 'g-', linewidth=1.5)
        # Mark the best period
        ax3.axvline(sliding_best_period, color='r', linestyle='--', alpha=0.7)
        ax3.scatter([sliding_best_period], [sliding_power[sliding_best_idx]], color='red', s=50, marker='o', zorder=5)
        ax3.set_xlabel('Period')
        ax3.set_ylabel('Power (1-FAP)')
        ax3.set_title(f'Method 2: Sliding Window\nBest Period: {sliding_best_period:.6f}')
        if plot_log_scale:
            ax3.set_xscale('log')
        # Ensure full range is visible
        ax3.set_xlim(min(periods), max(periods))
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Subtraction Method Periodogram
        ax4 = plt.subplot(gs[1, 2])
        ax4.plot(periods, subtraction_power, 'purple', linewidth=1.5)
        # Mark the best period
        ax4.axvline(subtraction_best_period, color='r', linestyle='--', alpha=0.7)
        ax4.scatter([subtraction_best_period], [subtraction_power[subtraction_best_idx]], color='red', s=50, marker='o', zorder=5)
        ax4.set_xlabel('Period')
        ax4.set_ylabel('Power (1-FAP)')
        ax4.set_title(f'Method 3: Subtraction\nBest Period: {subtraction_best_period:.6f}')
        if plot_log_scale:
            ax4.set_xscale('log')
        # Ensure full range is visible
        ax4.set_xlim(min(periods), max(periods))
        ax4.grid(True, alpha=0.3)
        
        # Add overall title
        if title:
            plt.suptitle(title, fontsize=16)
        else:
            plt.suptitle(f'Two-Stage NN_FAP Periodogram Analysis', fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save or show the periodogram figure
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Create separate phase-folded plots for each method's best period
        self.create_phase_folded_plot(time, flux, error, chunk_best_period, "Chunk Method", output_file)
        self.create_phase_folded_plot(time, flux, error, sliding_best_period, "Sliding Window Method", output_file)
        self.create_phase_folded_plot(time, flux, error, subtraction_best_period, "Subtraction Method", output_file)
        
        return "Periodogram and phase-folded plots created successfully"



    def analyze_file(self, file_path=None, output_dir=None, output_prefix=None):
        """
        Analyze a time series file and generate results.
        
        Parameters
        ----------
        file_path : str, optional
            Path to the input file. If None, uses config["input_file"]
        output_dir : str, optional
            Directory for output files. If None, uses config["output_dir"]
        output_prefix : str, optional
            Prefix for output filenames. If None, uses config["output_prefix"]
            
        Returns
        -------
        dict
            Result dictionary from find_periods
        """
        # Use parameters from config if not specified
        if file_path is None:
            file_path = self.config["input_file"]
        if output_dir is None:
            output_dir = self.config["output_dir"]
        if output_prefix is None:
            output_prefix = self.config["output_prefix"]
            
        if file_path is None:
            raise ValueError("No input file specified")
            
        # Set up output directory and file
        os.makedirs(output_dir, exist_ok=True)
        input_basename = os.path.basename(file_path).split('.')[0]
        output_file = os.path.join(output_dir, f"{output_prefix}_{input_basename}.png")
        
        logger.info(f"Input file: {file_path}")
        logger.info(f"Output file: {output_file}")
        
        # Read time series data
        logger.info(f"Reading data from {file_path}...")
        time, flux, error = self.read_time_series(file_path)
        logger.info(f"Loaded {len(time)} data points spanning {time.max() - time.min():.2f} days")
        
        # Check if we have enough data points
        if len(time) < 50:
            logger.error(f"Insufficient data points ({len(time)}) for reliable analysis. Need at least 50.")
            raise ValueError(f"Insufficient data points ({len(time)}) for reliable analysis. Need at least 50.")
        
        # Find periods using the two-stage method
        logger.info("Running period analysis...")
        result = self.find_periods(time, flux, error)
        best_period = result["best_period"]
        best_uncertainty = result["best_uncertainty"]
        logger.info(f"Best period: {best_period:.6f} ± {best_uncertainty:.6f} days")
        logger.info(f"Best period in hours: {best_period*24:.6f} ± {best_uncertainty*24:.6f} hours")
        
        # Plot results
        logger.info(f"Creating plots...")
        self.plot_results(time, flux, error, result, output_file)
        logger.info(f"Plots saved to {output_file}")
        
        # Save results to a file
        results_file = os.path.join(output_dir, f"{output_prefix}_{input_basename}_results.json")
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_result = {
                "best_period": float(result["best_period"]),
                "best_uncertainty": float(result["best_uncertainty"]),
                "chunk_best_period": float(result["chunk_best_period"]),
                "sliding_best_period": float(result["sliding_best_period"]),
                "subtraction_best_period": float(result["subtraction_best_period"]),
            }
            json.dump(serializable_result, f, indent=4)
        logger.info(f"Results saved to {results_file}")
        
        return result
        
    @staticmethod
    def process_directory(input_dir, config=None):
        """
        Process all compatible files in a directory.
        
        Parameters
        ----------
        input_dir : str
            Directory containing input files
        config : dict, optional
            Configuration dictionary. If None, default configuration is used
            
        Returns
        -------
        list
            List of result dictionaries for each file
        """
        import glob
        
        # Create instance with provided config
        nnp = NNPeriodogram(config)
        
        # Get all potential input files
        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
        fits_files = glob.glob(os.path.join(input_dir, "*.fits")) + glob.glob(os.path.join(input_dir, "*.fit"))
        txt_files = glob.glob(os.path.join(input_dir, "*.txt")) + glob.glob(os.path.join(input_dir, "*.dat"))
        
        all_files = csv_files + fits_files + txt_files
        logger.info(f"Found {len(all_files)} potential input files in {input_dir}")
        
        results = []
        
        # Process each file
        for i, file_path in enumerate(all_files):
            try:
                logger.info(f"Processing file {i+1}/{len(all_files)}: {os.path.basename(file_path)}")
                
                # Process the file
                result = nnp.analyze_file(file_path)
                results.append(result)
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Processed {len(results)} files from {input_dir}")
        return results

# Command-line interface functions to maintain backward compatibility

def main():
    """
    Main function to run the two-stage NN_FAP periodogram analyzer.
    Uses only the inlist file for configuration with no command-line arguments.
    """
    import os
    import sys
    
    # Default inlist path
    inlist_file = "inlist.txt"
    
    try:
        # Create NNPeriodogram instance from inlist file
        nnp = NNPeriodogram.from_inlist(inlist_file)
        
        # Validate required parameters
        if nnp.config["input_file"] is None:
            logger.error("No input file specified. Please set input_file in the inlist file.")
            sys.exit(1)
        
        # Check if input is a directory
        if os.path.isdir(nnp.config["input_file"]):
            NNPeriodogram.process_directory(nnp.config["input_file"], nnp.config)
        else:
            # Process single file
            nnp.analyze_file()
        
        logger.info("Analysis complete!")
        return 0
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())