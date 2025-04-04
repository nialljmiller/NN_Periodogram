#!/usr/bin/env python3
"""
Examples of how to use the NNPeriodogram class in your Python scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.timeseries import LombScargle

# Import the NNPeriodogram class
# For a local import (when not installed as a package)
try:
    # Try to import from package
    from NN_Periodogram import NNPeriodogram
except ImportError:
    # If not installed as a package, import directly from file
    from NNP import NNPeriodogram

def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created output directory: {directory}")

def basic_usage_example():
    """
    Basic example of using NNPeriodogram with manually provided data.
    """
    print("\n=== Basic Usage Example ===")
    
    # Create some synthetic time series data
    # A sinusoidal signal with period=0.5 days + noise
    np.random.seed(42)
    t = np.linspace(0, 10, 200)  # 10 days of data with 200 points
    true_period = 0.5  # days
    true_freq = 1.0 / true_period
    y = np.sin(2 * np.pi * true_freq * t) + 0.2 * np.random.randn(len(t))
    err = np.ones_like(t) * 0.2  # Constant error bars
    
    # Create a configuration dictionary with our parameters
    output_dir = "./results/example_basic"
    ensure_directory(output_dir)
    
    config = {
        "period_min": 0.1,          # Minimum period to search (days)
        "period_max": 1.0,          # Maximum period to search (days)
        "n_periods": 5000,          # Number of periods to sample
        "use_lombscargle_fallback": True,  # Use LombScargle since we don't have NN_FAP model here
        "output_dir": output_dir,   # Save results to this directory
        "output_prefix": "synthetic_data"  # Prefix for output filenames
    }
    
    # Create NNPeriodogram instance with our config
    nnp = NNPeriodogram(config)
    
    # Find periods using the two-stage method
    result = nnp.find_periods(t, y, err)
    
    # Print the best periods found by each method
    print(f"True period: {true_period:.6f} days")
    print(f"Chunk method best period: {result['chunk_best_period']:.6f} days")
    print(f"Sliding window method best period: {result['sliding_best_period']:.6f} days")
    print(f"Subtraction method best period: {result['subtraction_best_period']:.6f} days")
    print(f"Best period: {result['best_period']:.6f} ± {result['best_uncertainty']:.6f} days")
    
    # Plot the results (automatically saves to output_dir)
    output_file = os.path.join(output_dir, f"{config['output_prefix']}_periodogram.png")
    nnp.plot_results(t, y, err, result, output_file)
    print(f"Saved periodogram plots to {output_dir}")
    
    # Create phase-folded plot and save to file
    phase_fold_output = os.path.join(output_dir, f"{config['output_prefix']}_phase_folded.png")
    plt.figure(figsize=(10, 6))
    phase, folded_y, folded_err = nnp.phase_fold_lightcurve(t, y, err, result['best_period'])
    plt.errorbar(phase, folded_y, yerr=folded_err, fmt='.', alpha=0.5)
    plt.xlabel('Phase')
    plt.ylabel('Flux')
    plt.title(f'Phase-folded Light Curve (Period = {result["best_period"]:.6f} days)')
    plt.savefig(phase_fold_output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved additional phase-folded plot to {phase_fold_output}")
    
    return result


def comparison_with_lombscargle():
    """
    Compare NNPeriodogram with astropy's LombScargle.
    """
    print("\n=== Comparison with LombScargle ===")
    
    # Create some synthetic time series data (irregular sampling)
    np.random.seed(42)
    t = np.random.uniform(0, 20, 100)  # 20 days of randomly sampled data
    t.sort()  # Ensure time is sorted
    
    true_period = 1.3  # days
    true_freq = 1.0 / true_period
    y = np.sin(2 * np.pi * true_freq * t) + 0.2 * np.random.randn(len(t))
    err = np.ones_like(t) * 0.2
    
    # Setup output directory
    output_dir = "./results/example_comparison"
    ensure_directory(output_dir)
    
    # Run NNPeriodogram
    config = {
        "period_min": 0.5,
        "period_max": 2.0,
        "n_periods": 5000,
        "use_lombscargle_fallback": True,
        "output_dir": output_dir,
        "output_prefix": "comparison"
    }
    nnp = NNPeriodogram(config)
    nnp_result = nnp.find_periods(t, y, err)
    
    # Run LombScargle for comparison
    frequencies = np.linspace(1.0/config["period_max"], 1.0/config["period_min"], config["n_periods"])
    # Ensure frequencies are in ascending order
    frequencies = np.sort(frequencies)
    periods = 1.0 / frequencies
    ls = LombScargle(t, y, err)
    ls_power = ls.power(frequencies)
    ls_best_idx = np.argmax(ls_power)
    ls_best_period = periods[ls_best_idx]
    
    # Compare results
    print(f"True period: {true_period:.6f} days")
    print(f"NNPeriodogram best period: {nnp_result['best_period']:.6f} days")
    print(f"LombScargle best period: {ls_best_period:.6f} days")
    
    # Plot comparison and save
    comparison_output = os.path.join(output_dir, "method_comparison.png")
    plt.figure(figsize=(12, 6))
    
    # Plot NNPeriodogram results
    plt.subplot(1, 2, 1)
    plt.plot(nnp_result["primary_periods"], nnp_result["subtraction_power"], 'b-', label='Subtraction Method')
    plt.axvline(nnp_result['best_period'], color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Period (days)')
    plt.ylabel('Power (1-FAP)')
    plt.title('NNPeriodogram')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot LombScargle results
    plt.subplot(1, 2, 2)
    plt.plot(periods, ls_power, 'g-', label='LombScargle')
    plt.axvline(ls_best_period, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Period (days)')
    plt.ylabel('Power')
    plt.title('LombScargle')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(comparison_output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {comparison_output}")
    
    # Also save the standard NNPeriodogram plots
    nnp_output = os.path.join(output_dir, f"{config['output_prefix']}_periodogram.png")
    nnp.plot_results(t, y, err, nnp_result, nnp_output)
    
    return nnp_result, ls_best_period


def file_analysis_example(file_path):
    """
    Example of analyzing a time series file.
    
    Parameters
    ----------
    file_path : str
        Path to a time series file (CSV, FITS, or TXT)
    """
    print(f"\n=== File Analysis Example: {file_path} ===")
    
    # Extract filename for output directory
    file_basename = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = f"./results/{file_basename}_analysis"
    ensure_directory(output_dir)
    
    # Create NNPeriodogram instance with custom config
    config = {
        "period_min": 0.01,
        "period_max": 10.0,
        "use_lombscargle_fallback": True,
        "output_dir": output_dir,
        "output_prefix": file_basename
    }
    nnp = NNPeriodogram(config)
    
    # Analyze the file
    try:
        result = nnp.analyze_file(file_path)
        print(f"Analysis complete! Best period: {result['best_period']:.6f} days")
        print(f"Results saved in {config['output_dir']} directory")
        
        # The analyze_file method already creates plots, but we'll mention the locations
        periodogram_path = os.path.join(output_dir, f"{file_basename}_periodogram.png")
        phase_fold_path = os.path.join(output_dir, f"{file_basename}_folded_subtraction_method.png")
        
        print(f"Main periodogram plot: {periodogram_path}")
        print(f"Phase-folded plot for best period: {phase_fold_path}")
        
        return result
    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Run all examples.
    """
    print("=== NNPeriodogram Usage Examples ===")
    
    # Run the basic usage example
    basic_result = basic_usage_example()
    
    # Run the LombScargle comparison example
    nnp_result, ls_period = comparison_with_lombscargle()
    
    # Run the file analysis example if a file is provided
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        file_result = file_analysis_example(file_path)
    else:
        print("\nNo file provided for file analysis example. Skipping.")
        print("To run the file analysis example, provide a file path as a command-line argument:")
        print("    python example.py your_data_file.csv")
    
    print("\nAll examples completed. Results saved in the ./results/ directory.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())