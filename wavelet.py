import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet2
from scipy.interpolate import interp1d

def wavelet_analysis(time, flux, flux_err=None, dt=None, widths=None, f0=1.0, plot=True):
    """
    Perform wavelet analysis on a 1D light curve.
    
    Parameters:
    -----------
    time : array-like
        Time array (can be unevenly sampled).
    flux : array-like
        Flux or magnitude values.
    flux_err : array-like or None
        Optional flux uncertainties.
    dt : float or None
        Desired resampling interval. If None, uses median time spacing.
    widths : array-like or None
        Wavelet widths (scales). If None, uses log spacing.
    f0 : float
        Central frequency of Morlet wavelet (1.0 = good default).
    plot : bool
        Whether to show diagnostic plots.
    
    Returns:
    --------
    wavelet_power : 2D np.ndarray
        Wavelet power spectrum (scales x time).
    t_uniform : np.ndarray
        Resampled time array.
    periods : np.ndarray
        Approximate period associated with each wavelet scale.
    gws : np.ndarray
        Global Wavelet Spectrum (mean power vs period).
    """
    
    # Step 1: Uniform resampling
    if dt is None:
        dt = np.median(np.diff(time))
    
    t_uniform = np.arange(time.min(), time.max(), dt)
    interp_func = interp1d(time, flux, kind='linear', fill_value='extrapolate')
    y_uniform = interp_func(t_uniform)
    y_uniform -= np.mean(y_uniform)

    # Step 2: Define widths/scales
    if widths is None:
        widths = np.logspace(0.5, 2.5, 100)  # Adjust as needed
    
    # Step 3: Run CWT with Morlet
    wave = cwt(y_uniform, morlet2, widths, w=f0)
    power = np.abs(wave)**2
    periods = 2 * np.pi * widths / f0  # Rough mapping of width to period

    # Step 4: Global wavelet spectrum
    gws = power.mean(axis=1)

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})
        
        # Wavelet power
        im = axs[0].imshow(power, extent=[t_uniform[0], t_uniform[-1], periods[-1], periods[0]],
                           aspect='auto', cmap='magma')
        axs[0].set_ylabel("Period (days)")
        axs[0].set_title("Wavelet Power Spectrum")
        fig.colorbar(im, ax=axs[0], label="Power")

        # Global Wavelet Spectrum
        axs[1].plot(periods, gws, color='black')
        axs[1].set_ylabel("GWS Power")
        axs[1].set_xlabel("Period (days)")
        axs[1].set_xscale("log")
        axs[1].set_title("Global Wavelet Spectrum")
        
        plt.tight_layout()
        plt.show()

    return power, t_uniform, periods, gws
