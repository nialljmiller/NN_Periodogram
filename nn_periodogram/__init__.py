"""NN_Periodogram: neural-network false-alarm probabilities and periodograms for time series.

    import nn_periodogram
    p = nn_periodogram.periodogram(time, mag)        # search
    p.best_period, p.best_fap, p.peaks
    nn_periodogram.fap(time, mag, 0.5077)           # score given period(s)
"""

from .periodogram import Periodogram, fap, periodogram

__version__ = "0.1.0"
__all__ = ["fap", "periodogram", "Periodogram", "__version__"]
