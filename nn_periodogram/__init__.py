"""NN_Periodogram: neural-network false-alarm probabilities and periodograms for time series.

    import nn_periodogram
    p = nn_periodogram.periodogram(time, mag)        # search
    p.best_period, p.best_fap, p.peaks
    nn_periodogram.fap(time, mag, 0.5077)           # score given period(s)
    nn_periodogram.check_period(time, mag, 1.0155)  # is it P, 2P, P/2 or an alias?
"""

from .periodogram import PeriodCheck, Periodogram, check_period, fap, periodogram

__version__ = "1.1.0"
__all__ = ["fap", "periodogram", "check_period", "Periodogram", "PeriodCheck", "__version__"]
