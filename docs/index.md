# NN_Periodogram

Find the periods of variable stars in any survey's light curves, and tell real
periods from harmonics and aliases, with a small neural network that runs on a
GPU or a CPU.

```python
import nn_periodogram as nnp

p = nnp.periodogram(time, mag, err)          # find the period
p.best_period, p.best_fap, p.peaks

nnp.check_period(time, mag, 1.0155, err)     # is a known period right, 2P, P/2 or an alias?
nnp.fap(time, mag, [0.5077, 1.0155], err)    # false-alarm probability of given periods
```

```{toctree}
:maxdepth: 2

usage
devices
models
api
citing
```
