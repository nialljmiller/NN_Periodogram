# GPU and CPU

Every stage (classical periodograms, folding, binning, the network) runs on the
device you choose:

```python
nnp.periodogram(t, m, e, device="cuda")   # all on the GPU
nnp.periodogram(t, m, e, device="cpu")    # all on the CPU
nnp.periodogram(t, m, e, device="mps")    # Apple silicon (phases in float64 on the CPU)
```

The default is CUDA if available, then MPS, then CPU. Phases are always
computed in double precision (time x frequency over long baselines needs it);
Apple's MPS has no float64, so there they are computed on the CPU. Memory is
bounded at any light-curve length: work is chunked so that roughly 2^23 phases
are in flight at once.
