"""nifty-ls in its own Python process. On macOS nifty-ls (finufft) and PyTorch each ship
an OpenMP runtime and cannot share a process, so benchmarks call nifty-ls through this
worker, a separate interpreter that never imports torch. It times the periodogram itself,
so the pipe is not part of the timing."""
import pickle
import struct
import subprocess
import sys
import time


def _send(f, obj):
    b = pickle.dumps(obj, protocol=5); f.write(struct.pack("<Q", len(b))); f.write(b); f.flush()


def _recv(f):
    head = f.read(8)
    if len(head) < 8:          # the other end has gone
        return None
    n = struct.unpack("<Q", head)[0]
    return pickle.loads(f.read(n))


class Nifty:
    def __init__(self):
        self.proc = subprocess.Popen([sys.executable, "-I", __file__], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def power(self, t, m, e, f0, df, nf, backend="finufft"):
        """(power array, seconds spent inside nifty-ls); backend "finufft" (CPU) or "cufinufft" (GPU)."""
        _send(self.proc.stdin, (t, m, e, f0, df, nf, backend))
        return _recv(self.proc.stdout)

    def close(self):
        _send(self.proc.stdin, None); self.proc.wait()


if __name__ == "__main__":
    import nifty_ls
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    while True:
        job = _recv(inp)
        if job is None:
            break
        t, m, e, f0, df, nf, backend = job if len(job) == 7 else (*job, "finufft")
        s = time.perf_counter()
        p = nifty_ls.lombscargle(t, m, e, fmin=f0, fmax=f0 + df * (nf - 1), Nf=nf, backend=backend).power
        _send(out, (p, time.perf_counter() - s))
