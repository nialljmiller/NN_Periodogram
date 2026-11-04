"""Convert the 2022 Keras weights (_model.h5) to the NN_Periodogram PyTorch file and print its SHA-256."""
import hashlib, sys
import torch
from nn_periodogram.models import from_keras_h5

model = from_keras_h5(sys.argv[1])
torch.save(model.state_dict(), sys.argv[2])
print(hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest())
