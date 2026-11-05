#!/usr/bin/env bash
# Start the first training run once the big verified shards exist.
D=/media/nill/Musak/nn_periodogram
until [ -e $D/shards/vvv_real_verified.json ] && [ -e $D/shards/asassn_vsx_verified.json ]; do sleep 300; done
cd $D/code/training
echo "starting $(date)"
exec $HOME/.venvs/nnperiods/bin/python -u train.py --data $D/shards --out $D/runs/v1 --steps 60000 --batch 256
