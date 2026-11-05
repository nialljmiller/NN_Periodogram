#!/usr/bin/env bash
# Train v2 on every verified shard once TESS verification is done.
D=/media/nill/Musak/nn_periodogram
until [ -e $D/shards/tess_eb_verified.json ]; do sleep 300; done
cd $D/code/training
echo "starting $(date)"
exec $HOME/.venvs/nnperiods/bin/python -u train.py --data $D/shards --out $D/runs/v2 --steps 100000 --batch 256
