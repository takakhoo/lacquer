#!/bin/sh
# CBAM / FiLM ablation on a spectrogram U-Net, same task, data and budget as the from-scratch
# band-split run `art_small` (FMA-medium, artifact task, batch 8, lr 3e-4). Runs after the study, one GPU.
#   scripts/ablate_unet_lab.sh <gpu> <steps>
ROOT=${REMASTER_ROOT:-/scratch/$USER}
GPU=$1; N=$2
cd $ROOT/remaster
export MPLCONFIGDIR=$ROOT/.cache/mpl TORCH_HOME=$ROOT/.cache/torch CUDA_VISIBLE_DEVICES=$GPU
while ! grep -q "study done" logs/study.log 2>/dev/null; do sleep 60; done
run() {
  name=$1; shift
  echo "=== $name: $* ($(date))" >> logs/ablate.log
  $ROOT/venvs/remaster/bin/python -m remaster.train --data data/raw/fma_medium --rir data/raw/mit_ir --task artifact --arch unet \
    --batch 8 --lr 3e-4 --steps 40000 --stop-at $N --val-every 2000 --workers 10 --out runs/ablate/$name "$@" >> logs/ablate_$name.log 2>&1
  grep "\[val" logs/ablate_$name.log | tail -2 >> logs/ablate.log
}
run unet_plain
run unet_cbam --cbam
run unet_cbam_film --cbam --film
echo "=== ablation done $(date)" >> logs/ablate.log
