#!/bin/sh
# Model-selection study on the training box, one GPU, sequential. Branches the main fine-tune at a shared
# checkpoint and trains each variant for the same number of steps, so validation numbers are comparable.
#   scripts/study_lab.sh <gpu> <branch-step> <steps-per-variant>
ROOT=${REMASTER_ROOT:-/scratch/$USER}
GPU=$1; AT=$2; N=$3
cd $ROOT/remaster
export MPLCONFIGDIR=$ROOT/.cache/mpl TORCH_HOME=$ROOT/.cache/torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU
# wait for the main run to reach the branch point, then pause it
while [ "$(grep -c "step=$AT " logs/ft_bs.log)" -lt 1 ] && ! grep -q "\[val $AT\]" logs/ft_bs.log; do sleep 30; done
sleep 20; tmux kill-session -t ft_bs 2>/dev/null; sleep 5
mkdir -p runs/study && cp runs/ft_bs/last.pt runs/study/branch.pt
COMMON="--data data/raw/fma_medium data/raw/musdb18hq/train --data-weights 0.6 0.4 --p-identity 0.15 --rir data/raw/mit_ir --task artifact --arch msst_bs --msst-config pretrained/anvuew_bs.yaml --lr 1e-4 --warmup 300 --steps 60000 --val-every 1000 --workers 12 --stop-at $((AT + N))"
run() {
  name=$1; shift
  mkdir -p runs/study/$name && cp runs/study/branch.pt runs/study/$name/last.pt
  echo "=== $name: $* ($(date))" | tee -a logs/study.log
  until $ROOT/venvs/remaster/bin/python -m remaster.train $COMMON --out runs/study/$name --resume runs/study/$name/last.pt "$@" >> logs/study_$name.log 2>&1; do echo "retry $name" >> logs/study.log; sleep 60; done
  grep "\[val" logs/study_$name.log | tail -3 >> logs/study.log
}
run base   --batch 4
run seg8   --batch 2 --seg 8
run lowlr  --batch 4 --lr-scale 0.3
run logmag --batch 4 --w-log 1.5 --w-wave 5
echo "=== study done $(date)" >> logs/study.log
