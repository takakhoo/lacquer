#!/bin/sh
# Usage on the training box: scripts/train_lab.sh <run-name> [train args...]
# Runs in tmux session <run-name> on ONE GPU: the one with the most free memory (GPU 5 excluded, it is
# used by another of our jobs). Restarts from last.pt if the process dies (shared machine, OOM happens).
ROOT=${REMASTER_ROOT:-/scratch/$USER}
NAME=$1; shift
cd $ROOT/remaster && mkdir -p runs/$NAME logs
PIN=${GPU:-}
DATA=${DATA:-data/raw/fma_medium}
cat > runs/$NAME/run.sh <<EOF
#!/bin/sh
cd $ROOT/remaster
export MPLCONFIGDIR=$ROOT/.cache/mpl TORCH_HOME=$ROOT/.cache/torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  G=\${PIN:-$PIN}
  [ -z "\$G" ] && G=\$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | grep -v '^5,' | sort -t, -k2 -n | head -1 | cut -d, -f1)
  echo "=== starting on GPU \$G at \$(date)"
  CUDA_VISIBLE_DEVICES=\$G $ROOT/venvs/remaster/bin/python -m remaster.train --data $DATA --rir data/raw/mit_ir --out runs/$NAME --resume runs/$NAME/last.pt $* && break
  echo "=== crashed, retrying in 60s"; sleep 60
done
EOF
chmod +x runs/$NAME/run.sh
tmux new-session -d -s "$NAME" "runs/$NAME/run.sh 2>&1 | tee -a logs/$NAME.log"
