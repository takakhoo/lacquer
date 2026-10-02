#!/bin/sh
# After the selection study: pick the variant with the best final validation gain and continue the
# main fine-tune from its checkpoint with its settings (plus algorithmic reverb in the training mix).
ROOT=${REMASTER_ROOT:-/scratch/$USER}
cd $ROOT/remaster
while ! grep -q "study done" logs/study.log 2>/dev/null; do sleep 60; done
BEST=$($ROOT/venvs/remaster/bin/python - <<'PY'
import json, glob
best = None
for f in sorted(glob.glob("runs/study/*/val.jsonl")):
    last = json.loads(open(f).read().strip().splitlines()[-1])
    name = f.split("/")[2]
    print(name, last["step"], round(last["mean_sisdr_gain"], 3), file=__import__("sys").stderr)
    if best is None or last["mean_sisdr_gain"] > best[1]:
        best = (name, last["mean_sisdr_gain"])
print(best[0])
PY
)
case "$BEST" in
  seg8)   EXTRA="--batch 2 --seg 8" ;;
  lowlr)  EXTRA="--batch 4 --lr-scale 0.3" ;;
  logmag) EXTRA="--batch 4 --w-log 1.5 --w-wave 5" ;;
  *)      EXTRA="--batch 4" ;;
esac
echo "=== continuing main run from study/$BEST with: $EXTRA ($(date))" >> logs/study.log
cp runs/study/$BEST/last.pt runs/ft_bs/last.pt
cp runs/study/$BEST/best.pt runs/ft_bs/best.pt
DATA="data/raw/fma_medium data/raw/musdb18hq/train" GPU=3 scripts/train_lab.sh ft_bs --data-weights 0.6 0.4 --p-identity 0.15 --p-algo 0.35 \
  --task artifact --arch msst_bs --msst-config pretrained/anvuew_bs.yaml --init pretrained/anvuew_bs.ckpt --lr 1e-4 --warmup 300 --steps 60000 --val-every 1000 --workers 12 $EXTRA
