#!/usr/bin/env bash
# CrossPump experiment grid: datasets x baselines x seeds {42, 2024, 2025, 2026}.
#   bash scripts/xp_run_all.sh --datasets all --baselines all --seeds 42,2024,2025,2026
#   bash scripts/xp_run_all.sh --datasets synth --baselines crosspump,xp_counts_rule --seeds 42 --smoke --max-folds 1 --device cpu
# Options: --ablations (CrossPump ablations in configs/crosspump/ablations/) --platforms-sweep (E3 coverage study)
# Real datasets need their raw files under data_raw/crosspump/ (scripts/xp_prepare_data.py); a dataset whose panel cannot be
# built is skipped with a message, never faked.
set -u
cd "$(dirname "$0")/.."
SEEDS="42,2024,2025,2026"; DATASETS="all"; BASELINES="all"; EXTRA=""; ABLATIONS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2;; --datasets) DATASETS="$2"; shift 2;; --baselines) BASELINES="$2"; shift 2;;
    --ablations) ABLATIONS=1; shift;; --smoke) EXTRA="$EXTRA --smoke"; shift;; --max-folds) EXTRA="$EXTRA --max-folds $2"; shift 2;;
    --device) EXTRA="$EXTRA --device $2"; shift 2;; *) echo "unknown option $1"; exit 1;;
  esac
done
ALL_DATASETS="hu_pd sapienza_pd xu_pumpolymp ardia_pd pumpsense tgdataset_pd mirtaheri dex_wash rugpull1k memechain"
ALL_BASELINES="crosspump xp_counts_rule xp_mirtaheri21 xp_nghiem21 xp_victor21 xp_lamorgia23 xp_hu23 xp_bolz24 xp_perseus25 xp_pumpsense26 xp_moura26 xp_lroo26"
[[ "$DATASETS" == "all" ]] && DATASETS="$ALL_DATASETS" || DATASETS="${DATASETS//,/ }"
[[ "$BASELINES" == "all" ]] && BASELINES="$ALL_BASELINES" || BASELINES="${BASELINES//,/ }"
PY="${PYTHON:-python3}"
mkdir -p results/crosspump/logs
for ds in $DATASETS; do
  $PY -c "
import sys; sys.path.insert(0, 'scripts'); from xp_common import import_all, load_cfgs, build_panel
import argparse; import_all()
a = argparse.Namespace(dataset='$ds', baseline='crosspump', smoke=False, override=[], max_folds=None)
cfg, ds_cfg, _ = load_cfgs(a); build_panel(ds_cfg)" >/dev/null 2>results/crosspump/logs/${ds}_panel.err || { echo "[skip] $ds (raw data missing or loader unavailable; see results/crosspump/logs/${ds}_panel.err)"; continue; }
  for s in ${SEEDS//,/ }; do
    for b in $BASELINES; do
      echo "[run] $ds | $b | seed $s"
      $PY scripts/xp_run.py --dataset "$ds" --baseline "$b" --seed "$s" $EXTRA 2>&1 | tee -a "results/crosspump/logs/${ds}_${b}_s${s}.log" | grep -E "^\[|INFO|Error|error" | tail -3
      if [[ $ABLATIONS -eq 1 && "$b" == "crosspump" ]]; then
        for abl in configs/crosspump/ablations/*.yaml; do
          name=$(basename "$abl" .yaml)
          echo "[abl] $ds | crosspump/$name | seed $s"
          $PY scripts/xp_run.py --dataset "$ds" --baseline crosspump --seed "$s" --tag "abl_$name" \
            --override $($PY -c "import yaml,json; d=yaml.safe_load(open('$abl')); d.pop('_base_',None); d.pop('ablation',None); print(' '.join(f'baseline.{k}={json.dumps(v)}' for k,v in d.items()))") $EXTRA 2>&1 \
            | tee -a "results/crosspump/logs/${ds}_crosspump_${name}_s${s}.log" | grep -E "^\[|INFO|Error|error" | tail -3
        done
      fi
    done
  done
done
$PY scripts/xp_make_tables.py --results results/crosspump --fmt markdown
