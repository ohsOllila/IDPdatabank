#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NOISES=(10 20 30)
CUTOFFS=(600 800 1000)
INFERENCES=(laplace nuts)
NOISE_MODELS=(constant quadratic)

total=$(( ${#NOISES[@]} * ${#CUTOFFS[@]} * ${#INFERENCES[@]} * ${#NOISE_MODELS[@]} ))
run=0

for noise in "${NOISES[@]}"; do
    for cutoff in "${CUTOFFS[@]}"; do
        for inference in "${INFERENCES[@]}"; do
            for noise_model in "${NOISE_MODELS[@]}"; do
                run=$(( run + 1 ))
                out_dir="results/sweep/noise${noise}_cutoff${cutoff}_${inference}_${noise_model}"
                if compgen -G "${out_dir}/*/all_metrics.json" > /dev/null 2>&1; then
                    echo "[$run/$total] SKIP (already done): noise=$noise cutoff=$cutoff inference=$inference noise_model=$noise_model"
                    continue
                fi
                echo "[$run/$total] noise=$noise cutoff=$cutoff inference=$inference noise_model=$noise_model"
                python run.py --sample \
                    --max-noise "$noise" \
                    --cutoff "$cutoff" \
                    --inference "$inference" \
                    --noise-model "$noise_model" \
                    --out-dir "$out_dir"
            done
        done
    done
done

echo "Sweep complete."
