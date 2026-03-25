#!/bin/bash
# Submit a grid search over focal-loss (alpha, gamma) via the Makefile.
#
# Usage:
#   bash slurm/sweep-focal-loss.sh                # submit on biocluster (default)
#   bash slurm/sweep-focal-loss.sh --dry-run      # print commands without submitting
#
# All jobs run through `make submit` which handles TARGET, sbatch selection,
# and remote execution.  Override TARGET as usual:
#   TARGET=compute bash slurm/sweep-focal-loss.sh

set -euo pipefail

DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

ALPHAS=(0.80 0.90 0.95 0.99)
GAMMAS=(1.0 2.0 3.0 4.0)

echo "Sweep: ${#ALPHAS[@]} alphas × ${#GAMMAS[@]} gammas = $(( ${#ALPHAS[@]} * ${#GAMMAS[@]} )) jobs"
echo "  alphas: ${ALPHAS[*]}"
echo "  gammas: ${GAMMAS[*]}"
echo "  TARGET: ${TARGET:-cluster}"
echo ""

for alpha in "${ALPHAS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
        EXTRA="--focal_alpha $alpha --focal_gamma $gamma"

        if $DRY_RUN; then
            echo "[dry-run] make submit JOB=composition_focal EXTRA=\"$EXTRA\""
        else
            echo "Submitting: alpha=$alpha  gamma=$gamma"
            make submit JOB=composition_focal EXTRA="$EXTRA"
        fi
    done
done
