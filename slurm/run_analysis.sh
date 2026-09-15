#!/bin/bash
# THE one command that regenerates every figure and table in this repository.
#
#   sbatch slurm/run_analysis.sh
#
# Runs on the gpu partition without touching a GPU: the login node has Python
# 3.6 and cannot parse this code, and p201466 has no CPU allocation. Everything
# here reads results/raw/ and results/mined_p1/ and writes results/figures/.
# Nothing here can produce a measurement.
#
#SBATCH --job-name=kve-analysis
#SBATCH --account=p201466
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/kve_analysis_%j.out

set -euo pipefail

if ! type module >/dev/null 2>&1; then
    set +eu
    source /etc/profile
    set -eu
fi

find_repo() {
    local c
    for c in "${KVE_REPO:-}" "${SLURM_SUBMIT_DIR:-}" \
             "/home/users/u104240/apps/inference-work/kv-cache-envelope"; do
        if [[ -n "$c" && -f "${c}/env.sh" ]]; then printf '%s' "$c"; return 0; fi
    done
    return 1
}
HERE="$(find_repo)" || { echo "cannot locate repo root" >&2; exit 1; }
# shellcheck disable=SC1091
source "${HERE}/env.sh"

module load "$KVE_MOD_ENV"

PY="${KVE_VENV}/bin/python"
export PYTHONPATH="${HERE}/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

log() { printf '\n===== [analysis %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
log "host=$(hostname) job=${SLURM_JOB_ID}"

cd "$HERE"

log "syntax gate"
"$PY" -m compileall -q src analysis tests

log "unit tests"
"$PY" -m pytest tests -q

log "mine the sibling project's artifacts (read-only)"
"$PY" analysis/mine_p1.py

log "calculator validation table"
"$PY" analysis/make_tables.py

log "figure: predicted vs measured, with residual decomposition"
"$PY" analysis/plot_predicted_vs_measured.py

log "figure: capacity envelope"
"$PY" analysis/plot_capacity_envelope.py

log "figure: PagedAttention granularity, capacity and preemption"
"$PY" analysis/plot_paged_attention.py

log "verify every identifier in prose against ground truth"
"$PY" analysis/verify_identifiers.py

if [[ -x "${KVE_VENV}/bin/ruff" ]]; then
    log "ruff"
    "${KVE_VENV}/bin/ruff" check src analysis tests || true
fi

log "ANALYSIS COMPLETE"
ls -la "${HERE}/results/figures"
