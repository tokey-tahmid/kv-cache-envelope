#!/bin/bash
# Execute one envelope sweep on a GPU node.
#
#   sbatch slurm/run_sweep.sh configs/validate_7b.yaml
#   sbatch --array=0-2 slurm/run_sweep.sh configs/envelope_len.yaml   # strided slices
#
# Runs unpacked, one engine at a time on GPU 0. The sibling project measured
# packing four TP=1 replicas onto one node and found +150% TTFT and -34%
# throughput, so packing is not measurement-safe there. It is *less* harmful
# here, because this project measures a memory decision made at init rather
# than a latency, but the occupancy phases do measure throughput-sensitive
# preemption behaviour, so the same unpacked discipline applies.
#
#SBATCH --job-name=kve-sweep
#SBATCH --account=p201466
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/kve_sweep_%j.out

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

CONFIG="${1:?usage: run_sweep.sh <config.yaml> [extra args...]}"
shift || true
[[ "$CONFIG" = /* ]] || CONFIG="${HERE}/${CONFIG}"

module load "$KVE_MOD_ENV"
module load "$KVE_MOD_APPTAINER"

PY="${KVE_VENV}/bin/python"
export PYTHONPATH="${HERE}/src${PYTHONPATH:+:$PYTHONPATH}"

log() { printf '[sweep %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
log "host=$(hostname) job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID:-none}"
log "config=${CONFIG}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

# Syntax gate: the login node is Python 3.6 and cannot parse this code, so a
# typo would otherwise be discovered several minutes into a billed node.
"$PY" -m compileall -q "${HERE}/src" || { log "compileall FAILED"; exit 1; }

# Single GPU per engine. The node bills whole regardless, but a TP=1 engine that
# can see four cards will size its KV pool against the wrong device count.
export CUDA_VISIBLE_DEVICES=0

SLICE_ARG=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
    SLICE_ARG=(--slice "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}")
    log "array slice ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
    # Stagger: weights stream from Lustre on every launch and simultaneous
    # starts contend for the same files.
    if [[ "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
        sleep $(( SLURM_ARRAY_TASK_ID * 20 ))
    fi
fi

cd "$HERE"
"$PY" -m kv_cache_envelope.cli run "$CONFIG" "${SLICE_ARG[@]}" "$@"

log "SWEEP TASK COMPLETE"
