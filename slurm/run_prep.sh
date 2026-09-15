#!/bin/bash
# Non-GPU prep, which still needs a compute node: syntax gate, unit tests, HF
# config bundles for the calculator validation, and a dump of vLLM's block
# manager and scheduler out of the pinned container for the source-reading
# writeup.
#
#   sbatch slurm/run_prep.sh
#
# Runs on the `gpu` partition because p201466 has no CPU allocation at all
# (gres/cpun=0) and every cpu-partition QOS carries DenyOnLimit. None of this
# needs a GPU; it needs an account that can pay for a node.
#
#SBATCH --job-name=kve-prep
#SBATCH --account=p201466
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/kve_prep_%j.out

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
# BASH_SOURCE points at the psslurm spool copy under /var/spool/parastation, not
# at this file, so the repo must be located by SLURM_SUBMIT_DIR instead.
HERE="$(find_repo)" || { echo "cannot locate repo root" >&2; exit 1; }
# shellcheck disable=SC1091
source "${HERE}/env.sh"

module load "$KVE_MOD_ENV"
module load "$KVE_MOD_APPTAINER"

PY="${KVE_VENV}/bin/python"
export PYTHONPATH="${HERE}/src${PYTHONPATH:+:$PYTHONPATH}"

log() { printf '[prep %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
log "host=$(hostname) job=${SLURM_JOB_ID}"
"$PY" -V

# --- 1. syntax gate -----------------------------------------------------------
# The login node runs Python 3.6 and cannot parse this code, so a job is the
# first opportunity to find a typo. Costs seconds; catches what a local check
# normally would.
log "compileall gate"
"$PY" -m compileall -q "${HERE}/src" "${HERE}/analysis" "${HERE}/tests"
log "compileall OK"

# --- 2. unit tests ------------------------------------------------------------
log "pytest (pre-fetch: the fetched-config validation self-skips)"
"$PY" -m pytest "${HERE}/tests" -q 2>&1 | tail -30 || true

# --- 3. model config bundles --------------------------------------------------
log "fetching HF config bundles"
"$PY" "${HERE}/analysis/fetch_model_configs.py" || log "fetch returned non-zero"

# --- 4. unit tests again, now with the bundles present ------------------------
log "pytest (post-fetch: validates the calculator against the real configs)"
"$PY" -m pytest "${HERE}/tests" -q 2>&1 | tail -30

# --- 5. ruff ------------------------------------------------------------------
if [[ -x "${KVE_VENV}/bin/ruff" ]]; then
    log "ruff"
    "${KVE_VENV}/bin/ruff" check "${HERE}/src" "${HERE}/analysis" "${HERE}/tests" || true
else
    log "ruff not present in the shared venv; skipped"
fi

# --- 6. vLLM source, out of the pinned image ----------------------------------
# Read at the pinned digest, not from GitHub: the writeup must cite the version
# that produced the measurements, and the image is the only authority on that.
SRCDIR="${HERE}/docs/vllm_source"
mkdir -p "$SRCDIR"
VLLM_PKG="/usr/local/lib/python3.12/dist-packages/vllm"

log "locating vllm inside the pinned image"
apptainer exec --cleanenv "$KVE_SIF_VLLM" python3 -c \
    "import vllm, os; print(vllm.__version__); print(os.path.dirname(vllm.__file__))" \
    | tee "${SRCDIR}/VERSION.txt"

log "listing candidate source files"
apptainer exec --cleanenv "$KVE_SIF_VLLM" bash -c \
    "ls -la ${VLLM_PKG}/v1/core/ ${VLLM_PKG}/v1/core/ 2>/dev/null; echo ---; find ${VLLM_PKG} -name 'block_pool.py' -o -name 'kv_cache_manager.py' -o -name 'scheduler.py' -o -name 'kv_cache_utils.py' -o -name 'kv_cache_coordinator.py' -o -name 'single_type_kv_cache_manager.py' 2>/dev/null" \
    | tee "${SRCDIR}/FILELIST.txt"

for f in v1/core/block_pool.py v1/core/kv_cache_manager.py v1/core/sched/scheduler.py \
         v1/core/kv_cache_utils.py v1/core/kv_cache_coordinator.py \
         v1/core/single_type_kv_cache_manager.py v1/core/sched/output.py \
         v1/kv_cache_interface.py worker/gpu_worker.py; do
    out="${SRCDIR}/$(echo "$f" | tr '/' '_')"
    if apptainer exec --cleanenv "$KVE_SIF_VLLM" cat "${VLLM_PKG}/${f}" > "$out" 2>/dev/null; then
        log "dumped ${f} ($(wc -l < "$out") lines)"
    else
        rm -f "$out"
        log "ABSENT ${f}"
    fi
done

log "PREP COMPLETE"
