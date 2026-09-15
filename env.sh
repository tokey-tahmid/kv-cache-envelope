# Shared environment for kv-cache-envelope on MeluXina.
# Source from every slurm script: `source "${HERE}/env.sh"`.
#
# `module` is NOT initialised in a Slurm batch environment here; scripts must
# `set +eu; source /etc/profile; set -eu` before any `module load`, and must load
# env/release/2025.1 FIRST or the EasyBuild tree is not on MODULEPATH at all.

# Compute is billed to p201466 (GPU headroom, no CPU allocation at all, so every
# job runs on the gpu partition). Storage stays under p201362; filesystem access
# follows unix group membership, not the Slurm account, so the split needs no
# special handling.
export KVE_ACCOUNT="p201466"

KVE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KVE_REPO

# Bulk data lives on project storage, shared with P1: the vLLM image and the
# Qwen2.5 weights are already provisioned there and are deliberately reused
# rather than re-pulled. See provision.lock.json for digests and revisions.
export KVE_DATA="/mnt/tier2/project/p201362/ttahmid/inference-work"
export KVE_IMAGES="${KVE_DATA}/images"
export KVE_LOGS="${KVE_DATA}/logs"
export KVE_ENV="${KVE_DATA}/env"

export HF_HOME="${KVE_DATA}/hf_cache"
export HF_HUB_CACHE="${KVE_DATA}/hf_cache/hub"

export APPTAINER_CACHEDIR="${KVE_DATA}/apptainer_cache"
export APPTAINER_TMPDIR="${KVE_DATA}/apptainer_tmp"

export KVE_MOD_ENV="env/release/2025.1"
export KVE_MOD_APPTAINER="Apptainer/1.4.2-GCCcore-14.2.0"

export KVE_SIF_VLLM="${KVE_IMAGES}/vllm.sif"

# The harness python env provisioned for P1 (python 3.12.14, numpy/pandas/
# matplotlib/httpx/structlog/pytest/pyyaml). Reused rather than rebuilt: P2 has
# the same dependency surface and a second copy would drift.
export KVE_VENV="${KVE_ENV}/venv"

export KVE_LOCKFILE="${KVE_REPO}/provision.lock.json"

# P1's committed artifacts, mined for the PagedAttention deliverable. READ ONLY:
# nothing in this project ever writes under it.
export KVE_P1_REPO="$(cd "${KVE_REPO}/../llm-inference-bench" 2>/dev/null && pwd)"

mkdir -p "${KVE_LOGS}" 2>/dev/null || true
