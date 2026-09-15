"""Provenance capture for raw measurement artifacts.

Non-negotiable rule from the workspace CLAUDE.md: every result carries provenance,
raw artifacts are append-only, and nothing under ``results/raw/`` is ever produced
by anything but a real run.

This module is deliberately the *only* place that writes into ``results/raw/``.
It refuses to write an artifact whose provenance is incomplete, so a run that
could not identify its hardware or its backend version fails loudly rather than
producing an untraceable number.
"""

from __future__ import annotations

import functools
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import structlog

log = structlog.get_logger(__name__)

RunStatus = Literal["ok", "failed", "oom", "timeout", "launch_failed", "unsupported"]

# Fields that must be non-empty for an artifact to be writable at all. A run that
# cannot say what hardware or software produced it is not a measurement.
REQUIRED_FIELDS = (
    "run_id",
    "utc_timestamp",
    "hostname",
    "gpu_model",
    "gpu_count",
    "gpu_memory_gb",
    "driver_version",
    "backend_name",
    "model_id",
    "model_revision",
    "exact_command_line",
    "status",
)


def _run(cmd: list[str], timeout: float = 30.0) -> str | None:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


@functools.lru_cache(maxsize=8)
def git_sha(repo_root: Path) -> str | None:
    """Return HEAD sha, suffixed ``-dirty`` when the tree has uncommitted changes.

    The suffix matters: a figure regenerated from a dirty tree is not reproducible
    from a checkout, and the raw artifact should say so rather than imply it is.

    **Cached deliberately, so the answer is the state at process start.** Python
    imports the harness once, at launch, and the code that actually ran is
    whatever was on disk at that moment. Re-checking per artifact instead means a
    long sweep gets stamped `-dirty` because someone edited an unrelated file
    while it was running, which is both misleading (the running code was clean)
    and unfixable after the fact. Sampling once, at the point the modules were
    loaded, is the honest answer to "what produced this".

    It does not license editing the repo mid-sweep: an edit to a module that has
    not been imported yet would still change behaviour. It only stops an
    irrelevant edit from corrupting the record.
    """
    sha = _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    if sha is None:
        return None
    dirty = _run(["git", "-C", str(repo_root), "status", "--porcelain"])
    return f"{sha}-dirty" if dirty else sha


def query_gpus() -> dict[str, Any]:
    """Read GPU inventory and driver from nvidia-smi.

    Returns empty values rather than raising, so that a launch failure on a node
    without visible GPUs still produces a recorded failed run.
    """
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return {"gpu_model": None, "gpu_count": 0, "gpu_memory_gb": None, "driver_version": None}

    rows = [line.split(", ") for line in out.splitlines() if line.strip()]
    names = {r[0].strip() for r in rows}
    mem_mib = {int(float(r[1])) for r in rows}
    drivers = {r[2].strip() for r in rows}
    if len(names) > 1:
        log.warning("heterogeneous_gpus_on_node", models=sorted(names))
    return {
        "gpu_model": sorted(names)[0],
        "gpu_count": len(rows),
        # A100-40GB reports 40960 MiB; report GiB rounded so the figure captions
        # can say "40GB" without re-deriving it.
        "gpu_memory_gb": round(max(mem_mib) / 1024, 1),
        "driver_version": sorted(drivers)[0],
    }


def query_topology() -> str | None:
    """Capture ``nvidia-smi topo -m`` verbatim.

    Kept raw and whole: the TP scaling writeup has to correlate efficiency against
    the actual interconnect, and a parsed summary would lose the NIC affinity rows.
    """
    return _run(["nvidia-smi", "topo", "-m"])


def query_cuda_version() -> str | None:
    """CUDA version as reported by the driver (the ``CUDA Version:`` header)."""
    out = _run(["nvidia-smi"])
    if not out:
        return None
    for line in out.splitlines():
        if "CUDA Version:" in line:
            return line.split("CUDA Version:")[1].split("|")[0].strip()
    return None


def slurm_context() -> dict[str, Any]:
    """Everything Slurm tells us about this allocation, for cost attribution."""
    keys = {
        "slurm_job_id": "SLURM_JOB_ID",
        "slurm_array_job_id": "SLURM_ARRAY_JOB_ID",
        "slurm_array_task_id": "SLURM_ARRAY_TASK_ID",
        "slurm_nodelist": "SLURM_JOB_NODELIST",
        "slurm_partition": "SLURM_JOB_PARTITION",
        "slurm_nnodes": "SLURM_JOB_NUM_NODES",
        "slurm_cpus_per_task": "SLURM_CPUS_PER_TASK",
        "slurm_account": "SLURM_JOB_ACCOUNT",
    }
    return {k: os.environ.get(v) for k, v in keys.items()}


@dataclass
class Provenance:
    """The header block required by the workspace CLAUDE.md, plus MeluXina extras."""

    run_id: str
    utc_timestamp: str
    git_sha: str | None
    hostname: str
    slurm_job_id: str | None

    gpu_model: str | None
    gpu_count: int
    gpu_memory_gb: float | None
    driver_version: str | None
    cuda_version: str | None
    gpu_topology: str | None

    backend_name: str
    backend_version: str | None
    backend_image_digest: str | None

    model_id: str
    model_revision: str | None

    full_config_dict: dict[str, Any]
    exact_command_line: str
    repetition_index: int
    warmup: bool
    status: RunStatus

    slurm: dict[str, Any] = field(default_factory=dict)
    python_version: str = field(default_factory=platform.python_version)
    harness_version: str = "0.1.0"
    error: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


def build_provenance(
    *,
    repo_root: Path,
    backend_name: str,
    backend_version: str | None,
    backend_image_digest: str | None,
    model_id: str,
    model_revision: str | None,
    config: dict[str, Any],
    repetition_index: int,
    warmup: bool,
    status: RunStatus = "ok",
    error: str | None = None,
    notes: dict[str, Any] | None = None,
) -> Provenance:
    gpus = query_gpus()
    return Provenance(
        run_id=uuid.uuid4().hex[:16],
        utc_timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=git_sha(repo_root),
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        gpu_model=gpus["gpu_model"],
        gpu_count=gpus["gpu_count"],
        gpu_memory_gb=gpus["gpu_memory_gb"],
        driver_version=gpus["driver_version"],
        cuda_version=query_cuda_version(),
        gpu_topology=query_topology(),
        backend_name=backend_name,
        backend_version=backend_version,
        backend_image_digest=backend_image_digest,
        model_id=model_id,
        model_revision=model_revision,
        full_config_dict=config,
        exact_command_line=shlex.join(sys.argv),
        repetition_index=repetition_index,
        warmup=warmup,
        status=status,
        error=error,
        slurm=slurm_context(),
        notes=notes or {},
    )


class IncompleteProvenanceError(RuntimeError):
    """Raised when an artifact would be written without traceable provenance."""


def write_artifact(
    results_raw: Path,
    provenance: Provenance,
    measurements: dict[str, Any],
) -> Path:
    """Write one raw artifact. Append-only: refuses to overwrite an existing file.

    A failed, OOMed or timed-out run is written too, with its status and error.
    Failed configurations are data: an OOM at a given batch/context is a real
    boundary and belongs in the writeup, so nothing is silently dropped.
    """
    prov = asdict(provenance)
    missing = [f for f in REQUIRED_FIELDS if prov.get(f) in (None, "", [])]
    if missing:
        raise IncompleteProvenanceError(
            f"refusing to write artifact with missing provenance fields: {missing}"
        )

    results_raw.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{provenance.utc_timestamp.replace(':', '').replace('-', '')}"
        f"_{provenance.backend_name}"
        f"_{provenance.model_id.split('/')[-1]}"
        f"_rep{provenance.repetition_index}"
        f"_{provenance.run_id}"
    )
    path = results_raw / f"{stem}.json"
    if path.exists():
        raise FileExistsError(f"raw artifacts are append-only, refusing to overwrite {path}")

    payload = {"provenance": prov, "measurements": measurements}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    # Raw artifacts are never edited by hand; make that the filesystem's opinion too.
    path.chmod(0o440)
    log.info("artifact_written", path=str(path), status=provenance.status)
    return path
