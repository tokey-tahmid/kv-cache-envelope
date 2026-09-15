"""Load HuggingFace ``config.json`` files, with their revision, as ground truth.

Two sources, one shape of answer:

* a local snapshot directory under the provisioned HF cache, whose revision is
  the snapshot directory name;
* a repo on the Hub, fetched on a compute node (the login node has no network
  path to it and cannot even parse this file).

In both cases the revision is **read**, never typed. An earlier agent on the
sibling project fabricated a model revision by autocompleting a truncated hash,
so the rule here is that no identifier is ever constructed by hand:
:func:`load_local` takes it from the snapshot path and :func:`fetch_from_hub`
takes it from whatever ``huggingface_hub`` resolved.

Fetched configs are written to ``configs/model_configs/`` together with a
manifest so the calculator's validation runs offline afterwards, and so the
committed test inputs are upstream artifacts with a stated provenance rather than
values someone remembered.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class LoadedConfig:
    model_id: str
    revision: str | None
    config: dict[str, Any]
    source: str
    path: str | None = None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_local(snapshot_dir: Path, model_id: str) -> LoadedConfig:
    """Load a config from a provisioned HF snapshot directory.

    The snapshot directory name *is* the revision under the standard HF cache
    layout (``models--<org>--<name>/snapshots/<sha>/``). It is only reported as a
    revision when it actually looks like one, so a hand-made directory cannot
    masquerade as a pinned commit.
    """
    snapshot_dir = Path(snapshot_dir)
    cfg_path = snapshot_dir / "config.json"
    revision = snapshot_dir.name if _SHA_RE.match(snapshot_dir.name) else None
    if revision is None:
        log.warning(
            "snapshot_dir_is_not_a_revision", dir=str(snapshot_dir), model_id=model_id
        )
    return LoadedConfig(
        model_id=model_id,
        revision=revision,
        config=load_json(cfg_path),
        source="local_snapshot",
        path=str(cfg_path),
    )


def fetch_from_hub(model_id: str, cache_dir: Path | None = None) -> LoadedConfig:
    """Fetch ``config.json`` from the Hub and report the revision it resolved to.

    Compute-node only: the login node has no `huggingface_hub`. The revision comes
    back from the download path, which is the cache's record of what was actually
    fetched.
    """
    from huggingface_hub import hf_hub_download  # imported lazily: compute-node only

    local = hf_hub_download(
        repo_id=model_id,
        filename="config.json",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    local_path = Path(local)
    # .../models--org--name/snapshots/<sha>/config.json
    revision = None
    for part in local_path.parts:
        if _SHA_RE.match(part):
            revision = part
            break
    if revision is None:
        log.warning("hub_revision_not_recoverable", model_id=model_id, path=str(local_path))
    return LoadedConfig(
        model_id=model_id,
        revision=revision,
        config=load_json(local_path),
        source="huggingface_hub",
        path=str(local_path),
    )


def write_bundle(dest_dir: Path, loaded: LoadedConfig) -> Path:
    """Persist one config plus its provenance under ``configs/model_configs/``."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    slug = loaded.model_id.replace("/", "--")
    out = dest_dir / f"{slug}.json"
    out.write_text(
        json.dumps(
            {
                "model_id": loaded.model_id,
                "revision": loaded.revision,
                "source": loaded.source,
                "source_path": loaded.path,
                "config": loaded.config,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return out


def read_bundle(path: Path) -> LoadedConfig:
    """Read a bundle written by :func:`write_bundle` (offline, no network)."""
    data = load_json(Path(path))
    return LoadedConfig(
        model_id=data["model_id"],
        revision=data.get("revision"),
        config=data["config"],
        source=data.get("source", "bundle"),
        path=data.get("source_path"),
    )


def read_all_bundles(dest_dir: Path) -> list[LoadedConfig]:
    return [read_bundle(p) for p in sorted(Path(dest_dir).glob("*.json")) if p.name != "MANIFEST.json"]
