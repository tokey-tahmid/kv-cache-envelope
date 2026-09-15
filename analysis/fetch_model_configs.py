#!/usr/bin/env python3
"""Fetch HF ``config.json`` bundles used to validate the calculator.

    python analysis/fetch_model_configs.py            # compute node (has network)

Two of the three required models are already provisioned locally and are read
from their pinned snapshots rather than the Hub, so their revision is the one
recorded in ``provision.lock.json``. The remainder are fetched from the Hub on a
compute node (the login node has no network path and Python 3.6).

Every candidate is attempted and **the outcome of each is recorded**, success or
failure, into ``configs/model_configs/MANIFEST.json``. A model that could not be
fetched (gated, renamed, offline) appears there as a failure with its error, not
as an absence: an unexplained gap in a validation set looks like a choice.

Candidates are chosen to span the attention-shape axis the calculator has to get
right, not to be a tidy list:

* GQA at several ratios (the common case since 2023),
* at least one MHA model (where the query-head mistake is invisible),
* at least one MQA model (where it is largest),
* at least one MLA model, which the calculator must *refuse* rather than answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kv_cache_envelope.calculator import (  # noqa: E402
    UnsupportedArchitecture,
    kv_bytes_per_token,
    shape_from_config,
)
from kv_cache_envelope.hf_config import (  # noqa: E402
    LoadedConfig,
    fetch_from_hub,
    load_local,
    write_bundle,
)

DEST = REPO / "configs" / "model_configs"

# Locally provisioned, read from the pinned snapshot directory. The snapshot name
# is the revision; it is never typed here.
LOCAL = {
    "Qwen/Qwen2.5-7B-Instruct": "models--Qwen--Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct": "models--Qwen--Qwen2.5-32B-Instruct",
}
HF_HUB_CACHE = Path("/mnt/tier2/project/p201362/ttahmid/inference-work/hf_cache/hub")

# Hub candidates, ungated as far as is known; whichever succeed are recorded and
# whichever fail are recorded as failures.
HUB_CANDIDATES = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "microsoft/Phi-3-mini-4k-instruct",
    "tiiuae/falcon-7b",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "bigcode/starcoder2-7b",
    "allenai/OLMo-2-1124-7B-Instruct",
    "ibm-granite/granite-3.1-8b-instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "deepseek-ai/DeepSeek-V2-Lite",
]


def summarise(loaded: LoadedConfig) -> dict[str, object]:
    """Structural summary, or the reason the calculator refused to answer."""
    try:
        s = shape_from_config(
            loaded.config,
            model_id=loaded.model_id,
            revision=loaded.revision,
            config_path=loaded.path,
        )
    except UnsupportedArchitecture as exc:
        return {"supported": False, "reason": str(exc)}
    out = s.as_dict()
    out["supported"] = True
    try:
        out["kv_bytes_per_token_auto"] = kv_bytes_per_token(s, kv_cache_dtype="auto")
    except ValueError as exc:
        out["kv_bytes_per_token_auto"] = None
        out["dtype_error"] = str(exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hub", action="store_true", help="local snapshots only")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for model_id, cache_dir in LOCAL.items():
        snaps = sorted((HF_HUB_CACHE / cache_dir / "snapshots").glob("*"))
        if not snaps:
            records.append(
                {"model_id": model_id, "source": "local_snapshot", "status": "missing"}
            )
            print(f"  MISSING  {model_id}: no snapshot under {cache_dir}")
            continue
        loaded = load_local(snaps[0], model_id)
        path = write_bundle(DEST, loaded)
        records.append(
            {
                "model_id": model_id,
                "source": "local_snapshot",
                "status": "ok",
                "revision": loaded.revision,
                "bundle": path.name,
                "shape": summarise(loaded),
            }
        )
        print(f"  ok       {model_id}  rev={loaded.revision}")

    if not args.no_hub:
        for model_id in HUB_CANDIDATES:
            try:
                loaded = fetch_from_hub(model_id)
            except Exception as exc:  # every failure is data, not a reason to stop
                records.append(
                    {
                        "model_id": model_id,
                        "source": "huggingface_hub",
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"  FAILED   {model_id}: {type(exc).__name__}: {exc}")
                continue
            path = write_bundle(DEST, loaded)
            records.append(
                {
                    "model_id": model_id,
                    "source": "huggingface_hub",
                    "status": "ok",
                    "revision": loaded.revision,
                    "bundle": path.name,
                    "shape": summarise(loaded),
                }
            )
            print(f"  ok       {model_id}  rev={loaded.revision}")

    manifest = {
        "fetched_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "records": records,
    }
    (DEST / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    ok = [r for r in records if r.get("status") == "ok"]
    ratios = {
        r["model_id"]: r["shape"]["gqa_ratio"]  # type: ignore[index]
        for r in ok
        if isinstance(r.get("shape"), dict) and r["shape"].get("supported")  # type: ignore[union-attr]
    }
    distinct = sorted(set(ratios.values()))
    print(f"\n{len(ok)}/{len(records)} configs obtained; distinct GQA ratios: {distinct}")
    for m, r in sorted(ratios.items(), key=lambda kv: kv[1]):
        print(f"  {r:6.1f}x  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
