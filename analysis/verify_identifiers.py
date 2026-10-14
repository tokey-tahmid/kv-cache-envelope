#!/usr/bin/env python3
"""Verify every identifier written in prose against ground truth.

    python analysis/verify_identifiers.py     # exit 1 if anything is unverifiable

Guards the workspace's first non-negotiable rule at the one place it is easiest
to break by accident: prose. Figures and tables are generated from
``results/raw/`` and cannot drift, but a README is typed, and a long hex string
is exactly the kind of token that gets confidently completed rather than copied.

That is not hypothetical. An unattended run on the sibling project wrote a Qwen2.5-32B revision whose
first twelve characters matched the true one and whose tail was invented, because
figure captions truncate revisions to twelve characters and the rest was
reconstructed from nothing. Every measurement around it was real; only the
identifier was false, which is the most dangerous shape such an error can take.
The true value is in this repository's `provision.lock.json` and in every
artifact, and this script is what keeps prose tied to it.

Ground truth is the union of the provisioning lockfile and the provenance blocks
of every raw artifact. Anything hex-like and >=12 characters in a prose file must
prefix-match one of those.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROSE = ["README.md", "docs/notes.md", "docs/budget.md", "docs/PROGRESS.md"]


def ground_truth() -> set[str]:
    known: set[str] = set()
    lock = REPO / "provision.lock.json"
    if lock.exists():
        data = json.loads(lock.read_text())
        for v in data.get("models", {}).values():
            if v.get("revision"):
                known.add(v["revision"])
        for v in data.get("images", {}).values():
            for key in ("digest", "version"):
                if v.get(key):
                    known.add(str(v[key]).replace("sha256:", ""))
    # Also accept identifiers from the committed model-config bundles: a config
    # revision is ground truth for the calculator validation even though no
    # measurement was taken against that model.
    bundles = REPO / "configs" / "model_configs"
    if bundles.is_dir():
        for f in bundles.glob("*.json"):
            data = json.loads(f.read_text())
            for rec in data.get("records", []) if "records" in data else [data]:
                if rec.get("revision"):
                    known.add(str(rec["revision"]))

    for f in (REPO / "results" / "raw").glob("*.json"):
        p = json.loads(f.read_text()).get("provenance", {})
        for key in ("model_revision", "backend_version", "driver_version", "cuda_version"):
            if p.get(key):
                known.add(str(p[key]))
        for key in ("backend_image_digest", "git_sha"):
            if p.get(key):
                known.add(str(p[key]).replace("sha256:", "").replace("-dirty", ""))
    return known


def main() -> int:
    known = ground_truth()
    if not known:
        print("no ground truth available (no lockfile, no artifacts); nothing to verify")
        return 0

    problems: list[str] = []
    checked = 0
    for rel in PROSE:
        path = REPO / rel
        if not path.exists():
            continue
        text = path.read_text()
        for h in sorted(set(re.findall(r"\b[0-9a-f]{12,64}\b", text))):
            checked += 1
            if not any(k.startswith(h) or h.startswith(k) for k in known):
                problems.append(f"{rel}: {h}")

    print(f"checked {checked} identifier(s) against {len(known)} known values")
    if problems:
        print("\nUNVERIFIABLE IDENTIFIERS (not in the lockfile or any artifact):")
        for p in problems:
            print(f"  {p}")
        print("\nEvery identifier in prose must be copied from ground truth, never typed.")
        return 1
    print("all identifiers trace to the lockfile or a raw artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
