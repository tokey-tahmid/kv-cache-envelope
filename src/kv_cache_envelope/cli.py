"""Command line entry point.

    python -m kv_cache_envelope.cli run configs/envelope_7b.yaml [--slice i/n]
    python -m kv_cache_envelope.cli plan configs/envelope_7b.yaml
    python -m kv_cache_envelope.cli predict --model Qwen/Qwen2.5-7B-Instruct --seq-len 8192
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import structlog

from .calculator import kv_bytes, kv_bytes_per_token, max_concurrent_sequences
from .hf_config import read_all_bundles
from .sweep import SweepConfig, run_sweep

REPO = Path(__file__).resolve().parents[2]


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = SweepConfig.from_yaml(Path(args.config))
    lockfile = json.loads((REPO / "provision.lock.json").read_text())
    slice_index = slice_count = None
    if args.slice:
        i, _, n = args.slice.partition("/")
        slice_index, slice_count = int(i), int(n)
    written = run_sweep(
        cfg,
        repo_root=REPO,
        results_raw=REPO / "results" / "raw",
        sif=_env("KVE_SIF_VLLM", ""),
        data_dir=_env("KVE_DATA", "/mnt/tier2/project/p201362/ttahmid/inference-work"),
        log_dir=Path(_env("KVE_LOGS", "/tmp")),
        lockfile=lockfile,
        slice_index=slice_index,
        slice_count=slice_count,
        dry_run=args.dry_run,
    )
    print(f"artifacts written: {written}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    cfg = SweepConfig.from_yaml(Path(args.config))
    combos = cfg.expand()
    total = len(combos) * (cfg.repetitions + cfg.warmup_repetitions)
    print(f"sweep {cfg.name}: {len(combos)} configurations x "
          f"{cfg.repetitions + cfg.warmup_repetitions} launches = {total} launches")
    for i, c in enumerate(combos):
        print(f"  [{i:3d}] {c}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    bundles = {b.model_id: b for b in read_all_bundles(REPO / "configs" / "model_configs")}
    if args.model not in bundles:
        print(f"unknown model {args.model!r}; known: {sorted(bundles)}", file=sys.stderr)
        return 2
    from .calculator import shape_from_config

    b = bundles[args.model]
    shape = shape_from_config(b.config, model_id=b.model_id, revision=b.revision)
    per_token = kv_bytes_per_token(shape, kv_cache_dtype=args.kv_cache_dtype)
    total = kv_bytes(
        shape,
        seq_len=args.seq_len,
        batch=args.batch,
        kv_cache_dtype=args.kv_cache_dtype,
        block_size=args.block_size,
    )
    print(json.dumps(
        {
            "model_id": shape.model_id,
            "revision": shape.revision,
            "attention_kind": shape.attention_kind,
            "gqa_ratio": shape.gqa_ratio,
            "layers": shape.num_hidden_layers,
            "kv_heads": shape.num_key_value_heads,
            "attention_heads": shape.num_attention_heads,
            "head_dim": shape.head_dim,
            "kv_bytes_per_token": per_token,
            "kv_kib_per_token": per_token / 1024,
            "seq_len": args.seq_len,
            "batch": args.batch,
            "predicted_bytes": total,
            "predicted_gib": total / 1024**3,
            "sequences_in_budget": (
                max_concurrent_sequences(
                    shape,
                    int(args.budget_gib * 1024**3),
                    seq_len=args.seq_len,
                    kv_cache_dtype=args.kv_cache_dtype,
                    block_size=args.block_size,
                )
                if args.budget_gib
                else None
            ),
        },
        indent=2,
    ))
    return 0


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    ap = argparse.ArgumentParser(prog="kv-cache-envelope")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="execute a sweep on a GPU node")
    r.add_argument("config")
    r.add_argument("--slice", help="i/n: run every n-th configuration, offset i")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("plan", help="print the expanded configuration list")
    p.add_argument("config")
    p.set_defaults(func=cmd_plan)

    q = sub.add_parser("predict", help="analytical prediction for one model")
    q.add_argument("--model", required=True)
    q.add_argument("--seq-len", type=int, default=8192)
    q.add_argument("--batch", type=int, default=1)
    q.add_argument("--kv-cache-dtype", default="auto")
    q.add_argument("--block-size", type=int, default=None)
    q.add_argument("--budget-gib", type=float, default=None)
    q.set_defaults(func=cmd_predict)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
