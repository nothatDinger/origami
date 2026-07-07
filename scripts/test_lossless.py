#!/usr/bin/env python3
"""Run a small Origami lossless-codec round trip."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VLLM_ROOT = REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qat", "zlib", "raw"), default="qat")
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--chunk-bytes", type=int, default=1 << 20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--allow-zlib-fallback",
        action="store_true",
        help="allow backend=qat to fall back to zlib when QAT is unavailable",
    )
    args = parser.parse_args()

    if args.chunks <= 0 or args.chunk_bytes <= 0:
        raise SystemExit("--chunks and --chunk-bytes must be positive")

    os.environ.setdefault("ORIGAMI_LIGHT_IMPORT", "1")
    sys.path.insert(0, str(VLLM_ROOT))
    try:
        import torch
        from vllm.distributed.kv_transfer.kv_connector.v1.origami.lossless.cpu_qat import (
            CpuLosslessCodec,
        )
    except Exception as exc:
        raise SystemExit(
            "Could not import Origami lossless modules. Install dependencies with "
            "`bash scripts/setup_env.sh` or activate an environment with torch. "
            f"Original error: {exc}"
        ) from exc

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    chunks = [
        torch.randint(
            0,
            256,
            (int(args.chunk_bytes),),
            dtype=torch.uint8,
            generator=generator,
        )
        for _ in range(int(args.chunks))
    ]

    try:
        codec = CpuLosslessCodec(
            backend=args.backend,
            allow_zlib_fallback=bool(args.allow_zlib_fallback),
        )
    except Exception as exc:
        if args.backend == "qat":
            raise SystemExit(
                "QAT backend is not available. Run `python3 scripts/check_qat.py "
                "--try-extension` for diagnostics, or use `--backend zlib` for "
                f"the software lossless path. Original error: {exc}"
            ) from exc
        raise

    start = time.perf_counter()
    compressed = codec.compress_many(chunks)
    restored = codec.decompress_many(compressed, [args.chunk_bytes] * args.chunks)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    for index, (expected, actual) in enumerate(zip(chunks, restored)):
        if not torch.equal(expected.cpu(), actual.cpu()):
            raise SystemExit(f"round-trip mismatch in chunk {index}")

    input_bytes = sum(int(chunk.numel()) for chunk in chunks)
    compressed_bytes = sum(int(chunk.numel()) for chunk in compressed)
    ratio = compressed_bytes / input_bytes if input_bytes else 0.0
    print(
        "Origami lossless test passed: "
        f"backend={codec.backend} chunks={args.chunks} "
        f"input_bytes={input_bytes} compressed_bytes={compressed_bytes} "
        f"ratio={ratio:.4f} elapsed_ms={elapsed_ms:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
