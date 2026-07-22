# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Profile the plain-float (unquantized) Pi05 PaliGemma backbone on-device in
layer-range chunks, to get an FP16 latency baseline on devices where the
monolithic 18-layer float backbone does not fit.

Why this script exists in addition to `profile_pi05_float.py`:
the full float backbone (`Pi05PaliGemmaBackbone`, layer_range=(0, 18), ~2B
params, ~4GB in FP16) cannot be allocated on Dragonwing IQ-9075. Both paths
fail on memory:

    link    -> "Conversion to context binary failed with exit code 15"
    profile -> "QNN_COMMON_ERROR_MEM_ALLOC: Memory allocation related error."

The `mixed` (w8a16) build links and profiles fine because its weights are
roughly 4x smaller. To still get an FP16 number, this script exploits
`Pi05PaliGemmaBackboneBase`'s existing `layer_range` support: it builds one
model per contiguous layer chunk, compiles each to QNN_DLC, profiles each,
and sums the per-chunk latencies.

Two deliberate differences from the monolithic path:

1. Target runtime defaults to `qnn_dlc`, not `qnn_context_binary`. A single
   DLC per chunk needs no link step (see `run_collection_link` -- each
   component links alone, so there is no weight sharing to lose), which
   sidesteps the context-binary conversion failure entirely. Inference time
   is measured on the same HTP graph either way; only load/init differs.
2. The reported total is a SUM of chunk latencies. It is an estimate: it
   additionally pays inter-chunk `hidden_state` write/read that the fused
   18-layer graph would keep resident. Treat it as an upper bound.

Chunks are cut so that only the final chunk ends at layer 18, matching
`get_output_spec_static`, which emits `hidden_state_out` only when the range
does not end at 18.

Usage:
    conda run -n qc python profile_pi05_float_chunked.py \
        --device "Dragonwing IQ-9075 EVK" --chunk-size 3

    # Probe a single chunk first to find a size that fits in device memory:
    conda run -n qc python profile_pi05_float_chunked.py --ranges 0:6
"""

from __future__ import annotations

import argparse

import qai_hub as hub
import torch

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.pi05.model import Pi05PaliGemmaBackboneBase
from qai_hub_models.utils.export.compile import run_compile
from qai_hub_models.utils.export.profile import run_profile
from qai_hub_models.utils.export.upload import upload_source
from qai_hub_models.utils.qai_hub_helpers import assert_success_and_get_target_models

NUM_LAYERS = 18


def make_chunk_cls(start: int, end: int) -> type[Pi05PaliGemmaBackboneBase]:
    """
    Build a concrete backbone subclass pinned to [start, end).

    A distinct class per chunk gives each uploaded source model a distinct
    name on AI Hub (`serialize` names the ONNX after the class), which makes
    the job list readable. `return_hidden_state` is off only for a chunk that
    ends at the last layer, to stay consistent with `get_output_spec_static`.
    """
    returns_hidden = end != NUM_LAYERS

    def __init__(self, policy) -> None:  # noqa: N807
        Pi05PaliGemmaBackboneBase.__init__(
            self,
            policy,
            layer_range=(start, end),
            return_hidden_state=returns_hidden,
        )

    return type(
        f"Pi05PaliGemmaBackboneL{start}to{end}",
        (Pi05PaliGemmaBackboneBase,),
        {"__init__": __init__},
    )


def parse_ranges(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.ranges:
        out = []
        for r in args.ranges:
            start, _, end = r.partition(":")
            out.append((int(start), int(end)))
        return out
    size = args.chunk_size
    return [(s, min(s + size, NUM_LAYERS)) for s in range(0, NUM_LAYERS, size)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="Dragonwing IQ-9075 EVK")
    parser.add_argument("--checkpoint", default="DEFAULT")
    parser.add_argument(
        "--host-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load the float torch policy on before ONNX export.",
    )
    parser.add_argument(
        "--target-runtime",
        default=TargetRuntime.QNN_DLC.value,
        choices=[rt.value for rt in TargetRuntime],
        help="Defaults to qnn_dlc, which needs no link step.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=3, help="Layers per chunk (18 total)."
    )
    parser.add_argument(
        "--ranges",
        nargs="+",
        default=None,
        help='Explicit ranges, e.g. --ranges 0:6 6:12 12:18. Overrides --chunk-size.',
    )
    parser.add_argument("--model-name", default="pi05_float_backbone_chunked")
    args = parser.parse_args()

    device = hub.Device(args.device)
    target_runtime = TargetRuntime(args.target_runtime)
    ranges = parse_ranges(args)

    print(f"Loading float PI05Policy from checkpoint={args.checkpoint!r} ...")
    # load_checkpoint is lru_cached, so every chunk shares one set of float
    # weights rather than materializing the policy once per chunk.
    policy = Pi05PaliGemmaBackboneBase.torch_from_pretrained(
        checkpoint=args.checkpoint, host_device=torch.device(args.host_device)
    )

    print(f"Chunks: {ranges}")
    profile_jobs: dict[str, hub.client.ProfileJob] = {}

    for start, end in ranges:
        tag = f"l{start}to{end}"
        name = f"{args.model_name}_{tag}"
        print(f"\n=== chunk {tag} ===", flush=True)

        chunk = make_chunk_cls(start, end)(policy)
        input_spec = chunk.get_input_spec()

        print(f"[{tag}] exporting + uploading source ...", flush=True)
        source = upload_source(chunk, input_spec)

        print(f"[{tag}] compiling to {target_runtime.value} ...", flush=True)
        compile_job = run_compile(
            chunk, name, device, target_runtime, Precision.float, source, input_spec
        )
        target_model = assert_success_and_get_target_models(compile_job)

        print(f"[{tag}] submitting profile job ...", flush=True)
        profile_jobs[tag] = run_profile(
            name, device, chunk.get_hub_profile_options(target_runtime, ""), target_model
        )

    print("\n" + "=" * 70)
    print(f"FP16 chunked backbone -- {args.device}")
    print("=" * 70)

    total_us = 0.0
    failed: list[str] = []
    for tag, job in profile_jobs.items():
        job.wait()
        status = job.get_status()
        if not status.success:
            failed.append(tag)
            print(f"{tag:<12} FAILED: {status.message}  ({job.job_id})")
            continue
        us = job.download_profile()["execution_summary"]["estimated_inference_time"]
        total_us += us
        print(f"{tag:<12} {us / 1000:9.2f} ms   ({job.job_id})")

    if failed:
        print(
            f"\n{len(failed)} chunk(s) failed: {failed}. "
            "Retry those with a smaller --chunk-size."
        )
    else:
        print(f"\n{'SUM':<12} {total_us / 1000:9.2f} ms  (estimate; excludes fusion, "
              "includes inter-chunk hidden_state I/O)")


if __name__ == "__main__":
    main()
