# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Compile + profile the plain-float (unquantized) Pi05 collection on-device via
AI Hub, to get an FP16 latency baseline comparable to the `mixed`/`mixed_spin`
on-device numbers.

Why this script exists instead of `qai_hub_models.models.pi05.export`:
`pi05/code-gen.yaml` only lists `mixed` under `supported_precisions`, and
`pi05/__init__.py` registers `Model = Pi05CollectionQuantized` (the AIMET
quantizable collection). So the generated export CLI's `--precision` only
accepts `mixed` -- there is no `--precision float` path wired up for pi05.

This script drives the same low-level per-component export helpers
(`upload_collection_source`, `run_collection_compile`, `run_collection_link`,
`run_collection_profile`) that the generated pipeline uses internally, but
against `Pi05Collection` -- the plain torch float collection already used
by pi05/quantize.py to build calibration data -- instead of going through
`resolve_model_cls("pi05")`, which would always resolve to the quantized
collection regardless of `--precision`.

Usage:
    conda run -n qc python profile_pi05_float.py \
        --device "Dragonwing IQ-9075 EVK"

    # Profile only a subset of components:
    conda run -n qc python profile_pi05_float.py --components backbone vision_encoder
"""

from __future__ import annotations

import argparse

import qai_hub as hub
import torch

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.pi05.model import Pi05Collection
from qai_hub_models.utils.export.compile import run_collection_compile
from qai_hub_models.utils.export.link import run_collection_link
from qai_hub_models.utils.export.profile import run_collection_profile
from qai_hub_models.utils.export.summary import print_profile_summary
from qai_hub_models.utils.export.upload import upload_collection_source
from qai_hub_models.utils.qai_hub_helpers import assert_success_and_get_target_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="Dragonwing IQ-9075 EVK",
        help="AI Hub device name. Run `qai-hub list-devices` to see options.",
    )
    parser.add_argument(
        "--checkpoint",
        default="DEFAULT",
        help="Huggingface repo id or local directory with float weights.",
    )
    parser.add_argument(
        "--host-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help=(
            "Device to load/trace the float torch model on before uploading "
            "to AI Hub (does not affect the on-device profiling target). "
            "Defaults to cuda if available -- tracing pi05's backbone on CPU "
            "can be extremely slow."
        ),
    )
    parser.add_argument(
        "--target-runtime",
        default=TargetRuntime.QNN_CONTEXT_BINARY.value,
        choices=[rt.value for rt in TargetRuntime],
        help="Same runtime the mixed/mixed_spin builds compiled to, for a fair comparison.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=None,
        help="Subset of {vision_encoder, token_emb, action_expert, backbone}. Default: all.",
    )
    parser.add_argument("--model-name", default="pi05_float")
    args = parser.parse_args()

    device = hub.Device(args.device)
    target_runtime = TargetRuntime(args.target_runtime)

    print(
        f"Loading float Pi05Collection from checkpoint={args.checkpoint!r} "
        f"on host_device={args.host_device!r} ..."
    )
    model = Pi05Collection.from_pretrained(
        checkpoint=args.checkpoint, host_device=torch.device(args.host_device)
    )

    components = args.components or model.component_names
    input_specs = model.get_input_spec()

    print(f"Uploading source models for components: {components}")
    source_models = upload_collection_source(model, input_specs, components)

    print(
        f"Compiling for device={args.device!r}, "
        f"target_runtime={target_runtime.value}, precision=float ..."
    )
    compile_jobs = run_collection_compile(
        model,
        args.model_name,
        device,
        target_runtime,
        Precision.float,
        source_models,
        input_specs,
        components,
    )
    compiled = assert_success_and_get_target_models(compile_jobs)

    if target_runtime.uses_hub_link:
        print("Linking to context binary ...")
        link_jobs = run_collection_link(
            compiled, device, args.model_name, model, target_runtime
        )
        target_models = assert_success_and_get_target_models(link_jobs)
    else:
        target_models = compiled

    profile_opts = model.get_hub_profile_options(target_runtime, "")

    print("Submitting profile jobs ...")
    profile_jobs = run_collection_profile(
        args.model_name, device, profile_opts, target_models, components
    )

    print()
    print("=" * 70)
    print(f"FP16 latency -- {args.model_name} on {args.device}")
    print("=" * 70)
    for name in components:
        print(f"\n--- {name} ---")
        print_profile_summary(profile_jobs[name])


if __name__ == "__main__":
    main()
