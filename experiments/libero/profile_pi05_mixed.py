# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Compile + profile a quantized (mixed-precision) Pi05 collection on-device via
AI Hub, against a local checkpoint dir built by pi05/quantize.py (e.g.
build/pi05_mixed, build/pi05_mixed_spin, build/pi05_mixed_spin_actionw4).

Why this script exists instead of `qai_hub_models.models.pi05.export`:
`pi05/__init__.py` registers `Model = Pi05CollectionQuantized`, but the
generated export CLI has no `--checkpoint` override -- it always resolves the
default (HF-hosted) checkpoint. This script drives the same low-level
per-component export helpers (`upload_collection_source`,
`run_collection_compile`, `run_collection_link`, `run_collection_profile`)
against a `Pi05CollectionQuantized` loaded from an arbitrary local checkpoint
dir, so custom mixed-precision builds can be profiled on-device.

This measures on-device latency only (one profile job per component); it is
NOT the closed-loop LIBERO eval, which runs via QuantSim on a local GPU (see
run_libero_pi05.sh / run_libero_pi05_settings.sh).

Usage:
    conda run -n qc python profile_pi05_mixed.py \
        --checkpoint /home/jk656/ai-hub-models/build/pi05_mixed_spin_actionw4 \
        --device "Dragonwing IQ-9075 EVK"

    # Profile only a subset of components:
    conda run -n qc python profile_pi05_mixed.py \
        --checkpoint build/pi05_mixed_spin_actionw4 \
        --components backbone action_expert

SpinQuant rotations are folded into the checkpoint's weights, so there is no
flag to set here -- the profile reflects whatever the checkpoint was built with.
"""

from __future__ import annotations

import argparse

import qai_hub as hub
import torch

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.pi05.model import Pi05CollectionQuantized
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
        required=True,
        help=(
            "Local directory (or HF repo id) with a quantized checkpoint "
            "built by pi05/quantize.py, e.g. build/pi05_mixed_spin_actionw4."
        ),
    )
    parser.add_argument(
        "--use-spinquant-r1",
        action="store_true",
        help=(
            "Deprecated no-op. Rotations are folded into the checkpoint's "
            "weights and detected from its markers."
        ),
    )
    parser.add_argument(
        "--host-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help=(
            "Device to load the quantized torch/QuantSim model on before "
            "uploading to AI Hub (does not affect the on-device profiling "
            "target)."
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
    parser.add_argument("--model-name", default="pi05_mixed")
    args = parser.parse_args()

    device = hub.Device(args.device)
    target_runtime = TargetRuntime(args.target_runtime)

    if args.use_spinquant_r1:
        print(
            "NOTE: --use-spinquant-r1 is a no-op; rotations come from the "
            "checkpoint's markers."
        )
    print(
        f"Loading quantized Pi05CollectionQuantized from checkpoint={args.checkpoint!r} "
        f"on host_device={args.host_device!r} ..."
    )
    model = Pi05CollectionQuantized.from_pretrained(
        checkpoint=args.checkpoint,
        host_device=torch.device(args.host_device),
    )

    components = args.components or model.component_names
    input_specs = model.get_input_spec()

    print(f"Uploading source models for components: {components}")
    source_models = upload_collection_source(model, input_specs, components)

    print(
        f"Compiling for device={args.device!r}, "
        f"target_runtime={target_runtime.value}, precision=mixed ..."
    )
    compile_jobs = run_collection_compile(
        model,
        args.model_name,
        device,
        target_runtime,
        Precision.mixed,
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
    print(f"On-device latency -- {args.model_name} on {args.device}")
    print("=" * 70)
    for name in components:
        print(f"\n--- {name} ---")
        print_profile_summary(profile_jobs[name])


if __name__ == "__main__":
    main()
