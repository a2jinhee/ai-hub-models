# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""
CLI entrypoint for quantizing Pi05 components with AIMET-ONNX.

Uses mixed precision: vision_encoder=w8a16, backbone=w4a16, action_expert=w8a16.

Usage:
    python -m qai_hub_models.models.pi05.quantize --component vision_encoder
    python -m qai_hub_models.models.pi05.quantize --component backbone
    python -m qai_hub_models.models.pi05.quantize --component action_expert

    # Optionally rotate the backbone's weights with SpinQuant R1:
    python -m qai_hub_models.models.pi05.quantize --component backbone --use-spinquant-r1

    # Override the default per-component precision (e.g. action_expert at w4a16):
    python -m qai_hub_models.models.pi05.quantize --component action_expert --precision w4a16

If backbone is quantized with --use-spinquant-r1, the deployment/eval path
(Pi05App, e.g. pi05_libero_server.py) must also be run with R1 enabled, so
hidden_state is rotated to match the backbone's rotated weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from qai_hub_models import Precision
from qai_hub_models.models.pi05.app import Pi05App
from qai_hub_models.models.pi05.model import (
    MODEL_ID,
    Pi05ActionExpertQuantizable,
    Pi05Collection,
    Pi05PaliGemmaBackboneQuantizable,
    Pi05PaliGemmaVisionQuantizable,
)
from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader

# Per-component precision mapping
MIXED_PRECISION_MAP: dict[str, Precision] = {
    "vision_encoder": Precision.w8a16,
    "backbone": Precision.w4a16,
    "action_expert": Precision.w8a16,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize Pi05 components with AIMET-ONNX."
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["vision_encoder", "backbone", "action_expert"],
        default="vision_encoder",
        help="Component to quantize: 'vision_encoder', 'backbone', or 'action_expert'.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="DEFAULT_UNQUANTIZED",
        help="Huggingface repo id or local directory with custom weights.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=f"Directory where quantized checkpoint should be stored. Defaults to ./build/{MODEL_ID}_<precision>.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help=(
            "Override the default per-component precision from MIXED_PRECISION_MAP "
            "(e.g. 'w4a16', 'w8a16'). Defaults to the mixed-precision mapping."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples used to calibrate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="For reproducibility.",
    )
    parser.add_argument(
        "--host-device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="One of cpu, cuda. Run QuantSim calibration on this host device.",
    )
    parser.add_argument(
        "--use-spinquant-r1",
        action="store_true",
        help=(
            "Apply SpinQuant R1 rotation to the backbone's internal weights "
            "before calibration (only affects --component backbone; ignored "
            "otherwise). hidden_state calibration inputs are rotated to "
            "match. See spinquant_r1.py."
        ),
    )
    args = parser.parse_args()

    if args.use_spinquant_r1 and args.component != "backbone":
        print(
            f"--use-spinquant-r1 has no effect on --component {args.component} "
            "(only the backbone is rotated); ignoring."
        )

    torch.manual_seed(args.seed)

    host_device = torch.device(args.host_device)

    precision = (
        Precision.parse(args.precision)
        if args.precision
        else MIXED_PRECISION_MAP[args.component]
    )

    QCls: type
    if args.component == "vision_encoder":
        QCls = Pi05PaliGemmaVisionQuantizable
    elif args.component == "backbone":
        QCls = Pi05PaliGemmaBackboneQuantizable
    else:
        QCls = Pi05ActionExpertQuantizable

    print(f"Quantizing component={args.component} precision={precision}")

    from_pretrained_kwargs: dict = {}
    if args.component == "backbone":
        from_pretrained_kwargs["use_spinquant_r1"] = args.use_spinquant_r1

    component = QCls.from_pretrained(
        checkpoint=args.checkpoint,
        host_device=host_device,
        precision=precision,
        **from_pretrained_kwargs,
    )

    # Float collection whose components run float forward passes; used to build
    # calibration inputs for the quantizable component above. Shares the
    # lru_cached policy from load_checkpoint, so no duplicate float weights.
    fp_collection = Pi05Collection.from_pretrained(host_device=host_device)

    ds = Pi05App.get_calibration_data(
        fp_collection,
        args.component,
        num_samples=args.num_samples,
        use_spinquant_r1=args.use_spinquant_r1,
    )
    data_loader = dataset_entries_to_dataloader(ds)

    component.quantize(
        data_loader,
        num_samples=args.num_samples,
        use_seq_mse=True,
    )

    output_dir = args.output or str(Path() / "build" / f"{MODEL_ID}_mixed")
    component.save_calibrated_checkpoint(output_checkpoint=output_dir)


if __name__ == "__main__":
    main()
