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

    # Rotate with SpinQuant (R1 + R2). Both rotated components must be built:
    python -m qai_hub_models.models.pi05.quantize --component backbone \
        --use-spinquant -o build/pi05_mixed_spin_r1r2
    python -m qai_hub_models.models.pi05.quantize --component action_expert \
        --use-spinquant -o build/pi05_mixed_spin_r1r2

    # Rotations only, no seqMSE (isolates the rotations' contribution):
    python -m qai_hub_models.models.pi05.quantize --component backbone \
        --use-spinquant --no-seq-mse -o build/pi05_mixed_spin_r1r2_noseqmse

    # Override the default per-component precision (e.g. action_expert at w4a16):
    python -m qai_hub_models.models.pi05.quantize --component action_expert --precision w4a16

If backbone is quantized with --use-spinquant, the deployment/eval path
(Pi05App, e.g. pi05_libero_server.py) must also be run with R1 enabled, so
hidden_state is rotated to match the backbone's rotated weights.

--use-spinquant applies R1 (residual stream) and R2 (per-head V/o_proj) to the
backbone, and the matching R2 compensation to the action expert. R2 rotates the
per-layer V caches the backbone hands the expert, so the two components are
coupled and must be built together into the same checkpoint directory. Each
records what it was built with in a marker file (SPINQUANT_MARKER), and the
deployment path refuses a backbone/expert pair that disagree.
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
from qai_hub_models.models.pi05.spinquant_r1 import write_rotation_marker
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
        "--use-spinquant",
        "--use-spinquant-r1",
        dest="use_spinquant",
        action="store_true",
        help=(
            "Apply SpinQuant rotations (R1 + R2) before calibration. On the "
            "backbone this rotates the residual stream (R1) and the per-head "
            "V/o_proj path (R2); on the action expert it applies the matching "
            "R2 compensation. No effect on --component vision_encoder. "
            "Because R2 couples them, the backbone and action_expert must both "
            "be built with this flag. See spinquant_r1.py."
        ),
    )
    parser.add_argument(
        "--no-seq-mse",
        dest="use_seq_mse",
        action="store_false",
        help=(
            "Skip Sequential MSE weight-encoding optimization, leaving plain "
            "min-max encodings. Useful for isolating the effect of the "
            "SpinQuant rotations from that of seqMSE."
        ),
    )
    args = parser.parse_args()

    rotated_components = ("backbone", "action_expert")
    if args.use_spinquant and args.component not in rotated_components:
        print(
            f"--use-spinquant has no effect on --component {args.component} "
            f"(only {' and '.join(rotated_components)} are rotated); ignoring."
        )
    elif args.use_spinquant:
        other = next(c for c in rotated_components if c != args.component)
        print(
            f"\n  NOTE: R2 couples backbone and action_expert. Build '{other}' "
            f"with --use-spinquant\n  into the same checkpoint directory, or "
            "the pair will be refused at load time.\n"
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
        from_pretrained_kwargs["use_spinquant_r1"] = args.use_spinquant
        from_pretrained_kwargs["use_spinquant_r2"] = args.use_spinquant
    elif args.component == "action_expert":
        from_pretrained_kwargs["use_spinquant_r2"] = args.use_spinquant

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
        use_spinquant_r1=args.use_spinquant,
        use_spinquant_r2=args.use_spinquant,
    )
    data_loader = dataset_entries_to_dataloader(ds)

    if not args.use_seq_mse:
        print("Skipping Sequential MSE; using min-max weight encodings.")

    component.quantize(
        data_loader,
        num_samples=args.num_samples,
        use_seq_mse=args.use_seq_mse,
    )

    output_dir = args.output or str(Path() / "build" / f"{MODEL_ID}_mixed")
    component.save_calibrated_checkpoint(output_checkpoint=output_dir)

    if args.component in rotated_components:
        # save_calibrated_checkpoint writes into <output_dir>/<subfolder>, so
        # the marker belongs beside the component it describes -- that is where
        # Pi05CollectionQuantized.from_pretrained looks for it.
        marker = write_rotation_marker(
            Path(output_dir) / QCls.default_subfolder,
            r1=args.use_spinquant and args.component == "backbone",
            r2=args.use_spinquant,
        )
        print(f"Recorded SpinQuant rotations in {marker}")


if __name__ == "__main__":
    main()
