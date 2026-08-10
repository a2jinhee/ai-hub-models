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

    # Rotate with SpinQuant (R1 + R2 + R3 + R4, wherever each applies). All
    # three rotated components must be built into the same output directory:
    python -m qai_hub_models.models.pi05.quantize --component vision_encoder \
        --use-spinquant -o build/pi05_mixed_spin_r1r2r3r4
    python -m qai_hub_models.models.pi05.quantize --component backbone \
        --use-spinquant -o build/pi05_mixed_spin_r1r2r3r4
    python -m qai_hub_models.models.pi05.quantize --component action_expert \
        --use-spinquant -o build/pi05_mixed_spin_r1r2r3r4

    # Rotations only (seqMSE is off by default -- isolates the rotations'
    # contribution):
    python -m qai_hub_models.models.pi05.quantize --component backbone \
        --use-spinquant -o build/pi05_mixed_spin_noseqmse

    # Add seqMSE on top of rotations (cost is linear in the sample count):
    python -m qai_hub_models.models.pi05.quantize --component backbone \
        --use-spinquant --use-seq-mse --seq-mse-num-samples 10 -o build/pi05_fast_seqmse

    # Override the default per-component precision (e.g. action_expert at w4a16):
    python -m qai_hub_models.models.pi05.quantize --component action_expert --precision w4a16

--use-spinquant applies:

  * R1 (residual stream) to the backbone, and to the two components that
    produce its hidden_state -- the vision encoder's projector and the language
    embedding table inside token_emb. R1 is folded into weights everywhere, so
    the deployment path runs no rotation of its own and Pi05App needs no flag.
  * R1 to the action expert's own residual stream, independently of the
    backbone's R1 -- self-contained, no cross-component coupling. Most of it
    is a zero-cost static fold; a small, localized online rotation is paid
    only around the three AdaRMS norm calls per layer, since AdaRMS's
    scale/shift/gate is a genuine runtime computation. See
    apply_action_expert_r1 in spinquant_r1.py.
  * R2 (per-head V/o_proj) to the backbone, and the matching compensation to
    the action expert.
  * R3 (per-head Q/K, online rotation) to the backbone, and the matching
    compensation to the action expert. No effect on vision_encoder.
  * R4 (FFN down_proj input) to the backbone only -- self-contained, no
    action-expert coupling (not applied there; too slow currently). Recorded
    in the marker for record-keeping only, like action_expert's own R1.

R1 on vision_encoder/backbone and R2/R3 on backbone/action_expert couple those
components: R1 links vision_encoder -> backbone, R2 and R3 each independently
link backbone -> action_expert. So all rotated components must be built with
matching settings into the same checkpoint directory. Each records what it was
built with in a marker file (SPINQUANT_MARKER), and
Pi05CollectionQuantized.from_pretrained refuses a set that disagrees. token_emb
has no quantized artifact of its own; its R1 fold is re-derived at load time
from the vision encoder's marker. action_expert's own R1 has no such coupling
-- it's recorded in the same marker file alongside its R2/R3 flags, but only
for record-keeping, not cross-component validation.
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
from qai_hub_models.utils.quantization_aimet_onnx import DEFAULT_SEQ_MSE_NUM_SAMPLES

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
        "--seq-mse-num-samples",
        type=int,
        default=DEFAULT_SEQ_MSE_NUM_SAMPLES,
        help=(
            "Number of samples used for Sequential MSE weight-encoding "
            "optimization. Kept separate from --num-samples because seqMSE "
            "cost is linear in this count while activation calibration "
            "benefits from more samples."
        ),
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
            "Apply SpinQuant rotations (R1 + R2 + R3 + R4, wherever each "
            "applies) before calibration. On the backbone this rotates the "
            "residual stream (R1), the per-head V/o_proj path (R2), the "
            "per-head Q/K path (R3), and the FFN down_proj input (R4); on "
            "the action expert it applies its own independent "
            "residual-stream R1 plus the matching R2/R3 compensation for "
            "the backbone's R2/R3 (R4 is backbone-only). No effect on "
            "--component vision_encoder beyond R1. Because R2/R3 couple "
            "backbone and action_expert, both must be built with this flag "
            "together. See spinquant_r1.py."
        ),
    )
    parser.add_argument(
        "--use-seq-mse",
        dest="use_seq_mse",
        action="store_true",
        help=(
            "Apply Sequential MSE weight-encoding optimization instead of "
            "plain min-max encodings. Off by default so the effect of "
            "SpinQuant rotations can be checked in isolation from seqMSE's."
        ),
    )
    args = parser.parse_args()

    # R1 links vision_encoder -> backbone (shared rotation); on action_expert
    # it's an independent, self-contained rotation with no cross-component
    # coupling (see apply_action_expert_r1 in spinquant_r1.py). R2 and R3
    # each independently link backbone -> action_expert. R4 is backbone-only,
    # self-contained -- no action-expert counterpart.
    r1_components = ("vision_encoder", "backbone", "action_expert")
    r2_components = ("backbone", "action_expert")
    r3_components = ("backbone", "action_expert")
    r4_components = ("backbone",)
    rotated_components = ("vision_encoder", "backbone", "action_expert")
    if args.use_spinquant:
        others = [c for c in rotated_components if c != args.component]
        print(
            f"\n  NOTE: SpinQuant couples all of {', '.join(rotated_components)}.\n"
            f"  Build {' and '.join(repr(c) for c in others)} with --use-spinquant\n"
            "  into the same checkpoint directory, or the set will be refused at "
            "load time.\n"
        )

    torch.manual_seed(args.seed)

    # Resolve and validate the output directory up front. save_calibrated_checkpoint
    # runs only after calibration, so an unwritable path (a typo, or an unset shell
    # variable collapsing "$OUT/name" to "/name") would otherwise surface hours in
    # and discard the whole run.
    output_dir = Path(args.output or Path() / "build" / f"{MODEL_ID}_mixed").resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".write_probe"
        probe.touch()
        probe.unlink()
    except OSError as err:
        raise SystemExit(
            f"Output directory '{output_dir}' is not writable: {err}\n"
            "Check -o / --output (a leading '/' usually means a shell variable "
            "was empty)."
        ) from err
    print(f"Output directory: {output_dir}")

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
    if args.component in r1_components:
        from_pretrained_kwargs["use_spinquant_r1"] = args.use_spinquant
    if args.component in r2_components:
        from_pretrained_kwargs["use_spinquant_r2"] = args.use_spinquant
    if args.component in r3_components:
        from_pretrained_kwargs["use_spinquant_r3"] = args.use_spinquant
    if args.component in r4_components:
        from_pretrained_kwargs["use_spinquant_r4"] = args.use_spinquant

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
        use_spinquant_r3=args.use_spinquant,
    )
    data_loader = dataset_entries_to_dataloader(ds)

    if not args.use_seq_mse:
        print("Skipping Sequential MSE; using min-max weight encodings.")

    component.quantize(
        data_loader,
        num_samples=args.num_samples,
        use_seq_mse=args.use_seq_mse,
        seq_mse_num_samples=args.seq_mse_num_samples,
    )

    component.save_calibrated_checkpoint(output_checkpoint=str(output_dir))

    # save_calibrated_checkpoint writes into <output_dir>/<subfolder>, so the
    # marker belongs beside the component it describes -- that is where
    # Pi05CollectionQuantized.from_pretrained looks for it.
    marker = write_rotation_marker(
        Path(output_dir) / QCls.default_subfolder,
        r1=args.use_spinquant and args.component in r1_components,
        r2=args.use_spinquant and args.component in r2_components,
        r3=args.use_spinquant and args.component in r3_components,
        r4=args.use_spinquant and args.component in r4_components,
    )
    print(f"Recorded SpinQuant rotations in {marker}")


if __name__ == "__main__":
    main()
