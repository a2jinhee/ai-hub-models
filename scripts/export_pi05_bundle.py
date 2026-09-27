#!/usr/bin/env python3
"""Export every Pi05 component of one quantized checkpoint to QNN context
binaries, and write matching qnn-net-run inputs, into a single folder."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from qairt_local_compile import DEFAULT_ENV_SH, compile_checkpoint

from qai_hub_models.models.pi05 import MODEL_ID
from qai_hub_models.models.pi05.export import (
    DEFAULT_EXPORT_DEVICE,
    build_parser,
    export_model,
)
from qai_hub_models.models.pi05.model import (
    Pi05ActionExpert,
    Pi05PaliGemmaBackbone,
    Pi05PaliGemmaTokenEmbed,
    Pi05PaliGemmaVision,
)
from qai_hub_models.utils.input_spec import make_torch_inputs

# token_emb is excluded: it has no quantized artifact in a build folder.
DEFAULT_COMPONENTS = ["vision_encoder", "backbone", "action_expert"]
RUNTIME = "qnn_context_binary"
RUNTIME_DIR_NAME = f"{MODEL_ID}-{RUNTIME}"

# Classes whose static input specs shape the .raw tensors written for
# qnn-net-run. token_emb is listed here but not in DEFAULT_COMPONENTS: it has
# an input spec worth dumping even though there is nothing to compile for it.
INPUT_SPEC_COMPONENTS = {
    "vision_encoder": Pi05PaliGemmaVision,
    "token_emb": Pi05PaliGemmaTokenEmbed,
    "action_expert": Pi05ActionExpert,
    "backbone": Pi05PaliGemmaBackbone,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Build folder holding the per-component AIMET checkpoints, e.g. build/pi05_seqmse.",
    )
    p.add_argument(
        "--components",
        nargs="+",
        default=DEFAULT_COMPONENTS,
        choices=DEFAULT_COMPONENTS,
        help="Components to export. Defaults to all three.",
    )
    p.add_argument(
        "--output-root",
        default="export_assets",
        help="Where the per-experiment folder is created. Default: export_assets.",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Name of the per-experiment folder. Defaults to the checkpoint folder's name.",
    )
    device_group = p.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device",
        default=None,
        help="Hub device name. Defaults to the model's own default device.",
    )
    device_group.add_argument(
        "--chipset", default=None, help="Hub chipset, instead of --device."
    )
    p.add_argument(
        "--device-dir",
        default="/data/local/tmp/pi05",
        help=(
            "Path prefix AS SEEN ON THE DEVICE. The component name is appended, so "
            "input_list.txt refers to <device-dir>/<component>/<tensor>.raw."
        ),
    )
    p.add_argument(
        "--num-inferences",
        type=int,
        default=1,
        help="Input sets to generate per component.",
    )
    p.add_argument(
        "--profile", action="store_true", help="Also run the cloud profile job."
    )
    p.add_argument(
        "--inference",
        action="store_true",
        help="Also run the cloud inference comparison.",
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Compile with a local QAIRT SDK instead of AI Hub. Needs no Hub token.",
    )
    p.add_argument(
        "--sdk-env",
        default=DEFAULT_ENV_SH,
        help=f"With --local: shell script that puts the QAIRT tools on PATH. Default: {DEFAULT_ENV_SH}",
    )
    p.add_argument(
        "--skip-inputs", action="store_true", help="Do not write qnn_inputs."
    )
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Only write qnn_inputs; do not export.",
    )
    p.add_argument(
        "--collect-only",
        action="store_true",
        help=(
            "Do not export. Reorganize bundle folders a previous export already "
            "downloaded into the output folder."
        ),
    )
    return p.parse_args()


def write_component_inputs(
    component: str,
    output_dir: str | Path,
    num_inferences: int = 1,
    device_dir: str = ".",
) -> Path:
    """Write one component's .raw tensors + input_list.txt under output_dir/component.

    Inputs are random tensors (seed 42 + inference index) shaped by the
    component's static input spec, so they do not depend on any checkpoint.
    Returns the directory that was written.
    """
    spec = INPUT_SPEC_COMPONENTS[component].get_input_spec_static()
    out = Path(output_dir) / component
    out.mkdir(parents=True, exist_ok=True)

    lines = []
    for n in range(num_inferences):
        tensors = make_torch_inputs(spec, seed=42 + n)
        entries = []
        for (name, ts), t in zip(spec.items(), tensors, strict=False):
            arr = np.ascontiguousarray(t.numpy().astype(ts.dtype))
            fn = f"{name}_{n}.raw"
            arr.tofile(out / fn)
            entries.append(f"{name}:={device_dir.rstrip('/')}/{fn}")
            print(f"  {name:<22} {tuple(arr.shape)} {arr.dtype} -> {fn}")
        lines.append(" ".join(entries))

    (out / "input_list.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}/input_list.txt ({len(lines)} inference(s))")
    return out


def check_checkpoint(checkpoint: Path, components: list[str]) -> None:
    """Fail before any expensive work if a component's folder is missing."""
    if not checkpoint.is_dir():
        sys.exit(f"Checkpoint folder does not exist: {checkpoint}")
    missing = [c for c in components if not (checkpoint / c / "model.onnx").is_file()]
    if missing:
        sys.exit(
            f"No model.onnx under {checkpoint} for component(s): {', '.join(missing)}"
        )


def run_export(
    checkpoint: Path, components: list[str], staging: Path, args: argparse.Namespace
) -> Path:
    """Run the pi05 export pipeline into *staging*; return the downloaded bundle."""
    argv = [
        "--checkpoint",
        str(checkpoint),
        "--components",
        *components,
        "--runtime",
        RUNTIME,
        "--precision",
        "mixed",
        "--output-dir",
        str(staging),
    ]
    if args.device:
        argv += ["--device", args.device]
    elif args.chipset:
        argv += ["--chipset", args.chipset]
    if not args.profile:
        argv.append("--skip-profiling")
    if not args.inference:
        argv.append("--skip-inferencing")

    print(f"\n=== export: {' '.join(argv)}\n")
    export_args = build_parser().parse_args(argv)
    result = export_model(MODEL_ID, **vars(export_args))
    if result.download_path is None:
        sys.exit("Export finished without downloading a bundle.")
    return Path(result.download_path)


def collect_bundle(bundle: Path, runtime_dir: Path) -> list[str]:
    """Move each <component>.bin in *bundle* into its own folder under *runtime_dir*.

    A bundle holding one component keeps its metadata.json beside that
    component's binary; a multi-component bundle keeps it at the runtime
    folder's root. Returns the components that were moved.
    """
    runtime_dir.mkdir(parents=True, exist_ok=True)
    binaries = [
        p for p in sorted(bundle.iterdir()) if p.is_file() and p.suffix == ".bin"
    ]
    if not binaries:
        print(f"  no .bin in {bundle}, skipping")
        return []

    moved = []
    for src in binaries:
        dest_dir = runtime_dir / src.stem
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        dest.unlink(missing_ok=True)
        shutil.move(str(src), str(dest))
        moved.append(src.stem)
        print(f"  {src.stem}: {dest}")

    metadata = bundle / "metadata.json"
    if metadata.is_file():
        parent = runtime_dir / moved[0] if len(moved) == 1 else runtime_dir
        dest = parent / "metadata.json"
        dest.unlink(missing_ok=True)
        shutil.move(str(metadata), str(dest))

    leftovers = sorted(p.name for p in bundle.iterdir())
    if leftovers:
        print(f"  WARNING: left in {bundle}: {', '.join(leftovers)}")
    else:
        bundle.rmdir()
    return moved


def find_bundles(out_dir: Path, runtime_dir: Path) -> list[Path]:
    """Bundle folders a previous export downloaded, newest last."""
    return sorted(
        p
        for p in out_dir.glob(f"{RUNTIME_DIR_NAME}*")
        if p.is_dir() and p != runtime_dir
    )


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    components = list(dict.fromkeys(args.components))
    check_checkpoint(checkpoint, components)

    out_dir = Path(args.output_root) / (args.name or checkpoint.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = out_dir / RUNTIME_DIR_NAME

    if not args.skip_inputs:
        inputs_root = out_dir / "qnn_inputs"
        for component in components:
            print(f"\n=== inputs: {component}")
            write_component_inputs(
                component,
                inputs_root,
                num_inferences=args.num_inferences,
                device_dir=f"{args.device_dir.rstrip('/')}/{component}",
            )

    if args.collect_only:
        bundles = find_bundles(out_dir, runtime_dir)
        if not bundles:
            sys.exit(f"No {RUNTIME_DIR_NAME}* bundle folders to collect in {out_dir}")
        print(f"\n=== collecting {len(bundles)} existing bundle(s)")
        for bundle in bundles:
            collect_bundle(bundle, runtime_dir)
    elif args.local:
        # The local compiler writes straight into the final per-component
        # layout, so there is no downloaded bundle to collect afterwards.
        if args.chipset:
            sys.exit("--local selects its target with --device, not --chipset.")
        compile_checkpoint(
            checkpoint,
            components,
            runtime_dir,
            args.device or DEFAULT_EXPORT_DEVICE,
            Path(args.sdk_env),
        )
    elif not args.skip_export:
        staging = out_dir / ".export_staging"
        bundle = run_export(checkpoint, components, staging, args)
        collect_bundle(bundle, runtime_dir)
        if staging.is_dir() and not any(staging.iterdir()):
            staging.rmdir()

    print(f"\nDone: {out_dir}")


if __name__ == "__main__":
    main()
