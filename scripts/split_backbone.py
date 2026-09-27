#!/usr/bin/env python3
"""Compile the Pi05 backbone as several QNN context binaries instead of one."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import shape_inference
from qairt_local_compile import (
    _ONNX_DTYPES,
    COMPONENTS,
    DEFAULT_ENV_SH,
    MODEL_DISPLAY_NAME,
    MODEL_ID,
    clean_work,
    io_metadata,
    resolve_chipset,
    run,
    sdk_env,
    sdk_version,
    tensor_specs,
    write_htp_config,
)

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.model_metadata import (
    ChipsetAttributes,
    ModelFileMetadata,
    ModelMetadata,
    merge_input_metadata,
)
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models._shared.llm.split_onnx_utils.utils import split_onnx_by_names
from qai_hub_models.utils.input_spec import make_torch_inputs
from qai_hub_models.utils.onnx.helpers import ONNXBundle

COMPONENT = "backbone"

# Layer L's input_layernorm reads the hidden state entering that layer. Layer 0
# has no suffix, the rest are numbered, which is what makes the layer index
# recoverable from the node name.
_ENTRY_NODE = re.compile(r"^/input_layernorm(?:_(\d+))?/Cast$")


def layer_entry_tensors(graph: onnx.GraphProto) -> dict[int, str]:
    """The hidden-state tensor entering each transformer layer, by layer index."""
    entries: dict[int, str] = {}
    for node in graph.node:
        match = _ENTRY_NODE.match(node.name)
        if match is not None:
            entries[int(match.group(1) or 0)] = node.input[0]
    if not entries or sorted(entries) != list(range(len(entries))):
        sys.exit(
            "Could not read a contiguous run of layers off the backbone graph "
            f"(found layers {sorted(entries)}). The node naming this relies on "
            "may have changed."
        )
    return entries


def cut_layers(num_layers: int, splits: int) -> list[int]:
    """The layers to cut before, spreading *num_layers* evenly over *splits*."""
    if not 2 <= splits <= num_layers:
        sys.exit(f"--splits must be between 2 and {num_layers}, got {splits}")
    return [round(num_layers * i / splits) for i in range(1, splits)]


def rename_tensor(graph: onnx.GraphProto, old: str, new: str) -> None:
    """Rename one intermediate tensor everywhere the graph names it."""
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new
        for i, name in enumerate(node.output):
            if name == old:
                node.output[i] = new
    for value in graph.value_info:
        if value.name == old:
            value.name = new


def split_backbone(checkpoint: Path, work: Path, splits: int) -> list[ONNXBundle]:
    """Cut the backbone into *splits* ONNX models on its layer boundaries."""
    bundle = ONNXBundle.from_bundle_path(checkpoint / COMPONENT)
    model = onnx.load(str(bundle.onnx_graph_path), load_external_data=False)
    ir_version = model.ir_version
    # OnnxSplitter needs a ValueInfoProto for every tensor it cuts on, and the
    # exported graph carries none.
    model = shape_inference.infer_shapes(model, data_prop=True)

    entries = layer_entry_tensors(model.graph)
    boundaries = []
    for layer in cut_layers(len(entries), splits):
        boundary = f"hidden_state_l{layer}"
        rename_tensor(model.graph, entries[layer], boundary)
        boundaries.append(boundary)
    print(f"  {len(entries)} layers, cutting at: {', '.join(boundaries)}")

    out = work / "split"
    out.mkdir(parents=True, exist_ok=True)
    parts = split_onnx_by_names(
        bundle, COMPONENT, *boundaries, output_dir=out, onnxmodel=model
    )

    for part in parts:
        # make_model stamps the running ONNX release's IR version on a split;
        # the SDK's importer reads the one the export was written with.
        submodel = onnx.load(str(part.onnx_graph_path), load_external_data=False)
        submodel.ir_version = ir_version
        onnx.save(submodel, str(part.onnx_graph_path))
    return parts


def part_metadata(part: ONNXBundle) -> ModelFileMetadata:
    """Describe one split's I/O the way a downloaded bundle describes a component.

    Only the tensors the whole backbone declares carry semantic fields, so the
    input spec is merged over just those; a hidden state a cut introduces is an
    ordinary float tensor with nothing to add.
    """
    path = part.onnx_graph_path
    graph = onnx.load(str(path), load_external_data=False).graph
    metadata = ModelFileMetadata(
        inputs=tensor_specs(graph.input, path),
        outputs=tensor_specs(graph.output, path),
    )
    spec = COMPONENTS[COMPONENT].get_input_spec_static()
    merge_input_metadata(
        metadata,
        {name: t for name, t in spec.items() if name in metadata.inputs},
        TargetRuntime.QNN_CONTEXT_BINARY,
    )
    return metadata


def compile_part(
    name: str,
    part: ONNXBundle,
    runtime_dir: Path,
    work: Path,
    env: dict[str, str],
    htp_version: int,
    soc_model: int,
    float_bitwidth: int,
    keep_intermediates: bool,
) -> ModelFileMetadata:
    """Convert and prepare one split; return its I/O metadata."""
    graph = f"{MODEL_ID}_{name}"
    part_work = work / name
    part_work.mkdir(parents=True, exist_ok=True)
    dlc = part_work / f"{graph}.dlc"
    dest_dir = runtime_dir / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # A quantized checkpoint carries encodings, which the splitter subsets per
    # part; a float one does not, and is converted at --float_bitwidth instead.
    if part.aimet_encodings_path is not None:
        precision_args = ["--quantization_overrides", str(part.aimet_encodings_path)]
    else:
        precision_args = ["--float_bitwidth", str(float_bitwidth)]

    print(f"\n=== convert: {name}")
    run(
        [
            f"{env['QAIRT_BIN']}/qairt-converter",
            "--input_network",
            str(part.onnx_graph_path),
            *precision_args,
            "--preserve_io_datatype",
            "--output_path",
            str(dlc),
        ],
        env,
        part_work / "convert.log",
    )

    print(f"=== context binary: {name}")
    config = write_htp_config(part_work, graph, htp_version, soc_model)
    run(
        [
            f"{env['QAIRT_BIN']}/qnn-context-binary-generator",
            f"--model={env['QNN_TARGET_LIB']}/libQnnModelDlc.so",
            f"--backend={env['QNN_TARGET_LIB']}/libQnnHtp.so",
            f"--dlc_path={dlc}",
            f"--config_file={config}",
            f"--output_dir={dest_dir}",
            f"--binary_file={name}",
        ],
        env,
        part_work / "context.log",
    )

    binary = dest_dir / f"{name}.bin"
    if not binary.is_file():
        sys.exit(f"qnn-context-binary-generator did not write {binary}")
    print(f"  {name}: {binary} ({binary.stat().st_size / 2**20:.1f} MiB)")

    metadata = part_metadata(part)
    if not keep_intermediates:
        dlc.unlink(missing_ok=True)
    return metadata


def write_metadata(
    runtime_dir: Path,
    checkpoint: Path,
    parts_metadata: dict[str, ModelFileMetadata],
    chipset: ChipsetAttributes,
    qairt_version: str,
) -> None:
    """Rewrite the bundle's metadata.json around the split backbone.

    Components compiled before the split still belong in the bundle, so any
    whose binary is already there is described from its own ONNX graph. The
    unsplit backbone.bin entry, if a previous export left one, is dropped.
    """
    path = runtime_dir / "metadata.json"
    model_files: dict[str, ModelFileMetadata] = {}
    if path.is_file():
        model_files.update(ModelMetadata.from_json(path).model_files)

    for component in COMPONENTS:
        if component == COMPONENT or f"{component}.bin" in model_files:
            continue
        binary = runtime_dir / component / f"{component}.bin"
        onnx_path = checkpoint / component / "model.onnx"
        if binary.is_file() and onnx_path.is_file():
            model_files[f"{component}.bin"] = io_metadata(component, onnx_path)

    model_files.pop(f"{COMPONENT}.bin", None)
    model_files.update(parts_metadata)

    ModelMetadata(
        model_id=MODEL_ID,
        model_name=MODEL_DISPLAY_NAME,
        runtime=TargetRuntime.QNN_CONTEXT_BINARY,
        precision=Precision.mixed,
        tool_versions=ToolVersions(qairt=qairt_version),
        model_files=model_files,
        chipset_attributes=chipset,
    ).to_json(path)
    print(f"\nWrote {path} ({len(model_files)} model file(s))")


def write_part_inputs(
    names: list[str],
    parts: list[ONNXBundle],
    inputs_root: Path,
    device_dir: str,
    num_inferences: int,
) -> None:
    """Write .raw tensors and an input_list.txt for each split.

    Every layer reads the mask and rope tensors, so each split is given the
    same values for them as the whole backbone would get from
    export_pi05_bundle. A hidden state that a cut introduces is random here;
    on device it is the previous split's output.
    """
    spec = COMPONENTS[COMPONENT].get_input_spec_static()
    graphs = [
        onnx.load(str(part.onnx_graph_path), load_external_data=False).graph
        for part in parts
    ]
    lines: dict[str, list[str]] = {name: [] for name in names}

    for n in range(num_inferences):
        shared = {
            name: tensor.numpy().astype(t.dtype)
            for (name, t), tensor in zip(
                spec.items(), make_torch_inputs(spec, seed=42 + n), strict=False
            )
        }
        rng = np.random.default_rng(42 + n)
        for name, graph in zip(names, graphs, strict=False):
            out = inputs_root / name
            out.mkdir(parents=True, exist_ok=True)
            entries = []
            for value in graph.input:
                tensor_type = value.type.tensor_type
                dtype = np.dtype(_ONNX_DTYPES[tensor_type.elem_type])
                shape = tuple(d.dim_value for d in tensor_type.shape.dim)
                array = shared.get(value.name)
                if array is None:
                    array = rng.standard_normal(shape)
                array = np.ascontiguousarray(array.astype(dtype))
                raw = f"{value.name}_{n}.raw"
                array.tofile(out / raw)
                entries.append(f"{value.name}:={device_dir}/{name}/{raw}")
            lines[name].append(" ".join(entries))

    for name in names:
        listing = inputs_root / name / "input_list.txt"
        listing.write_text("\n".join(lines[name]) + "\n")
        print(f"  {name}: {listing}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint", required=True, help="Build folder holding backbone/."
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Per-experiment folder, e.g. export_assets/pi05_fp16.",
    )
    p.add_argument(
        "--splits",
        type=int,
        default=2,
        help="Number of context binaries to cut the backbone into. Default: 2.",
    )
    p.add_argument(
        "--device",
        default="Dragonwing IQ-9075 EVK",
        help="Device whose chipset the binaries are prepared for.",
    )
    p.add_argument(
        "--float-bitwidth",
        type=int,
        default=16,
        help="Float width for a checkpoint without encodings. Default: 16.",
    )
    p.add_argument(
        "--device-dir",
        default="/data/local/tmp/pi05",
        help="Path prefix AS SEEN ON THE DEVICE, used in input_list.txt.",
    )
    p.add_argument(
        "--num-inferences",
        type=int,
        default=1,
        help="Input sets to generate per split.",
    )
    p.add_argument(
        "--skip-inputs", action="store_true", help="Do not write qnn_inputs."
    )
    p.add_argument(
        "--sdk-env",
        default=DEFAULT_ENV_SH,
        help=f"Shell script that puts the QAIRT tools on PATH. Default: {DEFAULT_ENV_SH}",
    )
    p.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep the .qairt_work scratch tree, split models and DLCs included.",
    )
    args = p.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not (checkpoint / COMPONENT / "model.onnx").is_file():
        sys.exit(f"No {COMPONENT}/model.onnx under {checkpoint}")
    out_dir = Path(args.output_dir).resolve()
    runtime_dir = out_dir / f"{MODEL_ID}-qnn_context_binary"
    work = out_dir / ".qairt_work"

    chipset = resolve_chipset(args.device)
    env = sdk_env(Path(args.sdk_env))
    qairt_version = sdk_version(env)
    print(f"QAIRT SDK  : {env['QNN_SDK_ROOT']} ({qairt_version})")
    print(
        f"Target     : {args.device} / {chipset.name} "
        f"(v{chipset.htp_version}, soc_id {chipset.soc_model})"
    )

    print(f"\n=== split: {COMPONENT} into {args.splits}")
    parts = split_backbone(checkpoint, work, args.splits)
    names = [f"{COMPONENT}_{i + 1}_of_{args.splits}" for i in range(args.splits)]
    for name, part in zip(names, parts, strict=False):
        assert part.bundle_path.stem == name, (part.bundle_path, name)
        weights = sum(
            os.path.getsize(part.bundle_path / f)
            for f in os.listdir(part.bundle_path)
            if f.endswith(".data")
        )
        print(f"  {name}: {part.onnx_graph_path} ({weights / 2**30:.2f} GiB fp32)")

    assert chipset.htp_version is not None and chipset.soc_model is not None
    parts_metadata = {
        f"{name}.bin": compile_part(
            name,
            part,
            runtime_dir,
            work,
            env,
            chipset.htp_version,
            chipset.soc_model,
            args.float_bitwidth,
            args.keep_intermediates,
        )
        for name, part in zip(names, parts, strict=False)
    }
    write_metadata(runtime_dir, checkpoint, parts_metadata, chipset, qairt_version)

    if not args.skip_inputs:
        print("\n=== inputs")
        write_part_inputs(
            names,
            parts,
            out_dir / "qnn_inputs",
            args.device_dir.rstrip("/"),
            args.num_inferences,
        )

    if not args.keep_intermediates:
        clean_work(work)

    print(f"\nDone: {runtime_dir}")


if __name__ == "__main__":
    main()
