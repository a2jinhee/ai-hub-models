#!/usr/bin/env python3
"""Compile a Pi05 AIMET checkpoint to QNN context binaries with a local QAIRT SDK."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import onnx

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.model_metadata import (
    ChipsetAttributes,
    ModelFileMetadata,
    ModelMetadata,
    merge_input_metadata,
)
from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models.pi05.model import (
    Pi05ActionExpert,
    Pi05PaliGemmaBackbone,
    Pi05PaliGemmaVision,
)

DEFAULT_ENV_SH = os.environ.get("QAIRT_ENV_SH", "/home/jk656/qairt/env.sh")
MODEL_ID = "pi05"
MODEL_DISPLAY_NAME = "Pi0.5"

# ONNX TensorProto elem_type -> the dtype string metadata.json uses.
_ONNX_DTYPES = {
    onnx.TensorProto.FLOAT: "float32",
    onnx.TensorProto.FLOAT16: "float16",
    onnx.TensorProto.INT64: "int64",
    onnx.TensorProto.INT32: "int32",
    onnx.TensorProto.UINT8: "uint8",
    onnx.TensorProto.INT8: "int8",
    onnx.TensorProto.UINT16: "uint16",
    onnx.TensorProto.INT16: "int16",
    onnx.TensorProto.BOOL: "bool",
}

# The classes whose static specs describe each component's I/O. These are the
# same specs make_qnn_input_list.py shapes its .raw tensors from.
COMPONENTS = {
    "vision_encoder": Pi05PaliGemmaVision,
    "backbone": Pi05PaliGemmaBackbone,
    "action_expert": Pi05ActionExpert,
}


def sdk_env(env_sh: Path) -> dict[str, str]:
    """Return the environment a QAIRT tool needs, by sourcing *env_sh*."""
    if not env_sh.is_file():
        sys.exit(
            f"QAIRT env script not found: {env_sh}\n"
            "Install the QAIRT SDK and pass --sdk-env, or set QAIRT_ENV_SH."
        )
    dumped = subprocess.run(
        ["bash", "-c", f"source {shlex.quote(str(env_sh))} >/dev/null && env -0"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    env = dict(
        entry.split("=", 1) for entry in dumped.split("\0") if "=" in entry
    )
    for var in ("QNN_SDK_ROOT", "QAIRT_BIN", "QNN_TARGET_LIB"):
        if var not in env:
            sys.exit(f"{env_sh} did not export {var}.")
    return env


def sdk_version(env: dict[str, str]) -> str:
    """Read the SDK's full version out of its sdk.yaml.
    """
    sdk_yaml = Path(env["QNN_SDK_ROOT"]) / "sdk.yaml"
    fields = {}
    for line in sdk_yaml.read_text().splitlines():
        for key in ("version", "build_id"):
            if line.startswith(f"{key}:"):
                fields[key] = line.split(":", 1)[1].strip()
    missing = {"version", "build_id"} - fields.keys()
    if missing:
        sys.exit(f"{sdk_yaml} has no {', '.join(sorted(missing))} line")
    return f"{fields['version']}.{fields['build_id']}"


def run(cmd: list[str], env: dict[str, str], log: Path) -> None:
    """Run one SDK tool, teeing its output to *log*; exit on failure."""
    print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}")
    with log.open("w") as fh:
        proc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = log.read_text().splitlines()[-25:]
        sys.exit(
            f"\n{cmd[0]} failed (exit {proc.returncode}). Last lines of {log}:\n"
            + "\n".join(tail)
        )


def write_htp_config(
    work_dir: Path, graph: str, htp_version: int, soc_model: int
) -> Path:
    """Write the two JSON files qnn-context-binary-generator needs for HTP.
    """
    backend_config = work_dir / "htp_backend_config.json"
    backend_config.write_text(
        json.dumps(
            {
                "graphs": [{"graph_names": [graph]}],
                "devices": [
                    {"dsp_arch": f"v{htp_version}", "soc_id": soc_model}
                ],
            },
            indent=2,
        )
        + "\n"
    )

    config = work_dir / "htp_config.json"
    config.write_text(
        json.dumps(
            {
                "backend_extensions": {
                    "shared_library_path": "libQnnHtpNetRunExtensions.so",
                    "config_file_path": str(backend_config),
                }
            },
            indent=2,
        )
        + "\n"
    )
    return config


def tensor_specs(
    values: Iterable[onnx.ValueInfoProto], source: Path
) -> dict[str, TensorSpec]:
    """The shape and dtype of each ONNX value, as metadata.json records them."""
    specs: dict[str, TensorSpec] = {}
    for value in values:
        tensor_type = value.type.tensor_type
        dtype = _ONNX_DTYPES.get(tensor_type.elem_type)
        if dtype is None:
            sys.exit(
                f"{source}: unsupported ONNX elem_type "
                f"{tensor_type.elem_type} on tensor {value.name}"
            )
        shape = [d.dim_value for d in tensor_type.shape.dim]
        if not all(shape):
            sys.exit(
                f"{source}: tensor {value.name} has a dynamic dimension "
                f"({shape}); context binaries need static shapes."
            )
        specs[value.name] = TensorSpec(shape=tuple(shape), dtype=dtype)
    return specs


def io_metadata(component: str, onnx_path: Path) -> ModelFileMetadata:
    """Describe a component's I/O the way a downloaded bundle does.
    """
    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    metadata = ModelFileMetadata(
        inputs=tensor_specs(graph.input, onnx_path),
        outputs=tensor_specs(graph.output, onnx_path),
    )
    merge_input_metadata(
        metadata,
        COMPONENTS[component].get_input_spec_static(),
        TargetRuntime.QNN_CONTEXT_BINARY,
    )
    return metadata


def compile_component(
    component: str,
    checkpoint: Path,
    runtime_dir: Path,
    work_root: Path,
    env: dict[str, str],
    htp_version: int,
    soc_model: int,
    keep_intermediates: bool,
) -> ModelFileMetadata:
    """Convert and prepare one component; return its I/O metadata."""
    src = checkpoint / component
    onnx_path = src / "model.onnx"
    encodings = src / "model.encodings"
    if not onnx_path.is_file():
        sys.exit(f"Missing {onnx_path}")
    if not encodings.is_file():
        sys.exit(f"Missing {encodings}")

    graph = f"{MODEL_ID}_{component}"
    work = work_root / component
    work.mkdir(parents=True, exist_ok=True)
    dlc = work / f"{graph}.dlc"
    dest_dir = runtime_dir / component
    dest_dir.mkdir(parents=True, exist_ok=True)

    # --preserve_io_datatype keeps graph inputs and outputs float32.
    print(f"\n=== convert: {component}")
    run(
        [
            f"{env['QAIRT_BIN']}/qairt-converter",
            "--input_network", str(onnx_path),
            "--quantization_overrides", str(encodings),
            "--preserve_io_datatype",
            "--output_path", str(dlc),
        ],
        env,
        work / "convert.log",
    )

    print(f"=== context binary: {component}")
    config = write_htp_config(work, graph, htp_version, soc_model)
    run(
        [
            f"{env['QAIRT_BIN']}/qnn-context-binary-generator",
            f"--model={env['QNN_TARGET_LIB']}/libQnnModelDlc.so",
            f"--backend={env['QNN_TARGET_LIB']}/libQnnHtp.so",
            f"--dlc_path={dlc}",
            f"--config_file={config}",
            f"--output_dir={dest_dir}",
            f"--binary_file={component}",
        ],
        env,
        work / "context.log",
    )

    binary = dest_dir / f"{component}.bin"
    if not binary.is_file():
        sys.exit(f"qnn-context-binary-generator did not write {binary}")
    print(f"  {component}: {binary} ({binary.stat().st_size / 2**20:.1f} MiB)")

    if not keep_intermediates:
        dlc.unlink(missing_ok=True)

    return io_metadata(component, onnx_path)


def write_metadata(
    runtime_dir: Path,
    metadata_by_file: dict[str, ModelFileMetadata],
    chipset: ChipsetAttributes,
    qairt_version: str,
    components: list[str],
) -> None:
    metadata = ModelMetadata(
        model_id=MODEL_ID,
        model_name=MODEL_DISPLAY_NAME,
        runtime=TargetRuntime.QNN_CONTEXT_BINARY,
        precision=Precision.mixed,
        tool_versions=ToolVersions(qairt=qairt_version),
        model_files=metadata_by_file,
        chipset_attributes=chipset,
    )
    parent = runtime_dir / components[0] if len(components) == 1 else runtime_dir
    metadata.to_json(parent / "metadata.json")
    print(f"\nWrote {parent / 'metadata.json'}")


def resolve_chipset(device: str) -> ChipsetAttributes:
    """The chipset *device* runs, with the attributes offline prepare needs."""
    import qai_hub as hub

    chipset = ChipsetAttributes.from_hub_device(hub.Device(device))
    assert chipset is not None, f"No chipset attributes for device {device!r}"
    if chipset.htp_version is None or chipset.soc_model is None:
        sys.exit(
            f"Device {device!r} (chipset {chipset.name}) has no htp_version/"
            "soc_model recorded, so it cannot be targeted offline."
        )
    return chipset


def compile_checkpoint(
    checkpoint: Path,
    components: list[str],
    runtime_dir: Path,
    device: str,
    env_sh: Path,
    keep_intermediates: bool = False,
) -> None:
    """Compile every component locally and write the bundle's metadata.json."""
    chipset = resolve_chipset(device)
    env = sdk_env(env_sh)
    qairt_version = sdk_version(env)
    print(f"QAIRT SDK  : {env['QNN_SDK_ROOT']} ({qairt_version})")
    print(f"Target     : {device} / {chipset.name} "
          f"(v{chipset.htp_version}, soc_id {chipset.soc_model})")

    work_root = runtime_dir.parent / ".qairt_work"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    metadata_by_file = {
        f"{component}.bin": compile_component(
            component,
            checkpoint,
            runtime_dir,
            work_root,
            env,
            chipset.htp_version,
            chipset.soc_model,
            keep_intermediates,
        )
        for component in components
    }
    write_metadata(
        runtime_dir, metadata_by_file, chipset, qairt_version, components
    )
    if not keep_intermediates:
        shutil.rmtree(work_root, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, help="Build folder with per-component checkpoints.")
    p.add_argument(
        "--components",
        nargs="+",
        default=list(COMPONENTS),
        choices=list(COMPONENTS),
        help="Components to compile.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Folder the pi05-qnn_context_binary tree is written under.",
    )
    p.add_argument(
        "--device",
        default="Dragonwing IQ-9075 EVK",
        help="Device whose chipset the binaries are prepared for.",
    )
    p.add_argument(
        "--sdk-env",
        default=DEFAULT_ENV_SH,
        help=f"Shell script that puts the QAIRT tools on PATH. Default: {DEFAULT_ENV_SH}",
    )
    p.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep the .qairt_work scratch tree, DLCs included, instead of deleting it.",
    )
    args = p.parse_args()

    compile_checkpoint(
        Path(args.checkpoint).resolve(),
        list(dict.fromkeys(args.components)),
        Path(args.output_dir).resolve() / f"{MODEL_ID}-qnn_context_binary",
        args.device,
        Path(args.sdk_env),
        args.keep_intermediates,
    )


if __name__ == "__main__":
    main()
