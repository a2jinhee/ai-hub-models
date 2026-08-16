# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""pi05 inference server for decoupled LIBERO evaluation.

Drop-in replacement for FastWAM's fastwam_server.py that serves Qualcomm AI
Hub's pi05 (quantized `build/pi05_mixed` or float) over the SAME websocket
contract the FastWAM fastwam_libero_client.py speaks:

    client -> {"image": agentview_uint8_HWC, "wrist_image": wrist_uint8_HWC,
               "proprio": float32[8], "prompt": task_str}
    server -> {"action": float32[T, 7]}   # raw LIBERO env action space

Run this in the `qc` conda env (has qai_hub_models + aimet_onnx + lerobot).
The FastWAM client runs unchanged in the `libero` conda env. Usually launched
via run_libero_pi05.sh (this dir), which also drives the FastWAM sim client.

Usage (from the ai-hub-models repo root):
    conda run -n qc python experiments/libero/pi05_libero_server.py \
        --precision quantized \
        --checkpoint /home/jk656/ai-hub-models/build/pi05_mixed \
        --device cuda --port 23908 --gripper-mode direct

    # float baseline (same client, different server):
    conda run -n qc python experiments/libero/pi05_libero_server.py \
        --precision float --device cuda --port 23908 --gripper-mode direct

--precision: quantized vs float 
--component-precision: actual bitwidth precision to quantize
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import make_pi05_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

from qai_hub_models import Precision
from qai_hub_models.models.pi05.app import Pi05App, Pi05AppConfig
from qai_hub_models.models.pi05.model import (
    DEFAULT_COMPONENT_PRECISION,
    Pi05Collection,
    Pi05CollectionQuantized,
)
from qai_hub_models.models.pi05.spinquant_r1 import read_rotation_marker

# Make the local `deploy` package importable (websocket server + msgpack).
_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))
from deploy.websocket_policy_server import WebsocketPolicyServer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pi05_server")

HF_MODEL_ID = "lerobot/pi05_libero_finetuned"
DATASET_REPO_ID = "HuggingFaceVLA/libero"

_DEFAULT_PRECISION_HELP = ", ".join(
    f"{name}={precision}" for name, precision in DEFAULT_COMPONENT_PRECISION.items()
)

# pi05 camera keys (from config.input_features); the app ignores "empty_camera_0".
KEY_PRIMARY = "observation.images.image"  # agentview / base camera
KEY_WRIST = "observation.images.image2"  # wrist / eye-in-hand camera
KEY_STATE = "observation.state"  # 8-D proprio (app ignores it)


def _img_to_chw01(img_hwc: np.ndarray) -> torch.Tensor:
    """uint8 HWC [H,W,3] -> float32 CHW [1,3,H,W] in [0,1] (VISUAL=IDENTITY)."""
    arr = np.asarray(img_hwc)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


class Pi05InferenceServer:
    """Stateless policy: (image, wrist_image, proprio, prompt) -> action chunk."""

    def __init__(
        self,
        app: Pi05App,
        preprocessor,
        postprocessor,
        device: str,
        num_steps: int,
        gripper_mode: str,
        extra_image_flip: bool,
    ) -> None:
        self.app = app
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.num_steps = num_steps
        self.gripper_mode = gripper_mode
        self.extra_image_flip = extra_image_flip

    def _apply_gripper(self, action: np.ndarray) -> np.ndarray:
        g = action[..., -1]
        if self.gripper_mode == "direct":
            pass
        elif self.gripper_mode == "flip":
            g = -g
        elif self.gripper_mode == "binarize":
            g = np.sign(g)
        elif self.gripper_mode == "flip_binarize":
            g = -np.sign(g)
        else:
            raise ValueError(f"bad gripper_mode {self.gripper_mode}")
        action[..., -1] = g
        return action

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        if obs.get("reset", False):
            return {}

        agent = np.asarray(obs["image"])
        wrist = np.asarray(obs.get("wrist_image", obs["image"]))
        if self.extra_image_flip:
            agent = agent[::-1, ::-1]
            wrist = wrist[::-1, ::-1]
        proprio = np.asarray(obs["proprio"], dtype=np.float32)
        prompt = str(obs["prompt"])

        raw_batch = {
            KEY_PRIMARY: _img_to_chw01(agent).to(self.device),
            KEY_WRIST: _img_to_chw01(wrist).to(self.device),
            KEY_STATE: torch.from_numpy(proprio).float().unsqueeze(0).to(self.device),
            "task": [prompt],
        }

        batch = self.preprocessor(raw_batch)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        pred = self.app.predict_action_chunk(
            batch=batch, noise=None, num_steps=self.num_steps
        )
        actions = self.postprocessor(pred)
        if isinstance(actions, torch.Tensor):
            a = actions.detach().float().cpu().numpy()
        else:
            a = np.asarray(actions, dtype=np.float32)
        if a.ndim == 3:
            a = a[0]  # [T, D]
        a = a[:, :7].astype(np.float32)  # LIBERO env action = 7-D
        a = self._apply_gripper(a)
        return {"action": a}


def describe_rotations(checkpoint: str) -> dict[str, bool]:
    """Report the SpinQuant rotations baked into a quantized checkpoint.

    The markers written by quantize.py are the single source of truth --
    rotations live in the components' weights, so nothing needs to be enabled
    at load time. Pi05CollectionQuantized.from_pretrained refuses a set whose
    markers disagree, so this is purely for logging.
    """
    root = Path(checkpoint)
    return {
        "r1": read_rotation_marker(root / "backbone")["r1"],
        "r2": read_rotation_marker(root / "backbone")["r2"],
    }


def parse_component_precision(pairs: list[str] | None) -> dict[str, Precision]:
    """Parse ``--component-precision backbone=w4a8 ...`` into a Precision map.

    """
    parsed: dict[str, Precision] = {}
    for pair in pairs or []:
        name, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(
                f"--component-precision expects NAME=PRECISION, got {pair!r} "
                "(e.g. backbone=w4a8)."
            )
        name = name.strip()
        if name not in DEFAULT_COMPONENT_PRECISION:
            raise ValueError(
                f"Unknown component {name!r} in --component-precision. "
                f"Quantizable components: {sorted(DEFAULT_COMPONENT_PRECISION)}."
            )
        parsed[name] = Precision.parse(value.strip())
    return parsed


def build_app(
    precision: str,
    checkpoint: str,
    device: str,
    component_precision: dict[str, Precision] | None = None,
):
    logger.info("Loading PI05 policy (config/tokenizer) from %s ...", HF_MODEL_ID)
    policy = PI05Policy.from_pretrained(HF_MODEL_ID).to(device).eval()

    logger.info("Loading dataset stats from %s ...", DATASET_REPO_ID)
    ds = LeRobotDataset(DATASET_REPO_ID)
    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config=policy.config, dataset_stats=ds.meta.stats
    )

    logger.info("Loading %s collection ...", precision)
    if precision == "quantized":
        rotations = describe_rotations(checkpoint)
        logger.info(
            "Checkpoint SpinQuant rotations: r1=%s r2=%s (baked into weights).",
            rotations["r1"],
            rotations["r2"],
        )
        if component_precision:
            logger.info(
                "Component precision overrides: %s",
                {name: str(p) for name, p in component_precision.items()},
            )
        # Pass only the overrides: --component-precision overrides ckpt's markers.
        collection = Pi05CollectionQuantized.from_pretrained(
            checkpoint=checkpoint,
            host_device=device,
            component_precision=component_precision or None,
        )
        precision_info = {
            name: str(component._precision)
            for name, component in collection.components.items()
            if hasattr(component, "_precision")
        }
        logger.info("Component precision in use: %s", precision_info)
    else:
        collection = Pi05Collection.from_pretrained(host_device=device)
        precision_info = None

    app = Pi05App(
        config=Pi05AppConfig.from_policy(policy),
        **collection.components,
    ).eval()
    logger.info("Pi05App ready. image_keys=%s", app.image_keys)
    return app, preprocessor, postprocessor, precision_info


def main() -> None:
    ap = argparse.ArgumentParser(description="pi05 LIBERO inference server")
    ap.add_argument("--precision", choices=["quantized", "float"], default="quantized")
    ap.add_argument(
        "--checkpoint", default="/home/jk656/ai-hub-models/build/pi05_mixed"
    )
    ap.add_argument(
        "--component-precision",
        nargs="+",
        metavar="NAME=PRECISION",
        default=None,
        help=(
            "Override component's precision, e.g. 'backbone=w4a8'."
            "Not normally needed: quantize.py reads each component's"
            "precision in the ckpt and that is used automatically."
        ),
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=23908)
    ap.add_argument(
        "--num-steps", type=int, default=10, help="Flow-matching Euler steps"
    )
    ap.add_argument(
        "--gripper-mode",
        choices=["direct", "flip", "binarize", "flip_binarize"],
        default="direct",
        help="Transform applied to the last (gripper) action dim before returning.",
    )
    ap.add_argument(
        "--extra-image-flip",
        action="store_true",
        help="Rotate images 180 deg on the server (only if client does NOT already flip).",
    )
    ap.add_argument(
        "--use-spinquant-r1",
        action="store_true",
        help=(
            "Deprecated no-op. SpinQuant rotations are folded into the "
            "checkpoint's weights and read from its markers; nothing needs "
            "enabling at serve time."
        ),
    )
    args = ap.parse_args()

    if args.use_spinquant_r1:
        logger.warning(
            "--use-spinquant-r1 is a no-op and will be removed. Rotations are "
            "folded into the checkpoint's weights (vision projector, embedding "
            "table, backbone) and detected from its markers."
        )

    component_precision = parse_component_precision(args.component_precision)
    if component_precision and args.precision != "quantized":
        logger.warning(
            "--component-precision is ignored with --precision float (no "
            "QuantSim is built)."
        )

    app, preprocessor, postprocessor, precision_info = build_app(
        args.precision, args.checkpoint, args.device, component_precision
    )
    policy_server_impl = Pi05InferenceServer(
        app=app,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=args.device,
        num_steps=args.num_steps,
        gripper_mode=args.gripper_mode,
        extra_image_flip=args.extra_image_flip,
    )

    server = WebsocketPolicyServer(
        policy_server_impl,
        host=args.host,
        port=args.port,
        metadata={
            "model": "pi05",
            "precision": args.precision,
            "checkpoint": args.checkpoint if args.precision == "quantized" else "float",
            "component_precision": precision_info,
            "spinquant": (
                describe_rotations(args.checkpoint)
                if args.precision == "quantized"
                else {"r1": False, "r2": False}
            ),
            "num_steps": args.num_steps,
            "gripper_mode": args.gripper_mode,
        },
    )
    logger.info(
        "Starting pi05 server on %s:%d (precision=%s)",
        args.host,
        args.port,
        args.precision,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
