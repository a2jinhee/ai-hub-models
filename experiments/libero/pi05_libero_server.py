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

from qai_hub_models.models.pi05.app import Pi05App, Pi05AppConfig
from qai_hub_models.models.pi05.model import Pi05Collection, Pi05CollectionQuantized

# Make the local `deploy` package importable (websocket server + msgpack).
_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))
from deploy.websocket_policy_server import WebsocketPolicyServer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pi05_server")

HF_MODEL_ID = "lerobot/pi05_libero_finetuned"
DATASET_REPO_ID = "HuggingFaceVLA/libero"

# pi05 camera keys (from config.input_features); the app ignores "empty_camera_0".
KEY_PRIMARY = "observation.images.image"   # agentview / base camera
KEY_WRIST = "observation.images.image2"    # wrist / eye-in-hand camera
KEY_STATE = "observation.state"            # 8-D proprio (app ignores it)


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

        pred = self.app.predict_action_chunk(batch=batch, noise=None, num_steps=self.num_steps)
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


def build_app(
    precision: str, checkpoint: str, device: str, use_spinquant_r1: bool = False
):
    logger.info("Loading PI05 policy (config/tokenizer) from %s ...", HF_MODEL_ID)
    policy = PI05Policy.from_pretrained(HF_MODEL_ID).to(device).eval()

    logger.info("Loading dataset stats from %s ...", DATASET_REPO_ID)
    ds = LeRobotDataset(DATASET_REPO_ID)
    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config=policy.config, dataset_stats=ds.meta.stats
    )

    if use_spinquant_r1 and precision != "quantized":
        logger.warning(
            "--use-spinquant-r1 has no effect with --precision float (the float "
            "backbone's weights are never rotated); ignoring."
        )
        use_spinquant_r1 = False

    logger.info("Loading %s collection ...", precision)
    if precision == "quantized":
        collection = Pi05CollectionQuantized.from_pretrained(
            checkpoint=checkpoint, host_device=device, use_spinquant_r1=use_spinquant_r1
        )
    else:
        collection = Pi05Collection.from_pretrained(host_device=device)

    app = Pi05App(
        config=Pi05AppConfig.from_policy(policy),
        use_spinquant_r1=use_spinquant_r1,
        **collection.components,
    ).eval()
    logger.info("Pi05App ready. image_keys=%s", app.image_keys)
    return app, preprocessor, postprocessor


def main() -> None:
    ap = argparse.ArgumentParser(description="pi05 LIBERO inference server")
    ap.add_argument("--precision", choices=["quantized", "float"], default="quantized")
    ap.add_argument("--checkpoint", default="/home/jk656/ai-hub-models/build/pi05_mixed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=23908)
    ap.add_argument("--num-steps", type=int, default=10, help="Flow-matching Euler steps")
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
            "Must match how the checkpoint's backbone was quantized "
            "(see quantize.py --use-spinquant-r1). Only applies with "
            "--precision quantized."
        ),
    )
    args = ap.parse_args()

    app, preprocessor, postprocessor = build_app(
        args.precision, args.checkpoint, args.device, args.use_spinquant_r1
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
            "use_spinquant_r1": args.use_spinquant_r1 and args.precision == "quantized",
            "num_steps": args.num_steps,
            "gripper_mode": args.gripper_mode,
        },
    )
    logger.info("Starting pi05 server on %s:%d (precision=%s)", args.host, args.port, args.precision)
    server.serve_forever()


if __name__ == "__main__":
    main()
