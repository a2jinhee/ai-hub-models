"""FastWAM inference server for decoupled LIBERO evaluation.

Run this in the `fastwam` conda env (Terminal 1).
The client (fastwam_libero_client.py, this dir) runs in the `libero` conda
env (Terminal 2) -- see run_libero_pi05_settings.sh's sibling pi05 server
for the pi05 counterpart of this file.

This file lives in ai-hub-models but the FastWAM model code (`fastwam.*`)
and its hydra configs only exist in FASTWAM_REPO (default
/home/jk656/FastWAM-jk), so both are located there at import/run time. The
`deploy` websocket package, however, is local to this repo (ai-hub-models),
matching pi05_libero_server.py's usage.

Usage (from this repo's root):
    FASTWAM_REPO=/home/jk656/FastWAM-jk conda run -n fastwam python experiments/libero/fastwam_server.py \\
        task=libero_uncond_2cam224_1e-4 \\
        ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \\
        EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \\
        +server.port=23908
"""

import logging
import math
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

FASTWAM_REPO = Path(os.environ.get("FASTWAM_REPO", "/home/jk656/FastWAM-jk"))
if str(FASTWAM_REPO) not in sys.path:
    sys.path.insert(0, str(FASTWAM_REPO))

# Local (ai-hub-models) websocket package -- routed here instead of FastWAM's copy.
_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from deploy.websocket_policy_server import WebsocketPolicyServer
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _mixed_precision_to_dtype(mixed_precision: str) -> torch.dtype:
    key = str(mixed_precision).strip().lower()
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    return torch.float32


class FastWAMInferenceServer:
    """Stateless policy server: receives (image, wrist_image, proprio, prompt), returns action chunk."""

    def __init__(self, model: torch.nn.Module, processor: FastWAMProcessor, cfg: DictConfig):
        self.model = model
        self.processor = processor
        self.cfg = cfg

        video_size = cfg.data.train.get("video_size", [224, 224])
        self.input_h = int(video_size[0])
        self.input_w = int(video_size[1])
        self.concat = cfg.data.train.get("concat_multi_camera", "horizontal")
        self.num_cameras = processor.num_output_cameras

        action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
        self.action_horizon = (
            int(cfg.data.train.num_frames) - 1
            if action_horizon_cfg is None
            else int(action_horizon_cfg)
        )

        self.num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", 20))
        self.text_cfg_scale = float(cfg.EVALUATION.get("text_cfg_scale", 1.0))
        self.negative_prompt = str(cfg.EVALUATION.get("negative_prompt", ""))
        sigma_shift_cfg = cfg.EVALUATION.get("sigma_shift", None)
        self.sigma_shift = None if sigma_shift_cfg is None else float(sigma_shift_cfg)
        self.rand_device = str(cfg.EVALUATION.get("rand_device", "cpu"))
        self.tiled = bool(cfg.EVALUATION.get("tiled", False))
        self.binarize_gripper = bool(cfg.EVALUATION.get("binarize_gripper", False))
        self.model_device = str(cfg.EVALUATION.get("device", "cuda"))

        image_meta = processor.shape_meta["images"]
        self._image_meta = image_meta

        action_meta = processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected a single merged action key in shape_meta['action'].")
        self._action_key = action_meta[0]["key"]

        state_meta = processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected a single merged state key in shape_meta['state'].")
        self._state_key = state_meta[0]["key"]

        logger.info(
            "FastWAMInferenceServer ready: action_horizon=%d, num_inference_steps=%d, "
            "input_size=%dx%d (%d cameras, concat=%s)",
            self.action_horizon,
            self.num_inference_steps,
            self.input_h,
            self.input_w,
            self.num_cameras,
            self.concat,
        )

    def _preprocess_images(self, image: np.ndarray, wrist_image: np.ndarray) -> torch.Tensor:
        def _hw(meta):
            shape = meta["shape"]
            return int(shape[1]), int(shape[2])

        if self.num_cameras == 1:
            ph, pw = _hw(self._image_meta[0])
            rgb = _center_crop_resize(image, width=pw, height=ph)
        elif self.num_cameras == 2:
            ph, pw = _hw(self._image_meta[0])
            wh, ww = _hw(self._image_meta[1])
            primary = _center_crop_resize(image, width=pw, height=ph)
            wrist = _center_crop_resize(wrist_image, width=ww, height=wh)
            rgb = np.concatenate([primary, wrist], axis=1 if self.concat == "horizontal" else 0)
        else:
            raise ValueError(f"num_output_cameras must be 1 or 2, got {self.num_cameras}")

        x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0)
        x = x.to(device=self.model_device, dtype=self.model.torch_dtype)
        return x * (2.0 / 255.0) - 1.0

    def _normalize_proprio(self, proprio: np.ndarray) -> torch.Tensor:
        state_batch = {"state": {self._state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][self._state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        normalizer = self.processor.normalizer.normalizers["action"][self._action_key]
        action = action.to(dtype=torch.float32, device="cpu")
        return normalizer.backward(action).numpy()

    def infer(self, obs: dict) -> dict:
        if obs.get("reset", False):
            return {}

        image = np.asarray(obs["image"])
        wrist_image = np.asarray(obs.get("wrist_image", obs["image"]))
        proprio = np.asarray(obs["proprio"], dtype=np.float32)
        task_lang = str(obs["prompt"])
        prompt = DEFAULT_PROMPT.format(task=task_lang)

        input_image = self._preprocess_images(image, wrist_image)
        proprio_tensor = self._normalize_proprio(proprio)

        with torch.no_grad():
            pred = self.model.infer_action(
                prompt=prompt,
                input_image=input_image,
                action_horizon=self.action_horizon,
                proprio=proprio_tensor,
                negative_prompt=self.negative_prompt,
                text_cfg_scale=self.text_cfg_scale,
                num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift,
                rand_device=self.rand_device,
                tiled=self.tiled,
            )

        action = pred["action"]  # [T, D]
        action = self._denormalize_action(action)[0]  # [T, D] numpy

        # Undo the dataloader's gripper sign flip (0=close,1=open -> -1=open,+1=close for env)
        action[..., -1] = action[..., -1] * 2 - 1
        action[..., -1] = action[..., -1] * -1.0
        if self.binarize_gripper:
            action[..., -1] = np.sign(action[..., -1])

        return {"action": action.astype(np.float32)}


@hydra.main(version_base="1.3", config_path=str(FASTWAM_REPO / "configs"), config_name="sim_libero.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None. Pass ckpt=<path>.")

    try:
        port = int(OmegaConf.select(cfg, "server.port", default=23908))
        host = str(OmegaConf.select(cfg, "server.host", default="0.0.0.0"))
    except Exception:
        port = int(os.environ.get("FASTWAM_SERVER_PORT", 23908))
        host = os.environ.get("FASTWAM_SERVER_HOST", "0.0.0.0")

    model_device = str(cfg.EVALUATION.get("device", "cuda"))
    model_dtype = _mixed_precision_to_dtype(cfg.get("mixed_precision", "bf16"))

    logger.info("Loading FastWAM model from checkpoint: %s", cfg.ckpt)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    model.load_checkpoint(str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = None
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    if explicit is not None:
        dataset_stats_path = Path(os.path.expanduser(os.path.expandvars(str(explicit))))
    else:
        ckpt_path = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
        for parent in list(ckpt_path.parents)[:4]:
            candidate = parent / "dataset_stats.json"
            if candidate.exists():
                dataset_stats_path = candidate
                break

    if dataset_stats_path is None or not dataset_stats_path.exists():
        raise FileNotFoundError(
            "Cannot find dataset_stats.json. Pass EVALUATION.dataset_stats_path=<path>."
        )

    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logger.info("Loaded dataset stats from: %s", dataset_stats_path)

    policy = FastWAMInferenceServer(model, processor, cfg)
    server = WebsocketPolicyServer(
        policy,
        host=host,
        port=port,
        metadata={"model": "fastwam", "action_horizon": policy.action_horizon},
    )
    logger.info("Starting FastWAM server on %s:%d", host, port)
    server.serve_forever()


if __name__ == "__main__":
    main()
