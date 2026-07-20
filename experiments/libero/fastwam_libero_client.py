"""LIBERO evaluation client for any websocket policy server speaking the
same protocol as deploy/websocket_policy_server.py (model-agnostic).

Run this in the `libero` conda env AFTER starting the policy server
(e.g. pi05_libero_server.py in the `qc` env). Does not import the policy
model itself -- all inference happens on the server side.

Usage (from this repo's root):
    conda run -n libero python experiments/libero/fastwam_libero_client.py \\
        --port 23908 \\
        --libero-benchmarks libero_10 \\
        --test-num 50 \\
        --out-dir outputs/libero_eval

Benchmarks: libero_10 | libero_goal | libero_spatial | libero_object
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import imageio
import numpy as np
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from deploy.websocket_client_policy import WebsocketClientPolicy


# ---------------------------------------------------------------------------
# Libero helpers (inlined to avoid importing libero_utils which depends on fastwam)
# ---------------------------------------------------------------------------

def _get_libero_image(obs):
    """Extract and flip agentview + wrist images from raw libero obs dict."""
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return agentview, wrist


def _quat2axisangle(quat):
    """Convert quaternion (x,y,z,w) to axis-angle. From robosuite."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _extract_proprio(obs):
    """Build [eef_pos(3), axis_angle(3), gripper(1)] proprio from obs."""
    return np.concatenate((
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    )).astype(np.float32)


def _get_libero_dummy_action():
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _make_env(benchmark_instance, task_idx, resolution=256):
    env_args = {
        "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    for attempt in range(5):
        try:
            return OffScreenRenderEnv(**env_args)
        except Exception as e:
            print(f"  Env init failed (attempt {attempt + 1}/5): {e}")
            time.sleep(5)
    raise RuntimeError(f"Failed to create LIBERO env for task_idx={task_idx}")


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env,
    init_state,
    task_prompt: str,
    model: WebsocketClientPolicy,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
) -> tuple[bool, list]:
    """Run one episode. Returns (success, list_of_agentview_frames_for_video)."""
    env.reset()
    obs = env.set_init_state(init_state)

    replay_frames = []
    pending_actions = []
    done = False

    pbar = tqdm(total=max_steps + num_steps_wait, leave=False)
    for t in range(max_steps + num_steps_wait):
        pbar.update(1)

        # Warm-up: step with no-op while robot settles
        if t < num_steps_wait:
            obs, _, done, _ = env.step(_get_libero_dummy_action())
            continue

        # Request new action chunk when buffer is empty
        if not pending_actions:
            agentview, wrist = _get_libero_image(obs)
            proprio = _extract_proprio(obs)
            response = model.infer({
                "image": agentview,
                "wrist_image": wrist,
                "proprio": proprio,
                "prompt": task_prompt,
            })
            action_chunk = response["action"]  # [T, 7] float32 numpy
            pending_actions = action_chunk[:replan_steps].tolist()

        agentview, _ = _get_libero_image(obs)
        replay_frames.append(agentview)

        obs, _, done, _ = env.step(pending_actions.pop(0))
        if done:
            break

    pbar.close()
    return bool(done), replay_frames


# ---------------------------------------------------------------------------
# Task / benchmark runner
# ---------------------------------------------------------------------------

def _max_steps_for_suite(suite_name: str) -> int:
    table = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    return table.get(suite_name, 500)


def _save_video(frames, path, fps=24):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


ALL_BENCHMARKS = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def run_benchmark(
    libero_benchmark: str,
    model: WebsocketClientPolicy,
    out_dir: str,
    test_num: int,
    replan_steps: int,
    num_steps_wait: int,
    task_range: list | None,
    save_video: bool,
) -> dict:
    """Run one benchmark suite. Returns suite-level result dict."""
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()

    if task_range is None:
        task_ids = list(range(num_tasks))
    else:
        start, end = task_range
        task_ids = list(range(start, min(end, num_tasks)))

    print(f"\nBenchmark: {libero_benchmark} | tasks: {task_ids} | episodes/task: {test_num}")

    max_steps = _max_steps_for_suite(libero_benchmark)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = []

    for task_idx in tqdm(task_ids, desc=libero_benchmark):
        task = benchmark_instance.get_task(task_idx)
        task_prompt = task.language
        init_states = benchmark_instance.get_task_init_states(task_idx)

        env = _make_env(benchmark_instance, task_idx)

        task_results = {"task_idx": task_idx, "task_prompt": task_prompt, "successes": 0, "total": test_num}
        episode_bar = tqdm(range(test_num), desc=f"task {task_idx}", leave=False)

        for ep_idx in episode_bar:
            init_state = init_states[ep_idx % init_states.shape[0]]
            success, frames = run_episode(
                env=env,
                init_state=init_state,
                task_prompt=task_prompt,
                model=model,
                replan_steps=replan_steps,
                num_steps_wait=num_steps_wait,
                max_steps=max_steps,
            )
            if success:
                task_results["successes"] += 1

            succ_so_far = task_results["successes"]
            rate = succ_so_far / (ep_idx + 1)
            episode_bar.set_postfix({"succ_rate": f"{rate:.2f}"})
            print(
                f"  task {task_idx} ep {ep_idx}: {'SUCCESS' if success else 'FAIL'} "
                f"({succ_so_far}/{ep_idx + 1})"
            )

            if save_video and frames:
                safe_prompt = task_prompt.replace(" ", "_")[:40]
                vid_path = out_root / libero_benchmark / f"task{task_idx}_{safe_prompt}" / f"ep{ep_idx}_{success}.mp4"
                _save_video(frames, vid_path)

        env.close()

        task_results["success_rate"] = task_results["successes"] / test_num
        all_results.append(task_results)

        result_path = out_root / f"{libero_benchmark}_task{task_idx}.json"
        with open(result_path, "w") as f:
            json.dump(task_results, f, indent=2)
        print(f"Task {task_idx} done: {task_results['successes']}/{test_num} = {task_results['success_rate']:.2%}")

    overall_succ = sum(r["successes"] for r in all_results)
    overall_total = sum(r["total"] for r in all_results)
    suite_rate = overall_succ / overall_total if overall_total > 0 else 0.0

    print(f"\n===== {libero_benchmark} RESULTS =====")
    for r in all_results:
        print(f"  task {r['task_idx']} ({r['task_prompt'][:50]}): {r['successes']}/{r['total']} = {r['success_rate']:.2%}")
    print(f"  Suite: {overall_succ}/{overall_total} = {suite_rate:.2%}")

    summary_path = out_root / f"{libero_benchmark}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "benchmark": libero_benchmark,
            "results": all_results,
            "overall_successes": overall_succ,
            "overall_total": overall_total,
            "success_rate": suite_rate,
        }, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    return {"successes": overall_succ, "total": overall_total, "success_rate": suite_rate}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LIBERO evaluation client (model-agnostic websocket protocol)")
    parser.add_argument(
        "--libero-benchmarks", nargs="+",
        default=["libero_10"],
        help=(
            "Benchmark suite(s) to evaluate. Use 'all' to run all four standard suites "
            "(libero_spatial, libero_object, libero_goal, libero_10). "
            "Example: --libero-benchmarks libero_spatial libero_object"
        ),
    )
    parser.add_argument("--port", type=int, default=23908)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--test-num", type=int, default=50, help="Episodes per task")
    parser.add_argument("--out-dir", type=str, default="outputs/libero_eval")
    parser.add_argument("--replan-steps", type=int, default=10, help="Actions to execute before re-planning")
    parser.add_argument("--num-steps-wait", type=int, default=30, help="No-op warm-up steps at episode start")
    parser.add_argument("--task-range", type=int, nargs=2, default=None,
                        metavar=("START", "END"), help="[start, end) task ids to evaluate")
    parser.add_argument("--save-video", action="store_true", help="Save rollout videos")
    args = parser.parse_args()

    benchmarks = ALL_BENCHMARKS if args.libero_benchmarks == ["all"] else args.libero_benchmarks

    model = WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"Connected to server. Metadata: {model.get_server_metadata()}")

    suite_results = {}
    for bm in benchmarks:
        suite_results[bm] = run_benchmark(
            libero_benchmark=bm,
            model=model,
            out_dir=args.out_dir,
            test_num=args.test_num,
            replan_steps=args.replan_steps,
            num_steps_wait=args.num_steps_wait,
            task_range=args.task_range,
            save_video=args.save_video,
        )

    if len(suite_results) > 1:
        print("\n===== CROSS-SUITE SUMMARY =====")
        for bm, r in suite_results.items():
            print(f"  {bm:20s}: {r['successes']:4d}/{r['total']:4d} = {r['success_rate']:.2%}")
        overall_succ = sum(r["successes"] for r in suite_results.values())
        overall_total = sum(r["total"] for r in suite_results.values())
        print(f"  {'Average':20s}: {overall_succ:4d}/{overall_total:4d} = {overall_succ / overall_total:.2%}")

        cross_suite_path = Path(args.out_dir) / "cross_suite_summary.json"
        with open(cross_suite_path, "w") as f:
            json.dump({
                "suites": suite_results,
                "overall_successes": overall_succ,
                "overall_total": overall_total,
                "overall_success_rate": overall_succ / overall_total,
            }, f, indent=2)
        print(f"Cross-suite summary saved to: {cross_suite_path}")


if __name__ == "__main__":
    main()
