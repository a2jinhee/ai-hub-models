#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Evaluate pi05 on LIBERO across three quantization settings:
#
#   float    - unquantized baseline (Pi05Collection)
#   seqmse   - seq-MSE quantized vision_encoder + action_expert + backbone
#   mixed    - seq-MSE quantized vision_encoder + action_expert,
#              seq-MSE + SpinQuant (R1) quantized backbone (llm_backbone)
#
# CHECKPOINT_BASE (default build/pi05_mixed) is expected to contain:
#   vision_encoder/   action_expert/   backbone/
# `backbone` here is the seq-MSE-only backbone. The seq-MSE + SpinQuant
# backbone lives separately under SPIN_CHECKPOINT_BASE (default
# build/pi05_mixed_spin) as `backbone/`. Since pi05_libero_server.py always
# loads a component named `backbone` from the checkpoint dir, this script
# stages a small directory of symlinks per setting (vision_encoder/
# action_expert shared from CHECKPOINT_BASE, backbone pointed at the right
# variant) and passes that as --checkpoint.
#
# Both pieces are local to this repo (ai-hub-models), two conda envs:
#   * pi05 policy server  -> `qc`     env -> pi05_libero_server.py
#   * LIBERO sim client   -> `libero` env -> fastwam_libero_client.py (model-agnostic)
#
# USAGE
#   ai-hub-models/experiments/libero/run_libero_pi05_settings.sh [SETTING] [SUITE] [TEST_NUM] [TASK_RANGE]
#
#   SETTING     float | seqmse | mixed | all       (default: all)
#   SUITE       libero_object | libero_spatial | libero_goal | libero_10 | all
#                                                   (default: libero_object)
#   TEST_NUM    episodes per task                  (default: 5)
#   TASK_RANGE  "START END" (half-open) or "" for all tasks in the suite
#                                                   (default: "0 1"  -> only task 0)
#
# Everything else is overridable via env vars (defaults in the CONFIG block):
#   SERVER_GPU PORT NUM_STEPS GRIPPER_MODE CHECKPOINT_BASE OUT_ROOT
#   QAIHM_REPO FASTWAM_REPO CONDA_QC CONDA_LIBERO REPLAN_STEPS NUM_STEPS_WAIT
#   SAVE_VIDEO READY_TIMEOUT CLIENT_GPU
#
# EXAMPLES
#   # Smoke test: mixed setting, 1 task x 5 episodes on libero_object
#   ai-hub-models/experiments/libero/run_libero_pi05_settings.sh mixed libero_object 5 "0 1"
#
#   # Full LIBERO reporting protocol: all 3 settings, 50 ep/task, all 10 tasks
#   ai-hub-models/experiments/libero/run_libero_pi05_settings.sh all libero_object 50 ""
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- positional args ------------------------------------------------------
SETTING="${1:-all}"              # float | seqmse | mixed | all
SUITE="${2:-libero_object}"
TEST_NUM="${3:-5}"
TASK_RANGE="${4:-0 1}"           # "START END" or "" for all tasks

# ---- CONFIG (env-overridable) ---------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QAIHM_REPO="${QAIHM_REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"   # ai-hub-models root
FASTWAM_REPO="${FASTWAM_REPO:-/home/jk656/FastWAM-jk}"           # repo with the LIBERO sim client
CHECKPOINT_BASE="${CHECKPOINT_BASE:-${QAIHM_REPO}/build/pi05_mixed}"
SPIN_CHECKPOINT_BASE="${SPIN_CHECKPOINT_BASE:-${QAIHM_REPO}/build/pi05_mixed_spin}"  # holds the seq-MSE + SpinQuant backbone/
CONDA_QC="${CONDA_QC:-qc}"           # env with qai_hub_models + aimet_onnx + lerobot
CONDA_LIBERO="${CONDA_LIBERO:-libero}"  # env with the LIBERO simulator

SERVER_GPU="${SERVER_GPU:-2}"        # GPU the pi05 server runs on
CLIENT_GPU="${CLIENT_GPU:-}"         # optional: pin sim/rendering GPU (empty = default)
PORT="${PORT:-23908}"
NUM_STEPS="${NUM_STEPS:-10}"         # flow-matching Euler steps
GRIPPER_MODE="${GRIPPER_MODE:-direct}"   # validated: predictions match GT gripper
REPLAN_STEPS="${REPLAN_STEPS:-10}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-30}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"        # 1 -> pass --save-video (default off; set SAVE_VIDEO=1 to record)
READY_TIMEOUT="${READY_TIMEOUT:-600}"    # seconds to wait for server load
OUT_ROOT="${OUT_ROOT:-${QAIHM_REPO}/outputs/pi05_libero}"
STAGE_ROOT="${OUT_ROOT}/_ckpt_staging"

SERVER_PY="${SCRIPT_DIR}/pi05_libero_server.py"    # local (this repo, qc env)
CLIENT_PY="${SCRIPT_DIR}/fastwam_libero_client.py" # local (this repo, libero env) -- model-agnostic

mkdir -p "$OUT_ROOT" "$STAGE_ROOT"

# ---- helpers --------------------------------------------------------------
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run] stopping pi05 server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  # Ready == the server has finished loading pi05 and is serving. The server is
  # idle during this wait (client not started yet), so /healthz responds fast.
  local log="$1" deadline=$(( SECONDS + READY_TIMEOUT ))
  echo "[run] waiting up to ${READY_TIMEOUT}s for server on port ${PORT} (loading pi05 ~3 min)..."
  while (( SECONDS < deadline )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[run] ERROR: server process died during load. Last log lines:"; tail -30 "$log"; return 1
    fi
    if curl -sf -m 5 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
      echo "[run] server is serving on port ${PORT}."; return 0
    fi
    sleep 5
  done
  echo "[run] ERROR: server not ready within ${READY_TIMEOUT}s. Last log lines:"; tail -30 "$log"; return 1
}

# stage_checkpoint DEST BACKBONE_SUBDIR [BACKBONE_BASE]
# Builds a symlink tree DEST/{vision_encoder,action_expert,backbone}.
# vision_encoder/action_expert are always taken from CHECKPOINT_BASE; the
# backbone is taken from BACKBONE_BASE/BACKBONE_SUBDIR (BACKBONE_BASE defaults
# to CHECKPOINT_BASE -- used by the seq-MSE-only setting; the SpinQuant setting
# passes SPIN_CHECKPOINT_BASE). This is needed because Pi05CollectionQuantized
# always loads a component literally named `backbone` from the checkpoint dir.
stage_checkpoint() {
  local dest="$1" backbone_subdir="$2" backbone_base="${3:-${CHECKPOINT_BASE}}"
  local src_backbone="${backbone_base}/${backbone_subdir}"
  for part in vision_encoder action_expert; do
    if [[ ! -d "${CHECKPOINT_BASE}/${part}" ]]; then
      echo "[run] ERROR: missing ${CHECKPOINT_BASE}/${part}" >&2; return 1
    fi
  done
  if [[ ! -d "$src_backbone" ]]; then
    echo "[run] ERROR: backbone checkpoint dir not found: ${src_backbone}" >&2
    echo "[run]        (expected under BACKBONE_BASE=${backbone_base})" >&2
    return 1
  fi
  mkdir -p "$dest"
  ln -sfn "${CHECKPOINT_BASE}/vision_encoder" "${dest}/vision_encoder"
  ln -sfn "${CHECKPOINT_BASE}/action_expert" "${dest}/action_expert"
  ln -sfn "${src_backbone}" "${dest}/backbone"
}

run_one() {
  local setting="$1"
  local prec spin backbone_subdir checkpoint
  case "$setting" in
    float)
      prec=float; spin=0; checkpoint="${CHECKPOINT_BASE}"  # unused by server for float
      ;;
    seqmse)
      prec=quantized; spin=0; backbone_subdir="backbone"
      checkpoint="${STAGE_ROOT}/seqmse"
      stage_checkpoint "$checkpoint" "$backbone_subdir"
      ;;
    mixed)
      prec=quantized; spin=1; backbone_subdir="backbone"
      checkpoint="${STAGE_ROOT}/mixed"
      stage_checkpoint "$checkpoint" "$backbone_subdir" "$SPIN_CHECKPOINT_BASE"
      ;;
    *) echo "bad SETTING '${setting}' (use float|seqmse|mixed|all)"; return 2 ;;
  esac

  local out_dir="${OUT_ROOT}/${setting}"
  local server_log="${OUT_ROOT}/server_${setting}.log"
  mkdir -p "$out_dir"

  echo "=================================================================="
  echo "[run] setting=${setting}  precision=${prec}  spinquant_r1=${spin}  suite=${SUITE}  test_num=${TEST_NUM}  task_range='${TASK_RANGE}'"
  echo "[run] checkpoint=${checkpoint}"
  echo "[run] server_gpu=${SERVER_GPU}  port=${PORT}  num_steps=${NUM_STEPS}  gripper=${GRIPPER_MODE}"
  echo "[run] out_dir=${out_dir}  server_log=${server_log}"
  echo "=================================================================="

  # 1) launch the pi05 policy server (qc env) in the background
  local server_args=(
    --precision "${prec}"
    --checkpoint "${checkpoint}"
    --device cuda
    --port "${PORT}"
    --num-steps "${NUM_STEPS}"
    --gripper-mode "${GRIPPER_MODE}"
  )
  [[ "${spin}" == "1" ]] && server_args+=(--use-spinquant-r1)

  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
    conda run --no-capture-output -n "${CONDA_QC}" python "${SERVER_PY}" \
      "${server_args[@]}" \
      > "${server_log}" 2>&1 &
  SERVER_PID=$!
  echo "[run] server pid=${SERVER_PID}"

  # 2) wait until it is serving (or die trying)
  wait_for_server "${server_log}"

  # 3) run the LIBERO client (libero env) in the foreground
  local client_args=(
    --libero-benchmarks "${SUITE}"
    --port "${PORT}"
    --test-num "${TEST_NUM}"
    --replan-steps "${REPLAN_STEPS}"
    --num-steps-wait "${NUM_STEPS_WAIT}"
    --out-dir "${out_dir}"
  )
  if [[ -n "${TASK_RANGE// }" ]]; then
    # shellcheck disable=SC2206
    local tr=(${TASK_RANGE}); client_args+=(--task-range "${tr[0]}" "${tr[1]}")
  fi
  [[ "${SAVE_VIDEO}" == "1" ]] && client_args+=(--save-video)

  echo "[run] starting LIBERO client..."
  # IMPORTANT: only export CUDA_VISIBLE_DEVICES when CLIENT_GPU is set. Setting
  # it to an empty string breaks mujoco's EGL device parsing (int("") error).
  if [[ -n "${CLIENT_GPU}" ]]; then
    CUDA_VISIBLE_DEVICES="${CLIENT_GPU}" \
      conda run --no-capture-output -n "${CONDA_LIBERO}" python "${CLIENT_PY}" "${client_args[@]}"
  else
    conda run --no-capture-output -n "${CONDA_LIBERO}" python "${CLIENT_PY}" "${client_args[@]}"
  fi

  # 4) tear the server down before the next setting
  cleanup; SERVER_PID=""

  echo "[run] setting=${setting} DONE. Summary: ${out_dir}/${SUITE}_summary.json"
}

# ---- dispatch -------------------------------------------------------------
case "${SETTING}" in
  float|seqmse|mixed) run_one "${SETTING}" ;;
  all)                 run_one float; run_one seqmse; run_one mixed ;;
  *) echo "bad SETTING '${SETTING}' (use float|seqmse|mixed|all)"; exit 2 ;;
esac

echo "[run] ALL DONE. Results under: ${OUT_ROOT}"
