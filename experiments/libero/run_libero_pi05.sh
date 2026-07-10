#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate Qualcomm AI Hub pi05 (quantized `build/pi05_mixed` or float) on the
# LIBERO closed-loop benchmark, reusing the FastWAM model-agnostic websocket
# client (fastwam_libero_client.py) + `libero` conda env.
#
# Architecture (two repos + two envs, one box, decoupled over a websocket):
#   * pi05 policy server  -> ai-hub-models, `qc`     env (qai_hub_models + aimet_onnx + lerobot)
#   * LIBERO sim client   -> FastWAM,       `libero` env (mujoco/robosuite rollouts)
# The server (pi05_libero_server.py) is local to this repo; the client lives in
# FastWAM, located via FASTWAM_REPO (default /home/jk656/FastWAM-jk).
# This script launches the server, waits until it is serving, runs the client,
# and always tears the server down on exit.
#
# USAGE
#   ai-hub-models/experiments/libero/run_libero_pi05.sh [PRECISION] [SUITE] [TEST_NUM] [TASK_RANGE]
#
#   PRECISION   quantized | float | both      (default: quantized)
#   SUITE       libero_object | libero_spatial | libero_goal | libero_10 | all
#                                              (default: libero_object)
#   TEST_NUM    episodes per task             (default: 5)
#   TASK_RANGE  "START END" (half-open) or "" for all tasks in the suite
#                                              (default: "0 1"  -> only task 0)
#
# Everything else is overridable via env vars (defaults in the CONFIG block):
#   SERVER_GPU PORT NUM_STEPS GRIPPER_MODE CHECKPOINT OUT_ROOT
#   QAIHM_REPO FASTWAM_REPO CONDA_QC CONDA_LIBERO REPLAN_STEPS NUM_STEPS_WAIT
#   SAVE_VIDEO READY_TIMEOUT CLIENT_GPU
#
# EXAMPLES
#   # Smoke test (validated): quantized, 1 task x 5 episodes on libero_object
#   ai-hub-models/experiments/libero/run_libero_pi05.sh quantized libero_object 5 "0 1"
#
#   # Full LIBERO reporting protocol: 50 ep/task, all 10 tasks, both precisions
#   ai-hub-models/experiments/libero/run_libero_pi05.sh both libero_object 50 ""
#
#   # A different suite on a specific server GPU
#   SERVER_GPU=3 ai-hub-models/experiments/libero/run_libero_pi05.sh quantized libero_10 20 ""
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- positional args ------------------------------------------------------
PRECISION="${1:-quantized}"      # quantized | float | both
SUITE="${2:-libero_object}"
TEST_NUM="${3:-5}"
TASK_RANGE="${4:-0 1}"           # "START END" or "" for all tasks

# ---- CONFIG (env-overridable) ---------------------------------------------
# This script lives in ai-hub-models and drives two repos:
#   * pi05 server  -> ai-hub-models (QAIHM_REPO), qc env      -> local pi05_libero_server.py
#   * LIBERO client -> FastWAM (FASTWAM_REPO), libero env     -> fastwam_libero_client.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QAIHM_REPO="${QAIHM_REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"   # ai-hub-models root
FASTWAM_REPO="${FASTWAM_REPO:-/home/jk656/FastWAM-jk}"           # repo with the LIBERO sim client
CHECKPOINT="${CHECKPOINT:-${QAIHM_REPO}/build/pi05_mixed}"
CONDA_QC="${CONDA_QC:-qc}"           # env with qai_hub_models + aimet_onnx
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
OUT_ROOT="${OUT_ROOT:-${FASTWAM_REPO}/outputs/pi05_libero}"

SERVER_PY="${SCRIPT_DIR}/pi05_libero_server.py"                          # local (this repo, qc env)
CLIENT_PY="${FASTWAM_REPO}/experiments/libero/fastwam_libero_client.py"  # FastWAM (libero env)

# Run from the FastWAM repo root: the LIBERO client imports fastwam/experiments modules.
cd "$FASTWAM_REPO"
mkdir -p "$OUT_ROOT"

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

run_one() {
  local prec="$1"
  local out_dir="${OUT_ROOT}/${prec}"
  local server_log="${OUT_ROOT}/server_${prec}.log"
  mkdir -p "$out_dir"

  echo "=================================================================="
  echo "[run] precision=${prec}  suite=${SUITE}  test_num=${TEST_NUM}  task_range='${TASK_RANGE}'"
  echo "[run] server_gpu=${SERVER_GPU}  port=${PORT}  num_steps=${NUM_STEPS}  gripper=${GRIPPER_MODE}"
  echo "[run] out_dir=${out_dir}  server_log=${server_log}"
  echo "=================================================================="

  # 1) launch the pi05 policy server (qc env) in the background
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
    conda run --no-capture-output -n "${CONDA_QC}" python "${SERVER_PY}" \
      --precision "${prec}" \
      --checkpoint "${CHECKPOINT}" \
      --device cuda \
      --port "${PORT}" \
      --num-steps "${NUM_STEPS}" \
      --gripper-mode "${GRIPPER_MODE}" \
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

  # 4) tear the server down before the next precision
  cleanup; SERVER_PID=""

  echo "[run] precision=${prec} DONE. Summary: ${out_dir}/${SUITE}_summary.json"
}

# ---- dispatch -------------------------------------------------------------
case "${PRECISION}" in
  quantized|float) run_one "${PRECISION}" ;;
  both)            run_one quantized; run_one float ;;
  *) echo "bad PRECISION '${PRECISION}' (use quantized|float|both)"; exit 2 ;;
esac

echo "[run] ALL DONE. Results under: ${OUT_ROOT}"
