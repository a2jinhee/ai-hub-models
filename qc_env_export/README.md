# qc conda environment — replication bundle

Exported from `qc` on this machine (python 3.10.20). Recreates the env on another
server. Three packages are **installed from source separately** (not in the YAML):
`aimet-onnx`, `qai_hub_models`, `qai_hub_models_cli`.

## Files
- `environment.yml` — conda + pip deps, **excluding** the source-installed packages above and the machine-specific `prefix:` line. Use this to create the env.
- `environment.full.yml` — unmodified `conda env export` (reference only; its `qai_*` / `aimet-onnx` lines are NOT pip-installable because of the `+g<hash>` / `+cu121` local-version tags).
- `zz_onnxruntime_cuda_libs.sh` — activate hook that puts the pip `nvidia/*/lib` dirs on `LD_LIBRARY_PATH`. **Required** or onnxruntime-gpu silently falls back to CPU (can't find `libcudnn.so.9`).

## Steps on the other server

Assumes you've cloned your forks there, e.g. `~/ai-hub-models` (fork
`github.com/a2jinhee/ai-hub-models`) and your `aimet` fork.

```bash
# 1. Create the env
conda env create -f environment.yml          # env name from YAML: qc
conda activate qc

# 2. Install the onnxruntime cuDNN activate hook (do this while qc is active)
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cp zz_onnxruntime_cuda_libs.sh "$CONDA_PREFIX/etc/conda/activate.d/"
conda deactivate && conda activate qc        # re-activate so the hook runs

# 3. Install ai-hub-models from your fork (two editable subdir packages)
cd ~/ai-hub-models
pip install -e src
pip install -e cli

# 4. Install aimet from your fork
#    (original env used the prebuilt wheel aimet-onnx 2.26.0+cu121:
#     https://github.com/quic/aimet/releases/download/2.26.0/aimet_onnx-2.26.0+cu121-cp310-cp310-manylinux_2_34_x86_64.whl)
#    Build/install per your aimet fork's instructions, e.g.:
cd ~/aimet
pip install -e .        # or follow the repo's build steps
```

## Verify

```bash
conda activate qc
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# expect CUDAExecutionProvider in the list; if only CPU, the activate.d hook didn't run
python -c "import qai_hub_models, aimet_onnx; print('ok')"
```

## Notes
- The env pins CUDA-12 pip wheels (`nvidia-*-cu12`, `torch==2.7.1`, `onnxruntime-gpu==1.23.2`). The target server needs a compatible NVIDIA driver (CUDA 12.x capable).
- If `conda env create` is slow/conflicts on the pip section, the conda-level pins in `environment.yml` are exact-build strings; loosen them (drop the `=build` suffix) if the target has different base packages.
