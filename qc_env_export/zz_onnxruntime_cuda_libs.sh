#!/bin/sh
# Put this env's pip-installed NVIDIA CUDA-12 libraries (cuDNN, cuBLAS, cuFFT, ...)
# on the loader path so onnxruntime-gpu can dlopen libonnxruntime_providers_cuda.so.
# Without this, ORT fails to find libcudnn.so.9 and silently falls back to CPU.
_qc_nv_libs="$(ls -d "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
if [ -n "$_qc_nv_libs" ]; then
    export LD_LIBRARY_PATH="${_qc_nv_libs}${LD_LIBRARY_PATH}"
fi
unset _qc_nv_libs
