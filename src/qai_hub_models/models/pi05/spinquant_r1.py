# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
SpinQuant R1 rotation helper for Pi05's backbone.

`Pi05PaliGemmaBackboneQuantizable` takes an already-embedded `hidden_state`
as input rather than raw token ids (see Pi05PaliGemmaTokenEmbed), so R1
rotation of the backbone's internal weights (done via
`aimet_onnx.experimental.spinquant.apply_spinquant` in make_quant_sim)
requires `hidden_state` to arrive pre-rotated: hidden_state_rot = hidden_state @ R1.

Rather than rotating the weights that produce hidden_state (the vision
projector and the language embedding table, which live in components
that are shared -- via Pi05's lru_cache'd load_checkpoint -- with other,
unrotated components in the same process), we rotate the activation
directly wherever hidden_state is produced. This is mathematically
equivalent: rotating a writing layer's weight by R1 and then evaluating
it is identical to evaluating the unrotated layer and rotating its
output by R1, and R1 (acting on the hidden axis) commutes with
concatenation along the sequence axis. So rotating the vision projector
and embedding table weights individually and rotating their
concatenated output afterward produce the same result -- we just do the
latter, in app.py, since it touches no shared checkpoint state.
"""

from __future__ import annotations

import onnx
import torch
from aimet_onnx.experimental.spinquant.model_analysis import (
    get_decoder_block_boundaries,
    get_decoder_role_map,
    infer_hidden_size,
)
from aimet_onnx.experimental.spinquant.passes import R1RotationPass, SpinquantContext
from aimet_onnx.experimental.spinquant.transforms import hadamard_rotation_matrix
from aimet_onnx.meta.connectedgraph import ConnectedGraph


def get_r1_matrix(hidden_size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the R1 = H / sqrt(hidden_size) rotation matrix used throughout Pi05 R1."""
    return torch.from_numpy(hadamard_rotation_matrix(hidden_size)).to(dtype)


def rotate_activation(x: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
    """Rotate the last (hidden) dimension of an activation tensor: x @ R1."""
    return x @ R1.to(dtype=x.dtype, device=x.device)


def apply_backbone_r1(onnx_model: onnx.ModelProto) -> None:
    """
    Apply SpinQuant R1 to the backbone's internal weights (qkv, o_proj,
    gate_up, down_proj across all layers), in place.

    This calls the same analysis/rotation AIMET's public
    aimet_onnx.experimental.spinquant.apply_spinquant uses, but without its
    embedding/visual-model handling: Pi05's backbone graph has no in-graph
    embed_tokens or lm_head (it takes a pre-embedded hidden_state as
    input), so apply_spinquant's mandatory embedding-consistency check
    doesn't apply here -- the equivalent rotation of hidden_state is
    applied as an activation instead (see rotate_activation and its
    module docstring).
    """
    cg = ConnectedGraph(onnx_model)
    boundaries, active_norms = get_decoder_block_boundaries(onnx_model, cg)
    role_map = get_decoder_role_map(cg, boundaries, active_norms)
    hidden_size = infer_hidden_size(onnx_model, role_map)
    ctx = SpinquantContext(
        backbone_model=onnx_model,
        backbone_role_map=role_map,
        backbone_active_norms=active_norms,
        backbone_hidden_size=hidden_size,
    )
    rotation = R1RotationPass()
    rotation.validate(ctx)
    rotation.apply(ctx)
