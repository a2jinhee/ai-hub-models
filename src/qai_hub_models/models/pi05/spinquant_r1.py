# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
SpinQuant R1 / R2 rotation helpers for Pi05's backbone.

(The module keeps its historical `spinquant_r1` name; it now covers R2 too.)

`Pi05PaliGemmaBackboneQuantizable` takes an already-embedded `hidden_state`
as input rather than raw token ids (see Pi05PaliGemmaTokenEmbed), so R1
rotation of the backbone's internal weights (done via
`aimet_onnx.experimental.spinquant.apply_spinquant` in make_quant_sim)
requires `hidden_state` to arrive pre-rotated: hidden_state_rot = hidden_state @ R1.

That rotation is folded into the weights that *produce* hidden_state, which
is what AIMET does for the equivalent VLM topology (a backbone exported with
use_inputs_embeds=True): see `_rotate_merger_linear2` and
`_rotate_external_embedding` in aimet_onnx's R1RotationPass. hidden_state is

    prefix_emb = concat([img_embed, ..., lang_emb], dim=seq)

and R1 acts on the hidden axis, so it commutes with the sequence-axis concat.
Rotating each producer independently therefore yields a rotated concatenation:

  1. vision projector   -- apply_vision_r1, folds R1 into the multi_modal
     projector's weight *and* bias (the last MatMul+Add of the vision ONNX)
  2. language embedding -- rotate_embedding_weight, W <- W @ R1 on the
     [vocab, hidden] table, folded at Pi05PaliGemmaTokenEmbed build time
  3. zero padding for absent cameras -- 0 @ R1 == 0, nothing to do

Folding rather than rotating the activation at runtime matters for the device
export: an activation rotation is a [968, 2048] @ [2048, 2048] matmul that
lives in no exported graph, so it would land on the host between two NPU
components -- exactly the online-rotation cost R3/R4 were dropped to avoid.
Folding also puts R1 *before* the vision encoder's output quantizer, so the
vision->backbone activation is calibrated on rotated (outlier-suppressed)
values instead of unrotated ones.

The shared, lru_cache'd `load_checkpoint` policy is never mutated: the vision
rotation edits the exported ONNX ModelProto, and the embedding rotation is
held as a buffer on Pi05PaliGemmaTokenEmbed rather than written back into the
shared nn.Embedding.

R2 (per-head V/O rotation) carries an extra obligation on Pi05 that it does
not carry on a plain LLM. R2's correctness argument is that the rotation of
V's output channels cancels against the inverse rotation of o_proj's input
rows, because attention output is linear in V along head_dim. That
cancellation only covers the path *through* o_proj. Pi05's backbone is a
prefill KV *producer*: every block's V also leaves the graph as a per-layer
cache that the action expert consumes, and on that path nothing cancels.

So a backbone rotated by R2 is only correct when paired with an action expert
that compensates. Verified numerically, the compensation is three-part:

  1. backbone V rotated by R2      (apply_backbone_rotations, enable_r2=True)
  2. expert v_proj_sha rotated     (apply_action_expert_r2)
  3. expert o_proj absorbs R2.T    (apply_action_expert_r2)

Step 2 is required, not optional: SHAGemmaExpertAttention concatenates the
prefix v_cache with its own v_proj_sha output before attention, so absorbing
R2.T into o_proj alone un-rotates both halves and corrupts the suffix (a
39% error, versus 3.6e-07 when all three are applied).

Because of this coupling, an R2 backbone and its action expert must be built
and served as a pair. Each quantized component records which rotations it was
built with (see SPINQUANT_MARKER), and the deployment path refuses a mismatch.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import onnx
import torch
from aimet_onnx.experimental.spinquant.model_analysis import (
    DecoderModelRoleMap,
    attention_topology,
    get_decoder_block_boundaries,
    get_decoder_role_map,
    infer_hidden_size,
)
from aimet_onnx.experimental.spinquant.passes import (
    R1RotationPass,
    R2RotationPass,
    SpinquantContext,
)

# Private, but deliberate: reusing r2's own axis helper keeps the terminal
# v_proj below on exactly the same storage/axis convention as the pass.
from aimet_onnx.experimental.spinquant.passes.r2 import _get_rotated_axis_size
from aimet_onnx.experimental.spinquant.transforms import (
    block_diag_repeat,
    hadamard_rotation_matrix,
    rotate_linear_weight,
)
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.meta.operations import Op

# AIMET matches V projections by module name against
# `^(v_proj|value|v)(_sha)?(\.\d+)?$`, which accepts a dot-separated head index
# (`v_proj.1`) but not the underscore form Pi05's exporter emits (`v_proj_1`).
# Widened here to accept both. Written against aimet-onnx 2.34.0.
_V_MODULE_PATTERN_PI05 = re.compile(r"^(v_proj|value|v)(_sha)?([._]\d+)?$")

# Action expert op names: `/self_attn/v_proj_sha.<head>_<layer>/MatMul` and
# `/self_attn/o_proj_<layer>/MatMul`, where layer 0 carries no `_<layer>`
# suffix. The expert is a cross-attention stack with no decoder-block
# structure for AIMET's role map to walk, so its ops are selected by name.
_EXPERT_V_PATTERN = re.compile(r"^/self_attn/v_proj_sha\.\d+(_\d+)?/MatMul$")
_EXPERT_O_PATTERN = re.compile(r"^/self_attn/o_proj(_\d+)?/MatMul$")

# Records which rotations a quantized backbone's weights were baked with, so
# the deployment path can refuse a backbone/action-expert pair that disagree.
SPINQUANT_MARKER = "spinquant_rotations.json"


def write_rotation_marker(checkpoint_dir: str | Path, *, r1: bool, r2: bool) -> Path:
    """Record the rotations baked into a quantized checkpoint."""
    path = Path(checkpoint_dir) / SPINQUANT_MARKER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"r1": r1, "r2": r2}, indent=2) + "\n")
    return path


def read_rotation_marker(checkpoint_dir: str | Path) -> dict[str, bool]:
    """
    Return the rotations baked into a checkpoint.

    Missing marker means the checkpoint predates rotation tracking, which for
    every checkpoint built so far means no rotation was baked in.
    """
    path = Path(checkpoint_dir) / SPINQUANT_MARKER
    if not path.is_file():
        return {"r1": False, "r2": False}
    data = json.loads(path.read_text())
    return {"r1": bool(data.get("r1", False)), "r2": bool(data.get("r2", False))}


def get_r1_matrix(hidden_size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the R1 = H / sqrt(hidden_size) rotation matrix used throughout Pi05 R1."""
    return torch.from_numpy(hadamard_rotation_matrix(hidden_size)).to(dtype)


def get_r2_matrix(head_dim: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Return the R2 = H / sqrt(head_dim) per-head rotation matrix.

    Same construction as R1; the two differ only in the axis they act on --
    R1 on the residual stream (hidden_size), R2 per head (head_dim).
    """
    return torch.from_numpy(hadamard_rotation_matrix(head_dim)).to(dtype)


def rotate_activation(x: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
    """
    Rotate the last (hidden) dimension of an activation tensor: x @ R1.

    Only used on the host-side calibration path, where the float (unrotated)
    vision encoder and token embedder build backbone calibration inputs. The
    deployed path has R1 folded into those components' weights instead -- see
    apply_vision_r1 and rotate_embedding_weight.
    """
    return x @ R1.to(dtype=x.dtype, device=x.device)


def rotate_embedding_weight(weight: torch.Tensor) -> torch.Tensor:
    """
    Return the language embedding table rotated by R1: ``W @ R1``.

    Mirrors AIMET's `_rotate_external_embedding`, which rotates the embedding
    tensor of a VLM backbone exported with use_inputs_embeds=True. Pi05 keeps
    its embedding inside Pi05PaliGemmaTokenEmbed rather than in a separate
    embedding.pth, so the rotated table is returned for the caller to hold as a
    buffer -- the shared checkpoint's nn.Embedding is left untouched.

    Accumulates in float64 before casting back, matching AIMET's convention for
    weight-space rotations.

    Parameters
    ----------
    weight
        Embedding table of shape [vocab, hidden].

    Returns
    -------
    rotated : torch.Tensor
        ``weight @ R1``, same shape and dtype as ``weight``.
    """
    if weight.ndim != 2:
        raise ValueError(
            f"Embedding table must be [vocab, hidden], got {weight.shape}."
        )
    R1 = get_r1_matrix(weight.shape[-1], dtype=torch.float64).to(weight.device)
    return (weight.detach().to(torch.float64) @ R1).to(weight.dtype)


def _find_output_projection(onnx_model: onnx.ModelProto, cg: ConnectedGraph) -> Op:
    """
    Return the linear op that writes the vision encoder's single graph output.

    Walks back from the graph output through an optional bias `Add` to the
    producing MatMul/Gemm/Conv. AIMET's own `find_merger_linear2` matches Qwen
    PatchMerger module names, which Pi05's PaliGemma projector does not use, so
    the op is located structurally instead.
    """
    outputs = onnx_model.graph.output
    if len(outputs) != 1:
        raise ValueError(
            "apply_vision_r1 expects the vision encoder to have exactly one "
            f"graph output, found {[o.name for o in outputs]}."
        )
    producers = {out: n for n in onnx_model.graph.node for out in n.output}
    initializers = {i.name for i in onnx_model.graph.initializer}

    node = producers.get(outputs[0].name)
    if node is None:
        raise ValueError(
            f"Vision encoder output '{outputs[0].name}' has no producing node."
        )
    if node.op_type == "Add":
        # MatMul + bias Add: step back to the MatMul on the non-static input.
        activations = [i for i in node.input if i not in initializers]
        if len(activations) != 1:
            raise ValueError(
                f"Vision encoder output Add '{node.name}' should have exactly one "
                f"non-static input, found {activations}."
            )
        node = producers.get(activations[0])

    if node is None or node.op_type not in ("MatMul", "Gemm", "Conv"):
        found = node.op_type if node is not None else "None"
        raise ValueError(
            "Cannot locate the vision encoder's output projection: expected a "
            f"MatMul/Gemm/Conv writing the graph output, found '{found}'."
        )
    return cg.get_op_from_module_name(node.name)


def apply_vision_r1(
    onnx_model: onnx.ModelProto, *, hidden_size: int | None = None
) -> str:
    """
    Fold R1 into the vision encoder's output projection, in place.

    The vision encoder already includes PaliGemma's multi_modal_projector, so
    its graph output `img_embed` is in backbone hidden space. Rotating that
    projection as a writing layer (``W <- W @ R1``, and the bias likewise) makes
    the encoder emit embeddings already in the rotated residual stream, so the
    backbone's rotated weights consume them directly.

    This is the Pi05 analogue of AIMET's `_rotate_merger_linear2`, and uses
    AIMET's own `rotate_linear_weight`, which rotates the bias too when
    is_writing=True (its `get_bias_product` handles the MatMul-then-Add form
    this projector exports as).

    Parameters
    ----------
    onnx_model
        Vision encoder ONNX model. Mutated in place.
    hidden_size
        Expected backbone hidden size. Inferred from the projection's output
        axis when omitted; when given, a mismatch raises rather than silently
        rotating the wrong axis.

    Returns
    -------
    op_name : str
        Name of the op that was rotated.
    """
    cg = ConnectedGraph(onnx_model)
    op = _find_output_projection(onnx_model, cg)
    out_size = _get_rotated_axis_size(onnx_model, op, is_writing=True)
    if hidden_size is not None and out_size != hidden_size:
        raise ValueError(
            f"Vision projector '{op.name}' writes {out_size} channels, but the "
            f"backbone hidden size is {hidden_size}."
        )
    rotate_linear_weight(
        onnx_model, op, hadamard_rotation_matrix(out_size), is_writing=True
    )
    return op.name


@contextmanager
def _pi05_v_naming() -> Iterator[None]:
    """Temporarily widen AIMET's V allow-list to accept Pi05's `v_proj_1` names.

    Patching the module global is the least invasive option: it leaves the
    installed aimet_onnx untouched (so nothing else in the env changes
    behaviour) and is reverted on exit even if the pass raises.
    """
    original = attention_topology._V_MODULE_PATTERN
    attention_topology._V_MODULE_PATTERN = _V_MODULE_PATTERN_PI05
    try:
        yield
    finally:
        attention_topology._V_MODULE_PATTERN = original


def _is_v_projection(op_name: str) -> bool:
    """True iff `op_name`'s containing module name is a V projection."""
    parts = op_name.rsplit("/", 2)
    return len(parts) >= 2 and bool(_V_MODULE_PATTERN_PI05.match(parts[-2]))


def infer_backbone_head_dim(onnx_model: onnx.ModelProto) -> int:
    """
    Return head_dim for Pi05's backbone, read off the exported KV cache.

    AIMET's own infer_head_dim reads the last dimension of a `past_value`
    graph *input*, which a KV-cache-consuming decode graph has. Pi05's
    backbone is the mirror image -- a prefill graph that *produces* the cache
    -- so we read the same quantity off the rank-4 cache graph outputs
    instead. All of them must agree.
    """
    dims = set()
    for out in onnx_model.graph.output:
        shape = out.type.tensor_type.shape.dim
        if len(shape) != 4:
            continue
        last = shape[-1].dim_value
        if last > 0:
            dims.add(last)
    if len(dims) != 1:
        raise ValueError(
            "Cannot infer head_dim from the backbone's cache outputs: expected "
            f"exactly one distinct rank-4 last dimension, found {sorted(dims)}."
        )
    return dims.pop()


def _rotate_terminal_v_projections(
    onnx_model: onnx.ModelProto, role_map: DecoderModelRoleMap, head_dim: int
) -> list[str]:
    """
    Rotate V projections that the role map files under `lm_head`, in place.

    Pi05's final decoder layer produces only K/V -- it has no q_proj or o_proj,
    existing solely to hand the last layer's KV to the action expert. So
    get_decoder_role_map places its v_proj in `lm_head` rather than inside a
    block, and R2RotationPass (which only walks `blocks`) skips it. Left
    alone, 17 of the 18 exported V caches would be rotated and one would not.
    Rotating it here keeps every cache the action expert receives uniform, so
    the expert-side compensation can be applied identically to every layer.

    Parameters
    ----------
    onnx_model
        Backbone ONNX model. Mutated in place.
    role_map
        Role map produced by get_decoder_role_map.
    head_dim
        Per-head dimension of the rotated axis.

    Returns
    -------
    rotated : list[str]
        Names of the ops that were rotated.
    """
    R2 = hadamard_rotation_matrix(head_dim)
    rotated = []
    for op in role_map.lm_head:
        if not _is_v_projection(op.name):
            continue
        out_size = _get_rotated_axis_size(onnx_model, op, is_writing=True)
        if out_size % head_dim != 0:
            raise ValueError(
                f"Terminal V projection '{op.name}': output size {out_size} is "
                f"not divisible by head_dim={head_dim}."
            )
        rotate_linear_weight(
            onnx_model,
            op,
            block_diag_repeat(R2, out_size // head_dim),
            is_writing=True,
        )
        rotated.append(op.name)
    return rotated


def apply_backbone_rotations(
    onnx_model: onnx.ModelProto, *, enable_r1: bool = True, enable_r2: bool = False
) -> None:
    """
    Apply SpinQuant rotations to the backbone's internal weights, in place.

    R1 rotates the residual stream (qkv, o_proj, gate_up, down_proj across all
    layers). This calls the same analysis/rotation AIMET's public
    aimet_onnx.experimental.spinquant.apply_spinquant uses, but drives the
    passes directly rather than going through apply_spinquant: that entry point
    requires either an in-graph embed_tokens or an external embedding tensor,
    and Pi05 has neither -- its embedding lives in a separate exported
    component. The equivalent producer-side rotations are applied there instead,
    by apply_vision_r1 and rotate_embedding_weight.

    R2 additionally rotates each block's V output channels and o_proj input
    rows per head. It leaves the K caches bit-identical, but rotates the
    exported per-layer V caches -- which makes the resulting backbone valid
    only against an action expert built with apply_action_expert_r2.

    Parameters
    ----------
    onnx_model
        Backbone ONNX model. Mutated in place.
    enable_r1
        Apply the R1 residual-stream rotation (and its norm fusion). Defaults
        to True; disabling it is mainly useful for isolating R2 in tests, since
        R1 is not idempotent.
    enable_r2
        Also apply the R2 per-head V/O rotation.
    """
    cg = ConnectedGraph(onnx_model)
    boundaries, active_norms = get_decoder_block_boundaries(onnx_model, cg)
    role_map = get_decoder_role_map(cg, boundaries, active_norms)
    hidden_size = infer_hidden_size(onnx_model, role_map)
    head_dim = infer_backbone_head_dim(onnx_model) if enable_r2 else None
    ctx = SpinquantContext(
        backbone_model=onnx_model,
        backbone_role_map=role_map,
        backbone_active_norms=active_norms,
        backbone_hidden_size=hidden_size,
        backbone_head_dim=head_dim,
    )

    if enable_r1:
        rotation = R1RotationPass()
        rotation.validate(ctx)
        rotation.apply(ctx)

    if enable_r2:
        assert head_dim is not None
        with _pi05_v_naming():
            r2 = R2RotationPass()
            r2.validate(ctx)
            r2.apply(ctx)
        _rotate_terminal_v_projections(onnx_model, role_map, head_dim)


def apply_backbone_r1(onnx_model: onnx.ModelProto) -> None:
    """Apply R1 only. Thin wrapper kept for existing callers."""
    apply_backbone_rotations(onnx_model, enable_r2=False)


def infer_expert_head_dim(onnx_model: onnx.ModelProto) -> int:
    """
    Return head_dim for the action expert, read off its value_cache inputs.

    This is AIMET's own infer_head_dim semantics -- the last dimension of a
    cached-value graph input -- just under Pi05's `value_cache_*` naming
    rather than HF's `past_value`.
    """
    dims = set()
    for inp in onnx_model.graph.input:
        if "value_cache" not in inp.name:
            continue
        shape = inp.type.tensor_type.shape.dim
        if shape and shape[-1].dim_value > 0:
            dims.add(shape[-1].dim_value)
    if len(dims) != 1:
        raise ValueError(
            "Cannot infer head_dim from the action expert's value_cache inputs: "
            f"expected exactly one distinct last dimension, found {sorted(dims)}."
        )
    return dims.pop()


def apply_action_expert_r2(onnx_model: onnx.ModelProto) -> tuple[int, int]:
    """
    Apply the R2 compensation to the action expert, in place.

    Pairs with `apply_backbone_rotations(..., enable_r2=True)`: the backbone
    hands this expert per-layer V caches already rotated by R2, so o_proj must
    absorb the inverse. Its own v_proj_sha must be rotated to match, because
    SHAGemmaExpertAttention concatenates the prefix v_cache with its own
    v_proj_sha output *before* attention -- rotating o_proj alone would
    un-rotate both halves and corrupt the suffix.

    Uses the same axis conventions as AIMET's R2RotationPass (V on the writing
    axis, o_proj on the reading axis), so the two cancel exactly.

    Parameters
    ----------
    onnx_model
        Action expert ONNX model. Mutated in place.

    Returns
    -------
    counts : tuple[int, int]
        Number of (v_proj_sha, o_proj) ops rotated.
    """
    head_dim = infer_expert_head_dim(onnx_model)
    cg = ConnectedGraph(onnx_model)
    R2 = hadamard_rotation_matrix(head_dim)

    n_v = n_o = 0
    for node in onnx_model.graph.node:
        name = node.name or ""
        is_v = bool(_EXPERT_V_PATTERN.match(name))
        is_o = bool(_EXPERT_O_PATTERN.match(name))
        if not (is_v or is_o):
            continue
        op = cg.get_op_from_module_name(name)
        size = _get_rotated_axis_size(onnx_model, op, is_writing=is_v)
        if size % head_dim != 0:
            raise ValueError(
                f"Action expert op '{name}': rotated axis size {size} is not "
                f"divisible by head_dim={head_dim}."
            )
        rotate_linear_weight(
            onnx_model, op, block_diag_repeat(R2, size // head_dim), is_writing=is_v
        )
        n_v += is_v
        n_o += is_o

    if n_v == 0 or n_o == 0 or n_v != n_o:
        raise ValueError(
            "Action expert R2: expected a matching non-zero number of "
            f"v_proj_sha and o_proj ops, found v={n_v}, o={n_o}. The expert's "
            "op naming may have changed."
        )
    return n_v, n_o
