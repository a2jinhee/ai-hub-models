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
import types
from collections import deque
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
    insert_online_hadamard_node,
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
_EXPERT_Q_PATTERN = re.compile(r"^/self_attn/q_proj_sha\.\d+(_\d+)?/MatMul$")
_EXPERT_K_PATTERN = re.compile(r"^/self_attn/k_proj_sha\.\d+(_\d+)?/MatMul$")

# The expert's MLP keeps HF's plain GemmaMLP (only the backbone's is replaced
# with GemmaMLPSplitLinear), so its down_proj exports one MatMul per layer as
# `/mlp/down_proj_<layer>/MatMul`, layer 0 again carrying no `_<layer>` suffix.
_EXPERT_DOWN_PATTERN = re.compile(r"^/mlp/down_proj(_\d+)?/MatMul$")

# Records which rotations a quantized backbone's weights were baked with, so
# the deployment path can refuse a backbone/action-expert pair that disagree.
SPINQUANT_MARKER = "spinquant_rotations.json"


def write_rotation_marker(
    checkpoint_dir: str | Path,
    *,
    r1: bool,
    r2: bool,
    r3: bool = False,
    r4: bool = False,
) -> Path:
    """Record the rotations baked into a quantized checkpoint."""
    path = Path(checkpoint_dir) / SPINQUANT_MARKER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"r1": r1, "r2": r2, "r3": r3, "r4": r4}, indent=2) + "\n"
    )
    return path


def read_rotation_marker(checkpoint_dir: str | Path) -> dict[str, bool]:
    """
    Return the rotations baked into a checkpoint.

    Missing marker means the checkpoint predates rotation tracking, which for
    every checkpoint built so far means no rotation was baked in.
    """
    path = Path(checkpoint_dir) / SPINQUANT_MARKER
    if not path.is_file():
        return {"r1": False, "r2": False, "r3": False, "r4": False}
    data = json.loads(path.read_text())
    return {
        "r1": bool(data.get("r1", False)),
        "r2": bool(data.get("r2", False)),
        "r3": bool(data.get("r3", False)),
        "r4": bool(data.get("r4", False)),
    }


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


def get_r3_matrix(head_dim: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Return the R3 = H / sqrt(head_dim) per-head rotation matrix.

    Numerically identical to get_r2_matrix -- kept separately
    because R2 and R3 rotate different tensors (V/o_proj vs. Q/K).
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


def _index_consumers_by_tensor(
    onnx_model: onnx.ModelProto,
) -> dict[str, list[onnx.NodeProto]]:
    """Return {tensor_name: [nodes consuming it]} for every node input in the graph."""
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in onnx_model.graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return consumers


def _find_post_rope_hadamard_target(
    consumers_by_tensor: dict[str, list[onnx.NodeProto]],
    source_tensor: str,
) -> tuple[str, list[onnx.NodeProto]]:
    """
    Return the post-RoPE anchor for R3 insertion.

    - ``source_tensor`` is a q_proj/k_proj MatMul's output (pre-RoPE). 
    - HF-style rotate_half RoPE (Reshape -> Split -> two Mul/Sub/Add branches -> Concat)
    always merges back into exactly one Concat node before reaching any MatMul (QK^T)
    - verified against both the backbone's and action expert's exported graphs.
    - BFS forward from ``source_tensor``, collecting every Concat reached before
    any MatMul: exactly one is expected. 
    - Rotating its output rotates every downstream consumer in one insertion
    for K in the backbone this is both 1) the block's own QK^T input and 
    2) the exported k_cache_l{i} graph output, since both read the same post-Concat tensor.
    - :raises ValueError: if the walk reaches a MatMul before any Concat, or
        finds zero or more than one Concat.
    """
    visited: set[str] = set()
    queue: deque[str] = deque([source_tensor])
    concat_nodes: dict[str, onnx.NodeProto] = {}
    while queue:
        name = queue.popleft()
        if name in visited:
            continue
        visited.add(name)
        for node in consumers_by_tensor.get(name, []):
            if node.op_type == "MatMul":
                raise ValueError(
                    f"R3 anchor search from '{source_tensor}': reached MatMul "
                    f"'{node.name}' before any Concat -- expected a "
                    f"rotate_half RoPE merge in between."
                )
            if node.op_type == "Concat":
                concat_nodes[node.name] = node
                continue
            queue.extend(node.output)
    if len(concat_nodes) != 1:
        raise ValueError(
            f"R3 anchor search from '{source_tensor}': expected exactly one "
            f"post-RoPE Concat, found {len(concat_nodes)}: {sorted(concat_nodes)}."
        )
    concat_out = next(iter(concat_nodes.values())).output[0]
    consumers = consumers_by_tensor.get(concat_out, [])
    if not consumers:
        raise ValueError(
            f"R3 anchor search from '{source_tensor}': post-RoPE tensor "
            f"'{concat_out}' has no consumers."
        )
    return concat_out, consumers


def apply_backbone_rotations(
    onnx_model: onnx.ModelProto,
    *,
    enable_r1: bool = True,
    enable_r2: bool = False,
    enable_r3: bool = False,
    enable_r4: bool | None = None,
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

    R3 rotates each block's post-RoPE Q and K immediately before QK^T (see
    _rotate_backbone_r3). Both sides are activations with no weight to
    absorb an inverse into, so -- like R4 -- this is a live MatMul, not a
    fold. It makes the resulting backbone valid only against an action
    expert built with apply_action_expert_r3, the same coupling R2 has with
    apply_action_expert_r2. Defaults to False (unlike R4): enabling it
    without a matching action expert silently corrupts attention.

    R4 is self-contained; unlike R2/R3 it needs no cross-component pairing flag. 
    enable_r4 defaults to tracking enable_r1, so it turns on wherever SpinQuant (enable_r1)
    is enabled, while also allowing an explicit override.

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
    enable_r3
        Also apply the R3 per-head Q/K rotation (see _rotate_backbone_r3).
        Defaults to False -- requires a matching apply_action_expert_r3.
    enable_r4
        Also apply R4 (FFN down_proj input rotation -- see _rotate_backbone_r4).
        Defaults to None, which means "follow enable_r1"
        -- pass True/False explicitly to control it independently.
    """
    if enable_r4 is None:
        enable_r4 = enable_r1

    cg = ConnectedGraph(onnx_model)
    boundaries, active_norms = get_decoder_block_boundaries(onnx_model, cg)
    role_map = get_decoder_role_map(cg, boundaries, active_norms)
    hidden_size = infer_hidden_size(onnx_model, role_map)
    head_dim = infer_backbone_head_dim(onnx_model) if (enable_r2 or enable_r3) else None
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

    if enable_r4:
        # Must run after R1/R2: both are weight-only mutations that leave
        # role_map's Op/node identities valid, but R4 inserts new nodes (like
        # AIMET's own R3RotationPass), which invalidates role_map for any
        # pass run afterward.
        _rotate_backbone_r4(onnx_model, role_map)

    if enable_r3:
        assert head_dim is not None
        # Must run last: R3, like R4, inserts new nodes rather than mutating
        # existing weights, so it must follow every role-map-dependent pass
        # (R1/R2/R4 above) -- matching AIMET's own R3RotationPass discipline
        # ("R3 is the last pass in the pipeline").
        _rotate_backbone_r3(onnx_model, role_map, head_dim)


def _rotate_backbone_r4(
    onnx_model: onnx.ModelProto, role_map: DecoderModelRoleMap
) -> list[str]:
    """
    Insert an online R4 Hadamard rotation on each MLP down_proj's input, in place.

    R4 rotates the FFN intermediate activation (the Swish-gated up/gate
    product) immediately before down_proj, reducing outliers for activation
    quantization at that point. It sits downstream of Swish's nonlinearity, so
    -- unlike R1/R2 -- it cannot be folded into up_proj/gate_proj; it must be a
    live MatMul. aimet_onnx has no R4RotationPass to call, so this follows the
    same two-step pattern AIMET's own R3RotationPass uses by hand:
    insert_online_hadamard_node for the live rotation, then
    rotate_linear_weight(..., is_writing=False) so down_proj absorbs the
    compensating inverse and the float output is unchanged.

    Each op in role_map.blocks[i].down_proj is rotated independently, sized to
    its own input width -- this covers a single down_proj per block as well as
    a split one (e.g. GemmaMLPSplitLinear's per-chunk projections), since each
    chunk's rotation is self-contained and chunks are only combined after
    down_proj.

    Parameters
    ----------
    onnx_model
        Backbone ONNX model. Mutated in place.
    role_map
        Role map produced by get_decoder_role_map, from before R1/R2 ran (safe
        to reuse: R1/R2 only rewrite initializer values, not graph structure).

    Returns
    -------
    rotated : list[str]
        Names of the down_proj ops that were rotated.
    """
    initializers = {i.name for i in onnx_model.graph.initializer}
    rotated: list[str] = []
    for block_idx, block in enumerate(role_map.blocks):
        for chunk_idx, op in enumerate(block.down_proj):
            node = op.get_module()
            activations = [i for i in node.input if i not in initializers]
            if len(activations) != 1:
                raise ValueError(
                    f"down_proj op '{op.name}' (block {block_idx}): expected "
                    f"exactly one activation input, found {activations}."
                )
            in_size = _get_rotated_axis_size(onnx_model, op, is_writing=False)
            R4 = hadamard_rotation_matrix(in_size)
            insert_online_hadamard_node(
                onnx_model,
                target_tensor_name=activations[0],
                consumer_nodes=[node],
                head_dim=in_size,
                name_prefix=f"spinquant_block{block_idx}_down{chunk_idx}_R4",
            )
            rotate_linear_weight(onnx_model, op, R4, is_writing=False)
            rotated.append(op.name)
    return rotated


def _rotate_backbone_r3(
    onnx_model: onnx.ModelProto, role_map: DecoderModelRoleMap, head_dim: int
) -> list[str]:
    """
    Insert online R3 Hadamard rotations on Q and K, immediately after RoPE, in place.

    - R3 rotates the post-RoPE Q and K activations.
    Because ``(Q @ H) @ (K @ H)^T == Q @ K^T`` for H, the two rotations cancel 
    inside the attention matmul with no weight on either side to fuse into 
    (unlike R1/R2)-- so, like R4, this must be a live MatMul.

    - Every block in role_map.blocks gets both Q and K rotated. The last block 
    (role_map.lm_head; see _rotate_terminal_v_projections for why R2 special-cases
    the same block) has no q_proj/o_proj and exists only to emit K/V for the 
    action expert -- so only its K is rotated; 
    there is no in-block QK^T for a Q side to cancel against.

    - A backbone rotated this way is only correct paired with an action expert
    built with apply_action_expert_r: every k_cache_l{i} this emits becomes 
    the action expert's per-layer K-cache input, concatenated with the 
    expert's own RoPE'd K before its own QK^T. 
    The same coupling R2 already has for V/O.

    Parameters
    ----------
    onnx_model
        Backbone ONNX model. Mutated in place.
    role_map
        Role map produced by get_decoder_role_map, from before R4 ran.
    head_dim
        Per-head rotation size (same convention as R2).

    Returns
    -------
    rotated : list[str]
        Names of the q_proj/k_proj ops whose downstream RoPE output was rotated.
    """
    consumers_by_tensor = _index_consumers_by_tensor(onnx_model)
    rotated: list[str] = []

    def _rotate(op: Op, tag: str) -> None:
        source_tensor = op.get_module().output[0]
        target, consumers = _find_post_rope_hadamard_target(
            consumers_by_tensor, source_tensor
        )
        insert_online_hadamard_node(
            onnx_model,
            target_tensor_name=target,
            consumer_nodes=consumers,
            head_dim=head_dim,
            name_prefix=f"spinquant_{op.name.strip('/')}_{tag}_R3",
        )
        rotated.append(op.name)

    for block in role_map.blocks:
        q_ops = [op for op in block.qkv_linears if "/q_proj" in op.name]
        k_ops = [op for op in block.qkv_linears if "/k_proj" in op.name]
        if len(q_ops) != 1 or len(k_ops) != 1:
            raise ValueError(
                "R3 rotation: block expected exactly one q_proj and one "
                f"k_proj, found q={[o.name for o in q_ops]} "
                f"k={[o.name for o in k_ops]}."
            )
        _rotate(q_ops[0], "q")
        _rotate(k_ops[0], "k")

    terminal_k_ops = [op for op in role_map.lm_head if "/k_proj" in op.name]
    if len(terminal_k_ops) != 1:
        raise ValueError(
            "R3 rotation: expected exactly one terminal (lm_head) k_proj, "
            f"found {[op.name for op in terminal_k_ops]}."
        )
    _rotate(terminal_k_ops[0], "k")

    return rotated


def apply_backbone_r1(onnx_model: onnx.ModelProto) -> None:
    """Apply R1 (and R4, which follows it by default). Thin wrapper kept for existing callers."""
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


def apply_action_expert_r4(onnx_model: onnx.ModelProto) -> int:
    """
    Insert an online R4 Hadamard rotation on each MLP down_proj's input, in place.

    - The expert-side twin of `_rotate_backbone_r4`
    - self-contained: the inserted rotation and inverse folded into down_proj 
      both live inside this graph, so unlike R2/R3 there is no backbone counterpart 
      to pair with. 
    - Selected by op name rather than by a role map, for the same reason
      apply_action_expert_r2/r3 are: the expert is a cross-attention stack with
      no decoder-block structure for `get_decoder_role_map` to walk.
    - One rotation per layer, sized to that down_proj's own input width. The
      expert keeps HF's unsplit GemmaMLP, so that width is the full
      intermediate size (4096) rather than the backbone's per-chunk 2048 (see
      GemmaMLPSplitLinear): 18 [4096, 4096] fp32 initializers, +1.21 GB on a
      float export that is already 2.17 GB. That is the whole cost. Measured
      against the R1-folded float export on CPU: QuantSim construction 24s ->
      34s, calibration forward 0.7s -> 0.8s, seqMSE (2 samples) 301s -> 312s,
      plus a one-off ~25s to apply. An earlier attempt at expert R4 was
      abandoned as far too slow to quantize; none of that slowness reproduces
      through this path, so if it returns, suspect the surrounding harness
      (host device, external-data spilling on a >2 GB ModelProto) rather than
      the rotation itself.

    Parameters
    ----------
    onnx_model
        Action expert ONNX model. Mutated in place.

    Returns
    -------
    count : int
        Number of down_proj ops rotated.
    """
    cg = ConnectedGraph(onnx_model)
    initializers = {i.name for i in onnx_model.graph.initializer}

    # Snapshot the node list: insert_online_hadamard_node splices new nodes
    # into graph.node, and mutating a protobuf repeated field while iterating
    # it would skip or revisit entries.
    n_down = 0
    for node in list(onnx_model.graph.node):
        if not _EXPERT_DOWN_PATTERN.match(node.name or ""):
            continue
        activations = [i for i in node.input if i not in initializers]
        if len(activations) != 1:
            raise ValueError(
                f"Action expert down_proj '{node.name}': expected exactly one "
                f"activation input, found {activations}."
            )
        op = cg.get_op_from_module_name(node.name)
        in_size = _get_rotated_axis_size(onnx_model, op, is_writing=False)
        R4 = hadamard_rotation_matrix(in_size)
        insert_online_hadamard_node(
            onnx_model,
            target_tensor_name=activations[0],
            consumer_nodes=[node],
            head_dim=in_size,
            name_prefix=f"spinquant_{node.name.strip('/')}_R4",
        )
        rotate_linear_weight(onnx_model, op, R4, is_writing=False)
        n_down += 1

    if n_down == 0:
        raise ValueError(
            "Action expert R4: found no down_proj MatMul ops matching "
            f"'{_EXPERT_DOWN_PATTERN.pattern}'. The expert's op naming may "
            "have changed."
        )
    return n_down


def apply_action_expert_r3(onnx_model: onnx.ModelProto) -> tuple[int, int]:
    """
    Apply the R3 compensation to the action expert, in place.

    - Pairs with `apply_backbone_rotations(..., enable_r3=True)`
    - Backbone hands per-layer R3-rotated K caches (_rotate_backbone_r3)--
    the expert's post-RoPE Q and K must be rotated by same H for its QK^T
    to be correct (SHAGemmaExpertAttention concatenates the backbone's
    rotated K-cache with its own freshly-computed K before attention)
    - same failure mode apply_action_expert_r2 already guards against for V/O.

    Parameters
    ----------
    onnx_model
        Action expert ONNX model. Mutated in place.

    Returns
    -------
    counts : tuple[int, int]
        Number of (q_proj_sha, k_proj_sha) ops rotated.
    """
    head_dim = infer_expert_head_dim(onnx_model)
    consumers_by_tensor = _index_consumers_by_tensor(onnx_model)

    n_q = n_k = 0
    for node in onnx_model.graph.node:
        name = node.name or ""
        is_q = bool(_EXPERT_Q_PATTERN.match(name))
        is_k = bool(_EXPERT_K_PATTERN.match(name))
        if not (is_q or is_k):
            continue
        target, consumers = _find_post_rope_hadamard_target(
            consumers_by_tensor, node.output[0]
        )
        insert_online_hadamard_node(
            onnx_model,
            target_tensor_name=target,
            consumer_nodes=consumers,
            head_dim=head_dim,
            name_prefix=f"spinquant_{name.strip('/')}_R3",
        )
        n_q += is_q
        n_k += is_k

    if n_q == 0 or n_k == 0 or n_q % n_k != 0:
        raise ValueError(
            "Action expert R3: expected a non-zero q_proj_sha count that's a "
            f"multiple of the k_proj_sha count (GQA group size), found "
            f"q={n_q}, k={n_k}. The expert's op naming may have changed."
        )
    return n_q, n_k


class _R1WrappedAdaRMSNorm(torch.nn.Module):
    """
    Wrap an action-expert AdaRMSNorm so it operates correctly when its input
    lives in the R1-rotated residual-stream basis.

    AdaRMSNorm's pure-normalization step (division by RMS) is rotation-
    equivariant -- rotation preserves L2 norm, so rmsnorm(x @ R1) == rmsnorm(x)
    @ R1 exactly -- but its per-channel scale/shift modulation (derived from
    adarms_cond via a static nn.Linear; see
    transformers.models.gemma.modeling_gemma.GemmaRMSNorm.forward) is not: it
    was learned for the *unrotated* channel basis, and an elementwise multiply
    doesn't commute with a dense rotation matrix. So the input is un-rotated
    before calling the original norm, and its output is re-rotated after.
    `gate` is deliberately left un-rotated -- see apply_action_expert_r1's
    docstring for why it's consumed in the unrotated basis, not this one.

    Verified algebraically (not yet against the real checkpoint) in
    verify_r1_fold.py; see apply_action_expert_r1's docstring.
    """

    def __init__(self, norm: torch.nn.Module, r1: torch.Tensor) -> None:
        super().__init__()
        self.norm = norm
        self.register_buffer("r1", r1, persistent=False)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x_unrot = x @ self.r1.T
        normed, gate = self.norm(x_unrot, cond)
        return normed @ self.r1, gate


def _r1_expert_forward(
    self,
    full_att_4d: torch.Tensor,
    rope_emb_sin: torch.Tensor,
    rope_emb_cos: torch.Tensor,
    k_caches: list[torch.Tensor],
    v_caches: list[torch.Tensor],
    x_t: torch.Tensor,
    adarms_cond: torch.Tensor,
) -> torch.Tensor:
    """
    Replacement for Pi05ActionExpert.expert_forward, bound onto an R1-rotated
    expert instance by apply_action_expert_r1.

    Identical to the original except the two gated-residual-add lines rotate
    the (unrotated-basis) gated sublayer output into the rotated
    residual-stream basis before adding it to hidden_suffix -- see
    apply_action_expert_r1's docstring for why o_proj/mlp.down_proj are left
    unrotated and gate can't be folded into a static weight.
    """
    flow = self.model
    pg_we = flow.paligemma_with_expert
    gemma_layers = pg_we.gemma_expert.model.layers
    r1 = self._spinquant_r1

    device = next(flow.parameters()).device
    x_t = x_t.to(device)
    rope_emb_sin = rope_emb_sin.to(device)
    rope_emb_cos = rope_emb_cos.to(device)
    adarms_cond = adarms_cond.to(device)
    full_att_4d = full_att_4d.to(device)

    suffix_embs = flow.action_in_proj(x_t)
    hidden_suffix = suffix_embs.to(torch.float32)

    for layer_idx, layer in enumerate(gemma_layers):
        normed, gate = layer.input_layernorm(hidden_suffix, adarms_cond)

        out_emb = layer.self_attn(
            normed,
            rope_emb_sin,
            rope_emb_cos,
            k_caches[layer_idx].to(device),
            v_caches[layer_idx].to(device),
            full_att_4d,
        )

        if gate is None:
            out_emb = out_emb @ r1 + hidden_suffix
        else:
            out_emb = (out_emb * gate) @ r1 + hidden_suffix

        after_first_residual = out_emb

        out_emb, gate2 = layer.post_attention_layernorm(out_emb, adarms_cond)
        out_emb = layer.mlp(out_emb)

        if gate2 is None:
            hidden_suffix = out_emb @ r1 + after_first_residual
        else:
            hidden_suffix = (out_emb * gate2) @ r1 + after_first_residual

    # Final norm: un-rotate on entry like the other AdaRMS calls, but do NOT
    # re-rotate the output -- action_out_proj is deliberately left unmodified
    # (see apply_action_expert_r1's docstring), so it must be fed the norm's
    # natural, unrotated output, not a rotated one. Unlike input_layernorm /
    # post_attention_layernorm, this call is not wrapped with
    # _R1WrappedAdaRMSNorm for exactly this reason.
    hidden_unrot = hidden_suffix @ r1.T
    hidden_suffix, _ = pg_we.gemma_expert.model.norm(hidden_unrot, adarms_cond)
    suffix_only = hidden_suffix[:, -self._num_action_steps :]
    suffix_only = suffix_only.to(torch.float32)
    return flow.action_out_proj(suffix_only)


def apply_action_expert_r1(expert: torch.nn.Module, num_action_steps: int) -> None:
    """
    Fold an independent R1 rotation into the action expert's own residual
    stream, in place.

    Unlike apply_action_expert_r2 (which rotates already-exported ONNX
    initializers), this operates on the PyTorch module *before* export,
    because AdaRMSNorm's scale/shift/gate modulation is a genuine runtime
    computation (a function of adarms_cond, which is itself a live input to
    the exported graph -- the action expert's single ONNX graph is called
    once per Euler integration step with a different runtime time_step, not
    exported once per step; see app.py's inference loop). A purely static
    fold -- the kind backbone R1 and R2 compensation both are -- isn't
    possible here. Instead: the read/write linears that only depend on the
    rotated residual stream get a zero-added-cost static fold (identical in
    spirit to the backbone's), and a small, *localized* online rotation is
    paid only at the three AdaRMS norm calls per layer plus the two
    gated-residual-add sites -- not a rotation of every block, and no
    backbone/expert cache coupling the way R2 needs.

    Verified algebraically on synthetic random tensors reproducing this exact
    structure (GemmaRMSNorm's adaptive math, split per-head Q/K/V, gated
    residual adds): final output matches an unrotated float baseline to
    ~1e-13 (float64). NOT YET verified against the real checkpoint -- run the
    debug command in this repo's PR/session notes (forward-pass diff, float,
    no quantization) before trusting this against real weights.

    Convention (R1 orthogonal, hidden_size x hidden_size, independent of and
    unrelated to the backbone's own R1):
      - "Reading" layers (consume the rotated residual stream; output must be
        numerically unchanged): W_new = W_old @ R1. Applies to q_proj_sha,
        k_proj_sha, v_proj_sha (every head) and mlp.gate_proj / mlp.up_proj.
      - "Writing" layer that feeds the residual stream from a stream-external
        input with no gate involved: W_new = R1.T @ W_old. Applies only to
        action_in_proj.
      - o_proj and mlp.down_proj are left UNMODIFIED: their output is
        elementwise-gated by a runtime `gate` tensor before being added to the
        residual stream, and gate can neither be folded into a static weight
        (it depends on adarms_cond at runtime) nor commuted past a rotation
        (it's only meaningful in the original, unrotated basis). Instead,
        _r1_expert_forward applies gate where o_proj/mlp.down_proj already
        produce output (unrotated), then rotates the *gated* result before
        the residual add.
      - action_out_proj is left UNMODIFIED: it's fed the deliberately
        unrotated output of the final norm (see _r1_expert_forward), so there
        is no downstream basis for it to match.

    Parameters
    ----------
    expert
        A Pi05ActionExpert instance, already built from a checkpoint. Mutated
        in place: reading-layer weights are rotated, the three AdaRMSNorm
        modules are wrapped, and expert_forward is replaced with an R1-aware
        version bound to this instance only (other instances / the class
        itself are untouched).
    num_action_steps
        Number of trailing suffix positions expert_forward slices out after
        the final norm (NUM_ACTION_STEPS in qai_hub_models.models.pi05.model).
        Passed in rather than imported to avoid a circular import: that
        module imports this one.
    """
    flow = expert.model
    pg_we = flow.paligemma_with_expert
    gemma_layers = pg_we.gemma_expert.model.layers

    hidden_size = flow.action_in_proj.out_features
    r1 = get_r1_matrix(hidden_size, dtype=flow.action_in_proj.weight.dtype).to(
        flow.action_in_proj.weight.device
    )

    with torch.no_grad():
        flow.action_in_proj.weight.copy_(r1.T @ flow.action_in_proj.weight)
        if flow.action_in_proj.bias is not None:
            flow.action_in_proj.bias.copy_(flow.action_in_proj.bias @ r1)

        for layer in gemma_layers:
            self_attn = layer.self_attn
            for q in self_attn.q_proj_sha:
                q.weight.copy_(q.weight @ r1)
            for k in self_attn.k_proj_sha:
                k.weight.copy_(k.weight @ r1)
            for v in self_attn.v_proj_sha:
                v.weight.copy_(v.weight @ r1)
            # o_proj intentionally left unmodified -- see docstring.

            mlp = layer.mlp
            mlp.gate_proj.weight.copy_(mlp.gate_proj.weight @ r1)
            mlp.up_proj.weight.copy_(mlp.up_proj.weight @ r1)
            # down_proj intentionally left unmodified -- see docstring.

            layer.input_layernorm = _R1WrappedAdaRMSNorm(layer.input_layernorm, r1)
            layer.post_attention_layernorm = _R1WrappedAdaRMSNorm(
                layer.post_attention_layernorm, r1
            )

        # Final norm is intentionally left UNwrapped: _r1_expert_forward calls
        # it directly with an explicit un-rotate (matching entry behavior of
        # the wrapped norms above) but no re-rotation of its output -- see
        # action_out_proj note below and _r1_expert_forward.
        # action_out_proj intentionally left unmodified -- see docstring.

    expert._spinquant_r1 = r1
    expert._num_action_steps = num_action_steps
    expert.expert_forward = types.MethodType(_r1_expert_forward, expert)
