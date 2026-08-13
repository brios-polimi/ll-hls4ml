"""Compute-conscious hierarchical architecture probes for LLVM CDFGs.

The three variants deliberately change the composition law rather than merely
stacking more message-passing layers:

* ``sequence`` uses a bounded-memory GRU over instruction order in each block.
* ``block_attention`` uses global self-attention within each function.
* ``memory_dual`` keeps memory-like variables active and splits resource/timing
  composition before the block hierarchy.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax, to_dense_batch

from ll_hls4ml.io.schema import DERIVED_DEF_USE_EDGE, LABEL_KEYS, NODE_TYPES
from ll_hls4ml.models.hierarchical import _messages, _reverse_messages
from ll_hls4ml.models.input_projection import CDFGInputProjection
from ll_hls4ml.models.readout import (
    GlobalFeatureEncoder,
    GraphContextEncoder,
    SplitRegressionHead,
    multi_pool,
)


def _group_lengths(group: torch.Tensor, size: int) -> torch.Tensor:
    return torch.bincount(group, minlength=size)


def _length_buckets(
    lengths: torch.Tensor,
    budget: int,
    quadratic: bool = False,
) -> list[list[int]]:
    """Group ragged sequences without padding the whole graph to its maximum."""

    length_values = lengths.detach().cpu().tolist()
    ordered = sorted(
        (index for index, length in enumerate(length_values) if length),
        key=length_values.__getitem__,
        reverse=True,
    )
    buckets: list[list[int]] = []
    cursor = 0
    while cursor < len(ordered):
        maximum = int(length_values[ordered[cursor]]) + (1 if quadratic else 0)
        unit = maximum * maximum if quadratic else maximum
        width = max(1, budget // max(unit, 1))
        buckets.append(ordered[cursor : cursor + width])
        cursor += width
    return buckets


def _dense_group_chunk(
    state: torch.Tensor,
    group: torch.Tensor,
    group_ids: list[int],
    total_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dense ragged states plus original indices for one group bucket."""

    selected_groups = torch.as_tensor(group_ids, device=group.device)
    remap = torch.full((total_groups,), -1, dtype=torch.long, device=group.device)
    remap[selected_groups] = torch.arange(len(group_ids), device=group.device)
    local_group = remap[group]
    selected = (local_group >= 0).nonzero(as_tuple=False).flatten()
    order = torch.argsort(local_group[selected], stable=True)
    selected = selected[order]
    local_group = local_group[selected]
    dense, mask = to_dense_batch(
        state[selected], local_group, batch_size=len(group_ids)
    )
    return dense, mask, selected, selected_groups


class GatedInstructionLayer(nn.Module):
    """Keep control and def-use relations separate until after learned gates."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.control = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.def_use = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.control_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.def_use_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.update = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        state: torch.Tensor,
        control_edges: torch.Tensor,
        def_use_edges: torch.Tensor,
        control_features: torch.Tensor | None,
    ) -> torch.Tensor:
        control = F.gelu(
            _messages(
                self.control(state),
                control_edges,
                state.size(0),
                control_features,
            )
        )
        data = F.gelu(
            _messages(self.def_use(state), def_use_edges, state.size(0))
        )
        control = control * torch.sigmoid(
            self.control_gate(torch.cat([state, control], dim=-1))
        )
        data = data * torch.sigmoid(
            self.def_use_gate(torch.cat([state, data], dim=-1))
        )
        return self.norm(
            state + self.dropout(torch.tanh(self.update(control + data)))
        )


class GatedBlockLayer(nn.Module):
    """Directed CFG layer with independently gated predecessor/successor flow."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.forward_message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reverse_message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.forward_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.reverse_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.update = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, state: torch.Tensor, cfg_edges: torch.Tensor) -> torch.Tensor:
        predecessor = F.gelu(
            _messages(
                self.forward_message(state), cfg_edges, state.size(0)
            )
        )
        successor = F.gelu(
            _reverse_messages(
                self.reverse_message(state), cfg_edges, state.size(0)
            )
        )
        predecessor = predecessor * torch.sigmoid(
            self.forward_gate(torch.cat([state, predecessor], dim=-1))
        )
        successor = successor * torch.sigmoid(
            self.reverse_gate(torch.cat([state, successor], dim=-1))
        )
        return self.norm(
            state
            + self.dropout(torch.tanh(self.update(predecessor + successor)))
        )


class BlockSequenceGRU(nn.Module):
    """Forward GRU over every block, bucketed to cap padded activation memory."""

    def __init__(self, hidden_dim: int, token_budget: int = 16_384):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.token_budget = token_budget

    def forward(
        self,
        state: torch.Tensor,
        instruction_block: torch.Tensor,
        block_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = _group_lengths(instruction_block, block_count)
        context = torch.zeros_like(state)
        endpoints = state.new_zeros((block_count, state.size(-1)))
        for group_ids in _length_buckets(lengths, self.token_budget):
            dense, mask, selected, selected_groups = _dense_group_chunk(
                state, instruction_block, group_ids, block_count
            )
            encoded, _hidden = self.gru(dense)
            context[selected] = encoded[mask].to(context.dtype)
            last = lengths[selected_groups] - 1
            endpoints[selected_groups] = encoded[
                torch.arange(len(group_ids), device=state.device), last
            ].to(endpoints.dtype)
        gate = torch.sigmoid(self.gate(torch.cat([state, context], dim=-1)))
        return self.norm(state + gate * context), endpoints


def _sinusoidal_positions(length: int, width: int, reference: torch.Tensor):
    position = torch.arange(length, device=reference.device, dtype=torch.float32)
    frequency = torch.exp(
        torch.arange(0, width, 2, device=reference.device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(width, 1))
    )
    encoding = torch.zeros((length, width), device=reference.device)
    encoding[:, 0::2] = torch.sin(position[:, None] * frequency)
    encoding[:, 1::2] = torch.cos(
        position[:, None] * frequency[: encoding[:, 1::2].shape[1]]
    )
    return encoding.to(reference.dtype)


class FunctionBlockTransformer(nn.Module):
    """Global attention over blocks, independently within each function."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        layers: int,
        dropout: float,
        attention_budget: int = 131_072,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls, std=0.02)
        self.attention_budget = attention_budget

    def forward(
        self,
        block_state: torch.Tensor,
        block_function: torch.Tensor,
        function_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = _group_lengths(block_function, function_count)
        contextual = torch.zeros_like(block_state)
        summaries = block_state.new_zeros((function_count, block_state.size(-1)))
        buckets = _length_buckets(
            lengths, self.attention_budget, quadratic=True
        )
        for group_ids in buckets:
            dense, mask, selected, selected_groups = _dense_group_chunk(
                block_state, block_function, group_ids, function_count
            )
            dense = dense + _sinusoidal_positions(
                dense.size(1), dense.size(2), dense
            ).unsqueeze(0)
            cls = self.cls.expand(dense.size(0), -1, -1).to(dense.dtype)
            tokens = torch.cat([cls, dense], dim=1)
            padding = torch.cat(
                [
                    torch.zeros(
                        (mask.size(0), 1), dtype=torch.bool, device=mask.device
                    ),
                    ~mask,
                ],
                dim=1,
            )
            encoded = self.encoder(tokens, src_key_padding_mask=padding)
            contextual[selected] = encoded[:, 1:][mask].to(contextual.dtype)
            summaries[selected_groups] = encoded[:, 0].to(summaries.dtype)
        return contextual, summaries


class AttentionGroupPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, state, group, size):
        weight = softmax(self.score(state).squeeze(-1), group, num_nodes=size)
        return scatter(
            state * weight.unsqueeze(-1),
            group,
            dim=0,
            dim_size=size,
            reduce="sum",
        )


class PositiveGroupSum(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.contribution = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state, group, size):
        contribution = F.softplus(self.contribution(state))
        return torch.log1p(
            scatter(
                contribution,
                group,
                dim=0,
                dim_size=size,
                reduce="sum",
            )
        )


class SoftPathPropagation(nn.Module):
    """Tied, gated CFG recurrence with a soft predecessor maximum."""

    def __init__(self, hidden_dim: int, steps: int, dropout: float):
        super().__init__()
        self.steps = steps
        self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, state: torch.Tensor, edges: torch.Tensor):
        if edges.numel() == 0:
            # Every rank must touch the same parameters even when its batch has
            # no CFG edges; otherwise DDP sees an unfinished reduction.
            zero = sum(parameter.sum() * 0 for parameter in self.parameters())
            return state + zero
        for _step in range(self.steps):
            message = self.message(state[edges[0]])
            weight = softmax(
                self.score(message).squeeze(-1),
                edges[1],
                num_nodes=state.size(0),
            )
            incoming = scatter(
                message * weight.unsqueeze(-1),
                edges[1],
                dim=0,
                dim_size=state.size(0),
                reduce="sum",
            )
            state = self.norm(
                self.update(self.dropout(incoming), state)
            )
        return state


class DualTargetHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()

        def tower(outputs: int):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, outputs),
            )

        self.resource = tower(6)
        self.timing = tower(2)

    def forward(self, resource: torch.Tensor, timing: torch.Tensor):
        resource_output = self.resource(resource)
        return torch.cat(
            [resource_output[:, :4], self.timing(timing), resource_output[:, 4:]],
            dim=-1,
        )


class CDFGHierarchicalExperimental(nn.Module):
    """Shared implementation for three controlled hierarchy probes."""

    variants = {"sequence", "block_attention", "memory_dual"}

    def __init__(
        self,
        edge_pos_vocab_size: int,
        y_means: torch.Tensor,
        y_stds: torch.Tensor,
        instruction_vocab_size: int | None = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        instruction_num_layers: int | None = None,
        block_num_layers: int | None = None,
        dropout: float = 0.15,
        node_vocab_sizes: dict[str, int] | None = None,
        use_global_features: bool = True,
        use_context: bool = True,
        split_heads: bool = True,
        context_mode: str = "core",
        hurdle_heads: bool = True,
        hurdle_prediction_mode: str = "threshold",
        architecture: str = "sequence",
        attention_heads: int = 4,
        attention_layers: int = 2,
        cfg_recurrent_steps: int = 8,
        sequence_token_budget: int = 16_384,
        attention_pair_budget: int = 131_072,
    ):
        super().__init__()
        if architecture not in self.variants:
            raise ValueError(f"Unknown hierarchical architecture: {architecture}")
        if not split_heads or not hurdle_heads:
            raise ValueError(
                "experimental hierarchy probes require split_heads and hurdle_heads"
            )
        if instruction_vocab_size is None:
            if not node_vocab_sizes or "instruction" not in node_vocab_sizes:
                raise ValueError("instruction_vocab_size is required")
            instruction_vocab_size = node_vocab_sizes["instruction"]
        instruction_num_layers = (
            num_layers if instruction_num_layers is None else instruction_num_layers
        )
        block_num_layers = (
            num_layers if block_num_layers is None else block_num_layers
        )
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.use_global_features = use_global_features
        self.use_context = use_context
        self.hurdle_heads = True
        self.hurdle_prediction_mode = hurdle_prediction_mode
        self.register_buffer("y_means", y_means.clone())
        self.register_buffer("y_stds", y_stds.clone())

        self.input_proj = CDFGInputProjection(
            instruction_vocab_size, edge_pos_vocab_size, hidden_dim
        )
        self.instruction_input = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.instruction_layers = nn.ModuleList(
            GatedInstructionLayer(hidden_dim, dropout)
            for _ in range(instruction_num_layers)
        )
        if architecture == "memory_dual":
            self.memory_from_producer = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.memory_from_consumer = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.memory_update = nn.GRUCell(hidden_dim, hidden_dim)
            self.memory_to_instruction = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.memory_to_producer = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.memory_gate = nn.Linear(2 * hidden_dim, hidden_dim)
            self.resource_instruction_sum = PositiveGroupSum(hidden_dim)
            self.timing_instruction_pool = AttentionGroupPool(hidden_dim)
            self.resource_block_input = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.timing_block_input = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.resource_block_layer = GatedBlockLayer(hidden_dim, dropout)
            self.timing_path = SoftPathPropagation(
                hidden_dim, cfg_recurrent_steps, dropout
            )
            self.resource_function_sum = PositiveGroupSum(hidden_dim)
            self.timing_function_pool = AttentionGroupPool(hidden_dim)
            self.resource_function_input = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.timing_function_input = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.resource_callee_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.timing_callee_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.resource_call_gate = nn.Linear(2 * hidden_dim, hidden_dim)
            self.timing_call_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            self.callee_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.callee_gate = nn.Linear(2 * hidden_dim, hidden_dim)
            pooled_dim = 4 * hidden_dim + 1
            if architecture == "sequence":
                self.sequence = BlockSequenceGRU(
                    hidden_dim, sequence_token_budget
                )
                block_input_dim = pooled_dim + 3 * hidden_dim
            else:
                block_input_dim = pooled_dim + 2 * hidden_dim
            self.block_input = nn.Sequential(
                nn.Linear(block_input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.block_layers = nn.ModuleList(
                GatedBlockLayer(hidden_dim, dropout)
                for _ in range(block_num_layers)
            )
            if architecture == "block_attention":
                self.block_transformer = FunctionBlockTransformer(
                    hidden_dim,
                    attention_heads,
                    attention_layers,
                    dropout,
                    attention_pair_budget,
                )
            else:
                self.function_pool = AttentionGroupPool(hidden_dim)
            self.function_input = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )

        classifier_dim = hidden_dim
        if use_global_features:
            self.global_features = GlobalFeatureEncoder(
                instruction_vocab_size, hidden_dim
            )
            classifier_dim += hidden_dim
        if use_context:
            self.context_encoder = GraphContextEncoder(hidden_dim, context_mode)
            classifier_dim += hidden_dim
        self.output_dim = classifier_dim
        self.classifier = (
            DualTargetHead(classifier_dim, hidden_dim, dropout)
            if architecture == "memory_dual"
            else SplitRegressionHead(
                classifier_dim, hidden_dim, dropout, hurdle_heads=True
            )
        )

    @staticmethod
    def _owners(data):
        device = data["instruction"].x.device
        instruction_block = torch.full(
            (data["instruction"].num_nodes,), -1, dtype=torch.long, device=device
        )
        contains_instruction = data[
            ("block", "contains", "instruction")
        ].edge_index
        instruction_block[contains_instruction[1]] = contains_instruction[0]
        block_function = torch.full(
            (data["block"].num_nodes,), -1, dtype=torch.long, device=device
        )
        contains_block = data[("function", "contains", "block")].edge_index
        block_function[contains_block[1]] = contains_block[0]
        if (instruction_block < 0).any() or (block_function < 0).any():
            raise ValueError("Incomplete instruction/block/function containment")
        return instruction_block, block_function

    @staticmethod
    def _validate_depth_invariants(data):
        instruction_depth = data["instruction"].call_depth.long()
        block_depth = data["block"].call_depth.long()
        for edge_type, depth in (
            (("instruction", "control", "instruction"), instruction_depth),
            (DERIVED_DEF_USE_EDGE, instruction_depth),
            (("block", "control", "block"), block_depth),
        ):
            edges = data[edge_type].edge_index
            if edges.numel() and not torch.equal(depth[edges[0]], depth[edges[1]]):
                raise ValueError(
                    f"{edge_type} violates the intraprocedural depth invariant"
                )

    def _static_inputs(self, data):
        versions = torch.as_tensor(
            data.hierarchy_schema_version,
            device=data["instruction"].x.device,
        )
        if not bool(torch.all(versions == 2)):
            raise ValueError("Hierarchical tensor schema version 2 is required")
        self._validate_depth_invariants(data)
        x_dict = {node_type: data[node_type].x for node_type in NODE_TYPES}
        attr_types = (
            ("instruction", "control", "instruction"),
            ("variable", "operand", "instruction"),
            ("constant", "operand", "instruction"),
        )
        edge_attr_dict = {
            edge_type: data[edge_type].edge_attr.long()
            for edge_type in attr_types
        }
        base, edge_features = self.input_proj(x_dict, edge_attr_dict)
        pragma = base["pragma"]
        variable = base["variable"] + _messages(
            pragma,
            data[("pragma", "applies_to", "variable")].edge_index,
            data["variable"].num_nodes,
        )
        constant = base["constant"] + _messages(
            pragma,
            data[("pragma", "applies_to", "constant")].edge_index,
            data["constant"].num_nodes,
        )
        variable_operand = ("variable", "operand", "instruction")
        constant_operand = ("constant", "operand", "instruction")
        instruction = (
            base["instruction"]
            + _messages(
                pragma,
                data[("pragma", "applies_to", "instruction")].edge_index,
                data["instruction"].num_nodes,
            )
            + _messages(
                variable,
                data[variable_operand].edge_index,
                data["instruction"].num_nodes,
                edge_features[variable_operand],
            )
            + _messages(
                constant,
                data[constant_operand].edge_index,
                data["instruction"].num_nodes,
                edge_features[constant_operand],
            )
            + _reverse_messages(
                variable,
                data[("instruction", "defines", "variable")].edge_index,
                data["instruction"].num_nodes,
            )
        )
        block_pragma = _messages(
            pragma,
            data[("pragma", "applies_to", "block")].edge_index,
            data["block"].num_nodes,
        )
        function_pragma = _messages(
            pragma,
            data[("pragma", "applies_to", "function")].edge_index,
            data["function"].num_nodes,
        )
        return base, edge_features, instruction, variable, block_pragma, function_pragma

    def _generic_call_message(
        self,
        static_instruction,
        function_state,
        depth_calls,
        instruction_local,
        instruction_count,
    ):
        projected = self.callee_proj(function_state[depth_calls[1]])
        message = _messages(
            projected,
            torch.stack(
                [
                    torch.arange(depth_calls.size(1), device=depth_calls.device),
                    instruction_local[depth_calls[0]],
                ]
            ),
            instruction_count,
        )
        gate = torch.sigmoid(
            self.callee_gate(torch.cat([static_instruction, message], dim=-1))
        )
        return gate * message

    def _dual_call_message(
        self,
        static_instruction,
        resource_state,
        timing_state,
        depth_calls,
        instruction_local,
        instruction_count,
    ):
        call_edges = torch.stack(
            [
                torch.arange(depth_calls.size(1), device=depth_calls.device),
                instruction_local[depth_calls[0]],
            ]
        )
        resource = _messages(
            self.resource_callee_proj(resource_state[depth_calls[1]]),
            call_edges,
            instruction_count,
        )
        timing = _messages(
            self.timing_callee_proj(timing_state[depth_calls[1]]),
            call_edges,
            instruction_count,
        )
        resource *= torch.sigmoid(
            self.resource_call_gate(
                torch.cat([static_instruction, resource], dim=-1)
            )
        )
        timing *= torch.sigmoid(
            self.timing_call_gate(torch.cat([static_instruction, timing], dim=-1))
        )
        return resource + timing

    def _memory_exchange(
        self,
        data,
        instruction_state,
        instruction_ids,
        instruction_local,
        memory_state,
        edge_features,
    ):
        # Type slots 5:10 are array, pointer, stream, nnet-array, shift-register.
        memory_like = data["variable"].x[:, 5:10].bool().any(dim=-1)
        defines = data[("instruction", "defines", "variable")].edge_index
        operands = data[("variable", "operand", "instruction")].edge_index
        define_mask = (instruction_local[defines[0]] >= 0) & memory_like[defines[1]]
        operand_mask = (instruction_local[operands[1]] >= 0) & memory_like[operands[0]]
        local_defines = defines[:, define_mask]
        local_operands = operands[:, operand_mask]
        memory_ids = torch.unique(
            torch.cat([local_defines[1], local_operands[0]])
        )
        if not memory_ids.numel():
            # Retain zero-gradient dependencies for DDP ranks without memory nodes.
            zero = sum(
                parameter.sum() * 0
                for module in (
                    self.memory_from_producer,
                    self.memory_from_consumer,
                    self.memory_update,
                    self.memory_to_instruction,
                    self.memory_to_producer,
                    self.memory_gate,
                )
                for parameter in module.parameters()
            )
            return instruction_state + zero, memory_state
        memory_local = torch.full(
            (data["variable"].num_nodes,),
            -1,
            dtype=torch.long,
            device=instruction_state.device,
        )
        memory_local[memory_ids] = torch.arange(
            memory_ids.numel(), device=instruction_state.device
        )
        define_edges = torch.stack(
            [instruction_local[local_defines[0]], memory_local[local_defines[1]]]
        )
        consumer_edges = torch.stack(
            [instruction_local[local_operands[1]], memory_local[local_operands[0]]]
        )
        producer_message = _messages(
            self.memory_from_producer(instruction_state),
            define_edges,
            memory_ids.numel(),
        )
        consumer_message = _messages(
            self.memory_from_consumer(instruction_state),
            consumer_edges,
            memory_ids.numel(),
        )
        updated_memory = self.memory_update(
            F.gelu(producer_message + consumer_message),
            memory_state[memory_ids],
        )
        memory_state = memory_state.index_copy(0, memory_ids, updated_memory)
        operand_edges = torch.stack(
            [memory_local[local_operands[0]], instruction_local[local_operands[1]]]
        )
        operand_features = edge_features[
            ("variable", "operand", "instruction")
        ][operand_mask]
        incoming = _messages(
            self.memory_to_instruction(updated_memory),
            operand_edges,
            instruction_ids.numel(),
            operand_features,
        )
        producer_feedback = _reverse_messages(
            self.memory_to_producer(updated_memory),
            define_edges,
            instruction_ids.numel(),
        )
        memory_message = F.gelu(incoming + producer_feedback)
        gate = torch.sigmoid(
            self.memory_gate(
                torch.cat([instruction_state, memory_message], dim=-1)
            )
        )
        return instruction_state + gate * memory_message, memory_state

    def _append_graph_context(self, data, state):
        values = [state]
        if self.use_global_features:
            values.append(self.global_features(data))
        if self.use_context:
            values.append(self.context_encoder(data))
        return torch.cat(values, dim=-1)

    def forward(self, data):
        (
            base,
            edge_features,
            static_instruction,
            memory_state,
            block_pragma,
            function_pragma,
        ) = self._static_inputs(data)
        instruction_block, block_function = self._owners(data)
        call_depth = data["function"].call_depth.long()
        instruction_depth = data["instruction"].call_depth.long()
        block_depth = data["block"].call_depth.long()
        reachable = data["function"].is_reachable.bool()
        calls = data[("instruction", "calls", "function")].edge_index
        control_type = ("instruction", "control", "instruction")
        control_edges = data[control_type].edge_index
        def_use_edges = data[DERIVED_DEF_USE_EDGE].edge_index
        cfg_edges = data[("block", "control", "block")].edge_index

        function_state = torch.zeros_like(base["function"], dtype=torch.float32)
        if self.architecture == "memory_dual":
            resource_function_state = function_state.clone()
            timing_function_state = function_state.clone()

        max_depth = int(call_depth[reachable].max().item())
        for depth in range(max_depth + 1):
            function_ids = ((call_depth == depth) & reachable).nonzero().flatten()
            instruction_ids = (instruction_depth == depth).nonzero().flatten()
            block_ids = (block_depth == depth).nonzero().flatten()
            instruction_local = torch.full_like(instruction_depth, -1)
            instruction_local[instruction_ids] = torch.arange(
                instruction_ids.numel(), device=instruction_ids.device
            )
            block_local = torch.full_like(block_depth, -1)
            block_local[block_ids] = torch.arange(
                block_ids.numel(), device=block_ids.device
            )
            function_local = torch.full_like(call_depth, -1)
            function_local[function_ids] = torch.arange(
                function_ids.numel(), device=function_ids.device
            )

            depth_calls = calls[:, instruction_depth[calls[0]] == depth]
            static_local = static_instruction[instruction_ids]
            if self.architecture == "memory_dual":
                callee = self._dual_call_message(
                    static_local,
                    resource_function_state,
                    timing_function_state,
                    depth_calls,
                    instruction_local,
                    instruction_ids.numel(),
                )
            else:
                callee = self._generic_call_message(
                    static_local,
                    function_state,
                    depth_calls,
                    instruction_local,
                    instruction_ids.numel(),
                )
            instruction_state = self.instruction_input(static_local + callee)
            control_mask = instruction_depth[control_edges[0]] == depth
            local_control = instruction_local[control_edges[:, control_mask]]
            def_use_mask = instruction_depth[def_use_edges[0]] == depth
            local_def_use = instruction_local[def_use_edges[:, def_use_mask]]
            for layer in self.instruction_layers:
                instruction_state = layer(
                    instruction_state,
                    local_control,
                    local_def_use,
                    edge_features[control_type][control_mask],
                )
            if self.architecture == "memory_dual":
                instruction_state, memory_state = self._memory_exchange(
                    data,
                    instruction_state,
                    instruction_ids,
                    instruction_local,
                    memory_state,
                    edge_features,
                )

            instruction_group = block_local[instruction_block[instruction_ids]]
            cfg_mask = block_depth[cfg_edges[0]] == depth
            local_cfg = block_local[cfg_edges[:, cfg_mask]]
            if self.architecture == "memory_dual":
                resource_summary = self.resource_instruction_sum(
                    instruction_state, instruction_group, block_ids.numel()
                )
                timing_summary = self.timing_instruction_pool(
                    instruction_state, instruction_group, block_ids.numel()
                )
                resource_block = self.resource_block_input(
                    torch.cat(
                        [
                            base["block"][block_ids],
                            block_pragma[block_ids],
                            resource_summary,
                        ],
                        dim=-1,
                    )
                )
                timing_block = self.timing_block_input(
                    torch.cat(
                        [
                            base["block"][block_ids],
                            block_pragma[block_ids],
                            timing_summary,
                        ],
                        dim=-1,
                    )
                )
                resource_block = self.resource_block_layer(
                    resource_block, local_cfg
                )
                timing_block = self.timing_path(timing_block, local_cfg)
                block_group = function_local[block_function[block_ids]]
                resource_pool = self.resource_function_sum(
                    resource_block, block_group, function_ids.numel()
                )
                timing_pool = self.timing_function_pool(
                    timing_block, block_group, function_ids.numel()
                )
                resource_function = self.resource_function_input(
                    torch.cat(
                        [
                            base["function"][function_ids],
                            function_pragma[function_ids],
                            resource_pool,
                        ],
                        dim=-1,
                    )
                )
                timing_function = self.timing_function_input(
                    torch.cat(
                        [
                            base["function"][function_ids],
                            function_pragma[function_ids],
                            timing_pool,
                        ],
                        dim=-1,
                    )
                )
                resource_function_state = resource_function_state.index_copy(
                    0, function_ids, resource_function
                )
                timing_function_state = timing_function_state.index_copy(
                    0, function_ids, timing_function
                )
                continue

            instruction_pool = multi_pool(
                instruction_state,
                instruction_group,
                block_ids.numel(),
            )
            block_parts = [
                base["block"][block_ids],
                block_pragma[block_ids],
                instruction_pool,
            ]
            if self.architecture == "sequence":
                instruction_state, endpoint = self.sequence(
                    instruction_state,
                    instruction_group,
                    block_ids.numel(),
                )
                instruction_pool = multi_pool(
                    instruction_state,
                    instruction_group,
                    block_ids.numel(),
                )
                block_parts = [
                    base["block"][block_ids],
                    block_pragma[block_ids],
                    instruction_pool,
                    endpoint,
                ]
            block_state = self.block_input(torch.cat(block_parts, dim=-1))
            for layer in self.block_layers:
                block_state = layer(block_state, local_cfg)
            block_group = function_local[block_function[block_ids]]
            if self.architecture == "block_attention":
                block_state, function_pool = self.block_transformer(
                    block_state, block_group, function_ids.numel()
                )
            else:
                function_pool = self.function_pool(
                    block_state, block_group, function_ids.numel()
                )
            current_function = self.function_input(
                torch.cat(
                    [
                        base["function"][function_ids],
                        function_pragma[function_ids],
                        function_pool,
                    ],
                    dim=-1,
                )
            )
            function_state = function_state.index_copy(
                0, function_ids, current_function
            )

        entries = data["function"].is_entry.bool()
        function_batch = getattr(data["function"], "batch", None)
        if function_batch is None:
            function_batch = torch.zeros(
                data["function"].num_nodes,
                dtype=torch.long,
                device=entries.device,
            )
        if self.architecture == "memory_dual":
            resource_root = scatter(
                resource_function_state[entries],
                function_batch[entries],
                dim=0,
                dim_size=data.num_graphs,
                reduce="mean",
            )
            timing_root = scatter(
                timing_function_state[entries],
                function_batch[entries],
                dim=0,
                dim_size=data.num_graphs,
                reduce="mean",
            )
            resource_features = self._append_graph_context(data, resource_root)
            timing_features = self._append_graph_context(data, timing_root)
            return self.classifier(resource_features, timing_features)

        root = scatter(
            function_state[entries],
            function_batch[entries],
            dim=0,
            dim_size=data.num_graphs,
            reduce="mean",
        )
        return self.classifier(self._append_graph_context(data, root))


class CDFGHierarchicalSequence(CDFGHierarchicalExperimental):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, architecture="sequence", **kwargs)


class CDFGHierarchicalBlockAttention(CDFGHierarchicalExperimental):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, architecture="block_attention", **kwargs)


class CDFGHierarchicalMemoryDual(CDFGHierarchicalExperimental):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, architecture="memory_dual", **kwargs)
