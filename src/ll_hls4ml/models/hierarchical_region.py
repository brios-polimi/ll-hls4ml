"""Loop-region hierarchical encoder for schema-v3 LLVM CDFGs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax

from ll_hls4ml.io.schema import (
    DERIVED_DEF_USE_EDGE,
    LOOP_HIERARCHY_SCHEMA_VERSION,
    REGION_NODE_TYPES,
)
from ll_hls4ml.models.hierarchical import (
    CDFGHierarchical,
    _messages,
    _reverse_messages,
)
from ll_hls4ml.models.input_projection import CDFGRegionInputProjection
from ll_hls4ml.models.readout import multi_pool


def _cardinality_stats(
    source: torch.Tensor,
    edge_index: torch.Tensor,
    target_count: int,
    edge_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate signed-sum, mean, max, std, and log-degree statistics."""

    hidden_dim = source.size(-1)
    if edge_index.numel() == 0:
        zero = source.sum(dim=0, keepdim=True)
        if edge_features is not None:
            zero = zero + edge_features.sum(dim=0, keepdim=True)
        zero = zero.expand(target_count, hidden_dim) * 0
        return torch.cat([zero, zero, zero, zero, zero[:, :1]], dim=-1)

    values = source[edge_index[0]]
    if edge_features is not None:
        values = values + edge_features
    target = edge_index[1]
    summed = scatter(values, target, dim=0, dim_size=target_count, reduce="sum")
    mean = scatter(values, target, dim=0, dim_size=target_count, reduce="mean")
    maximum = scatter(values, target, dim=0, dim_size=target_count, reduce="max")
    mean_square = scatter(
        values.square(), target, dim=0, dim_size=target_count, reduce="mean"
    )
    counts = scatter(
        torch.ones_like(target, dtype=values.dtype),
        target,
        dim=0,
        dim_size=target_count,
        reduce="sum",
    ).unsqueeze(-1)
    present = counts > 0
    maximum = torch.where(present, maximum, torch.zeros_like(maximum))
    std = (mean_square - mean.square()).clamp_min(0).add(1e-6).sqrt()
    std = torch.where(present, std, torch.zeros_like(std))
    signed_log_sum = torch.sign(summed) * torch.log1p(summed.abs())
    return torch.cat(
        [signed_log_sum, mean, maximum, std, torch.log1p(counts)], dim=-1
    )


class CardinalityInstructionFlowLayer(nn.Module):
    """Bidirectional typed instruction messages that retain multiplicity."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.control_forward = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.control_reverse = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.data_forward = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.data_reverse = nn.Linear(hidden_dim, hidden_dim, bias=False)
        pooled_dim = 4 * hidden_dim + 1
        self.message = nn.Linear(4 * pooled_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        state: torch.Tensor,
        control_edges: torch.Tensor,
        def_use_edges: torch.Tensor,
        control_features: torch.Tensor | None,
    ) -> torch.Tensor:
        control_reverse = control_edges.flip(0)
        data_reverse = def_use_edges.flip(0)
        statistics = torch.cat(
            [
                _cardinality_stats(
                    self.control_forward(state),
                    control_edges,
                    state.size(0),
                    control_features,
                ),
                _cardinality_stats(
                    self.control_reverse(state),
                    control_reverse,
                    state.size(0),
                    control_features,
                ),
                _cardinality_stats(
                    self.data_forward(state), def_use_edges, state.size(0)
                ),
                _cardinality_stats(
                    self.data_reverse(state), data_reverse, state.size(0)
                ),
            ],
            dim=-1,
        )
        message = self.message(statistics)
        return self.norm(state + self.dropout(F.relu(message)))


class CardinalityBlockFlowLayer(nn.Module):
    """Bidirectional basic-block CFG messages that retain branch cardinality."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.forward_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reverse_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        pooled_dim = 4 * hidden_dim + 1
        self.message = nn.Linear(2 * pooled_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, state: torch.Tensor, cfg_edges: torch.Tensor) -> torch.Tensor:
        statistics = torch.cat(
            [
                _cardinality_stats(
                    self.forward_proj(state), cfg_edges, state.size(0)
                ),
                _cardinality_stats(
                    self.reverse_proj(state), cfg_edges.flip(0), state.size(0)
                ),
            ],
            dim=-1,
        )
        return self.norm(
            state + self.dropout(F.relu(self.message(statistics)))
        )


class HardwareAlignedPool(nn.Module):
    """Keep additive resource and smooth critical-path summaries separate."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.resource = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.timing = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.critical_score = nn.Linear(hidden_dim, 1, bias=False)
        self.sharing = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self, state: torch.Tensor, owner: torch.Tensor, size: int
    ) -> torch.Tensor:
        hidden_dim = state.size(-1)
        if state.numel() == 0:
            dependency = (
                self.resource(state).sum()
                + self.timing(state).sum()
                + self.critical_score(state).sum()
                + self.sharing(state).sum()
            )
            zero = state.new_zeros((size, hidden_dim)) + dependency * 0
            return torch.cat([zero, zero, zero, zero, zero[:, :1]], dim=-1)

        resource = F.softplus(self.resource(state))
        resource_sum = scatter(
            resource, owner, dim=0, dim_size=size, reduce="sum"
        )
        additive = torch.log1p(resource_sum)

        weights = softmax(
            self.critical_score(state).squeeze(-1), owner, num_nodes=size
        ).unsqueeze(-1)
        timing = F.softplus(self.timing(state))
        critical = torch.log1p(
            scatter(
                weights * timing, owner, dim=0, dim_size=size, reduce="sum"
            )
        )

        sharing = self.sharing(state)
        mean = scatter(sharing, owner, dim=0, dim_size=size, reduce="mean")
        mean_square = scatter(
            sharing.square(), owner, dim=0, dim_size=size, reduce="mean"
        )
        counts = scatter(
            torch.ones_like(owner, dtype=state.dtype),
            owner,
            dim=0,
            dim_size=size,
            reduce="sum",
        ).unsqueeze(-1)
        present = counts > 0
        std = (mean_square - mean.square()).clamp_min(0).add(1e-6).sqrt()
        std = torch.where(present, std, torch.zeros_like(std))
        return torch.cat(
            [additive, critical, mean, std, torch.log1p(counts)], dim=-1
        )


class SmoothPathSummary(nn.Module):
    """Damped max-plus relaxation over directed control/data dependencies."""

    def __init__(self, hidden_dim: int, steps: int):
        super().__init__()
        self.steps = steps
        self.local_delay = nn.Linear(hidden_dim, 1)
        self.projection = nn.Linear(3, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.raw_decay = nn.Parameter(torch.zeros(()))

    def forward(self, state: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        delay = F.softplus(self.local_delay(state))
        arrival = delay
        decay = torch.sigmoid(self.raw_decay)
        for _ in range(self.steps):
            predecessor = scatter(
                arrival[edges[0]],
                edges[1],
                dim=0,
                dim_size=state.size(0),
                reduce="max",
            )
            counts = scatter(
                torch.ones_like(edges[1], dtype=state.dtype),
                edges[1],
                dim=0,
                dim_size=state.size(0),
                reduce="sum",
            ).unsqueeze(-1)
            predecessor = torch.where(
                counts > 0, predecessor, torch.zeros_like(predecessor)
            )
            arrival = delay + decay * predecessor
        path_features = torch.cat(
            [
                torch.log1p(delay),
                torch.log1p(arrival),
                torch.log1p((arrival - delay).clamp_min(0)),
            ],
            dim=-1,
        )
        return self.norm(state + self.projection(path_features))


class CDFGHierarchicalRegion(CDFGHierarchical):
    """Compose instructions, blocks, nested loops, and functions leaf-to-root."""

    def __init__(
        self,
        *args,
        cardinality_messages: bool = True,
        composition: str = "generic",
        critical_path_steps: int = 3,
        **kwargs,
    ):
        if composition not in {"generic", "hardware_aligned"}:
            raise ValueError(
                "composition must be 'generic' or 'hardware_aligned'"
            )
        if critical_path_steps < 1:
            raise ValueError("critical_path_steps must be positive")
        super().__init__(*args, **kwargs)
        self.composition = composition
        hidden_dim = self.callee_proj.in_features
        instruction_layers = len(self.instruction_layers)
        block_layers = len(self.block_layers)
        edge_pos_vocab_size = self.input_proj.edge_pos_emb.num_embeddings - 1
        instruction_vocab_size = self.input_proj.instruction_emb.num_embeddings
        self.input_proj = CDFGRegionInputProjection(
            instruction_vocab_size, edge_pos_vocab_size, hidden_dim
        )
        if cardinality_messages:
            dropout = self.instruction_layers[0].dropout.p if instruction_layers else 0.0
            self.instruction_layers = nn.ModuleList(
                CardinalityInstructionFlowLayer(hidden_dim, dropout)
                for _ in range(instruction_layers)
            )
            block_dropout = self.block_layers[0].dropout.p if block_layers else 0.0
            self.block_layers = nn.ModuleList(
                CardinalityBlockFlowLayer(hidden_dim, block_dropout)
                for _ in range(block_layers)
            )
        pooled_dim = 4 * hidden_dim + 1
        self.loop_input = nn.Sequential(
            nn.Linear(2 * pooled_dim + 2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.function_input = nn.Sequential(
            nn.Linear(2 * pooled_dim + 2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        if composition == "hardware_aligned":
            self.composition_pool = HardwareAlignedPool(hidden_dim)
            self.instruction_path = SmoothPathSummary(
                hidden_dim, critical_path_steps
            )
            self.block_path = SmoothPathSummary(hidden_dim, critical_path_steps)
            self.loop_gate = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid()
            )
        else:
            self.composition_pool = None
            self.instruction_path = None
            self.block_path = None
            self.loop_gate = None

    def _pool(
        self, state: torch.Tensor, owner: torch.Tensor, size: int
    ) -> torch.Tensor:
        if self.composition_pool is None:
            return multi_pool(state, owner, size)
        return self.composition_pool(state, owner, size)

    def encode(self, data):
        versions = torch.as_tensor(
            data.hierarchy_schema_version,
            device=data["instruction"].x.device,
        )
        if not bool(torch.all(versions == LOOP_HIERARCHY_SCHEMA_VERSION)):
            raise ValueError("Region model requires hierarchical tensor schema 3")
        x_dict = {node_type: data[node_type].x for node_type in REGION_NODE_TYPES}
        edge_attr_dict = {
            edge_type: data[edge_type].edge_attr.long()
            for edge_type in (
                ("instruction", "control", "instruction"),
                ("variable", "operand", "instruction"),
                ("constant", "operand", "instruction"),
            )
        }
        base, edge_features = self.input_proj(x_dict, edge_attr_dict)
        instruction_block, block_function = self._owners(data)
        call_depth = data["function"].call_depth.long()
        instruction_depth = data["instruction"].call_depth.long()
        block_depth = data["block"].call_depth.long()
        loop_call_depth = data["loop"].call_depth.long()
        loop_nesting_depth = data["loop"].nesting_depth.long()
        reachable_functions = data["function"].is_reachable.bool()

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
        direct_pragma = _messages(
            pragma,
            data[("pragma", "applies_to", "instruction")].edge_index,
            data["instruction"].num_nodes,
        )
        variable_operand = ("variable", "operand", "instruction")
        constant_operand = ("constant", "operand", "instruction")
        static_instruction = (
            base["instruction"]
            + direct_pragma
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

        function_state = torch.zeros_like(base["function"], dtype=torch.float32)
        loop_state = torch.zeros_like(base["loop"], dtype=torch.float32)
        calls = data[("instruction", "calls", "function")].edge_index
        control = ("instruction", "control", "instruction")
        block_cfg = data[("block", "control", "block")].edge_index
        control_edges = data[control].edge_index
        def_use_edges = data[DERIVED_DEF_USE_EDGE].edge_index
        block_pragma = _messages(
            pragma,
            data[("pragma", "applies_to", "block")].edge_index,
            data["block"].num_nodes,
        )
        loop_pragma = _messages(
            pragma,
            data[("pragma", "applies_to", "loop")].edge_index,
            data["loop"].num_nodes,
        )
        function_pragma = _messages(
            pragma,
            data[("pragma", "applies_to", "function")].edge_index,
            data["function"].num_nodes,
        )
        loop_blocks = data[("loop", "contains", "block")].edge_index
        loop_children = data[("loop", "contains", "loop")].edge_index
        function_loops = data[("function", "contains", "loop")].edge_index
        block_in_loop = torch.zeros(
            data["block"].num_nodes, dtype=torch.bool, device=block_depth.device
        )
        block_in_loop[loop_blocks[1]] = True

        max_depth = int(call_depth[reachable_functions].max().item())
        for depth in range(max_depth + 1):
            function_ids = (
                (call_depth == depth) & reachable_functions
            ).nonzero(as_tuple=False).flatten()
            instruction_ids = (instruction_depth == depth).nonzero(
                as_tuple=False
            ).flatten()
            block_ids = (block_depth == depth).nonzero(as_tuple=False).flatten()
            loop_ids = (loop_call_depth == depth).nonzero(as_tuple=False).flatten()

            instruction_local = torch.full(
                (data["instruction"].num_nodes,),
                -1,
                dtype=torch.long,
                device=instruction_ids.device,
            )
            instruction_local[instruction_ids] = torch.arange(
                instruction_ids.numel(), device=instruction_ids.device
            )
            block_local = torch.full(
                (data["block"].num_nodes,),
                -1,
                dtype=torch.long,
                device=block_ids.device,
            )
            block_local[block_ids] = torch.arange(
                block_ids.numel(), device=block_ids.device
            )
            function_local = torch.full(
                (data["function"].num_nodes,),
                -1,
                dtype=torch.long,
                device=function_ids.device,
            )
            function_local[function_ids] = torch.arange(
                function_ids.numel(), device=function_ids.device
            )

            call_mask = instruction_depth[calls[0]] == depth
            depth_calls = calls[:, call_mask]
            callee_message = _messages(
                self.callee_proj(function_state[depth_calls[1]]),
                torch.stack(
                    [
                        torch.arange(depth_calls.size(1), device=depth_calls.device),
                        instruction_local[depth_calls[0]],
                    ]
                ),
                instruction_ids.numel(),
            )
            instruction_state = self.instruction_input(
                static_instruction[instruction_ids] + callee_message
            )
            control_mask = instruction_depth[control_edges[0]] == depth
            local_control = instruction_local[control_edges[:, control_mask]]
            def_use_mask = instruction_depth[def_use_edges[0]] == depth
            local_def_use = instruction_local[def_use_edges[:, def_use_mask]]
            for layer in self.instruction_layers:
                instruction_state = layer(
                    instruction_state,
                    local_control,
                    local_def_use,
                    edge_features[control][control_mask],
                )
            if self.instruction_path is not None:
                instruction_state = self.instruction_path(
                    instruction_state,
                    torch.cat([local_control, local_def_use], dim=1),
                )

            instruction_pool = self._pool(
                instruction_state,
                block_local[instruction_block[instruction_ids]],
                block_ids.numel(),
            )
            current_block = self.block_input(
                torch.cat(
                    [
                        base["block"][block_ids],
                        block_pragma[block_ids],
                        instruction_pool,
                    ],
                    dim=-1,
                )
            )
            cfg_mask = block_depth[block_cfg[0]] == depth
            local_cfg = block_local[block_cfg[:, cfg_mask]]
            block_state = current_block
            for layer in self.block_layers:
                block_state = layer(block_state, local_cfg)
            if self.block_path is not None:
                block_state = self.block_path(block_state, local_cfg)

            if loop_ids.numel():
                max_loop_depth = int(loop_nesting_depth[loop_ids].max().item())
                for nesting_depth in range(max_loop_depth, -1, -1):
                    region_ids = loop_ids[
                        loop_nesting_depth[loop_ids] == nesting_depth
                    ]
                    if not region_ids.numel():
                        continue
                    region_local = torch.full(
                        (data["loop"].num_nodes,),
                        -1,
                        dtype=torch.long,
                        device=region_ids.device,
                    )
                    region_local[region_ids] = torch.arange(
                        region_ids.numel(), device=region_ids.device
                    )
                    block_mask = (
                        (loop_nesting_depth[loop_blocks[0]] == nesting_depth)
                        & (loop_call_depth[loop_blocks[0]] == depth)
                    )
                    depth_loop_blocks = loop_blocks[:, block_mask]
                    region_block_pool = self._pool(
                        block_state[block_local[depth_loop_blocks[1]]],
                        region_local[depth_loop_blocks[0]],
                        region_ids.numel(),
                    )
                    child_mask = (
                        (loop_nesting_depth[loop_children[0]] == nesting_depth)
                        & (loop_call_depth[loop_children[0]] == depth)
                    )
                    depth_children = loop_children[:, child_mask]
                    child_pool = self._pool(
                        loop_state[depth_children[1]],
                        region_local[depth_children[0]],
                        region_ids.numel(),
                    )
                    if self.loop_gate is not None:
                        gate = 0.5 + self.loop_gate(
                            torch.cat(
                                [
                                    base["loop"][region_ids],
                                    loop_pragma[region_ids],
                                ],
                                dim=-1,
                            )
                        )
                        region_block_pool = region_block_pool.clone()
                        child_pool = child_pool.clone()
                        region_block_pool[:, :-1] *= gate.repeat(1, 4)
                        child_pool[:, :-1] *= gate.repeat(1, 4)
                    current_loop = self.loop_input(
                        torch.cat(
                            [
                                base["loop"][region_ids],
                                loop_pragma[region_ids],
                                region_block_pool,
                                child_pool,
                            ],
                            dim=-1,
                        )
                    )
                    loop_state = loop_state.index_copy(
                        0, region_ids, current_loop
                    )

            direct_blocks = block_ids[~block_in_loop[block_ids]]
            direct_block_pool = self._pool(
                block_state[block_local[direct_blocks]],
                function_local[block_function[direct_blocks]],
                function_ids.numel(),
            )
            top_mask = (
                (call_depth[function_loops[0]] == depth)
                & reachable_functions[function_loops[0]]
            )
            depth_top_loops = function_loops[:, top_mask]
            top_loop_pool = self._pool(
                loop_state[depth_top_loops[1]],
                function_local[depth_top_loops[0]],
                function_ids.numel(),
            )
            current_function = self.function_input(
                torch.cat(
                    [
                        base["function"][function_ids],
                        function_pragma[function_ids],
                        direct_block_pool,
                        top_loop_pool,
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
                device=function_state.device,
            )
        entry_pool = self._pool(
            function_state[entries], function_batch[entries], data.num_graphs
        )
        reachable_pool = self._pool(
            function_state[reachable_functions],
            function_batch[reachable_functions],
            data.num_graphs,
        )
        graph_state = self.root_readout(
            torch.cat([entry_pool, reachable_pool], dim=-1)
        )
        if data["loop"].num_nodes == 0:
            loop_dependency = sum(
                parameter.sum()
                for parameter in (
                    *self.input_proj.loop_proj.parameters(),
                    *self.loop_input.parameters(),
                    *(() if self.loop_gate is None else self.loop_gate.parameters()),
                )
            )
            graph_state = graph_state + loop_dependency * 0
        features = [graph_state]
        if self.use_global_features:
            features.append(self.global_features(data))
        if self.use_context:
            features.append(self.context_encoder(data))
        return torch.cat(features, dim=-1)
