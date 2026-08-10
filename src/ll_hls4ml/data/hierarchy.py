"""Small, model-independent helpers for strict bottom-up function scheduling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionSchedule:
    depth: tuple[int, ...]
    roots: tuple[bool, ...]
    entry: tuple[bool, ...]
    reachable: tuple[bool, ...]


def function_schedule(
    num_functions: int,
    calls: list[tuple[int, int]],
    instruction_counts: list[int] | None = None,
) -> FunctionSchedule:
    """Return leaf-first depths and roots for an acyclic caller→callee graph."""

    callees = [set() for _ in range(num_functions)]
    called = set()
    for caller, callee in calls:
        if not (0 <= caller < num_functions and 0 <= callee < num_functions):
            raise ValueError(f"Invalid function call {caller} -> {callee}")
        callees[caller].add(callee)
        called.add(callee)

    depth: list[int | None] = [None] * num_functions
    active: set[int] = set()

    def visit(function: int) -> int:
        if depth[function] is not None:
            return depth[function]
        if function in active:
            raise ValueError(
                "Recursive function-call component found; strict bottom-up "
                "encoding requires an acyclic HLS call graph"
            )
        active.add(function)
        value = 0
        if callees[function]:
            value = 1 + max(visit(callee) for callee in callees[function])
        active.remove(function)
        depth[function] = value
        return value

    for function in range(num_functions):
        visit(function)
    roots = tuple(function not in called for function in range(num_functions))
    weights = instruction_counts or [1] * num_functions
    if len(weights) != num_functions:
        raise ValueError("instruction_counts must match num_functions")

    def reachable(function: int) -> set[int]:
        result = {function}
        for callee in callees[function]:
            result.update(reachable(callee))
        return result

    root_ids = [index for index, is_root in enumerate(roots) if is_root]
    entry_id = max(
        root_ids,
        key=lambda function: (
            sum(weights[item] for item in reachable(function)),
            -function,
        ),
        default=None,
    )
    entry_reachable = reachable(entry_id) if entry_id is not None else set()
    return FunctionSchedule(
        depth=tuple(int(value) for value in depth),
        roots=roots,
        entry=tuple(function == entry_id for function in range(num_functions)),
        reachable=tuple(
            function in entry_reachable
            for function in range(num_functions)
        ),
    )
