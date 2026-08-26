"""Small transform registry for reproducible graph augmentation experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol

from torch_geometric.data import HeteroData


class GraphTransform(Protocol):
    def __call__(self, graph: HeteroData) -> HeteroData: ...


_TRANSFORMS: dict[str, Callable[..., GraphTransform]] = {}


def register(name: str):
    """Register a transform factory by stable configuration name."""

    def decorator(factory: Callable[..., GraphTransform]):
        if name in _TRANSFORMS:
            raise KeyError(f"Graph transform already registered: {name}")
        _TRANSFORMS[name] = factory
        return factory

    return decorator


def resolve(name: str, **kwargs) -> GraphTransform:
    if name not in _TRANSFORMS:
        raise KeyError(f"Unknown graph transform {name!r}; available: {sorted(_TRANSFORMS)}")
    return _TRANSFORMS[name](**kwargs)


class Compose:
    def __init__(self, transforms: Iterable[GraphTransform]):
        self.transforms = tuple(transforms)

    def __call__(self, graph: HeteroData) -> HeteroData:
        for transform in self.transforms:
            graph = transform(graph)
        return graph
