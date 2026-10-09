
from __future__ import annotations

from dataclasses import dataclass

from ..graph import Graph
from .base import TransformationPass


@dataclass
class _PassEntry:
    transformation: TransformationPass
    enabled: bool = True


class PassPipeline:
    """Run graph transformation passes in a configurable order."""

    def __init__(self) -> None:
        self._entries: list[_PassEntry] = []

    @property
    def passes(self) -> tuple[TransformationPass, ...]:
        """Return the configured passes in execution order."""
        return tuple(
            entry.transformation for entry in self._entries
        )

    def add(
        self,
        transformation: TransformationPass,
        *,
        enabled: bool = True,
    ) -> None:
        """Append a transformation to the pipeline."""
        if any(
            entry.transformation.name == transformation.name
            for entry in self._entries
        ):
            raise ValueError(
                f"Pass already registered: {transformation.name}"
            )

        self._entries.append(
            _PassEntry(
                transformation=transformation,
                enabled=enabled,
            )
        )

    def set_enabled(self, pass_name: str, enabled: bool) -> None:
        """Enable or disable a registered pass by name."""
        for entry in self._entries:
            if entry.transformation.name == pass_name:
                entry.enabled = enabled
                return

        raise KeyError(f"Pass not found: {pass_name}")

    def run(self, graph: Graph) -> Graph:
        """Execute enabled passes sequentially."""
        current_graph = graph

        for entry in self._entries:
            if not entry.enabled:
                continue

            result = entry.transformation.apply(current_graph)

            if not isinstance(result, Graph):
                raise TypeError(
                    f"{entry.transformation.name}.apply() "
                    "must return a Graph"
                )

            result.validate()
            current_graph = result

        return current_graph