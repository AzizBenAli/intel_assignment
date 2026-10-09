
import pytest

from model_graph import Graph
from model_graph.transforms import PassPipeline, TransformationPass


class RecordingPass(TransformationPass):
    def __init__(self, label: str, history: list[str]) -> None:
        self.label = label
        self.history = history

    @property
    def name(self) -> str:
        return self.label

    def apply(self, graph: Graph) -> Graph:
        self.history.append(self.label)
        return graph


def test_pipeline_runs_passes_in_order() -> None:
    history: list[str] = []
    pipeline = PassPipeline()

    pipeline.add(RecordingPass("first", history))
    pipeline.add(RecordingPass("second", history))
    pipeline.add(RecordingPass("third", history))

    graph = Graph()
    result = pipeline.run(graph)

    assert history == ["first", "second", "third"]
    assert result is graph


def test_pipeline_skips_disabled_passes() -> None:
    history: list[str] = []
    pipeline = PassPipeline()

    pipeline.add(RecordingPass("first", history))
    pipeline.add(
        RecordingPass("second", history),
        enabled=False,
    )
    pipeline.add(RecordingPass("third", history))

    pipeline.run(Graph())

    assert history == ["first", "third"]


def test_pipeline_can_reenable_a_pass() -> None:
    history: list[str] = []
    pipeline = PassPipeline()

    pipeline.add(
        RecordingPass("highlight", history),
        enabled=False,
    )

    pipeline.set_enabled("highlight", True)
    pipeline.run(Graph())

    assert history == ["highlight"]


def test_pipeline_rejects_duplicate_pass_names() -> None:
    pipeline = PassPipeline()
    pipeline.add(RecordingPass("first", []))

    with pytest.raises(ValueError, match="already registered"):
        pipeline.add(RecordingPass("first", []))


def test_pipeline_rejects_unknown_pass_name() -> None:
    pipeline = PassPipeline()

    with pytest.raises(KeyError, match="Pass not found"):
        pipeline.set_enabled("missing", True)


def test_pipeline_rejects_invalid_pass_result() -> None:
    class BrokenPass(TransformationPass):
        def apply(self, graph: Graph) -> Graph:
            return None  # type: ignore[return-value]

    pipeline = PassPipeline()
    pipeline.add(BrokenPass())

    with pytest.raises(TypeError, match="must return a Graph"):
        pipeline.run(Graph())