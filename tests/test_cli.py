import pytest

from model_graph.cli import build_parser


def test_cli_default_arguments():
    parser = build_parser()
    args = parser.parse_args(["model.xml"])

    assert str(args.model) == "model.xml"
    assert str(args.output) == "graph.svg"
    assert args.no_passes is False
    assert args.passes is None
    assert args.highlight == []


def test_cli_accepts_custom_output():
    parser = build_parser()
    args = parser.parse_args(
        ["model.xml", "--output", "result.svg"]
    )

    assert str(args.output) == "result.svg"


def test_cli_can_disable_all_passes():
    parser = build_parser()
    args = parser.parse_args(
        ["model.xml", "--no-passes"]
    )

    assert args.no_passes is True


def test_cli_preserves_requested_pass_order():
    parser = build_parser()
    args = parser.parse_args(
        [
            "model.xml",
            "--pass",
            "hide-constants",
            "--pass",
            "highlight",
        ]
    )

    assert args.passes == ["hide-constants", "highlight"]


def test_cli_accepts_multiple_highlight_names():
    parser = build_parser()
    args = parser.parse_args(
        [
            "model.xml",
            "--highlight",
            "input",
            "--highlight",
            "add",
        ]
    )

    assert args.highlight == ["input", "add"]


def test_cli_rejects_unknown_pass():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["model.xml", "--pass", "unknown"]
        )
