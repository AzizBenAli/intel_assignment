from __future__ import annotations

import argparse
from pathlib import Path

from .parser import OpenVINOParser
from .renderer import SVGRenderer
from .transforms import (
    ConstantHidingPass,
    NodeHighlightingPass,
    PassPipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-graph",
        description="Visualize an OpenVINO model as an SVG graph.",
    )

    parser.add_argument(
        "model",
        type=Path,
        help="Path to the OpenVINO model XML file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("graph.svg"),
        help="Output SVG path (default: graph.svg).",
    )
    parser.add_argument(
        "--no-passes",
        action="store_true",
        help="Disable all transformation passes.",
    )
    parser.add_argument(
        "--pass",
        dest="passes",
        action="append",
        choices=["highlight", "hide-constants"],
        help=(
            "Pass to enable. Can be specified more than once. "
            "Default: enable both passes."
        ),
    )
    parser.add_argument(
        "--highlight",
        action="append",
        default=[],
        metavar="NODE_NAME",
        help="Highlight a node with this exact name (repeatable).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    argument_parser = build_parser()
    args = argument_parser.parse_args(argv)

    if not args.model.is_file():
        argument_parser.error(
            f"Model XML file does not exist: {args.model}"
        )

    if args.model.suffix.lower() != ".xml":
        argument_parser.error("The model must be an OpenVINO .xml file.")

    if args.output.suffix.lower() != ".svg":
        argument_parser.error("The output filename must end with .svg.")

    try:
        graph = OpenVINOParser().parse(args.model)

        if not args.no_passes:
            pipeline = PassPipeline()

            selected_passes = (
                args.passes
                if args.passes is not None
                else ["highlight", "hide-constants"]
            )

            for pass_name in selected_passes:
                if pass_name == "highlight":
                    pipeline.add(NodeHighlightingPass(args.highlight))
                elif pass_name == "hide-constants":
                    pipeline.add(ConstantHidingPass())

            graph = pipeline.run(graph)

        output_path = SVGRenderer().render(graph, args.output)

    except Exception as exc:
        argument_parser.exit(
            1,
            f"model-graph: error: {exc}\n",
        )

    print(f"SVG created: {output_path}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
