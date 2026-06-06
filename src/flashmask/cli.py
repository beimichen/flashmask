"""Unified ``flashmask`` command-line entrypoint.

Thin dispatcher over the module CLIs. The detector trainer is Hydra-driven and
the apps own their own arg parsing, so those are invoked directly via
``python -m ...`` (the dispatcher prints the exact command).
"""

from __future__ import annotations

import sys

_HELP = """\
flashmask <command> [args]

Data
  scrape            Download a PDF corpus (arXiv / PMC / CORE)
  pdf2img           Render PDF pages to images
  synth             Generate synthetic text-on-background data

Modeling
  train-classifier  Train the text/diagram classifier
  evaluate          Evaluate a detector on the test split
  export            Export a detector to ONNX (+ parity check)

Active learning
  active mine       Rank an unlabeled pool by detector uncertainty (+ --stage)

Serving
  serve             Launch the FastAPI service (uvicorn)

Hydra / app entrypoints (run directly):
  python -m flashmask.modeling.train_detector train=finetune_real data=real
  streamlit run apps/label_tool.py
  python apps/demo.py
"""


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(_HELP)
        return

    command, rest = argv[0], argv[1:]
    if command == "scrape":
        from flashmask.data.scrape import main as run
    elif command == "pdf2img":
        from flashmask.data.pdf_to_image import main as run
    elif command == "synth":
        from flashmask.data.synthetic import main as run
    elif command == "train-classifier":
        from flashmask.modeling.train_classifier import main as run
    elif command == "evaluate":
        from flashmask.modeling.evaluate import main as run
    elif command == "export":
        from flashmask.modeling.export import main as run
    elif command == "active":
        from flashmask.active.mine import main as run

        # support "flashmask active mine ..." (mine is the only subcommand for now)
        if rest and rest[0] == "mine":
            rest = rest[1:]
    elif command == "serve":
        return _serve(rest)
    else:
        print(f"Unknown command: {command}\n\n{_HELP}")
        sys.exit(2)
    run(rest)


def _serve(rest: list[str]) -> None:
    import uvicorn

    host = rest[0] if rest else "127.0.0.1"
    uvicorn.run("flashmask.serving.api:app", host=host, port=8000)


if __name__ == "__main__":
    main()
