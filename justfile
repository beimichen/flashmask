# flashmask task runner — `just --list` to see everything.
# Requires uv (https://docs.astral.sh/uv). Most recipes wrap `uv run`.

set dotenv-load := true

# Show available recipes
default:
    @just --list

# Install the project + dev tools into a local .venv
setup:
    uv sync

# Install with every optional extra (train, label, scrape)
setup-all:
    uv sync --all-extras

# Lint + format-check (what CI runs)
lint:
    uv run ruff check .
    uv run ruff format --check .

# Auto-fix lint issues and format
format:
    uv run ruff check --fix .
    uv run ruff format .

# Static type check
typecheck:
    uv run pyright

# Run the test suite
test:
    uv run pytest

# Generate the committed placeholder sample dataset
sample:
    uv run python scripts/make_sample_images.py

# Launch the Gradio demo (needs models/detector.onnx — see models/README.md)
demo:
    uv run python apps/demo.py

# Launch the FastAPI service
serve:
    uv run flashmask serve

# Launch the Streamlit labeling tool
label:
    uv run --extra label streamlit run apps/label_tool.py

# Phase 1: pretrain the detector on synthetic data
train-pretrain:
    uv run --extra train python -m flashmask.modeling.train_detector train=pretrain_synthetic data=synthetic

# Phase 2: fine-tune the detector on real data from the phase-1 checkpoint
train-finetune weights="runs/pretrain_synthetic/weights/best.pt":
    uv run --extra train python -m flashmask.modeling.train_detector train=finetune_real data=real model.weights={{weights}}

# Evaluate a detector on the real test split
evaluate weights data:
    uv run --extra train python -m flashmask.modeling.evaluate -w {{weights}} -d {{data}}

# Export a detector to ONNX (+ parity check if --parity-image given)
export weights:
    uv run --extra train python -m flashmask.modeling.export -w {{weights}}

# Active learning: rank an unlabeled pool by detector uncertainty, stage winners
active-mine pool top_k="50":
    uv run flashmask active mine --pool {{pool}} --top-k {{top_k}} --stage
