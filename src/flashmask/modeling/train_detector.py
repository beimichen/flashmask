"""One configurable YOLO trainer — the single replacement for the 9 old scripts.

Every former ``train_text_detector_*.py`` variant differed only in
hyperparameters, model size, data, and which training *phase* it ran. Those are
all expressed as Hydra config groups now, so the two-phase recipe is two config
selections rather than two copy-pasted files:

    # Phase 1 — pretrain on synthetic data (heavy augmentation)
    python -m flashmask.modeling.train_detector train=pretrain_synthetic data=synthetic

    # Phase 2 — fine-tune on real data (light augmentation), from phase-1 weights
    python -m flashmask.modeling.train_detector train=finetune_real data=real \
        model.weights=runs/pretrain_synthetic/weights/best.pt

The resolved config is logged to MLflow (when installed) so each run is fully
reproducible from its recorded parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf

from flashmask.config import PROJECT_ROOT, paths, set_seed
from flashmask.data.dataset import build_yolo_dataset

_CONFIG_DIR = str(PROJECT_ROOT / "configs")


def _resolve_data_yaml(cfg: DictConfig) -> str:
    """Use an existing ``data.yaml`` or build one from review metadata on the fly."""
    if cfg.data.get("yaml"):
        return str(Path(cfg.data.yaml).expanduser())
    metadata = Path(cfg.data.metadata).expanduser()
    out_dir = Path(cfg.data.get("out_dir", paths.data_processed / cfg.data.name))
    return str(build_yolo_dataset(metadata, out_dir, tuple(cfg.data.ratios)))


def train(cfg: DictConfig) -> dict:
    """Train a detector for one phase and return the validation metrics."""
    from ultralytics import YOLO

    set_seed(cfg.seed)
    data_yaml = _resolve_data_yaml(cfg)
    model = YOLO(cfg.model.weights)

    overrides = cast(dict, OmegaConf.to_container(cfg.train.args, resolve=True))
    results = model.train(
        data=data_yaml,
        imgsz=cfg.model.imgsz,
        seed=cfg.seed,
        deterministic=True,
        project=str(Path(cfg.project_dir)),
        name=cfg.run_name,
        exist_ok=True,
        **overrides,
    )

    _log_mlflow(cfg, results)
    return getattr(results, "results_dict", {})


def _log_mlflow(cfg: DictConfig, results) -> None:
    try:
        import mlflow
    except ImportError:
        return
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(
            {
                "model": cfg.model.weights,
                "imgsz": cfg.model.imgsz,
                "seed": cfg.seed,
                **cfg.train.args,
            }
        )
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "resolved_config.yaml")
        for k, v in getattr(results, "results_dict", {}).items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k.replace("(B)", "").replace("/", "_"), float(v))


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    metrics = train(cfg)
    print({k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))})


if __name__ == "__main__":
    main()
