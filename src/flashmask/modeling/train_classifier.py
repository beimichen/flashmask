"""One configurable image classifier trainer (text/non-text and diagram/photo).

Merges the old ``train_text_classifier*.py`` and ``train_diagram_classifier.py``
into a single transfer-learning trainer. The backbone, input size, optional Canny
edge augmentation, and class-balanced sampling are all parameters, and the best
model is exported to ONNX for the inference pipeline.

Expects an ``ImageFolder`` layout::  <data_dir>/{train,val}/<class>/*.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from flashmask.config import set_seed


@dataclass
class ClassifierConfig:
    data_dir: Path
    backbone: str = "resnet18"  # or "mobilenet_v3_small"
    input_size: int = 256
    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-4
    edge_aug_p: float = 0.0  # text classifier used ~0.1-0.3; diagram used 0.0
    balance_classes: bool = True
    output: Path = Path("models/classifier.onnx")
    seed: int = 42


def _to_edge_image(img):
    """Canny-edge augmentation (module-level so DataLoader workers can pickle it)."""
    import cv2
    import numpy as np
    from PIL import Image

    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return Image.fromarray(np.stack([cv2.Canny(gray, 50, 150)] * 3, axis=-1))


def _build_model(backbone: str, num_classes: int):
    import torch.nn as nn
    from torchvision import models

    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


def train_classifier(cfg: ClassifierConfig) -> float:
    """Train, keep the best-val-accuracy weights, export to ONNX. Returns best acc."""
    import torch
    import torch.nn as nn
    from torch.optim import Adam, lr_scheduler
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, transforms

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.Resize((cfg.input_size, cfg.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.RandomApply([transforms.Lambda(_to_edge_image)], p=cfg.edge_aug_p),
            transforms.ToTensor(),
            norm,
        ]
    )
    val_tf = transforms.Compose(
        [transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), norm]
    )

    sets = {
        p: datasets.ImageFolder(str(cfg.data_dir / p), tf)
        for p, tf in (("train", train_tf), ("val", val_tf))
    }
    train_sampler = None
    if cfg.balance_classes:
        counts = [0] * len(sets["train"].classes)
        for _, label in sets["train"].imgs:
            counts[label] += 1
        weights = [1.0 / counts[label] for _, label in sets["train"].imgs]
        train_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    loaders = {
        "train": DataLoader(
            sets["train"],
            cfg.batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            num_workers=4,
        ),
        "val": DataLoader(sets["val"], cfg.batch_size, shuffle=False, num_workers=4),
    }
    print("Classes:", sets["train"].classes)

    model = _build_model(cfg.backbone, len(sets["train"].classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_acc, best_state = 0.0, model.state_dict()
    for epoch in range(cfg.epochs):
        for phase in ("train", "val"):
            model.train() if phase == "train" else model.eval()
            running, correct = 0.0, 0
            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running += loss.item() * inputs.size(0)
                correct += int((outputs.argmax(1) == labels).sum())
            if phase == "train":
                scheduler.step()
            acc = correct / len(sets[phase])
            print(
                f"epoch {epoch + 1}/{cfg.epochs} {phase}: loss={running / len(sets[phase]):.4f} acc={acc:.4f}"
            )
            if phase == "val" and acc > best_acc:
                best_acc, best_state = acc, model.state_dict()

    model.load_state_dict(best_state)
    _export_onnx(model, cfg.output, cfg.input_size, device)
    print(f"Best val acc: {best_acc:.4f} -> {cfg.output}")
    return best_acc


def _export_onnx(model, output: Path, input_size: int, device) -> None:
    import torch

    model.eval()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train a text/diagram image classifier.")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--edge-aug-p", type=float, default=0.0)
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("models/classifier.onnx"))
    args = ap.parse_args(argv)
    train_classifier(
        ClassifierConfig(
            data_dir=args.data_dir,
            backbone=args.backbone,
            input_size=args.input_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            edge_aug_p=args.edge_aug_p,
            balance_classes=not args.no_balance,
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()
