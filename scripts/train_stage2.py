#!/usr/bin/env python3
"""Train ResNet18 ROI classifier for stage-2 defect typing."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from config import STAGE2_ROOT, UNIFIED_CLASSES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_dir = STAGE2_ROOT / "train"
    val_dir = STAGE2_ROOT / "val"
    if not train_dir.exists():
        raise SystemExit(f"Missing {train_dir}. Run prepare_stage2_roi.py first.")

    tf_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.485, 0.456, 0.406]),
    ])
    tf_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.485, 0.456, 0.406]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=tf_train)
    val_ds = datasets.ImageFolder(val_dir, transform=tf_val) if val_dir.exists() else None
    print("Classes:", train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0) if val_ds else None
    )

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    out_dir = STAGE2_ROOT.parent / "runs" / "stage2_cls"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        msg = f"epoch {epoch+1}/{args.epochs} train_loss={loss_sum/len(train_loader):.4f}"
        if val_loader:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(args.device), y.to(args.device)
                    pred = model(x).argmax(1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            acc = correct / max(total, 1)
            msg += f" val_acc={acc:.4f}"
            if acc > best_acc:
                best_acc = acc
                torch.save(
                    {"model": model.state_dict(), "classes": train_ds.classes},
                    out_dir / "best.pt",
                )
        print(msg)

    torch.save({"model": model.state_dict(), "classes": train_ds.classes}, out_dir / "last.pt")
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()
