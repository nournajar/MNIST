"""Evaluate a saved MNIST checkpoint on the test set.

Usage:
    python evaluate.py [--checkpoint PATH] [--data-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import MNISTNet


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = MNISTNet(num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    model, checkpoint = load_model(Path(args.checkpoint), device)
    print(
        f"loaded {args.checkpoint} (trained epoch {checkpoint['epoch']}, "
        f"checkpoint test_acc={checkpoint['test_acc']:.4f})"
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((checkpoint["input_mean"],), (checkpoint["input_std"],)),
        ]
    )
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2)

    confusion = torch.zeros(10, 10, dtype=torch.long)
    correct = 0
    total = 0
    for images, labels in loader:
        logits = model(images.to(device))
        preds = logits.argmax(dim=1).cpu()
        for t, p in zip(labels, preds):
            confusion[t.item(), p.item()] += 1
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"test accuracy on {total} images: {acc:.4f} ({correct}/{total})\n")
    print("per-class recall:")
    for cls in range(10):
        tp = confusion[cls, cls].item()
        support = confusion[cls].sum().item()
        print(f"  {cls}: {tp / support:.4f} ({tp}/{support})")
    print("\nconfusion matrix (rows=true, cols=pred):")
    print("      " + "".join(f"{c:>6d}" for c in range(10)))
    for cls in range(10):
        print(f"  {cls:>3d} " + "".join(f"{confusion[cls, c].item():>6d}" for c in range(10)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved MNIST model")
    p.add_argument("--checkpoint", default="model/mnist_cnn.pt")
    p.add_argument("--data-dir", default="data")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
