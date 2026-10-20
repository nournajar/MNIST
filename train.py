"""Train a CNN on the MNIST dataset and evaluate it on the test set.

Usage:
    python train.py [--epochs N] [--batch-size N] [--lr F] [--data-dir DIR] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_CLASSES = 10
INPUT_MEAN = 0.1307
INPUT_STD = 0.3081


class MNISTNet(nn.Module):
    """Small convolutional network: 2 conv blocks + 1 fully-connected head."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(data_dir: Path, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    transform_train = transforms.Compose(
        [
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((INPUT_MEAN,), (INPUT_STD,)),
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((INPUT_MEAN,), (INPUT_STD,)),
        ]
    )
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform_eval)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    test_loader = DataLoader(
        test_set, batch_size=512, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total, total_loss / total


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.threads)

    train_loader, test_loader = make_loaders(Path(args.data_dir), args.batch_size, args.workers)
    model = MNISTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1), gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    print(
        f"device={device} threads={torch.get_num_threads()} "
        f"train={len(train_loader.dataset)} test={len(test_loader.dataset)} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    history: list[dict] = []
    best_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        scheduler.step()

        test_acc, test_loss = evaluate(model, test_loader, device)
        train_loss = running_loss / seen
        best_acc = max(best_acc, test_acc)
        elapsed = time.time() - start
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 4),
            "seconds": round(elapsed, 1),
        }
        history.append(record)
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}  ({elapsed:.1f}s)",
            flush=True,
        )

        if test_acc >= best_acc:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": "MNISTNet",
                    "num_classes": NUM_CLASSES,
                    "test_acc": test_acc,
                    "epoch": epoch,
                    "input_mean": INPUT_MEAN,
                    "input_std": INPUT_STD,
                },
                out_path,
            )

    results = {
        "best_test_acc": round(best_acc, 4),
        "final_test_acc": history[-1]["test_acc"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": str(device),
        "model_path": str(out_path),
        "model_params": sum(p.numel() for p in model.parameters()),
        "history": history,
    }
    results_path = out_path.with_suffix(".json")
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nbest test accuracy: {best_acc:.4f}")
    print(f"model saved to {out_path}")
    print(f"results saved to {results_path}")
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a CNN on MNIST")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="model/mnist_cnn.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
