"""Web interface for the trained MNIST model.

Run:
    python app.py               # serves http://127.0.0.1:5000
    python app.py --port 8080 --host 0.0.0.0

Endpoints:
    GET  /                      drawing UI (static/index.html)
    GET  /api/info              model metadata (accuracy, epoch, classes)
    GET  /api/samples           sample digits from the MNIST test set
    POST /api/predict           {"image": "<data URL png/jpg>"} -> prediction
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from torchvision import transforms

from evaluate import load_model

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DEFAULT_CHECKPOINT = ROOT / "model" / "mnist_cnn.pt"
MNIST_DIR = ROOT / "data"

# Drawing canvas is white ink on a black background (MNIST convention),
# but preprocessing below also copes with dark ink on a light canvas.
INK_THRESHOLD = 40          # gray-level considered as stroke
TARGET_BOX = 20             # MNIST digits fit inside a 20x20 box
CANVAS_SIZE = 28

app = Flask(__name__, static_folder=None)

_state: dict = {}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def init_model(checkpoint_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    model, checkpoint = load_model(checkpoint_path, device)
    _state.update(
        model=model,
        device=device,
        checkpoint=checkpoint,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((checkpoint["input_mean"],), (checkpoint["input_std"],)),
            ]
        ),
    )
    print(
        f"model loaded from {checkpoint_path} "
        f"(epoch {checkpoint['epoch']}, test_acc {checkpoint['test_acc']:.4f}, device {device})"
    )


def preprocess(raw: bytes) -> tuple[torch.Tensor, Image.Image]:
    """Turn an arbitrary drawn image into a 1x1x28x28 MNIST-style tensor.

    Steps: grayscale -> figure out ink polarity -> ink bounding box -> fit the
    digit into a 20x20 box (keeping aspect ratio, like MNIST) -> centre it in a
    28x28 black canvas -> normalise with the training statistics.
    """
    try:
        img = Image.open(io.BytesIO(raw)).convert("L")
        img.load()
    except Exception as exc:
        raise ValueError("could not decode image") from exc
    if img.width < 4 or img.height < 4:
        raise ValueError("image too small")

    # keep the mask computation cheap on huge uploads
    if img.width * img.height > 400_000:
        ratio = (400_000 / (img.width * img.height)) ** 0.5
        img = img.resize((max(4, int(img.width * ratio)), max(4, int(img.height * ratio))), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.uint8)
    background_is_dark = int((arr < 128).sum()) > arr.size / 2
    ink = arr if background_is_dark else 255 - arr  # digit bright on black

    mask = ink > INK_THRESHOLD
    if not mask.any():
        raise ValueError("empty image: no stroke detected")

    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    digit = Image.fromarray(ink[y0 : y1 + 1, x0 : x1 + 1])

    # fit inside a TARGET_BOX x TARGET_BOX box, keep aspect ratio (like MNIST)
    dw, dh = digit.size
    scale = TARGET_BOX / max(dw, dh)
    new_w, new_h = max(1, round(dw * scale)), max(1, round(dh * scale))
    digit = digit.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    canvas.paste(digit, ((CANVAS_SIZE - new_w) // 2, (CANVAS_SIZE - new_h) // 2))

    tensor = _state["transform"](canvas).unsqueeze(0)
    return tensor, canvas


@torch.no_grad()
def predict_tensor(tensor: torch.Tensor) -> dict:
    model = _state["model"]
    device = _state["device"]
    logits = model(tensor.to(device))
    probs = torch.softmax(logits, dim=1)[0].cpu()
    top = torch.topk(probs, k=3)
    return {
        "pred": int(probs.argmax()),
        "confidence": round(float(probs.max()), 4),
        "probs": [round(float(p), 4) for p in probs],
        "top3": [{"label": int(i), "prob": round(float(p), 4)} for p, i in zip(top.values, top.indices)],
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/info")
def info():
    ck = _state["checkpoint"]
    return jsonify(
        {
            "test_acc": ck["test_acc"],
            "epoch": ck["epoch"],
            "num_classes": ck["num_classes"],
            "arch": ck["arch"],
            "device": str(_state["device"]),
            "model_path": DEFAULT_CHECKPOINT.name,
        }
    )


@app.get("/api/samples")
def samples():
    """Return a fixed set of test digits (image + true label) for one-click demo."""
    if "samples" in _state:
        return jsonify(_state["samples"])

    from torchvision import datasets

    try:
        test_set = datasets.MNIST(MNIST_DIR, train=False, download=False)
    except RuntimeError:
        return jsonify({"error": "MNIST test set not found"}), 404

    step = max(len(test_set) // 24, 1)  # 24 fixed samples spread over the test set
    out = []
    for i in range(24):
        image, label = test_set[i * step]
        if not isinstance(image, Image.Image):  # no transform -> already a PIL image
            image = transforms.ToPILImage()(image)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        out.append(
            {
                "id": i,
                "label": int(label),
                "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            }
        )
    _state["samples"] = out
    return jsonify(out)


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    data_url: str = payload.get("image", "")
    if not data_url or "," not in data_url:
        return jsonify({"error": "missing 'image' (data URL) in JSON body"}), 400
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
    except Exception:
        return jsonify({"error": "invalid base64 image"}), 400

    try:
        tensor, canvas = preprocess(raw)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "could not decode image"}), 400

    result = predict_tensor(tensor)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    result["preview"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return jsonify(result)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST web interface")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_model(Path(args.checkpoint))
    print(f"serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
