"""End-to-end tests for the MNIST web interface (run while app.py is serving).

    python test_api.py [--base http://127.0.0.1:5000]

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request

from PIL import Image, ImageOps

FAILURES: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    line = f"[{mark}] {label}"
    if detail and not condition:
        line += f" -- {detail}"
    print(line)
    if not condition:
        FAILURES.append(label)


def get_json(base: str, path: str) -> tuple[int, dict | list]:
    try:
        with urllib.request.urlopen(base + path, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def post_json(base: str, path: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main(base: str) -> int:
    # 1. UI served
    try:
        with urllib.request.urlopen(base + "/", timeout=30) as r:
            html = r.read().decode()
            check(r.status == 200, "GET / -> 200")
    except Exception as exc:
        check(False, "GET / -> 200", str(exc))
        return 1
    check('id="board"' in html and "/api/predict" in html, "page contains canvas + predict call")
    check("api/samples" in html and "api/info" in html, "page calls /api/samples and /api/info")

    # 2. model info
    status, info = get_json(base, "/api/info")
    check(status == 200 and abs(info.get("test_acc", 0) - 0.9908) < 1e-6,
          "GET /api/info -> 200 with test_acc 0.9908", f"got {status} {info}")

    # 3. samples
    status, samples = get_json(base, "/api/samples")
    check(status == 200 and isinstance(samples, list) and len(samples) == 24,
          "GET /api/samples -> 24 samples", f"got {status} ({len(samples) if isinstance(samples, list) else samples})")
    if not isinstance(samples, list) or not samples:
        return 1
    check(all("label" in s and s["image"].startswith("data:image/png") for s in samples),
          "samples carry label + PNG data URL")

    # 4. accuracy of the whole interface (server-side preprocessing included)
    correct = 0
    results = []
    for s in samples:
        status, res = post_json(base, "/api/predict", {"image": s["image"]})
        if status != 200:
            check(False, f"predict sample {s['id']} -> 200", f"{status} {res}")
            return 1
        results.append((s["label"], res["pred"]))
        correct += int(res["pred"] == s["label"])
        check(len(res["probs"]) == 10 and abs(sum(res["probs"]) - 1) < 0.01,
              f"probs well formed for sample {s['id']}", str(res["probs"]))
        check(res["preview"].startswith("data:image/png"), f"preview returned for sample {s['id']}")
    acc = correct / len(samples)
    check(acc >= 0.9, f"UI accuracy on {len(samples)} test samples >= 90%",
          f"{correct}/{len(samples)} = {acc:.2%}")
    print(f"       -> {correct}/{len(samples)} correct ({acc:.1%}) ; "
          f"errors: {[f'{t}->{p}' for t, p in results if t != p]}")

    # 5. polarity: dark ink on white canvas must give the same prediction
    first = samples[0]
    ink_on_white = ImageOps.invert(Image.open(
        io.BytesIO(base64.b64decode(first["image"].split(",", 1)[1]))
    ).convert("L"))
    status, res_inv = post_json(base, "/api/predict", {"image": to_data_url(ink_on_white)})
    status2, res_ref = post_json(base, "/api/predict", {"image": first["image"]})
    check(status == 200 and res_inv["pred"] == res_ref["pred"],
          "inverted polarity gives the same prediction",
          f"{res_inv.get('pred')} vs {res_ref.get('pred')}")

    # 6. stroke drawn in a corner (canvas larger than the digit) is re-centred
    big = Image.new("RGB", (340, 340), (0, 0, 0))
    big.paste(ink_on_white.resize((60, 60)).convert("RGB"), (10, 260))
    status, res_corner = post_json(base, "/api/predict", {"image": to_data_url(big)})
    check(status == 200 and isinstance(res_corner.get("pred"), int),
          "small digit in a corner still predicts", str(res_corner))

    # 7. error cases
    status, res = post_json(base, "/api/predict", {})
    check(status == 400 and "error" in res, "missing image -> 400", f"{status} {res}")

    status, res = post_json(base, "/api/predict", {"image": "data:image/png;base64,AAAA"})
    check(status == 400 and "error" in res, "undecodable image -> 400", f"{status} {res}")

    blank = to_data_url(Image.new("RGB", (340, 340), (0, 0, 0)))
    status, res = post_json(base, "/api/predict", {"image": blank})
    check(status == 400 and "error" in res, "empty canvas -> 400 with message",
          f"{status} {res}")
    print(f"       -> empty canvas message: {res.get('error')}")

    # 8. big white canvas (UI default ink polarity is white on black, but also
    #    test a white background: blank white must be rejected too)
    status, res = post_json(base, "/api/predict", {"image": to_data_url(Image.new("RGB", (340, 340), (255, 255, 255)))})
    check(status == 400, "blank white canvas -> 400", f"{status} {res}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): " + "; ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:5000")
    args = p.parse_args()
    sys.exit(main(args.base.rstrip("/")))
