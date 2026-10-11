"""Rescue Mission person detector: the ONNX model (model_files/model_onnx.zip -> best.onnx, a YOLOv8m-worldv2 fine-tuned
on Unity renders of the burning building, one class: person) behind the HTTP interface scan_server.py expects.

    GET  /health                                -> {"ok": true, "engine": "onnx-person", "targets": ["person"], ...}
    POST /detect?target=all&conf=0.4  (JPEG)    -> {"objects": [{"category": "person", "confidence", "bbox": {x, y, w, h}}],
                                                    "detected", "confidence", "bbox", "latencyMs", "error", "engine"}
bbox is normalized to the image, origin top-left. Everything runs locally (onnxruntime on the CPU).

Setup and run (port 8767 is scan_server.py's --rescue-vision default):
    python3.13 -m venv .venv-rescue && .venv-rescue/bin/pip install onnxruntime numpy pillow
    unzip -o ../model_files/model_onnx.zip -d /tmp/onnx && cp /tmp/onnx/best.onnx models/rescue_people.onnx
    .venv-rescue/bin/python person_server.py
"""

import argparse
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "models", "rescue_people.onnx")


class PersonEngine:
    name = "onnx-person"

    def __init__(self, weights, threads=0):
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(weights, opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])                     # 640
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = meta.get("names", "{0: 'person'}")
        self.weights = weights
        self.lock = threading.Lock()                      # one frame at a time: all cores go to that frame
        self.frames, self.total_ms = 0, 0.0

    def _letterbox(self, image):
        """Resize keeping the aspect ratio and pad to size x size with gray 114 (as Ultralytics does)."""
        w, h = image.size
        s = self.size / max(w, h)
        nw, nh = round(w * s), round(h * s)
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        px, py = (self.size - nw) // 2, (self.size - nh) // 2
        canvas.paste(image.resize((nw, nh), Image.BILINEAR) if (nw, nh) != (w, h) else image, (px, py))
        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return x, s, px, py

    def detect(self, image, conf, iou=0.5):
        """[(confidence, (x, y, w, h) normalized to the original image)], best first, after NMS."""
        W, H = image.size
        x, s, px, py = self._letterbox(image)
        t0 = time.perf_counter()
        with self.lock:
            out = self.session.run(None, {self.input_name: x})[0][0]   # (4 + classes, anchors)
        self.frames += 1
        self.total_ms += (time.perf_counter() - t0) * 1000
        scores = out[4:].max(axis=0)
        keep = scores >= conf
        if not keep.any():
            return []
        cx, cy, bw, bh = out[0][keep], out[1][keep], out[2][keep], out[3][keep]
        scores = scores[keep]
        x1, y1 = (cx - bw / 2 - px) / s, (cy - bh / 2 - py) / s
        x2, y2 = (cx + bw / 2 - px) / s, (cy + bh / 2 - py) / s
        x1, x2 = np.clip(x1, 0, W), np.clip(x2, 0, W)
        y1, y2 = np.clip(y1, 0, H), np.clip(y2, 0, H)
        order = scores.argsort()[::-1]
        kept = []
        areas = (x2 - x1) * (y2 - y1)
        while order.size:
            i = order[0]
            kept.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            ov = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][ov < iou]
        return [(float(scores[i]), (float(x1[i] / W), float(y1[i] / H), float((x2[i] - x1[i]) / W), float((y2[i] - y1[i]) / H)))
                for i in kept]


def run(engine, data, conf):
    t0 = time.perf_counter()
    res = {"detected": False, "category": "person", "confidence": 0.0, "hasBox": False,
           "bbox": {"x": 0, "y": 0, "w": 0, "h": 0}, "engine": engine.name, "latencyMs": 0.0, "error": "", "objects": []}
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        dets = engine.detect(image, conf)
        res["objects"] = [{"category": "person", "label": "person", "confidence": round(c, 4),
                           "bbox": dict(zip("xywh", (round(v, 4) for v in b)))} for c, b in dets]
        if dets:
            best = res["objects"][0]
            res.update(detected=True, confidence=best["confidence"], hasBox=True, bbox=best["bbox"])
    except Exception as e:                                # report, never crash the server
        res["error"] = f"{type(e).__name__}: {e}"
    res["latencyMs"] = round((time.perf_counter() - t0) * 1000, 1)
    return res


def make_handler(engine, default_conf):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlparse(self.path).path != "/health":
                return self._send(404, {"error": "not found"})
            avg = engine.total_ms / engine.frames if engine.frames else None
            self._send(200, {"ok": True, "engine": engine.name, "targets": ["person"], "model": os.path.basename(engine.weights),
                             "input": engine.size, "frames": engine.frames, "avg_inference_ms": avg and round(avg, 1)})

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/detect":
                return self._send(404, {"error": "not found"})
            q = parse_qs(u.query)
            try:
                conf = float(q.get("conf", [default_conf])[0])
            except ValueError:
                conf = default_conf
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 20 * 1024 * 1024:
                return self._send(400, {"error": "missing or oversized body"})
            res = run(engine, self.rfile.read(n), conf)
            print(f"[person] {len(res['objects'])} person(s)"
                  + (f" best {res['confidence']:.2f}" if res["objects"] else "")
                  + f"  {res['latencyMs']:.0f} ms" + (f"  ERROR {res['error']}" if res["error"] else ""), flush=True)
            self._send(200, res)

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Rescue Mission ONNX person detector")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25, help="default minimum confidence if the request gives none")
    ap.add_argument("--threads", type=int, default=0, help="onnxruntime intra-op threads (0: all cores)")
    args = ap.parse_args()
    if not os.path.exists(args.weights):
        sys.exit(f"Model not found: {args.weights}\nunzip -o ../model_files/model_onnx.zip -d /tmp/onnx && "
                 f"cp /tmp/onnx/best.onnx {args.weights}")
    engine = PersonEngine(args.weights, args.threads)
    run(engine, _warmup_jpeg(engine.size), 0.5)             # first run allocates; keep it out of the mission
    print(f"Person model: {args.weights} ({engine.names}, {engine.size}x{engine.size})", flush=True)
    print(f"Vision server: engine={engine.name} on http://{args.host}:{args.port}  (POST /detect, GET /health)", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(engine, args.conf)).serve_forever()


def _warmup_jpeg(size):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (90, 90, 90)).save(buf, format="JPEG")
    return buf.getvalue()


if __name__ == "__main__":
    main()
