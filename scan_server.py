"""Serves the website, stores area-scan images, and runs each one through the AI model (replaces
`python3 -m http.server` for the scan demo).

    python3 scan_server.py                  # http://localhost:8000, images in image_store/scans/
    python3 scan_server.py --port 8000 --root image_store/scans --vision http://127.0.0.1:8765 --conf 0.25

The model runs in VisionServer/server.py (start it first; see VisionServer/README.md). Every saved capture is queued,
and one background worker sends it to POST <vision>/detect?target=all and stores the answer:
  - an object of the scan's target class → the image moves to <scan>/detections/ and is listed with its box and
    estimated ground position in <scan>/detections/detections.json
  - nothing of the target class → the image file is deleted (its metadata stays in the manifest)
  - the model could not be reached or failed → the image is kept, marked "failed" (never counted as negative)

API (JSON in, JSON out):
    POST /api/scans                      create a scan            -> manifest (scan_id, scan_region_id, ...)
    POST /api/scans/<id>/images          {"image_b64", metadata}   -> manifest entry (inference is queued)
    POST /api/scans/<id>/finish          summary {"status", ...}   -> manifest
    POST /api/scans/<id>/ground_truth    evaluation only (stored apart from the manifest)
    POST /api/scans/<id>/retry           re-queue images whose inference failed or that were never sent
    GET  /api/scans                      list of scans
    GET  /api/scans/<id>                 manifest
    GET  /api/scans/<id>/detections      model results + inference counts
    GET  /api/scans/<id>/image/<image_id> the image file (if it was kept)
    GET  /api/vision                     is the model server reachable
Rescue Mission (same calls under /api/rescue, frames in image_store/rescue/, ONNX person model at --rescue-vision):
    POST /api/rescue/<id>/frames         a drone camera frame -> analyzed now -> {"status", "detections"}
    POST /api/rescue/<id>/localize       Unity's 3-D position for each detection (ray through the box center)
    GET  /api/rescue/vision              is the person model server reachable
Narration (ElevenLabs, key from ELEVENLABS_API_KEY or .env, never sent to the browser): /api/narration/..., see
narration/README.md.
Everything else is a static file from the repository (city.html, drone-city/, ...). Standard library only.
"""

import argparse
import json
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from image_store.scans import ScanError, ScanStore, decode_image_payload
from narration.hub import NarrationHub

REPO = Path(__file__).resolve().parent
MAX_BODY = 20 * 1024 * 1024
ROUTE = re.compile(r"^/api/(?P<kind>scans|rescue)(?:/(?P<id>[^/]+)(?:/(?P<action>images|frames|finish|ground_truth|detections|retry|localize)"
                   r"|/image/(?P<image>[0-9a-f]{32}))?)?/?$")
# Scan target (website) -> category in the model's answers (VisionServer ItemsEngine).
MODEL_CATEGORY = {"wallet": "wallet", "keys": "key", "phone": "phone", "purse": "purse", "person": "person"}
# The model was trained on views from 2.3-5 m; only images whose camera was 2.5-4 m above the ground go to it.
AI_ALTITUDE = (2.45, 4.05)


def log(msg):
    print(f"[scan_server] {msg}", file=sys.stderr, flush=True)


class Inference:
    """One worker thread and a bounded queue: images are analyzed in capture order, one at a time."""

    def __init__(self, store, vision_url, conf=0.25, keep_negatives=False, timeout=30.0, max_queue=2000):
        self.store, self.vision = store, vision_url.rstrip("/")
        self.conf, self.keep_negatives, self.timeout = conf, keep_negatives, timeout
        self.q = queue.Queue(maxsize=max_queue)
        # Waiting/running images live only in memory (the manifest is written once per image, with the result).
        self.lock = threading.Lock()
        self.pending, self.running = set(), None
        threading.Thread(target=self._run, name="inference", daemon=True).start()

    def submit(self, scan_id, image_id):
        with self.lock:
            if (scan_id, image_id) in self.pending or self.running == (scan_id, image_id):
                return
            self.pending.add((scan_id, image_id))
        try:
            self.q.put_nowait((scan_id, image_id))
        except queue.Full:
            with self.lock:
                self.pending.discard((scan_id, image_id))
            self.store.mark_inference(scan_id, image_id, "failed", error="inference queue full")

    def live_counts(self, scan_id):
        with self.lock:
            queued = sum(1 for s, _ in self.pending if s == scan_id)
            running = 1 if self.running and self.running[0] == scan_id else 0
        return queued, running

    def process_now(self, scan_id, image_id):
        """Rescue Mission: analyze one frame right away (the caller waits) and return its result."""
        with self.lock:
            self.running = (scan_id, image_id)
        try:
            self._process(scan_id, image_id)
        finally:
            with self.lock:
                self.running = None
        e = next(e for e in self.store.manifest(scan_id)["images"] if e["image_id"] == image_id)
        inf = e.get("inference") or {}
        dets = [d for d in self.store.detections(scan_id)["detections"] if d["image_id"] == image_id]
        return {"image_id": image_id, "status": inf.get("status"), "error": inf.get("error"),
                "detections": [{k: d[k] for k in ("detection_id", "category", "confidence", "bbox")} for d in dets]}

    def health(self):
        try:
            with urllib.request.urlopen(self.vision + "/health", timeout=3) as r:
                return dict(json.loads(r.read()), ok=True, url=self.vision)
        except Exception as e:
            return {"ok": False, "url": self.vision, "error": f"{type(e).__name__}: {e}"}

    def _run(self):
        while True:
            scan_id, image_id = self.q.get()
            with self.lock:
                self.pending.discard((scan_id, image_id))
                self.running = (scan_id, image_id)
            try:
                self._process(scan_id, image_id)
            except Exception as e:                         # never let the worker die
                log(f"inference {scan_id}/{image_id} crashed: {e}")
                try:
                    self.store.mark_inference(scan_id, image_id, "failed", error=f"{type(e).__name__}: {e}")
                except Exception:
                    pass
            finally:
                with self.lock:
                    self.running = None

    def _process(self, scan_id, image_id):
        entry = next((e for e in self.store.manifest(scan_id)["images"] if e["image_id"] == image_id), {})
        alt = (entry.get("ground_footprint") or {}).get("altitude")
        if entry.get("ground_footprint") is not None and (alt is None or not AI_ALTITUDE[0] <= alt <= AI_ALTITUDE[1]):
            self.store.mark_inference(scan_id, image_id, "skipped_altitude", altitude=alt,
                                      reason="camera outside the 2.5-4 m band the model was trained for")
            return
        if (entry.get("inference") or {}).get("status") in ("positive", "negative"):
            return                                         # already analyzed: never twice
        target = MODEL_CATEGORY.get((self.store.manifest(scan_id).get("target_category") or "").lower())
        image = self.store.read_image(scan_id, image_id)
        result, error = None, None
        for attempt in range(3):                           # the model server may be busy or restarting
            try:
                req = urllib.request.Request(f"{self.vision}/detect?target=all&conf={self.conf}", data=image,
                                             headers={"Content-Type": "image/jpeg"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    result = json.loads(r.read())
                error = result.get("error") or None
                if not error:
                    break
            except (urllib.error.URLError, OSError, ValueError) as e:
                error = f"{type(e).__name__}: {e}"
            time.sleep(1.0 + attempt)
        if error or result is None:
            self.store.mark_inference(scan_id, image_id, "failed", error=error or "no response")
            log(f"inference failed {scan_id}/{image_id}: {error}")
            return
        if target is None:
            self.store.mark_inference(scan_id, image_id, "failed", error="scan has no known target category",
                                      objects=result.get("objects", []))
            return
        relevant = [o for o in result.get("objects", []) if o.get("category") == target and o.get("confidence", 0) >= self.conf]
        e = self.store.apply_inference(scan_id, image_id, result, relevant, keep_negatives=self.keep_negatives)
        if relevant:
            log(f"{scan_id} #{e['sequence_number']}: {len(relevant)} × {target} "
                f"(best {max(o['confidence'] for o in relevant):.2f})")


class Handler(SimpleHTTPRequestHandler):
    store: ScanStore = None
    inference: Inference = None
    rescue_store: ScanStore = None          # Rescue Mission frames (image_store/rescue/)
    rescue_inference: Inference = None      # the ONNX person detector (second VisionServer)
    narration: NarrationHub = None          # spoken mission updates (narration/)

    def _pick(self, m):
        """(store, inference) for /api/scans or /api/rescue."""
        return (self.rescue_store, self.rescue_inference) if m["kind"] == "rescue" else (self.store, self.inference)
    extensions_map = dict(SimpleHTTPRequestHandler.extensions_map, **{".wasm": "application/wasm", ".data": "application/octet-stream"})

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _narration(self, method):
        if not self.narration or not self.path.startswith("/api/narration"):
            return False
        n = int(self.headers.get("Content-Length") or 0) if method == "POST" else 0
        if n > MAX_BODY:
            self._json(413, {"error": "oversized body"})
            return True
        code, ctype, body = self.narration.handle(method, self.path.split("?", 1)[0], self.rfile.read(n) if n > 0 else None)
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if ctype == "application/json" else "max-age=3600")
        self.end_headers()
        self.wfile.write(body)
        return True

    def do_GET(self):
        if self._narration("GET"):
            return
        path = self.path.split("?", 1)[0]
        if "ground_truth" in path:
            return self._json(404, {"error": "not found"})
        if path == "/api/vision":
            return self._json(200, self.inference.health() if self.inference else {"ok": False, "error": "AI disabled"})
        if path == "/api/rescue/vision":
            return self._json(200, self.rescue_inference.health() if self.rescue_inference else {"ok": False, "error": "AI disabled"})
        m = ROUTE.match(path)
        if not m:
            return super().do_GET()
        store, inference = self._pick(m)
        try:
            if m["id"] is None:
                return self._json(200, {"scans": store.list_scans()})
            if m["image"]:
                data = store.read_image(m["id"], m["image"])
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if m["action"] is None:
                return self._json(200, store.manifest(m["id"]))
            if m["action"] == "detections":
                d = store.detections(m["id"])
                if inference:
                    q, r = inference.live_counts(m["id"])
                    c = d["inference"]
                    c["queued"], c["running"] = q, r
                    c["not_sent"] = max(0, c["not_sent"] - q - r)
                return self._json(200, d)
            return self._json(405, {"error": "use POST"})
        except ScanError as e:
            return self._json(404, {"error": str(e)})

    def do_POST(self):
        if self._narration("POST"):
            return
        m = ROUTE.match(self.path.split("?", 1)[0])
        if not m or m["image"]:
            return self._json(404, {"error": "not found"})
        store, inference = self._pick(m)
        try:
            sid, action = m["id"], m["action"]
            if action == "retry":            # failed inferences, and saved images never sent (e.g. server restarted)
                n = 0
                for e in store.images(sid):
                    if (e.get("inference") or {}).get("status") in (None, "failed") and inference:
                        inference.submit(sid, e["image_id"]); n += 1
                return self._json(200, {"requeued": n})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self._json(413 if n > MAX_BODY else 400, {"error": "missing or oversized body"})
            payload = json.loads(self.rfile.read(n))
            if not isinstance(payload, dict):
                raise ScanError("body must be a JSON object")
            if sid is None:
                return self._json(201, store.create_scan(payload))
            if action == "frames" and m["kind"] == "rescue":
                image = decode_image_payload(payload)
                entry = store.add_image(sid, image, payload)
                if not inference:
                    return self._json(201, {"image_id": entry["image_id"], "status": "not_sent", "detections": []})
                if entry.get("duplicate"):
                    return self._json(201, {"image_id": entry["image_id"], "status": "duplicate", "detections": []})
                res = inference.process_now(sid, entry["image_id"])
                return self._json(201 if res["status"] != "failed" else 502, res)
            if action == "localize" and m["kind"] == "rescue":
                n = store.set_world_positions(sid, payload.get("image_id"), payload.get("positions") or [])
                return self._json(200, {"updated": n})
            if action == "images":
                image = decode_image_payload(payload)
                entry = self.store.add_image(sid, image, payload)
                if self.inference and entry["capture_status"] == "saved" and not entry.get("duplicate"):
                    self.inference.submit(sid, entry["image_id"])   # file is written: safe to analyze
                return self._json(201, entry)
            if action == "finish":
                return self._json(200, store.finish_scan(sid, payload))
            if action == "ground_truth":
                store.set_ground_truth(sid, payload)
                return self._json(200, {"ok": True})
            return self._json(404, {"error": "not found"})
        except (ScanError, ValueError) as e:
            return self._json(400, {"error": str(e)})
        except OSError as e:
            log(f"write failed: {e}")
            return self._json(500, {"error": f"write failed: {e}"})

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/") or "/detections" in self.path or self.path.startswith("/api/narration/") or self.path == "/api/vision":
            return
        if (self.path.endswith("/images") or self.path.endswith("/frames") or self.path.endswith("/localize")) \
                and len(args) > 1 and str(args[1]).startswith("2"):
            return                                     # one line per photo is noise; failures are still logged
        sys.stderr.write("[scan_server] " + fmt % args + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--root", default=str(REPO / "image_store" / "scans"), help="where scans are stored")
    ap.add_argument("--vision", default="http://127.0.0.1:8765", help="VisionServer URL")
    ap.add_argument("--conf", type=float, default=0.25, help="minimum model confidence for a detection")
    ap.add_argument("--keep-negatives", action="store_true", help="keep images where the target was not found")
    ap.add_argument("--no-ai", action="store_true", help="store images only, no inference")
    ap.add_argument("--rescue-vision", default="http://127.0.0.1:8767", help="VisionServer running the ONNX person model")
    ap.add_argument("--rescue-conf", type=float, default=0.4, help="minimum person confidence (Rescue Mission)")
    ap.add_argument("--rescue-root", default=str(REPO / "image_store" / "rescue"), help="where Rescue Mission frames go")
    args = ap.parse_args(argv)
    Handler.store = ScanStore(args.root)
    Handler.rescue_store = ScanStore(args.rescue_root)
    Handler.narration = NarrationHub(detections=Handler.rescue_store.detections)
    st = Handler.narration.tts.status()
    log("Narration: " + (f"ElevenLabs configured, voice {st['voice_id'] or repr(st['voice_name'])}, model {st['model_id']}"
                         if st["configured"] else "ELEVENLABS_API_KEY not set, captions only (no audio)"))
    if not args.no_ai:
        Handler.rescue_inference = Inference(Handler.rescue_store, args.rescue_vision, args.rescue_conf)
        h = Handler.rescue_inference.health()
        log(f"Rescue person model at {args.rescue_vision}: " + (f"ready ({h.get('targets')})" if h["ok"] else f"NOT reachable ({h['error']})"))
        Handler.inference = Inference(Handler.store, args.vision, args.conf, args.keep_negatives)
        h = Handler.inference.health()
        log(f"AI model at {args.vision}: " + (f"ready ({h.get('engine')})" if h["ok"] else f"NOT reachable ({h['error']})"))
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=str(REPO)))
    print(f"Scan server on http://{args.host}:{args.port}/city.html  (images in {Handler.store.root})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
