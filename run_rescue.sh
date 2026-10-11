#!/bin/bash
# Starts the Rescue Mission services: the ONNX person model (port 8767) and the website (port 8000).
# First run sets up VisionServer/.venv-rescue and unpacks the model. Then open http://localhost:8000 →
# Launch Drone → Rescue Mission (or http://localhost:8000/rescue.html?seed=12345&fire=off for the demo setup).
set -e
cd "$(dirname "$0")"
V=VisionServer/.venv-rescue
if [ ! -x "$V/bin/python" ]; then
  PY=$(command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3)
  "$PY" -m venv "$V"
  "$V/bin/pip" install -q onnxruntime numpy pillow
fi
if [ ! -f VisionServer/models/rescue_people.onnx ]; then
  mkdir -p VisionServer/models
  T=$(mktemp -d); unzip -o -q model_files/model_onnx.zip -d "$T"; mv "$T/best.onnx" VisionServer/models/rescue_people.onnx; rm -rf "$T"
fi
"$V/bin/python" VisionServer/person_server.py &
PERSON=$!
trap 'kill $PERSON 2>/dev/null' EXIT
until curl -s http://127.0.0.1:8767/health > /dev/null; do sleep 0.5; done
python3 scan_server.py "$@"
