# Rescue Mission: run, test, and how it works

## Run

```bash
./run_rescue.sh          # starts the ONNX person model (port 8767), then the website (port 8000)
```

Open **http://localhost:8000** → **Launch Drone** → **Rescue Mission**. The first run creates
`VisionServer/.venv-rescue` (onnxruntime, numpy, pillow) and unpacks the model from `model_files/model_onnx.zip`
to `VisionServer/models/rescue_people.onnx`. Both are git-ignored.

Manual start (two terminals):

```bash
VisionServer/.venv-rescue/bin/python VisionServer/person_server.py   # ONNX person model, http://127.0.0.1:8767
python3 scan_server.py                                               # website + frame storage, http://127.0.0.1:8000
```

URL options (`rescue.html?...`):

| Option | Effect |
|---|---|
| `seed=12345` | Builds a fixed building: the same layout, furniture, people and starting fires every time. Without it, each load is random. |
| `fire=off` | No fire spread during the mission. The starting fires still burn. The page says "FIRE SPREAD OFF". This is a demo mode. |

The page shows a tutorial on every load. Use 1× / 5× / 10× at any time.

## What the mission does

- Two drones search a four-storey building and navigate by sonar.
- Each drone takes AI camera frames: 640 × 640, 70° FOV, 30° down, with a headlight. This matches the model's
  training camera. The frames go to `scan_server.py`, which passes them to `VisionServer/person_server.py`.
  That server runs the ONNX YOLOv8m-worldv2 person model locally on the CPU (about 90–110 ms per frame).
- **Search coverage comes from the camera only.** A spot counts as searched when a frame that was sent to the model covered it:
  - the spot is inside the frame,
  - it is 0.6–3.5 m away (people are detected 93% of the time at 1–2 m, 60% at 3–4 m, 35% at 4–5 m),
  - the line of sight reaches the spot's surface.

  Flying past a spot does not count. If a frame fails to reach the model, the spots it covered become unsearched again.
- **Search targets are viewpoints.** A drone flies toward the nearest unsearched spot of its share. It stops about 2 m
  away once it has a clear view of the spot, then hovers until a frame has been aimed at it (at most 3 s). Spots it can't
  reach or see are skipped for the current pass. Among nearby spots it prefers ones inside larger unsearched areas, so it sweeps an area at a time instead of zigzagging back for single leftover spots. The drones never stop early: after each pass they go back for anything still unsearched (and retry spots they gave up on), and once everything reachable is covered they search the building again, until the timer runs out. Drones take the
  stairs only when a floor has at least 20 spots left.
- **Choosing the frame direction.** The gimbal turns to the direction (checked every 30°) that covers the most unsearched
  spots. Unconfirmed sightings that would be seen from a new position add extra weight.
- **Location of a detection.** A ray goes from the frame's camera pose through the box center to the first surface it hits.
  Detections whose estimated physical size is outside 0.25–2.2 m are ignored. Detections within 1.5 m of each other on the
  same floor are grouped into one sighting.
- **Second look.** A sighting with confidence ≥ 0.8 that is not yet confirmed sends its drone back once. The drone flies to
  spots 1.5–3 m around the sighting so it can look from a new position.
- **Confirmed:** the person was detected from at least 2 different drone positions, and the best confidence is ≥ 0.92.
  In the test missions, 548 person-sized detections were analyzed: none away from a real person scored ≥ 0.92, and one
  (a toilet) scored 0.907.
- **Success:** every trapped person has a confirmed sighting within 1.5 m. The simulation's true positions are used only to
  check this. They are never shown or used as detections, and they are stored in `ground_truth.json`, which is never served.
- **Failure:**
  - **20:00 runs out:** the building collapses.
- **Speed:** 1×/5×/10× sets `Time.timeScale`, so the clock, drones, fire and collapse all scale together. If the model can't
  keep up with the frames the drones need, the simulation slows down. The header then shows "AI-limited: N×".

## Tests

```bash
U=/Applications/Unity/Hub/Editor/6000.6.4f1/Unity.app/Contents/MacOS/Unity     # Unity must be closed
$U -batchmode -quit -projectPath Building2 -executeMethod RescueTests.ModelOnPeople -logFile model.log   # needs person_server.py
$U -batchmode -quit -projectPath Building2 -executeMethod RescueTests.MissionStates -logFile states.log
git checkout -- Building2/Assets/Scenes/DroneSearchDemo.unity    # the tests regenerate the scene file; keep the committed one
$U -batchmode -quit -projectPath Building2 -executeMethod RescueWebBuild.BuildBatch -logFile build.log   # → rescue-sim/
```

## Tutorial video

`rescue-media/collapse.mp4` (11 s, 620 KB) is the mission's own failure ending: the real `FireSpread` and `BuildingCollapse`,
rendered from outside by an editor script. To regenerate it:

```bash
COLLAPSE_OUT=/tmp/collapse $U -batchmode -quit -projectPath Building2 -executeMethod CollapseVideo.Record
ffmpeg -framerate 30 -i /tmp/collapse/frame_%04d.jpg -vf scale=960:-2 -c:v libx264 -pix_fmt yuv420p -crf 26 -movflags +faststart rescue-media/collapse.mp4
```
