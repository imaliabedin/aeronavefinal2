# Rescue Mission narration (ElevenLabs)

Spoken mission updates for the Rescue Mission. The server decides **what** to say (from the real simulation state and
AI detections); the browser only plays it. The ElevenLabs key stays on the server.

## Setup

```bash
cp .env.example .env            # put ELEVENLABS_API_KEY in it (and ELEVENLABS_VOICE_ID if you have it)
python3 -m narration.smoke      # live check: prints the voice found, saves narration/cache/smoke.mp3, plays it
python3 scan_server.py          # narration is mounted automatically under /api/narration
```

Without a key everything still runs: lines come back with `audio_url: null` (captions only), and the server log says
`ELEVENLABS_API_KEY not set`.

**Voice.** `ELEVENLABS_VOICE_ID` wins. Otherwise `ELEVENLABS_VOICE_NAME` (default "Richard Basston") is looked up in the
account's voices (`GET /v1/voices`); a voice from the public library must first be added to the account. If it is not
found, synthesis fails with `voice_not_found` and lists the available names. It never falls back to another voice.

## Connecting a UI (for the new Rescue Mission page)

```html
<script src="narration/aeronav-narration.js"></script>
<script>
  const narr = AeroNavNarration.attach({
    onLine: u => showCaption(u.text, u.priority),   // every line, audio or not
    onStatus: s => showNarrationStatus(s),          // {ok, audio, error}
  });
  startButton.addEventListener('click', () => narr.unlock());   // browsers need a click before audio can play
  muteButton.onclick = () => narr.setMuted(true);
</script>
```

That is all: it listens to the `window` "rescue" events Unity already sends (`RescueMission.Emit`), posts a snapshot at
most once a second (immediately when the mission state changes), and plays lines one at a time, most important first.
A critical line interrupts a lower one; lines that waited longer than `max_delay_s` are skipped.

Other calls: `narr.say(text)` (manual line), `narr.status()`, `narr.detach()`.

## HTTP API (scan_server.py)

| Call | |
|---|---|
| `GET /api/narration/status` | `{configured, voice_id, voice_name, voice_verified, voice_note, model_id, last_error}`; never the key |
| `POST /api/narration/state` | body: the "rescue" event detail as Unity sends it → `{utterances: [...], audio, error}` |
| `GET /api/narration/audio/<key>.mp3` | MP3 for a line the server produced (unknown keys → 404) |
| `POST /api/narration/say` | `{text}` → one utterance |

Utterance: `{id, mission_id, key, kind, priority, text, max_delay_s, audio_url}`.
`kind`: mission, person, possible, time, fire, search, drone, coverage, outcome. `priority`: 0 critical … 3 low.
Errors: 503 `missing_key`, 429 `rate_limited` (with `retry_after_s`), 504 `timeout`, 502 `unauthorized`/`api_error`,
424 `voice_not_found`, 402 `payment_required` (e.g. free plan + library voice).

## What gets said

| Priority | Event | Source |
|---|---|---|
| 0 | Person confirmed (drone, floor, running count) | `/api/rescue/<id>/detections`, same rule as rescue.html: ≥ 2 drone positions, confidence ≥ 0.9 |
| 0 | Building collapsing; mission successful / failed (with time used, coverage, cut-off spots, people) | `collapsing`, `state` |
| 1 | Mission start (drones and floors, time); search complete or "everything reachable, N spots not searched" | `state`, `search_complete`, `unseen`, `cut_off_by_fire` |
| 1–2 | Time left: 10, 5, 2, 1 min, 30 s, with coverage and "behind pace" when coverage < elapsed time | `remaining_s`, `coverage` |
| 1–2 | Fire covers 10 / 25 / 50 / 75 % (from 50 %: spots still unsearched) | `fire_share`, `unseen` |
| 2 | Possible person (one view, ≥ 0.8); drone finished its part | detections, drone `status` |
| 3 | Coverage 25 / 50 / 75 / 90 %; drone moved to another floor | `coverage`, drone `floor` |

Each line is said at most once per mission. Normal/low lines need 6 / 12 s of quiet (wall time) and are dropped, not
queued, when it is too soon: at 10× speed stale progress is not worth saying. Several of one kind in a single update
(a drone climbing two floors) collapse to the latest. Audio is cached in `narration/cache/` (git-ignored).

## Tests

```bash
python3 -m unittest narration.test_narration -v    # fake ElevenLabs server; no key or network needed
```
