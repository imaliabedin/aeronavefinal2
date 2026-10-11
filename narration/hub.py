"""HTTP side of the narration (mounted by scan_server.py under /api/narration). See narration/README.md.

    GET  /api/narration/status            TTS configuration (never the key), voice check
    POST /api/narration/state             a Rescue Mission state snapshot -> {"utterances": [...]} to speak now
    GET  /api/narration/audio/<key>.mp3   the audio for an utterance (generated on first request, then cached)
    POST /api/narration/say               {"text"} -> one utterance (testing; only text the server registered is ever
                                          synthesized through /audio, so the browser cannot spend credits freely)
"""

import json
import re
import sys
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable, Dict, Optional, Tuple

from .director import MissionNarrator
from .tts import ElevenLabsTTS, TTSError, load_dotenv

AUDIO = re.compile(r"^/api/narration/audio/(?P<key>[0-9a-f]{24})\.mp3$")
Response = Tuple[int, str, bytes]


def _json(code: int, obj) -> Response:
    return code, "application/json", json.dumps(obj).encode()


class NarrationHub:
    def __init__(self, tts: Optional[ElevenLabsTTS] = None, detections: Optional[Callable[[str], Dict]] = None):
        if tts is None:
            load_dotenv()
            tts = ElevenLabsTTS()
        self.tts, self.detections = tts, detections
        self.narrators: Dict[str, MissionNarrator] = {}
        self.texts: Dict[str, str] = {}            # audio key -> text, only for lines the server produced
        self._lock = threading.Lock()

    # ---------- core (usable without HTTP) ----------

    def _speakable(self, u: Dict) -> Dict:
        key = self.tts.key_for(u["text"])
        self.texts[key] = u["text"]
        u["audio_url"] = f"/api/narration/audio/{key}.mp3" if self.tts.configured else None
        if self.tts.configured:
            self.tts.prefetch(u["text"])               # audio is usually ready before the browser asks
        return u

    def on_state(self, snapshot: Dict) -> Dict:
        mid = snapshot.get("mission_id") or "local"
        dets = None
        if self.detections and snapshot.get("mission_id") and "sightings" not in snapshot:
            try:
                dets = self.detections(snapshot["mission_id"])
            except Exception:
                dets = None                            # detections unavailable: narrate the rest
        with self._lock:
            n = self.narrators.get(mid)
            if n is None:
                # A new mission replaces the pre-start "local" narrator, keeping what it already said.
                n = self.narrators.pop("local", None) or MissionNarrator(mid)
                n.mission_id = mid
                self.narrators[mid] = n
            out = [self._speakable(u) for u in n.update(snapshot, dets)]
        return {"utterances": out, "audio": self.tts.configured, "error": None if self.tts.configured else
                "ELEVENLABS_API_KEY not set: captions only"}

    def say(self, text: str) -> Dict:
        text = " ".join((text or "").split())
        if not text or len(text) > 600:
            raise TTSError("bad_text", "text must be 1-600 characters")
        return self._speakable({"id": 0, "key": "say", "kind": "manual", "priority": 1, "text": text, "max_delay_s": 30})

    # ---------- HTTP ----------

    def handle(self, method: str, path: str, body: Optional[bytes]) -> Optional[Response]:
        """Returns (status, content type, body), or None when the path is not a narration route."""
        if not path.startswith("/api/narration"):
            return None
        try:
            if method == "GET" and path == "/api/narration/status":
                st = self.tts.status()
                if self.tts.configured and st["voice_verified"] is None:
                    try:
                        self.tts.resolve_voice()
                    except TTSError:
                        pass
                    st = self.tts.status()
                return _json(200, st)
            m = AUDIO.match(path)
            if method == "GET" and m:
                text = self.texts.get(m["key"])
                if text is None:
                    return _json(404, {"error": "unknown audio key"})
                if not self.tts.configured:
                    return _json(503, {"error": "missing_key", "message": "ELEVENLABS_API_KEY not set"})
                try:
                    return 200, "audio/mpeg", self.tts.get(text, wait=25)
                except FutureTimeout:
                    return _json(504, {"error": "timeout", "message": "audio still generating"})
            if method == "POST" and path in ("/api/narration/state", "/api/narration/say"):
                payload = json.loads(body or b"{}")
                if not isinstance(payload, dict):
                    return _json(400, {"error": "body must be a JSON object"})
                if path.endswith("/say"):
                    return _json(200, self.say(payload.get("text", "")))
                return _json(200, self.on_state(payload))
            return _json(404, {"error": "not found"})
        except TTSError as e:
            print(f"[narration] {e.code}: {e}", file=sys.stderr, flush=True)     # messages never contain the key
            code = {"missing_key": 503, "rate_limited": 429, "timeout": 504, "bad_text": 400,
                    "voice_not_found": 424, "unauthorized": 502, "payment_required": 402}.get(e.code, 502)
            return _json(code, e.to_dict())
        except ValueError as e:
            return _json(400, {"error": f"bad JSON: {e}"})
        except Exception as e:                                  # narration must never take the mission page down
            print(f"[narration] internal error: {e!r}", file=sys.stderr, flush=True)
            return _json(500, {"error": "internal", "message": type(e).__name__})
