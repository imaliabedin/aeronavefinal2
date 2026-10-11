"""Server-side ElevenLabs text-to-speech. Standard library only, like scan_server.py.

The API key never leaves this process: it is read from ELEVENLABS_API_KEY (or a git-ignored .env in the repository
root), sent only to api.elevenlabs.io, and never logged or returned to the browser.

    ELEVENLABS_API_KEY      required for audio
    ELEVENLABS_VOICE_ID     voice to use (takes precedence over the name)
    ELEVENLABS_VOICE_NAME   voice to look up by name when no ID is set (default "Richard Basston")
    ELEVENLABS_MODEL_ID     default eleven_flash_v2_5 (lowest latency)

Generated audio is cached on disk by (voice, model, text), so a repeated line costs one request.
"""

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

API = "https://api.elevenlabs.io"
REPO = Path(__file__).resolve().parent.parent
DEFAULT_VOICE_NAME = "Richard Basston"
DEFAULT_MODEL = "eleven_flash_v2_5"
OUTPUT_FORMAT = "mp3_44100_128"
MAX_CHARS = 600


class TTSError(Exception):
    """code: missing_key | voice_not_found | unauthorized | payment_required | rate_limited | timeout | api_error | network | bad_text"""

    def __init__(self, code: str, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.code, self.retry_after = code, retry_after

    def to_dict(self) -> Dict:
        d = {"error": self.code, "message": str(self)}
        if self.retry_after is not None:
            d["retry_after_s"] = self.retry_after
        return d


def load_dotenv(path: Path = REPO / ".env") -> None:
    """Minimal KEY=VALUE reader. Existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class ElevenLabsTTS:
    def __init__(self, api_key: Optional[str] = None, voice_id: Optional[str] = None, voice_name: Optional[str] = None,
                 model_id: Optional[str] = None, cache_dir: Path = REPO / "narration" / "cache",
                 api: str = API, timeout: float = 15.0):
        self.api_key = api_key if api_key is not None else os.environ.get("ELEVENLABS_API_KEY", "").strip()
        self.voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID", "").strip() or None
        self.voice_name = voice_name or os.environ.get("ELEVENLABS_VOICE_NAME", "").strip() or DEFAULT_VOICE_NAME
        self.model_id = model_id or os.environ.get("ELEVENLABS_MODEL_ID", "").strip() or DEFAULT_MODEL
        self.cache_dir, self.api, self.timeout = Path(cache_dir), api.rstrip("/"), timeout
        self.voice_verified: Optional[bool] = None    # None: not checked yet
        self.voice_note = "not checked"
        self.last_error: Optional[Dict] = None
        self._pool = ThreadPoolExecutor(max_workers=1)   # one request at a time: stays under ElevenLabs concurrency limits
        self._jobs: Dict[str, Future] = {}
        self._lock = threading.Lock()

    # ---------- status ----------

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def status(self) -> Dict:
        """Safe to send to the browser: no key, only whether one is set."""
        return {"configured": self.configured, "voice_id": self.voice_id, "voice_name": self.voice_name,
                "voice_verified": self.voice_verified, "voice_note": self.voice_note, "model_id": self.model_id,
                "last_error": self.last_error}

    # ---------- HTTP ----------

    def _request(self, method: str, path: str, body: Optional[Dict] = None, accept: str = "application/json") -> bytes:
        if not self.api_key:
            raise TTSError("missing_key", "ELEVENLABS_API_KEY is not set (environment or .env)")
        req = urllib.request.Request(self.api + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"xi-api-key": self.api_key, "Accept": accept,
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            detail = _detail(e)
            if e.code == 401:
                raise TTSError("unauthorized", f"ElevenLabs rejected the API key or its permissions: {detail}")
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                raise TTSError("rate_limited", f"ElevenLabs rate limit / quota: {detail}",
                               retry_after=float(ra) if ra and ra.replace(".", "", 1).isdigit() else None)
            if e.code == 402:
                raise TTSError("payment_required", f"ElevenLabs plan does not allow this: {detail}")
            if e.code == 404:
                raise TTSError("voice_not_found", f"ElevenLabs: not found ({detail})")
            raise TTSError("api_error", f"ElevenLabs HTTP {e.code}: {detail}")
        except (TimeoutError, OSError) as e:      # socket.timeout is an OSError subclass
            if "timed out" in str(e).lower():
                raise TTSError("timeout", f"ElevenLabs did not answer within {self.timeout:.0f} s")
            raise TTSError("network", f"Cannot reach ElevenLabs: {e}")

    # ---------- voice ----------

    def resolve_voice(self) -> str:
        """The voice ID to use. With ELEVENLABS_VOICE_ID set, it is checked to exist. Otherwise the configured voice
        name is looked up in the account's voices. Never falls back to a different voice."""
        if self.voice_id and self.voice_verified:
            return self.voice_id
        if self.voice_id:
            try:
                v = json.loads(self._request("GET", f"/v1/voices/{urllib.parse.quote(self.voice_id)}"))
                self.voice_name, self.voice_verified = v.get("name") or self.voice_name, True
                self.voice_note = f"voice {self.voice_id} exists: {self.voice_name}"
            except TTSError as e:
                if e.code == "voice_not_found":
                    self.voice_verified, self.voice_note = False, f"voice ID {self.voice_id} not found in this account"
                elif e.code == "unauthorized":
                    # Key may lack voices_read but still be allowed to synthesize: use the ID as configured.
                    self.voice_note = f"could not verify voice {self.voice_id} (key lacks voices_read); used as configured"
                    return self.voice_id
                raise
            return self.voice_id
        voices = json.loads(self._request("GET", "/v1/voices")).get("voices", [])
        want = self.voice_name.casefold()
        match = next((v for v in voices if (v.get("name") or "").casefold() == want), None) \
            or next((v for v in voices if (v.get("name") or "").casefold().startswith(want)), None)
        if not match:
            self.voice_verified = False
            names = ", ".join(sorted(v.get("name", "?") for v in voices)[:25])
            self.voice_note = f'"{self.voice_name}" is not in this account\'s voices ({names}). Set ELEVENLABS_VOICE_ID.'
            raise TTSError("voice_not_found", self.voice_note)
        self.voice_id, self.voice_name, self.voice_verified = match["voice_id"], match.get("name"), True
        self.voice_note = f'found "{self.voice_name}" ({self.voice_id})'
        return self.voice_id

    # ---------- speech ----------

    def key_for(self, text: str) -> str:
        return hashlib.sha256(f"{self.voice_id or self.voice_name}|{self.model_id}|{text}".encode()).hexdigest()[:24]

    def synthesize(self, text: str) -> bytes:
        """MP3 bytes for `text`, from the disk cache or ElevenLabs. Raises TTSError."""
        text = " ".join((text or "").split())
        if not text or len(text) > MAX_CHARS:
            raise TTSError("bad_text", f"text must be 1-{MAX_CHARS} characters")
        voice = self.resolve_voice()
        path = self.cache_dir / f"{self.key_for(text)}.mp3"
        if path.exists():
            return path.read_bytes()
        q = urllib.parse.urlencode({"output_format": OUTPUT_FORMAT})
        body = {"text": text, "model_id": self.model_id}
        try:
            try:
                audio = self._request("POST", f"/v1/text-to-speech/{voice}?{q}", body, accept="audio/mpeg")
            except TTSError as e:
                if e.code not in ("rate_limited", "timeout", "api_error"):
                    raise
                time.sleep(min(e.retry_after or 1.5, 5))       # one retry, then give up (narration is time-sensitive)
                audio = self._request("POST", f"/v1/text-to-speech/{voice}?{q}", body, accept="audio/mpeg")
        except TTSError as e:
            self.last_error = e.to_dict()
            raise
        if len(audio) < 256:
            self.last_error = {"error": "api_error", "message": f"ElevenLabs returned only {len(audio)} bytes"}
            raise TTSError("api_error", self.last_error["message"])
        self.last_error = None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(audio)
        tmp.replace(path)
        return audio

    def prefetch(self, text: str) -> Future:
        """Start synthesizing in the background (one worker); `get` waits for the same job."""
        k = self.key_for(" ".join(text.split()))
        with self._lock:
            f = self._jobs.get(k)
            if f is None or (f.done() and f.exception() is not None):
                f = self._jobs[k] = self._pool.submit(self.synthesize, text)
            return f

    def get(self, text: str, wait: float = 20.0) -> bytes:
        return self.prefetch(text).result(timeout=wait)


def _detail(e: urllib.error.HTTPError) -> str:
    try:
        d = json.loads(e.read() or b"{}").get("detail")
        if isinstance(d, dict):
            return d.get("message") or d.get("status") or json.dumps(d)[:200]
        return str(d)[:200] if d else e.reason
    except Exception:
        return str(e.reason)
