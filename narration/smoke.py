"""Live ElevenLabs check: python3 -m narration.smoke ["text"]

Uses ELEVENLABS_API_KEY (environment or .env) and the configured voice. Prints whether the voice was found, writes
the audio to narration/cache/smoke.mp3 and, on macOS, plays it with afplay. Exit code 0 only if real audio came back.
"""

import shutil
import subprocess
import sys

from narration.tts import ElevenLabsTTS, TTSError, load_dotenv


def main():
    load_dotenv()
    text = " ".join(sys.argv[1:]) or "AeroNav narration check. Drone two confirmed a person on floor three."
    t = ElevenLabsTTS()
    print(f"key set: {t.configured} | voice id: {t.voice_id} | voice name: {t.voice_name!r} | model: {t.model_id}")
    try:
        audio = t.synthesize(text)
    except TTSError as e:
        print(f"FAILED ({e.code}): {e}\nvoice check: {t.voice_note}")
        return 1
    out = t.cache_dir / "smoke.mp3"
    out.write_bytes(audio)
    print(f"OK: {len(audio)} bytes of audio from voice {t.voice_name!r} ({t.voice_id}) -> {out}\nvoice check: {t.voice_note}")
    if shutil.which("afplay"):
        subprocess.run(["afplay", str(out)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
