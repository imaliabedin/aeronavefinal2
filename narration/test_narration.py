"""python3 -m unittest narration.test_narration -v

Runs against a local fake ElevenLabs server; no real key or network is used. For a live check see narration/smoke.py.
"""

import json
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from narration.director import CRITICAL, MissionNarrator
from narration.hub import NarrationHub
from narration.tts import ElevenLabsTTS, TTSError

MP3 = b"ID3" + b"\x00" * 1000


class FakeEleven(BaseHTTPRequestHandler):
    voices = [{"voice_id": "rb123", "name": "Richard Basston"}, {"voice_id": "x9", "name": "Other"}]
    mode = "ok"             # ok | 429once | 401 | slow
    calls = []

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", headers=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        FakeEleven.calls.append(("GET", self.path, self.headers.get("xi-api-key")))
        if FakeEleven.mode == "401":
            return self._send(401, b'{"detail":{"status":"invalid_api_key"}}')
        if self.path == "/v1/voices":
            return self._send(200, json.dumps({"voices": FakeEleven.voices}).encode())
        vid = self.path.rsplit("/", 1)[-1]
        v = next((v for v in FakeEleven.voices if v["voice_id"] == vid), None)
        return self._send(200, json.dumps(v).encode()) if v else self._send(404, b'{"detail":"voice_not_found"}')

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n))
        FakeEleven.calls.append(("POST", self.path, body))
        if FakeEleven.mode == "429once":
            FakeEleven.mode = "ok"
            return self._send(429, b'{"detail":"too_many_concurrent_requests"}', headers=[("Retry-After", "0.1")])
        if FakeEleven.mode == "slow":
            time.sleep(1.5)
        self._send(200, MP3, "audio/mpeg")


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeEleven)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.api = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        FakeEleven.mode, FakeEleven.calls = "ok", []
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def tts(self, **kw):
        kw.setdefault("api_key", "test-key")
        return ElevenLabsTTS(api=self.api, cache_dir=Path(self.tmp.name), timeout=kw.pop("timeout", 5), **kw)


class TTSTests(Base):
    def test_resolves_richard_basston_by_name_and_caches(self):
        t = self.tts()
        self.assertEqual(t.synthesize("Mission started."), MP3)
        self.assertEqual(t.voice_id, "rb123")
        self.assertTrue(t.voice_verified)
        t.synthesize("Mission started.")
        self.assertEqual(sum(1 for c in FakeEleven.calls if c[0] == "POST"), 1)       # second time from cache
        post = next(c for c in FakeEleven.calls if c[0] == "POST")
        self.assertTrue(post[1].startswith("/v1/text-to-speech/rb123?"))
        self.assertEqual(post[2]["model_id"], "eleven_flash_v2_5")

    def test_missing_voice_is_reported_not_substituted(self):
        t = self.tts(voice_name="Nobody Here")
        with self.assertRaises(TTSError) as e:
            t.synthesize("hello")
        self.assertEqual(e.exception.code, "voice_not_found")
        self.assertFalse(t.voice_verified)
        self.assertFalse(any(c[0] == "POST" for c in FakeEleven.calls))

    def test_configured_voice_id(self):
        t = self.tts(voice_id="x9")
        t.synthesize("hi there")
        self.assertEqual(t.voice_name, "Other")
        with self.assertRaises(TTSError):
            self.tts(voice_id="nope").synthesize("hi")

    def test_missing_key(self):
        t = self.tts(api_key="")
        self.assertFalse(t.configured)
        with self.assertRaises(TTSError) as e:
            t.synthesize("hi")
        self.assertEqual(e.exception.code, "missing_key")

    def test_rate_limit_retried_once(self):
        FakeEleven.mode = "429once"
        self.assertEqual(self.tts(voice_id="rb123").synthesize("retry me"), MP3)

    def test_unauthorized_and_timeout(self):
        FakeEleven.mode = "401"
        with self.assertRaises(TTSError) as e:
            self.tts().synthesize("hi")
        self.assertEqual(e.exception.code, "unauthorized")
        FakeEleven.mode = "slow"
        t = self.tts(voice_id="rb123", timeout=0.4)
        with self.assertRaises(TTSError) as e:
            t.synthesize("slow one")
        self.assertEqual(e.exception.code, "timeout")

    def test_key_never_in_status(self):
        st = json.dumps(self.tts(api_key="sk_secret_value").status())
        self.assertNotIn("sk_secret_value", st)


def snap(**kw):
    s = {"type": "rescue", "state": "running", "remaining_s": 1200, "mission_s": 1200, "mission_id": "m1",
         "coverage": 0.0, "unseen": 500, "closed_off": 20, "search_complete": False, "cut_off_by_fire": 0,
         "fire_share": 0.02, "collapsing": False, "error": None,
         "drones": [{"id": 1, "floor": 1, "status": "Searching"}, {"id": 2, "floor": 4, "status": "Searching"}]}
    s.update(kw)
    return s


def dets(n_positions, conf=0.95, ids=("d1", "d2")):
    ds = [{"detection_id": i, "drone_id": 2, "floor": 3, "camera_position": [k * 3.0, 1, 0],
           "world_position": {"plausible": True, "floor": 3}} for k, i in enumerate(ids[:n_positions])]
    return {"detections": ds, "sightings": [{"detection_ids": [d["detection_id"] for d in ds], "best_confidence": conf,
                                             "best_detection_id": ds[0]["detection_id"], "world_position": {"x": 1, "z": 2},
                                             "floor": 3}]}


class DirectorTests(unittest.TestCase):
    def setUp(self):
        self.t = [0.0]
        self.n = MissionNarrator("m1", clock=lambda: self.t[0])

    def say(self, s, d=None, dt=20.0):
        self.t[0] += dt
        return [u["text"] for u in self.n.update(s, d)]

    def test_mission_story(self):
        self.assertEqual(self.say(snap(state="ready")), [])
        start = self.say(snap())
        self.assertTrue(start[0].startswith("Mission started. Drone 1 on floor 1 and drone 2 on floor 4"))
        self.assertEqual(self.say(snap()), [])                                   # nothing changed: silence
        possible = self.say(snap(), dets(1, 0.85))
        self.assertIn("Possible person on floor 3", possible[0])
        confirmed = self.n.update(snap(), dets(2))
        self.assertEqual(confirmed[0]["priority"], CRITICAL)
        self.assertIn("Drone 2 confirmed a person on floor 3", confirmed[0]["text"])
        self.assertEqual(self.say(snap(), dets(2)), [])                          # same person not repeated
        fire = self.say(snap(fire_share=0.3, coverage=0.1, remaining_s=900))
        self.assertTrue(any("quarter of the building is now on fire" in x for x in fire), fire)
        warn = self.say(snap(remaining_s=290, coverage=0.3, fire_share=0.3))
        self.assertIn("5 minutes remaining. 30 percent of the building searched. Search is behind pace.", warn)
        end = self.say(snap(state="failed", remaining_s=0, coverage=0.4, unseen=100, cut_off_by_fire=12))
        self.assertIn("Mission failed", end[0])
        self.assertIn("One person confirmed", end[0])

    def test_bursts_are_coalesced_and_rate_limited(self):
        self.say(snap(state="ready")); self.say(snap())
        # 10x speed: drone 1 climbs two floors and coverage passes two milestones in one update
        s = snap(coverage=0.55, drones=[{"id": 1, "floor": 3, "status": "Searching"}, {"id": 2, "floor": 4, "status": "Searching"}])
        out = self.say(s)
        self.assertEqual(len(out), 1)                                            # one low-priority line, not three
        s2 = snap(coverage=0.8, drones=s["drones"])
        self.assertEqual(self.say(s2, dt=1.0), [])                               # too soon after the last one
        self.assertEqual(self.say(snap(coverage=0.8, drones=s["drones"]), dt=30), [])   # dropped, not deferred

    def test_collapse_and_success(self):
        self.say(snap(state="ready")); self.say(snap())
        self.assertIn("Warning: the building is collapsing.", self.say(snap(collapsing=True)))
        self.assertIn("Search complete", self.say(snap(collapsing=True, search_complete=True, unseen=0))[0])
        ok = self.say(snap(state="successful", remaining_s=400, collapsing=True, search_complete=True, unseen=0))
        self.assertTrue(ok[0].startswith("Mission successful. Every reachable part of the building was searched in 13 minutes 20 seconds"))


class UnitySightingsTests(unittest.TestCase):
    def test_uses_unity_confirmed_flag(self):
        n = MissionNarrator("m1", clock=lambda: 0.0)
        n.update(snap(state="ready", sightings=[])); n.update(snap(sightings=[]))
        poss = [{"id": 7, "floor": 2, "best": 0.85, "confirmed": False, "drone": 1}]
        self.assertIn("Possible person on floor 2", n.update(snap(sightings=poss))[0]["text"])
        conf = [{"id": 7, "floor": 2, "best": 0.95, "confirmed": True, "drone": 1}]
        out = n.update(snap(sightings=conf))
        self.assertEqual(out[0]["text"], "Drone 1 confirmed a person on floor 2. Mark this location for the rescue team.")
        self.assertEqual(n.update(snap(sightings=conf)), [])
        # Not confirmed by Unity -> never announced as confirmed, even with high confidence.
        self.assertFalse(any(u["kind"] == "person" for u in n.update(snap(sightings=conf + [{"id": 8, "best": 0.99, "confirmed": False}]))))


class ServerTests(Base):
    """scan_server.py with the narration mounted, ElevenLabs replaced by the fake."""

    def test_state_to_audio_through_scan_server(self):
        import scan_server
        from image_store.scans import ScanStore
        from functools import partial
        scan_server.Handler.rescue_store = ScanStore(self.tmp.name + "/rescue")
        scan_server.Handler.narration = NarrationHub(tts=self.tts(), detections=lambda mid: {"detections": [], "sightings": []})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(scan_server.Handler, directory=self.tmp.name))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            def post(path, obj):
                req = urllib.request.Request(base + path, json.dumps(obj).encode(), {"Content-Type": "application/json"})
                return json.loads(urllib.request.urlopen(req).read())
            post("/api/narration/state", snap(state="ready", mission_id=None))
            out = post("/api/narration/state", snap())
            self.assertTrue(out["audio"])
            u = out["utterances"][0]
            with urllib.request.urlopen(base + u["audio_url"]) as r:
                self.assertEqual(r.headers["Content-Type"], "audio/mpeg")
                self.assertEqual(r.read(), MP3)
            st = json.loads(urllib.request.urlopen(base + "/api/narration/status").read())
            self.assertEqual((st["voice_id"], st["voice_verified"]), ("rb123", True))
            self.assertNotIn("test-key", json.dumps(st))
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base + "/api/narration/audio/" + "0" * 24 + ".mp3")
            self.assertEqual(e.exception.code, 404)                            # only server-made lines are spoken
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
