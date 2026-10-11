"""Turns Rescue Mission state into short spoken lines: what changed, why it matters, and in what order.

Input is exactly what Unity already sends the page (RescueMission.Emit, the window "rescue" event detail). Person
sightings come from that state's `sightings` list (with Unity's own `confirmed` flag) when present, otherwise from
ScanStore.detections(). Each call to `update` compares the new state with the previous one and
returns the lines worth saying now, most important first.

Priorities: 0 critical (person confirmed, collapse, outcome), 1 high (start, time warnings, search complete),
2 normal (fire milestones, unconfirmed sighting, drone done), 3 low (coverage milestones, drone changes floor).
Duplicates: every line has a key and is said at most once per mission. Low/normal lines are rate-limited in wall time
(at 10x speed several milestones can pass in a second) and, when several of one kind arrive together, only the
latest is kept.
"""

import itertools
import time
from typing import Callable, Dict, List, Optional

CRITICAL, HIGH, NORMAL, LOW = 0, 1, 2, 3
FIRE_STEPS = (0.10, 0.25, 0.50, 0.75)
COVERAGE_STEPS = (0.25, 0.50, 0.75, 0.90)
TIME_WARNINGS = (600, 300, 120, 60, 30)
MIN_GAP_S = {NORMAL: 6.0, LOW: 12.0}      # wall-clock gap since the last line of this priority or lower
NUMBERS = "zero one two three four five six seven eight nine ten".split()
_ids = itertools.count(1)


def confirmed_people(detections: Optional[Dict]):
    """Same rule as rescue.html pollAI: a person is confirmed when the sighting was seen from >= 2 drone positions
    with best confidence >= 0.9; 'possible' is one position at >= 0.8. Only plausibly person-sized hits count."""
    if not detections:
        return [], []
    by_id = {d["detection_id"]: d for d in detections.get("detections", [])}

    def positions(g):
        out = set()
        for i in g["detection_ids"]:
            d = by_id.get(i) or {}
            if not (d.get("world_position") or {}).get("plausible"):
                continue
            p = d.get("camera_position")
            out.add(",".join(str(round(v * 2)) for v in p) if p else i)
        return len(out)

    all_ = [g for g in detections.get("sightings", []) if g.get("detection_ids") and g.get("world_position")]
    confirmed = [g for g in all_ if positions(g) >= 2 and (g.get("best_confidence") or 0) >= 0.9]
    possible = [g for g in all_ if g not in confirmed and positions(g) >= 1 and (g.get("best_confidence") or 0) >= 0.8]
    return confirmed, possible


def _n(k: int) -> str:
    return NUMBERS[k] if 0 <= k < len(NUMBERS) else str(k)


def _clock(s: float) -> str:
    s = max(0, int(round(s)))
    m, s = divmod(s, 60)
    if m and s:
        return f"{m} minute{'s' * (m != 1)} {s} seconds"
    return f"{m} minute{'s' * (m != 1)}" if m else f"{s} seconds"


def _pct(x: float) -> str:
    return f"{round(x * 100)} percent"


def _sighting_floor(g: Dict, by_id: Dict) -> Optional[int]:
    if g.get("floor"):
        return g["floor"]
    d = by_id.get(g.get("best_detection_id")) or {}
    return (d.get("world_position") or {}).get("floor") or d.get("floor")


class MissionNarrator:
    """One per mission. Not thread-safe; NarrationHub serializes calls."""

    def __init__(self, mission_id: Optional[str], clock: Callable[[], float] = time.monotonic):
        self.mission_id, self.clock = mission_id, clock
        self.prev: Optional[Dict] = None
        self.said: set = set()
        # detection_ids of sightings already spoken: stable even when sightings regroup as frames arrive
        self.spoken_confirmed: set = set()
        self.spoken_possible: set = set()
        self.people = 0
        self.last_at: Dict[int, float] = {}
        self.history: List[Dict] = []

    # ---------- public ----------

    def update(self, s: Dict, detections: Optional[Dict] = None) -> List[Dict]:
        p = self.prev or {}
        cands: List[Dict] = []
        self._mission(s, p, cands)
        self._people(s, detections, cands)
        self._time(s, p, cands)
        self._fire(s, p, cands)
        self._search(s, p, cands)
        self._drones(s, p, cands)
        self.prev = s
        return self._select(cands)

    # ---------- rules ----------

    def _add(self, cands, key, priority, text, kind, replaces=None):
        if key in self.said:
            return
        cands.append({"key": key, "priority": priority, "text": text, "kind": kind, "group": replaces})

    def _mission(self, s, p, c):
        st, was = s.get("state"), p.get("state")
        if st == "running" and was != "running" and "start" not in self.said:
            floors = " and ".join(f"drone {d['id']} on floor {d['floor']}" for d in s.get("drones", []))
            self._add(c, "start", HIGH, f"Mission started. {floors.capitalize() if floors else 'The drones are'} "
                      f"searching the burning building. {_clock(s.get('remaining_s', 0))} on the clock.", "mission")
        if s.get("collapsing") and not p.get("collapsing"):
            self._add(c, "collapse", CRITICAL, "Warning: the building is collapsing.", "mission")
        if st in ("successful", "failed") and was != st:
            people = f"{_n(self.people).capitalize()} {'person' if self.people == 1 else 'people'} confirmed."
            if st == "successful":
                used = s.get("mission_s", 0) - s.get("remaining_s", 0)
                closed = s.get("closed_off", 0)
                text = (f"Mission successful. Every reachable part of the building was searched in {_clock(used)}. {people}"
                        + (f" {closed} spots could not be reached." if closed else ""))
            else:
                text = (f"Mission failed. Time ran out with {_pct(s.get('coverage', 0))} of the building searched. "
                        f"{s.get('unseen', 0)} spots were not reached and {s.get('cut_off_by_fire', 0)} were cut off by fire. {people}")
            self._add(c, "outcome", CRITICAL, text, "outcome")

    def _people(self, s, detections, c):
        if isinstance(s.get("sightings"), list):
            return self._people_from_unity(s["sightings"], c)
        if not detections:
            return
        by_id = {d["detection_id"]: d for d in detections.get("detections", [])}
        confirmed, possible = confirmed_people(detections)
        self.people = max(self.people, len(confirmed))
        for g in confirmed:
            ids = set(g["detection_ids"])
            if ids & self.spoken_confirmed:
                self.spoken_confirmed |= ids
                continue
            self.spoken_confirmed |= ids
            floor = _sighting_floor(g, by_id)
            drone = (by_id.get(g.get("best_detection_id")) or {}).get("drone_id")
            where = f" on floor {floor}" if floor else ""
            who = f"Drone {drone} confirmed" if drone else "Confirmed"
            total = len(confirmed)
            tail = f" That is {_n(total)} {'person' if total == 1 else 'people'} confirmed so far." if total > 1 else ""
            self._add(c, "person:" + min(ids), CRITICAL, f"{who} a person{where}. Mark this location for the rescue team.{tail}", "person")
        for g in possible:
            ids = set(g["detection_ids"])
            if ids & (self.spoken_possible | self.spoken_confirmed):
                self.spoken_possible |= ids
                continue
            self.spoken_possible |= ids
            floor = _sighting_floor(g, by_id)
            self._add(c, "possible:" + min(ids), NORMAL,
                      f"Possible person{f' on floor {floor}' if floor else ''}. Waiting for a second view to confirm.",
                      "possible", replaces="possible")

    def _people_from_unity(self, sightings, c):
        """Newer Rescue builds send their own sightings with a `confirmed` flag (the page's rule): use it as is."""
        confirmed = [g for g in sightings if g.get("confirmed")]
        self.people = max(self.people, len(confirmed))
        for g in confirmed:
            sid = str(g.get("id"))
            if sid in self.spoken_confirmed:
                continue
            self.spoken_confirmed.add(sid)
            total = len(confirmed)
            tail = f" That is {_n(total)} {'person' if total == 1 else 'people'} confirmed so far." if total > 1 else ""
            who = f"Drone {g['drone']} confirmed" if g.get("drone") else "Confirmed"
            where = f" on floor {g['floor']}" if g.get("floor") else ""
            self._add(c, f"person:{sid}", CRITICAL, f"{who} a person{where}. Mark this location for the rescue team.{tail}", "person")
        for g in sightings:
            sid = str(g.get("id"))
            if g.get("confirmed") or (g.get("best") or 0) < 0.8 or sid in self.spoken_possible or sid in self.spoken_confirmed:
                continue
            self.spoken_possible.add(sid)
            where = f" on floor {g['floor']}" if g.get("floor") else ""
            self._add(c, f"possible:{sid}", NORMAL, f"Possible person{where}. Waiting for a second view to confirm.",
                      "possible", replaces="possible")

    def _time(self, s, p, c):
        if s.get("state") != "running" or not p:
            return
        now = s.get("remaining_s", 0)
        before = p.get("remaining_s", now)
        crossed = [t for t in TIME_WARNINGS if before > t >= now]
        if not crossed:
            return
        t = min(crossed)
        cov = s.get("coverage", 0)
        elapsed = 1 - now / max(1, s.get("mission_s", 1))
        pace = " Search is behind pace." if cov < elapsed - 0.05 and not s.get("search_complete") else ""
        self._add(c, f"time:{t}", HIGH if t <= 300 else NORMAL,
                  f"{_clock(t)} remaining. {_pct(cov)} of the building searched.{pace}", "time")

    def _fire(self, s, p, c):
        now, before = s.get("fire_share", 0), p.get("fire_share", 0) if p else 0
        crossed = [f for f in FIRE_STEPS if before < f <= now]
        if not crossed or s.get("state") != "running":
            return
        f = max(crossed)
        words = {0.10: "Fire is spreading. A tenth", 0.25: "A quarter", 0.50: "Half", 0.75: "Three quarters"}[f]
        rest = " of the building is now on fire."
        if f >= 0.5 and s.get("unseen", 0) > 0:
            rest += f" {s['unseen']} spots still need searching before they are cut off."
        self._add(c, f"fire:{f}", HIGH if f >= 0.5 else NORMAL, words + rest, "fire")

    def _search(self, s, p, c):
        if s.get("state") != "running":
            return
        cov, before = s.get("coverage", 0), p.get("coverage", 0) if p else 0
        crossed = [x for x in COVERAGE_STEPS if before < x <= cov]
        if crossed and not s.get("search_complete"):
            x = max(crossed)
            self._add(c, f"coverage:{x}", LOW,
                      f"{_pct(x).capitalize()} of the building searched, {_clock(s.get('remaining_s', 0))} left.", "coverage")
        if s.get("search_complete") and not p.get("search_complete"):
            left, cut = s.get("unseen", 0), s.get("cut_off_by_fire", 0)
            if left + cut > 0:
                text = (f"The drones have searched everything they can reach, but {left + cut} spots were not searched"
                        + (f", {cut} of them cut off by fire." if cut else ".") + " The search is not complete.")
            else:
                text = "Search complete. Every reachable part of the building has been checked."
            self._add(c, "search_complete", HIGH, text, "search")

    def _drones(self, s, p, c):
        if s.get("state") != "running" or not p:
            return
        before = {d["id"]: d for d in p.get("drones", [])}
        for d in s.get("drones", []):
            b = before.get(d["id"])
            if not b:
                continue
            if d["floor"] != b["floor"]:
                self._add(c, f"floor:{d['id']}:{d['floor']}", LOW, f"Drone {d['id']} is now searching floor {d['floor']}.",
                          "drone", replaces=f"floor:{d['id']}")
            if d["status"] == "Done" and b["status"] != "Done":
                self._add(c, f"done:{d['id']}", NORMAL, f"Drone {d['id']} has finished its part of the building.", "drone")

    # ---------- selection ----------

    def _select(self, cands: List[Dict]) -> List[Dict]:
        # Same group in one update (drone climbed two floors at 10x): keep only the latest.
        latest: Dict[str, Dict] = {}
        for x in cands:
            if x["group"]:
                latest[x["group"]] = x
        cands = [x for x in cands if not x["group"] or latest[x["group"]] is x]
        cands.sort(key=lambda x: x["priority"])
        now, out = self.clock(), []
        for x in cands:
            gap = MIN_GAP_S.get(x["priority"])
            if gap is not None:
                recent = max((t for pr, t in self.last_at.items() if pr >= x["priority"]), default=-1e9)
                if now - recent < gap or any(o["priority"] >= NORMAL for o in out):
                    self.said.add(x["key"])        # dropped, not deferred: stale progress news is not worth saying
                    continue
            self.said.add(x["key"])
            self.last_at[x["priority"]] = now
            u = {"id": next(_ids), "mission_id": self.mission_id, "key": x["key"], "kind": x["kind"],
                 "priority": x["priority"], "text": x["text"],
                 # The player may drop a line it could not start within this time (it would be stale).
                 "max_delay_s": {CRITICAL: 60, HIGH: 20, NORMAL: 10, LOW: 6}[x["priority"]]}
            out.append(u)
            self.history.append(u)
        return out
