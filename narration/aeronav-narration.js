// AeroNav mission narration, browser side. No API key here: audio comes from the site's own server (scan_server.py).
//
//   <script src="narration/aeronav-narration.js"></script>
//   const narr = AeroNavNarration.attach({ onLine: u => showCaption(u.text), onStatus: s => ... });
//   startButton.onclick = () => { narr.unlock(); ... };   // browsers only allow audio after a user click
//
// It listens to the Rescue Mission's window "rescue" events (sent by Unity), forwards a snapshot to the server at
// most every `intervalMs` (and immediately when the mission state changes), and plays the returned lines one at a time:
// most important first, a critical line interrupts a less important one, stale lines are dropped. See narration/README.md.
(function () {
  'use strict';

  function attach(opts) {
    opts = Object.assign({ base: '', intervalMs: 1000, eventName: 'rescue', onLine: null, onStatus: null, muted: false }, opts || {});
    const audio = new Audio();
    const queue = [];
    let current = null, last = null, lastSent = 0, lastState = null, inFlight = false, timer = null, muted = !!opts.muted;
    const status = (s) => opts.onStatus && opts.onStatus(s);

    function post(snapshot) {
      inFlight = true; lastSent = Date.now(); lastState = snapshot.state;
      fetch(opts.base + '/api/narration/state', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(snapshot) })
        .then(r => r.json())
        .then(d => { (d.utterances || []).forEach(enqueue); status({ ok: true, audio: d.audio, error: d.error }); })
        .catch(e => status({ ok: false, audio: false, error: 'narration server not reachable: ' + e.message }))
        .finally(() => { inFlight = false; });
    }

    function onEvent(ev) {
      const s = ev.detail;
      if (!s || s.type !== 'rescue') return;
      last = s;
      if (inFlight) return;
      if (s.state !== lastState || Date.now() - lastSent >= opts.intervalMs) post(s);
      else if (!timer) timer = setTimeout(() => { timer = null; if (last && !inFlight) post(last); }, opts.intervalMs);
    }

    function enqueue(u) {
      u.queuedAt = Date.now();
      if (opts.onLine) opts.onLine(u);                          // captions show immediately, audio or not
      if (current && u.priority === 0 && current.priority > 0) { audio.pause(); current = null; }
      queue.push(u);
      queue.sort((a, b) => a.priority - b.priority || a.id - b.id);
      playNext();
    }

    function playNext() {
      if (current) return;
      while (queue.length) {
        const u = queue.shift();
        if (Date.now() - u.queuedAt > u.max_delay_s * 1000) continue;   // too late to be useful
        if (muted || !u.audio_url) continue;
        current = u;
        audio.src = opts.base + u.audio_url;
        audio.play().catch(e => { if (e.name === 'AbortError') return; status({ ok: true, audio: false, error: 'audio blocked: ' + e.message }); done(u); });
        return;
      }
    }
    function done(u) { if (u && u !== current) return; current = null; playNext(); }
    audio.addEventListener('ended', () => done(current));
    audio.addEventListener('error', () => { status({ ok: true, audio: false, error: 'audio failed to load (see /api/narration/status)' }); done(current); });

    window.addEventListener(opts.eventName, onEvent);
    return {
      // Call from a click handler (e.g. Start mission) so the browser allows playback.
      unlock() { audio.muted = true; audio.play().catch(() => {}).finally(() => { audio.pause(); audio.muted = false; }); },
      setMuted(m) { muted = !!m; if (muted) { audio.pause(); current = null; queue.length = 0; } },
      say(text) {
        return fetch(opts.base + '/api/narration/say', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) })
          .then(r => r.json()).then(u => { if (u.text) enqueue(u); return u; });
      },
      status() { return fetch(opts.base + '/api/narration/status').then(r => r.json()); },
      detach() { window.removeEventListener(opts.eventName, onEvent); audio.pause(); queue.length = 0; },
    };
  }

  window.AeroNavNarration = { attach };
})();
