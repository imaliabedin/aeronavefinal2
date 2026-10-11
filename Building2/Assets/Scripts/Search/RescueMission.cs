using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using Drone2.Building;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.Rendering.Universal;

/// The website's Rescue Mission on top of the existing two-drone search (SearchCoordinator, SearchDrone, sonar,
/// SearchGrid). Adds what a mission needs and nothing else:
///   - a 20:00 countdown of simulated time, started by the page's Start button (not when the page loads)
///   - the objective: find the trapped people. MISSION SUCCESSFUL when every trapped person has been confirmed by the
///     AI (a confirmed sighting within sightingRadius of the person; the simulation's true positions are used only to
///     check the outcome, never as detections). MISSION FAILED only when time runs out (the building collapses): the
///     drones keep searching, pass after pass, until the last second.
///   - the existing FireSpread throughout, and the existing BuildingCollapse at the deadline
///   - speed 1x / 5x / 10x (Time.timeScale, so the clock, drones, fire and collapse all scale together). With aiPacing
///     the simulation slows down while the model cannot keep up with the frames the drones need ("AI-limited").
///   - camera search: a spot counts as searched only when an AI frame covered it (inside the camera's view, within
///     inspectMin-inspectRange, the line of sight reaching that spot's surface) and the frame went to the model.
///     Flying near a spot does not count. Each frame's direction is chosen (gimbal yaw) to cover the most spots not
///     yet searched, plus unconfirmed sightings that need a view from a new position.
///   - the AI: each frame goes to scan_server.py, which runs the ONNX person detector. The AI camera sits at the
///     drone's camera mount and copies the camera the model was trained with (Editor/Dataset/PersonDatasetGenerator:
///     640 x 640, 70 deg FOV, 30 deg down, a spot headlight dimmed near walls, the scene's own lighting). Each
///     detection's box center is cast as a ray from that frame's camera pose into the scene; the first hit is the
///     estimated 3-D position (and floor). Detections of plausible person size are grouped into sightings by
///     position; a sighting is confirmed when it was detected from confirmViews different drone positions with a
///     best confidence of at least confirmConfidence.
/// Commands (SendMessage "RescueControls"): StartMission, SetSpeed("1"|"5"|"10"), SetFireSpread("off"), SetSeed("n").
/// State goes to the page as JSON through RescueBridge.jslib (window "rescue" events). In the Editor it attaches only
/// if RESCUE_MISSION=1.
public class RescueMission : MonoBehaviour
{
    public enum Phase { Ready, Running, Successful, Failed }
    public const int PeopleInMission = 3, PeopleCandidates = 9;

    public float missionSeconds = 20f * 60f;
    [Tooltip("Simulated seconds between AI frames of one drone (only when there is something new to look at).")]
    public float captureEvery = 1f;
    [Tooltip("Frames of one drone waiting for the model at a time.")]
    public int maxInFlight = 1;
    [Tooltip("Slow the whole simulation down (clock, drones, fire) while the model can't keep up with the selected speed.")]
    public bool aiPacing = true;
    public int captureSize = 640;           // the model's input size
    [Header("AI camera (as in the model's training images)")]
    public float aiFov = 70f, aiPitch = 30f;
    public float headlightIntensity = 2.25f, headlightRange = 12f, headlightAngle = 80f;
    [Header("What a frame searches")]
    [Tooltip("Horizontal distance (m) from the camera at which a spot counts as searched by a frame. Test missions, people " +
             "inside the frame (occluded ones included): detected 93% at 1-2 m, 70% at 2-3 m, 60% at 3-4 m, 35% at 4-5 m.")]
    public float inspectMin = 0.6f, inspectRange = 3.5f;
    [Tooltip("Camera search: a floor is worth a trip by the stairs only with at least this many spots left.")]
    public int minFloorWork = 20;
    [Tooltip("No frame unless its best direction covers at least this many spots not yet searched (or a sighting to verify).")]
    public int minGain = 4;
    public float yawStep = 30f;
    [Tooltip("Weight (in spots) of an unconfirmed sighting seen from a new position, when choosing a frame's direction.")]
    public float verifyBonus = 30f;
    [Header("Confirmation")]
    [Tooltip("Test missions (548 person-sized detections): at >= 0.92 no detection away from a real person, at 0.90 one " +
             "(a toilet, 0.907); real people mostly score 0.93-0.98.")]
    public float confirmConfidence = 0.92f;
    public int confirmViews = 2;
    [Tooltip("Detections within this distance (m, same floor) are one sighting; a person is found by a confirmed " +
             "sighting within this distance.")]
    public float sightingRadius = 1.5f;
    [Tooltip("A sighting at least this confident but not confirmed sends its drone back for a second look (once).")]
    public float verifyConfidence = 0.8f;
    [Tooltip("Fire spread for the mission (FireSpread's own default is 2 s between steps). 10 s: slow enough that the fire " +
             "rarely seals rooms off before the drones get there.")]
    public float fireInterval = 10f;
    public string serverUrl;

    public Phase State { get; private set; } = Phase.Ready;
    public float MissionTime { get; private set; }
    public float Remaining => Mathf.Max(0f, missionSeconds - MissionTime);
    public string MissionId { get; private set; }

    SearchCoordinator coord;
    FireSpread fire;
    int initialFires;
    Camera aiCam;
    Light headlight;
    RenderTexture aiRT;
    Texture2D aiTex;
    BuildingCollapse collapse;
    float speed = 1f, aiScale = 1f, emitTimer;
    bool starved, cameraOn, fireOff;
    readonly Dictionary<int, float> nextCapture = new();
    readonly Dictionary<int, int> inFlight = new(), lastGain = new();
    int secondLooks, framesSent, framesFailed, framesUnfit, framesNoGain, detections, implausible, sequence, passes = 1;
    byte[,,] cover;                          // frames sent to the model that covered each spot
    [Tooltip("No AI frame when a surface is closer than this straight ahead, or a fire closer than fireClearance: " +
             "the model's training images always had a clear view (camera 0.25-0.3 m from walls and furniture).")]
    public float minClearView = 0.8f, fireClearance = 1.5f;
    [Tooltip("Estimated physical size (m) a detection must have to count as a person.")]
    public float minPersonSize = 0.25f, maxPersonSize = 2.2f;
    string lastError, endReason;
    bool creating;
    readonly List<Sighting> sightings = new();
    // Evaluation only: where the people really are (to check the outcome and score the AI, never as detections).
    readonly List<Vector3> truth = new();
    readonly List<int> truthFloor = new();
    int[] visibleFrames;
    float[] closestView;

    class Sighting
    {
        public int id, floor, drone, frames;
        public Vector3 sum;
        public int n;
        public readonly HashSet<Vector3Int> views = new();
        public float best, distance;
        public string detectionId, imageId;
        public Box box;
        public bool confirmed, verifyRequested;
        public Vector3 Pos => sum / Mathf.Max(1, n);
    }

#if UNITY_WEBGL && !UNITY_EDITOR
    [DllImport("__Internal")] static extern void RescueEmit(string json);
#else
    static void RescueEmit(string json) { }
#endif

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Attach()
    {
        bool web = Application.platform == RuntimePlatform.WebGLPlayer;
        bool forced = System.Environment.GetEnvironmentVariable("RESCUE_MISSION") == "1";
        var c = Object.FindAnyObjectByType<SearchCoordinator>();
        if ((!web && !forced) || c == null) return;
        var m = new GameObject("RescueControls").AddComponent<RescueMission>();
        m.coord = c;
        c.autoStart = false;          // the mission starts when the user presses Start
        // Candidates for the trapped people: PickPeople keeps the PeopleInMission best placed ones (this runs before the
        // generator's Start, so the first building gets them too).
        c.generator.minPeople = c.generator.maxPeople = PeopleCandidates;
        c.generator.Generated += m.PickPeople;
        c.allowKeys = false;          // no rebuilding or pausing the building mid-mission
    }

    void Awake()
    {
        if (string.IsNullOrEmpty(serverUrl))
        {
            var page = Application.absoluteURL;
            serverUrl = Application.platform == RuntimePlatform.WebGLPlayer && !string.IsNullOrEmpty(page)
                ? page.Substring(0, page.LastIndexOf('/') + 1)
                : System.Environment.GetEnvironmentVariable("SCAN_SERVER") ?? "http://127.0.0.1:8000/";
        }
        if (!serverUrl.EndsWith("/")) serverUrl += "/";
    }

    void Start()
    {
        if (coord == null) coord = Object.FindAnyObjectByType<SearchCoordinator>();
        Time.timeScale = speed;
        // At 10x a frame covers ~0.3 s of mission time; allow up to 1 s so a slower frame rate doesn't stretch the
        // mission clock (the coordinator still moves the drones in 0.05 s steps).
        Time.maximumDeltaTime = 1f;
    }

    /// Editor tests: use this coordinator without the web page.
    public void InitForTest(SearchCoordinator c) => coord = c;

    // ---------- commands from the page ----------

    public void StartMission()
    {
        if (State != Phase.Ready || creating || coord == null || coord.Grid == null) return;
        creating = true;
        StartCoroutine(Post("api/rescue", MissionRecordJson(), r =>
        {
            creating = false;
            if (r.result == UnityWebRequest.Result.Success)
                MissionId = JsonUtility.FromJson<Created>(r.downloadHandler.text).scan_id;
            else lastError = "AI server not reachable (" + r.error + "): the mission runs, but frames are not analyzed.";
            Begin(MissionId != null);
            if (MissionId != null) StartCoroutine(Post("api/rescue/" + MissionId + "/ground_truth", TruthJson(), null));
            Emit();
        }));
    }

    /// "off": no fire spread this mission (the building's starting fires still burn). Only before the start.
    public void SetFireSpread(string value)
    {
        if (State == Phase.Ready && value == "off") { fireInterval = 1e9f; fireOff = true; Emit(); }
    }

    /// A fixed building (layout, furniture, people, fires) for repeatable missions: rebuilds it from this seed.
    public void SetSeed(string value)
    {
        if (State != Phase.Ready || creating || coord == null || coord.generator == null) return;
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int seed) || seed <= 0) return;
        var gen = coord.generator;
        // The old building is destroyed at the end of the frame; hide it now so its colliders don't reach the new search.
        for (int i = gen.transform.childCount - 1; i >= 0; i--)
            if (gen.transform.GetChild(i).name == "Building") gen.transform.GetChild(i).gameObject.SetActive(false);
        gen.seed = seed;
        gen.Generate();             // the coordinator sets the search up again (Generated event)
        Emit();
    }

    public void SetSpeed(string value)
    {
        if (!float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float k)) return;
        speed = aiScale = Mathf.Clamp(k, 1f, 10f);
        Time.timeScale = speed;
        Emit();
    }

    [System.Serializable] class Created { public string scan_id; }

    // ---------- mission ----------

    void Update()
    {
        Step(Time.deltaTime);
        if (aiPacing && State == Phase.Running && cameraOn)
        {
            // Frames wanted but the model is still busy: slow down (quickly); otherwise speed back up to the selection.
            float udt = Mathf.Min(Time.unscaledDeltaTime, 0.1f);
            aiScale = starved ? Mathf.Max(1f, aiScale * Mathf.Exp(-3f * udt)) : Mathf.Min(speed, aiScale * Mathf.Exp(1f * udt));
            Time.timeScale = Mathf.Min(speed, aiScale);
        }
        else if (Time.timeScale != speed) Time.timeScale = speed;
        if ((emitTimer -= Time.unscaledDeltaTime) <= 0f) { emitTimer = 0.25f; Emit(); }
    }

    /// Editor tests: start without the website or the AI server (no frames: the drones search by proximity).
    public void BeginOffline() => Begin(false);

    /// Editor tests: camera search without the website, AI server or rendering: each frame's spots count as searched
    /// at once (no detections), to test the search and coverage logic.
    public void BeginSimulatedFrames() { simulateFrames = true; Begin(true); }
    bool simulateFrames;
    public int FramesSent => framesSent;

    void Begin(bool camera)
    {
        fire = new FireSpread(coord.generator, coord, new FireSpread.Settings { interval = fireInterval }, coord.generator.seed + 7);
        initialFires = coord.Fires.Count;
        MissionTime = 0f;
        State = Phase.Running;
        cameraOn = camera;
        var g = coord.Grid;
        cover = new byte[g.floors, g.nx, g.nz];
        truth.Clear();
        truthFloor.Clear();
        foreach (var p in coord.generator.people)
        {
            if (p == null) continue;
            truth.Add(p.Bounds.center);
            truthFloor.Add(g.StoreyOf(p.Bounds.min.y + 0.1f));
        }
        visibleFrames = new int[truth.Count];
        closestView = new float[truth.Count];
        for (int i = 0; i < closestView.Length; i++) closestView[i] = float.PositiveInfinity;
        foreach (var d in coord.drones) d.cameraSearch = camera;
        coord.minFloorWork = camera ? minFloorWork : 1;
        coord.Begin();
    }

    public bool Collapsing => collapse != null;
    public int CutOff => CutOffByFire();
    public string EndReason => endReason;
    public int PeopleTotal => truth.Count;
    public float SearchedShare => Searched(out _);

    /// Advances the mission by dt of mission time (Update in play; tests call it directly).
    public void Step(float dt)
    {
        if (State == Phase.Running)
        {
            MissionTime += dt;
            fire?.Step(dt);
            starved = false;
            if (truth.Count > 0 && PeopleFound() == truth.Count)
            {
                State = Phase.Successful;              // checked before the deadline: both can't happen
                endReason = "all_found";
                coord.Halt();
            }
            else if (MissionTime >= missionSeconds)
            {
                MissionTime = missionSeconds;
                State = Phase.Failed;
                endReason = "time";
                coord.Halt();
                collapse = new BuildingCollapse(coord.generator, fire, coord.drones, coord.generator.seed + 11);
            }
            else
            {
                CaptureFrames();
                KeepSearching();
            }
            if (State != Phase.Running)
            {
                Emit();
                if (MissionId != null) StartCoroutine(Post("api/rescue/" + MissionId + "/ground_truth", TruthJson(), null));
            }
        }
        collapse?.Step(dt);
    }

    readonly Dictionary<int, float> restartAt = new();

    /// No drone stops while someone is still missing. A drone that has run out of work (Done) gets more at once: it
    /// takes over part of another drone's remaining area, or else tries its own left-over and given-up spots again,
    /// or, when its whole area has been covered, searches that area again (a person missed once can be found on the
    /// next look). What the page shows as searched (cover) is kept.
    void KeepSearching()
    {
        foreach (var d in coord.drones)
        {
            if (d.Status != SearchDrone.State.Done) continue;
            if (restartAt.TryGetValue(d.id, out float at) && MissionTime < at) continue;
            restartAt[d.id] = MissionTime + 2f;          // nothing it can reach at all: don't restart every frame
            coord.minFloorWork = 1;                      // after the first pass even a few spots on a floor are worth the trip
            if (!cameraOn) { if (!coord.ClaimWork(d) && !HasOwnWork(d.id)) SearchAgain(d.id); }
            else if (TakeUncovered(d) == 0) SearchAgain(d.id);   // only when no camera has missed anything anywhere
            passes++;
            coord.Restart(d);
        }
    }

    /// The spots no camera frame has covered yet (wherever they are, whoever's area, given up or not) become this
    /// drone's targets, except those on a floor where another drone is still busy (that drone is on them).
    int TakeUncovered(SearchDrone d)
    {
        var g = coord.Grid;
        int n = 0;
        for (int f = 0; f < g.floors; f++)
        {
            if (coord.drones.Exists(o => o != d && o.Status != SearchDrone.State.Done && o.Floor == f && f != d.Floor)) continue;
            for (int x = 0; x < g.nx; x++)
                for (int z = 0; z < g.nz; z++)
                {
                    var st = g.state[f, x, z];
                    if (st == SearchGrid.Spot.None || st == SearchGrid.Spot.Solid || cover[f, x, z] > 0) continue;
                    g.state[f, x, z] = SearchGrid.Spot.Unseen;
                    g.owner[f, x, z] = d.id;
                    n++;
                }
        }
        return n;
    }

    bool HasOwnWork(int drone)
    {
        var g = coord.Grid;
        for (int f = 0; f < g.floors; f++)
            for (int x = 0; x < g.nx; x++)
                for (int z = 0; z < g.nz; z++)
                    if (g.owner[f, x, z] == drone && (g.state[f, x, z] == SearchGrid.Spot.Unseen || g.state[f, x, z] == SearchGrid.Spot.Unreachable)) return true;
        return false;
    }

    void SearchAgain(int drone)
    {
        var g = coord.Grid;
        for (int f = 0; f < g.floors; f++)
            for (int x = 0; x < g.nx; x++)
                for (int z = 0; z < g.nz; z++)
                    if (g.owner[f, x, z] == drone && g.state[f, x, z] == SearchGrid.Spot.Seen) g.state[f, x, z] = SearchGrid.Spot.Unseen;
    }

    /// Share of the open floor (not inside walls or furniture) that some frame has covered.
    float Searched(out int left)
    {
        var g = coord.Grid;
        left = 0;
        if (!cameraOn || cover == null) { left = g.Count(SearchGrid.Spot.Unseen); return coord.Coverage(); }
        int open = 0, done = 0;
        for (int f = 0; f < g.floors; f++)
            for (int x = 0; x < g.nx; x++)
                for (int z = 0; z < g.nz; z++)
                {
                    var st = g.state[f, x, z];
                    if (st == SearchGrid.Spot.None || st == SearchGrid.Spot.Solid) continue;
                    open++;
                    if (cover[f, x, z] > 0) done++; else left++;
                }
        return open == 0 ? 1f : done / (float)open;
    }

    int InFlightTotal()
    {
        int n = 0;
        foreach (var v in inFlight.Values) n += v;
        return n;
    }

    /// Trapped people with a confirmed AI sighting near them (evaluation against the simulation's true positions).
    int PeopleFound()
    {
        int n = 0;
        for (int i = 0; i < truth.Count; i++)
            if (MatchOf(i) != null) n++;
        return n;
    }

    Sighting MatchOf(int person)
    {
        foreach (var s in sightings)
            if (s.confirmed && s.floor == truthFloor[person] && Flat(s.Pos - truth[person]) <= sightingRadius) return s;
        return null;
    }

    int PersonNear(Sighting s)
    {
        for (int i = 0; i < truth.Count; i++)
            if (s.floor == truthFloor[i] && Flat(s.Pos - truth[i]) <= sightingRadius) return i;
        return -1;
    }

    static float Flat(Vector3 v) => new Vector2(v.x, v.z).magnitude;

    /// Spots the drones gave up on (Unreachable) because a fire that started during the mission blocks them.
    int CutOffByFire()
    {
        var g = coord.Grid;
        if (g == null || coord.Fires.Count <= initialFires) return 0;
        const float near = 2f;
        int n = 0;
        for (int f = 0; f < g.floors; f++)
            for (int x = 0; x < g.nx; x++)
                for (int z = 0; z < g.nz; z++)
                {
                    if (g.state[f, x, z] != SearchGrid.Spot.Unreachable) continue;
                    var w = g.World(f, x, z);
                    for (int i = initialFires; i < coord.Fires.Count; i++)
                    {
                        var fi = coord.Fires[i];
                        if (fi.floor != f) continue;
                        float dx = fi.position.x - w.x, dz = fi.position.z - w.z;
                        if (dx * dx + dz * dz <= near * near) { n++; break; }
                    }
                }
        return n;
    }
    int cutOffCache;

    // ---------- AI frames ----------

    void CaptureFrames()
    {
        if (!cameraOn) return;
        foreach (var d in coord.drones)
        {
            if (d == null || d.cam == null) continue;
            if (d.Status == SearchDrone.State.Idle || d.Status == SearchDrone.State.Climbing || d.Status == SearchDrone.State.Done) continue;
            bool looking = d.LookTarget != null && !d.LookDone;   // hovering to look at a spot: a frame now
            if (!looking && nextCapture.TryGetValue(d.id, out float due) && MissionTime < due) continue;
            inFlight.TryGetValue(d.id, out int busy);
            if (busy >= maxInFlight)
            {
                // The model is still busy with this drone's last frame: wait (and slow down if there is something to see).
                if (looking || lastGain.TryGetValue(d.id, out int lg) && lg >= minGain) starved = true;
                continue;
            }
            nextCapture[d.id] = MissionTime + captureEvery;
            var view = BestView(d, out bool unfit);
            lastGain[d.id] = view.cells == null ? 0 : Mathf.RoundToInt(view.gain);
            if (view.cells == null)
            {
                if (unfit) framesUnfit++; else framesNoGain++;
                continue;
            }
            if (looking)
            {
                var lt = d.LookTarget.Value - d.cam.transform.position;
                if (Mathf.Abs(Mathf.DeltaAngle(Mathf.Atan2(lt.x, lt.z) * Mathf.Rad2Deg, view.yaw)) <= aiFov * 0.5f - 3f) d.LookDone = true;
            }
            if (simulateFrames) { Cover(view.cells, d.id, true); framesSent++; }
            else StartCoroutine(SendFrame(d, view));
        }
    }

    struct View { public float yaw, gain; public List<Vector3Int> cells; }
    readonly List<(Vector3Int cell, float bearing, float weight)> candidates = new();

    /// The direction (gimbal yaw) whose frame would search the most spots not yet searched, plus unconfirmed
    /// sightings seen from a position that has not detected them yet. cells == null: no frame worth taking.
    View BestView(SearchDrone d, out bool unfit)
    {
        unfit = false;
        var g = coord.Grid;
        int f = d.Floor;
        var cam = d.cam.transform.position;
        float floorY = g.origin.y + f * g.floorHeight;
        candidates.Clear();
        var c = g.CellOf(cam);
        int r = Mathf.CeilToInt(inspectRange / SearchGrid.Cell);
        float maxDown = aiPitch + aiFov * 0.5f - 2f, minDown = aiPitch - aiFov * 0.5f + 2f;
        for (int x = Mathf.Max(1, c.x - r); x <= Mathf.Min(g.nx - 2, c.x + r); x++)
            for (int z = Mathf.Max(1, c.y - r); z <= Mathf.Min(g.nz - 2, c.y + r); z++)
            {
                var st = g.state[f, x, z];
                if (st == SearchGrid.Spot.None || st == SearchGrid.Spot.Solid) continue;
                var w = g.World(f, x, z);
                float dx = w.x - cam.x, dz = w.z - cam.z, hd = Mathf.Sqrt(dx * dx + dz * dz);
                if (hd < inspectMin || hd > inspectRange) continue;
                float down = Mathf.Atan2(cam.y - floorY, hd) * Mathf.Rad2Deg;
                if (down > maxDown || down < minDown) continue;           // above or below the frame
                if (!SeesSpot(cam, w.x, w.z, floorY)) continue;
                candidates.Add((new Vector3Int(x, f, z), Mathf.Atan2(dx, dz) * Mathf.Rad2Deg, st == SearchGrid.Spot.Seen ? 0f : 1f));
            }
        if (d.LookTarget != null)
        {
            var lt = d.LookTarget.Value - cam;
            candidates.Add((new Vector3Int(-1, -1, -1), Mathf.Atan2(lt.x, lt.z) * Mathf.Rad2Deg, 2f * verifyBonus));
        }
        var key = ViewKey(cam);
        foreach (var s in sightings)
        {
            if (s.confirmed || s.floor != f || s.best < 0.6f || s.views.Contains(key)) continue;
            var to = s.Pos - cam;
            float hd = Flat(to);
            if (hd < 0.8f || hd > inspectRange + 0.5f) continue;
            if (Physics.Raycast(cam, to.normalized, out var hit, to.magnitude, SearchCoordinator.WorldMask, QueryTriggerInteraction.Ignore)
                && hit.distance < to.magnitude - 0.6f) continue;
            candidates.Add((new Vector3Int(-1, -1, -1), Mathf.Atan2(to.x, to.z) * Mathf.Rad2Deg, verifyBonus));
        }

        float half = aiFov * 0.5f - 3f;
        int steps = Mathf.Max(1, Mathf.RoundToInt(360f / yawStep));
        var order = new List<(float yaw, float gain)>();
        for (int k = 0; k < steps; k++)
        {
            float yaw = k * yawStep, gain = 0f;
            foreach (var cd in candidates)
                if (Mathf.Abs(Mathf.DeltaAngle(cd.bearing, yaw)) <= half) gain += cd.weight;
            order.Add((yaw, gain));
        }
        order.Sort((a, b) => b.gain.CompareTo(a.gain));
        foreach (var (yaw, gain) in order)
        {
            if (gain < minGain) break;
            if (!ClearView(d, yaw)) { unfit = true; continue; }
            var cells = new List<Vector3Int>();
            foreach (var cd in candidates)
                if (cd.cell.x >= 0 && Mathf.Abs(Mathf.DeltaAngle(cd.bearing, yaw)) <= half) cells.Add(cd.cell);
            return new View { yaw = yaw, gain = gain, cells = cells };
        }
        return new View { gain = order.Count > 0 ? order[0].gain : 0f };
    }

    /// The camera sees a spot when the line of sight toward its floor ends at that spot: on the floor, or on whatever
    /// lies on top of it there (a bed, a sofa, a person). A ray stopped earlier (a wall, a sofa back) does not count.
    public static bool SeesSpot(Vector3 cam, float x, float z, float floorY)
    {
        var target = new Vector3(x, floorY + 0.05f, z);
        var dir = target - cam;
        float len = dir.magnitude;
        if (!Physics.Raycast(cam, dir / len, out var hit, len + 0.1f, SearchCoordinator.WorldMask, QueryTriggerInteraction.Ignore)) return true;
        float hx = hit.point.x - x, hz = hit.point.z - z;
        return hx * hx + hz * hz <= 0.4f * 0.4f;
    }

    static Vector3Int ViewKey(Vector3 p) => new(Mathf.RoundToInt(p.x * 2f), Mathf.RoundToInt(p.y * 2f), Mathf.RoundToInt(p.z * 2f));

    /// Frame quality gate (the conditions of the model's training images): nothing right in front of the lens and no
    /// fire filling the view.
    bool ClearView(SearchDrone d, float yaw)
    {
        var pos = d.cam.transform.position;
        var fwd = Quaternion.Euler(aiPitch, yaw, 0f) * Vector3.forward;
        if (Physics.Raycast(pos, fwd, minClearView, SearchCoordinator.WorldMask, QueryTriggerInteraction.Ignore)) return false;
        if (Physics.CheckSphere(pos, 0.3f, SearchCoordinator.WorldMask, QueryTriggerInteraction.Ignore)) return false;
        foreach (var f in coord.Fires)
        {
            var dv = f.position - pos; dv.y = 0f;
            if (f.floor == d.Floor && dv.magnitude < fireClearance) return false;
        }
        return true;
    }

    void Cover(List<Vector3Int> cells, int droneId, bool add)
    {
        var g = coord.Grid;
        foreach (var c in cells)
        {
            int f = c.y, x = c.x, z = c.z;
            if (add)
            {
                if (cover[f, x, z] < 255) cover[f, x, z]++;
                if (g.state[f, x, z] == SearchGrid.Spot.Unseen || g.state[f, x, z] == SearchGrid.Spot.Unreachable)
                {
                    g.state[f, x, z] = SearchGrid.Spot.Seen;
                    g.seenBy[f, x, z] = droneId;
                }
            }
            else if (cover[f, x, z] > 0 && --cover[f, x, z] == 0 && g.state[f, x, z] == SearchGrid.Spot.Seen)
                g.state[f, x, z] = SearchGrid.Spot.Unseen;           // the only frame covering it never reached the model
        }
    }

    struct Pose { public Vector3 pos; public Quaternion rot; public float fov; public int floor; public float time; }

    IEnumerator SendFrame(SearchDrone d, View view)
    {
        inFlight[d.id] = (inFlight.TryGetValue(d.id, out int n) ? n : 0) + 1;
        Cover(view.cells, d.id, true);
        var camPos = d.cam.transform.position;
        byte[] jpg = CaptureFrom(camPos, view.yaw, out var camRot);
        var pose = new Pose { pos = camPos, rot = camRot, fov = aiFov, floor = d.Floor, time = MissionTime };
        TallyTruthInView(pose);
        int seq = ++sequence;
        var sb = new StringBuilder("{");
        sb.Append("\"sequence_number\":").Append(seq);
        sb.Append(",\"timestamp\":").Append(Q(System.DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ", CultureInfo.InvariantCulture)));
        sb.Append(",\"sim_time\":").Append(F(pose.time));
        sb.Append(",\"drone_id\":").Append(d.id + 1).Append(",\"floor\":").Append(pose.floor + 1);
        sb.Append(",\"drone_position\":").Append(V(d.transform.position)).Append(",\"camera_position\":").Append(V(pose.pos));
        sb.Append(",\"camera_orientation\":{\"forward\":").Append(V(pose.rot * Vector3.forward)).Append(",\"image_up\":").Append(V(pose.rot * Vector3.up)).Append('}');
        sb.Append(",\"camera\":{\"vertical_fov_deg\":").Append(F(pose.fov)).Append(",\"width\":").Append(captureSize)
          .Append(",\"height\":").Append(captureSize).Append(",\"format\":\"jpg\",\"pitch_deg\":").Append(F(aiPitch)).Append(",\"yaw_deg\":").Append(F(view.yaw))
          .Append(",\"spots_searched\":").Append(view.cells.Count)
          .Append(",\"mount\":\"AI camera at the drone camera mount, gimbal turned to the most unsearched spots, tilted down, with headlight (matches the model's training camera)\"}");
        sb.Append(",\"image_b64\":").Append(Q(System.Convert.ToBase64String(jpg))).Append('}');
        string reply = null;
        yield return Post("api/rescue/" + MissionId + "/frames", sb.ToString(), r =>
        {
            if (r.result == UnityWebRequest.Result.Success) { framesSent++; reply = r.downloadHandler.text; }
            else { framesFailed++; lastError = "Frame not analyzed: " + r.error; }
        });
        if (reply == null) Cover(view.cells, d.id, false);            // not analyzed: those spots are not searched
        inFlight[d.id]--;
        if (reply == null) yield break;
        var res = JsonUtility.FromJson<FrameResult>(reply);
        if (res == null || res.detections == null || res.detections.Length == 0) yield break;
        detections += res.detections.Length;
        // Locate each detection: a ray from this frame's camera pose through the box center, into the scene.
        var loc = new StringBuilder("{\"image_id\":").Append(Q(res.image_id)).Append(",\"positions\":[");
        for (int i = 0; i < res.detections.Length; i++)
        {
            var det = res.detections[i];
            var b = det.bbox;
            float u = b.x + b.w * 0.5f, v = b.y + b.h * 0.5f;
            float t = Mathf.Tan(pose.fov * 0.5f * Mathf.Deg2Rad);   // square frame: same horizontally
            var dir = pose.rot * new Vector3((2f * u - 1f) * t, (1f - 2f * v) * t, 1f).normalized;
            loc.Append(i > 0 ? "," : "").Append("{\"detection_id\":").Append(Q(det.detection_id));
            if (Physics.Raycast(pose.pos, dir, out var hit, 40f, SearchCoordinator.WorldMask, QueryTriggerInteraction.Ignore))
            {
                bool person = hit.collider.GetComponentInParent<TrappedPerson>() != null;
                // Physical size of the box at that distance: a person (standing, sitting or lying) is 0.25-2.2 m.
                // Bigger boxes are doorways, corridors and stairs the model mistakes for people.
                float span = 2f * hit.distance * t;
                float sw = b.w * span, sh = b.h * span, big = Mathf.Max(sw, sh);
                bool plausible = big <= maxPersonSize && big >= minPersonSize;
                int floor = coord.Grid.StoreyOf(hit.point.y + 0.1f);
                loc.Append(",\"available\":true,\"x\":").Append(F(hit.point.x)).Append(",\"y\":").Append(F(hit.point.y)).Append(",\"z\":").Append(F(hit.point.z))
                   .Append(",\"floor\":").Append(floor + 1).Append(",\"distance_m\":").Append(F(hit.distance))
                   .Append(",\"ray_hit\":").Append(Q(person ? "person collider" : "scene geometry"))
                   .Append(",\"size_m\":[").Append(F(sw)).Append(',').Append(F(sh)).Append("],\"plausible\":").Append(plausible ? "true" : "false")
                   .Append(",\"method\":\"ray from the frame's camera pose through the box center to the first scene surface\"");
                if (plausible) AddSighting(floor, hit.point, pose.pos, det, res.image_id, d.id, hit.distance);
                else implausible++;
            }
            else loc.Append(",\"available\":false,\"reason\":\"ray hit nothing within 40 m\"");
            loc.Append('}');
        }
        loc.Append("]}");
        yield return Post("api/rescue/" + MissionId + "/localize", loc.ToString(), null);
    }

    /// Groups a located, person-sized detection with earlier ones at the same place (same floor, within sightingRadius).
    void AddSighting(int floor, Vector3 at, Vector3 camPos, Det det, string imageId, int drone, float distance)
    {
        Sighting s = null;
        float bestD = sightingRadius;
        foreach (var o in sightings)
        {
            if (o.floor != floor) continue;
            float dd = Flat(o.Pos - at);
            if (dd <= bestD) { bestD = dd; s = o; }
        }
        if (s == null) { s = new Sighting { id = sightings.Count + 1, floor = floor }; sightings.Add(s); }
        s.sum += at;
        s.n++;
        s.frames++;
        s.views.Add(ViewKey(camPos));
        if (det.confidence > s.best)
        {
            s.best = det.confidence; s.detectionId = det.detection_id; s.imageId = imageId; s.box = det.bbox;
            s.drone = drone; s.distance = distance;
        }
        s.confirmed = s.views.Count >= confirmViews && s.best >= confirmConfidence;
        if (!s.confirmed && s.best >= verifyConfidence && !s.verifyRequested) RequestSecondLook(s);
    }

    /// A strong sighting not yet confirmed: the spots 1.5-3 m around it with a clear line of sight to it become search
    /// targets again for the drone that saw it, so it comes back and looks from a new position (the frame direction
    /// then favors the sighting, verifyBonus). The person still has to be detected again, at confirmConfidence, to count.
    void RequestSecondLook(Sighting s)
    {
        s.verifyRequested = true;
        var d = coord.drones.Find(o => o.id == s.drone);
        var g = coord.Grid;
        if (d == null || !cameraOn) return;
        var c = g.CellOf(s.Pos);
        int r = Mathf.CeilToInt(3f / SearchGrid.Cell), n = 0;
        for (int x = Mathf.Max(1, c.x - r); x <= Mathf.Min(g.nx - 2, c.x + r); x++)
            for (int z = Mathf.Max(1, c.y - r); z <= Mathf.Min(g.nz - 2, c.y + r); z++)
            {
                if (g.state[s.floor, x, z] != SearchGrid.Spot.Seen) continue;
                var w = g.World(s.floor, x, z);
                float hd = Flat(w - s.Pos);
                if (hd < 1.5f || hd > 3f) continue;
                var to = s.Pos - w;
                if (Physics.Raycast(w, to.normalized, out var hit, to.magnitude, SearchCoordinator.WorldMask, QueryTriggerInteraction.Ignore)
                    && hit.distance < to.magnitude - 0.6f) continue;
                g.state[s.floor, x, z] = SearchGrid.Spot.Unseen;
                g.owner[s.floor, x, z] = d.id;
                d.Unvisit(s.floor, x, z);
                n++;
            }
        secondLooks++;
        Debug.Log($"Second look at sighting {s.id} (floor {s.floor + 1}, {s.best:0.00}): {n} spots for drone {d.id + 1}");
    }

    /// Evaluation only: which trapped people this frame could show (inside the frame, within 6 m, in clear view).
    void TallyTruthInView(Pose pose)
    {
        float t = Mathf.Tan(pose.fov * 0.5f * Mathf.Deg2Rad);
        var inv = Quaternion.Inverse(pose.rot);
        for (int i = 0; i < truth.Count; i++)
        {
            if (truthFloor[i] != pose.floor) continue;
            var to = truth[i] - pose.pos;
            var local = inv * to;
            if (local.z <= 0.1f || Mathf.Abs(local.x / local.z) > t || Mathf.Abs(local.y / local.z) > t || to.magnitude > 6f) continue;
            if (Physics.Raycast(pose.pos, to.normalized, out var hit, to.magnitude + 0.5f, SearchCoordinator.WorldMask, QueryTriggerInteraction.Ignore)
                && hit.collider.GetComponentInParent<TrappedPerson>() == null) continue;
            visibleFrames[i]++;
            closestView[i] = Mathf.Min(closestView[i], Flat(to));
        }
    }

    [System.Serializable] class Box { public float x, y, w, h; }
    [System.Serializable] class Det { public string detection_id; public float confidence; public Box bbox; }
    [System.Serializable] class FrameResult { public string image_id; public string status; public Det[] detections; }

    /// The drone camera's real view, square at the model's input size, without the sonar/drone overlay layers.
    void EnsureAiCamera()
    {
        if (aiCam != null) return;
        var go = new GameObject("AI Camera");
        aiCam = go.AddComponent<Camera>();
        aiCam.enabled = false;                                   // rendered on demand
        aiCam.fieldOfView = aiFov;
        aiCam.aspect = 1f;
        aiCam.nearClipPlane = 0.05f;
        aiCam.farClipPlane = 200f;
        aiCam.cullingMask = SearchCoordinator.WorldMask;         // no drones, sonar overlays or schematic
        var data = aiCam.GetUniversalAdditionalCameraData();
        data.renderPostProcessing = true;
        data.antialiasing = AntialiasingMode.SubpixelMorphologicalAntiAliasing;
        data.antialiasingQuality = AntialiasingQuality.High;
        var lightGo = new GameObject("AI Headlight");
        lightGo.transform.SetParent(go.transform, false);
        headlight = lightGo.AddComponent<Light>();
        headlight.type = LightType.Spot;
        headlight.color = new Color(1f, 0.96f, 0.88f);
        headlight.range = headlightRange;
        headlight.spotAngle = headlightAngle;
        headlight.innerSpotAngle = headlightAngle * 0.5f;
        headlight.shadows = LightShadows.None;
        headlight.enabled = false;                               // on only while the AI frame renders
        aiRT = new RenderTexture(captureSize, captureSize, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB) { antiAliasing = 1 };
        aiCam.targetTexture = aiRT;
        aiTex = new Texture2D(captureSize, captureSize, TextureFormat.RGB24, false, false);
    }

    /// One AI frame from the drone's camera mount: the given heading, 30 deg down, headlight on and
    /// dimmed when a surface is close (the training images' auto-exposure stand-in), then halved while the frame
    /// is blown out. Returns the JPEG and the exact pose used (for locating detections).
    /// An AI frame from any pose (also used by the editor test RescueTests).
    public byte[] CaptureFrom(Vector3 pos, float yaw, out Quaternion rot)
    {
        EnsureAiCamera();
        rot = Quaternion.Euler(aiPitch, yaw, 0f);
        aiCam.transform.SetPositionAndRotation(pos, rot);
        const float fullAt = 3f;
        float nearest = fullAt;
        var t = aiCam.transform;
        foreach (var dir in new[] { t.forward, t.forward + 0.35f * t.right, t.forward - 0.35f * t.right, t.forward + 0.35f * t.up, t.forward - 0.35f * t.up })
            if (Physics.Raycast(t.position, dir.normalized, out var hit, fullAt, SearchCoordinator.WorldMask, QueryTriggerInteraction.Ignore))
                nearest = Mathf.Min(nearest, hit.distance);
        float k = nearest / fullAt;
        headlight.intensity = headlightIntensity * Mathf.Clamp(k * k, 0.08f, 1f);
        headlight.enabled = true;
        try
        {
            coord.WithSceneLook(RenderAi);
            for (int retake = 0; retake < 4 && BlownOut() > 0.04f; retake++)
            {
                headlight.intensity *= 0.5f;
                coord.WithSceneLook(RenderAi);
            }
        }
        finally { headlight.enabled = false; }
        return aiTex.EncodeToJPG(95);
    }

    void RenderAi()
    {
        aiCam.Render();
        var prev = RenderTexture.active;
        RenderTexture.active = aiRT;
        aiTex.ReadPixels(new Rect(0, 0, captureSize, captureSize), 0, 0, false);
        aiTex.Apply(false);
        RenderTexture.active = prev;
    }

    float BlownOut()
    {
        var px = aiTex.GetPixelData<byte>(0);
        int blown = 0, n = px.Length / 3;
        for (int i = 0; i < px.Length; i += 3)
            if (px[i] > 235 && px[i + 1] > 235 && px[i + 2] > 235) blown++;
        return (float)blown / n;
    }

    /// After every (re)build: of the candidate people, keep the PeopleInMission that a drone can really find. A person
    /// counts by the spots a drone can fly to (reachable from the stairs at flight height) 1.2-3.2 m away with a clear
    /// line of sight to them; people near a starting fire don't count, and upright poses (the model's easiest) score
    /// higher. One per floor where possible. The rest are removed before the mission starts.
    void PickPeople(BuildingGenerator gen)
    {
        if (coord == null) coord = Object.FindAnyObjectByType<SearchCoordinator>();
        var g = coord != null ? coord.Grid : null;
        if (g == null || State != Phase.Ready) return;
        Physics.SyncTransforms();
        var reach = new bool[g.floors, g.nx, g.nz];
        var queue = new Queue<Vector3Int>();
        for (int f = 0; f < g.floors; f++)
        {
            var s0 = g.CellOf(g.StairEntry(f));
            reach[f, s0.x, s0.y] = true;
            queue.Enqueue(new Vector3Int(s0.x, f, s0.y));
        }
        while (queue.Count > 0)
        {
            var c = queue.Dequeue();
            foreach (var st in new[] { new Vector2Int(1, 0), new Vector2Int(-1, 0), new Vector2Int(0, 1), new Vector2Int(0, -1) })
            {
                int x = c.x + st.x, z = c.z + st.y, f = c.y;
                if (!g.InInterior(x, z) || reach[f, x, z]) continue;
                var w = g.World(f, x, z);
                if (Physics.CheckSphere(w, 0.25f, SearchCoordinator.WorldMask, QueryTriggerInteraction.Ignore)) continue;
                if (coord.Fires.Exists(fi => fi.floor == f && Flat(fi.position - w) < 1.5f)) continue;
                reach[f, x, z] = true;
                queue.Enqueue(new Vector3Int(x, f, z));
            }
        }
        var scored = new List<(TrappedPerson p, int floor, float score)>();
        foreach (var p in gen.people)
        {
            if (p == null) continue;
            var b = p.Bounds;
            int f = g.StoreyOf(b.min.y + 0.1f);
            if (coord.Fires.Exists(fi => fi.floor == f && Flat(fi.position - b.center) < 3f)) { scored.Add((p, f, -1f)); continue; }
            var c = g.CellOf(b.center);
            int views = 0, r = Mathf.CeilToInt(3.2f / SearchGrid.Cell);
            for (int x = Mathf.Max(1, c.x - r); x <= Mathf.Min(g.nx - 2, c.x + r); x++)
                for (int z = Mathf.Max(1, c.y - r); z <= Mathf.Min(g.nz - 2, c.y + r); z++)
                {
                    if (!reach[f, x, z]) continue;
                    var w = g.World(f, x, z);
                    float hd = Flat(w - b.center);
                    if (hd < 1.2f || hd > 3.2f) continue;
                    var to = b.center - w;
                    if (Physics.Raycast(w, to.normalized, out var hit, to.magnitude + 0.5f, SearchCoordinator.WorldMask, QueryTriggerInteraction.Ignore)
                        && hit.collider.GetComponentInParent<TrappedPerson>() == p) views++;
                }
            bool upright = p.pose == PersonPose.Standing || p.pose == PersonPose.Waving || p.pose == PersonPose.SittingOnFurniture;
            scored.Add((p, f, views * (upright ? 1.5f : 1f)));
        }
        scored.Sort((a, b) => b.score.CompareTo(a.score));
        var keep = new List<TrappedPerson>();
        var floorsUsed = new HashSet<int>();
        foreach (var e in scored)                                   // the best one on each floor first...
            if (keep.Count < PeopleInMission && e.score >= 8f && floorsUsed.Add(e.floor)) keep.Add(e.p);
        foreach (var e in scored)                                   // ...then the best of the rest
            if (keep.Count < PeopleInMission && !keep.Contains(e.p) && e.score > 0f) keep.Add(e.p);
        foreach (var e in scored)                                   // never fewer than PeopleInMission
            if (keep.Count < PeopleInMission && !keep.Contains(e.p)) keep.Add(e.p);
        foreach (var e in scored)
        {
            if (keep.Contains(e.p)) continue;
            gen.people.Remove(e.p);
            e.p.gameObject.SetActive(false);                       // its colliders leave now; the object at the end of the frame
            Destroy(e.p.gameObject);
        }
        Debug.Log($"Trapped people: kept {keep.Count} of {scored.Count} candidates (" +
                  string.Join(", ", scored.FindAll(e => keep.Contains(e.p)).ConvertAll(e => $"floor {e.floor + 1} {e.p.pose} {e.score:0}")) + ")");
        Emit();
    }

    void OnDestroy()
    {
        if (coord != null && coord.generator != null) coord.generator.Generated -= PickPeople;
        if (aiRT != null) { aiRT.Release(); Destroy(aiRT); }
        if (aiTex != null) Destroy(aiTex);
    }

    // ---------- server ----------

    IEnumerator Post(string path, string json, System.Action<UnityWebRequest> done)
    {
        using var r = new UnityWebRequest(serverUrl + path, "POST")
        {
            uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json)),
            downloadHandler = new DownloadHandlerBuffer(),
            timeout = 30,
        };
        r.SetRequestHeader("Content-Type", "application/json");
        yield return r.SendWebRequest();
        done?.Invoke(r);
    }

    string MissionRecordJson() =>
        "{\"target_category\":\"person\",\"region\":{\"building\":\"Building2 four-storey burning building\",\"floors\":" + coord.generator.floors +
        ",\"seed\":" + coord.generator.seed + "},\"config\":{\"mission_seconds\":" + F(missionSeconds) + ",\"capture_every_s\":" + F(captureEvery) +
        ",\"drones\":" + coord.drones.Count + ",\"fire_spread\":" + (fireOff ? "false" : "true") + ",\"inspect_range_m\":" + F(inspectRange) +
        ",\"confirm\":{\"views\":" + confirmViews + ",\"confidence\":" + F(confirmConfidence) + "}" +
        "},\"coordinate_frame\":\"Unity world meters (x east, y up, z north); floors numbered from 1\"}";

    /// Evaluation only (stored apart from the frames and never served): where the people really are, and at the end
    /// how often each was inside a frame and whether the AI confirmed them.
    string TruthJson()
    {
        var sb = new StringBuilder("{\"seed\":").Append(coord.generator.seed).Append(",\"people\":[");
        for (int i = 0; i < truth.Count; i++)
        {
            var c = truth[i];
            sb.Append(i > 0 ? "," : "").Append("{\"x\":").Append(F(c.x)).Append(",\"y\":").Append(F(c.y)).Append(",\"z\":").Append(F(c.z))
              .Append(",\"floor\":").Append(truthFloor[i] + 1)
              .Append(",\"visible_frames\":").Append(visibleFrames[i]).Append(",\"closest_m\":").Append(F(closestView[i]))
              .Append(",\"confirmed\":").Append(MatchOf(i) != null ? "true" : "false").Append('}');
        }
        sb.Append("],\"state\":").Append(Q(State.ToString().ToLowerInvariant())).Append(",\"end_reason\":").Append(Q(endReason))
          .Append(",\"mission_time_s\":").Append(F(MissionTime)).Append(",\"passes\":").Append(passes)
          .Append(",\"frames\":{\"sent\":").Append(framesSent).Append(",\"failed\":").Append(framesFailed).Append(",\"no_gain\":").Append(framesNoGain)
          .Append(",\"unfit\":").Append(framesUnfit).Append(",\"detections\":").Append(detections).Append(",\"implausible\":").Append(implausible).Append('}')
          .Append(",\"second_looks\":").Append(secondLooks).Append(",\"false_confirmed\":").Append(FalseConfirmed()).Append('}');
        return sb.ToString();
    }

    int FalseConfirmed()
    {
        int n = 0;
        foreach (var s in sightings) if (s.confirmed && PersonNear(s) < 0) n++;
        return n;
    }

    // ---------- page ----------

    void Emit()
    {
        if (coord == null || coord.Grid == null) return;
        var g = coord.Grid;
        float searched = Searched(out int unseen);
        int seen = g.Count(SearchGrid.Spot.Seen), closed = g.Count(SearchGrid.Spot.Unreachable);
        bool ended = State == Phase.Successful || State == Phase.Failed;
        var sb = new StringBuilder("{\"type\":\"rescue\"");
        sb.Append(",\"state\":").Append(Q(State.ToString().ToLowerInvariant())).Append(",\"end_reason\":").Append(Q(endReason));
        sb.Append(",\"remaining_s\":").Append(F(Remaining)).Append(",\"mission_s\":").Append(F(missionSeconds));
        sb.Append(",\"speed\":").Append(F(speed)).Append(",\"effective_speed\":").Append(F(State == Phase.Running ? Time.timeScale : speed));
        sb.Append(",\"seed\":").Append(coord.generator != null ? coord.generator.seed : 0).Append(",\"fire_spread\":").Append(fireOff ? "false" : "true");
        sb.Append(",\"mission_id\":").Append(Q(MissionId)).Append(",\"camera_search\":").Append(cameraOn ? "true" : "false").Append(",\"pass\":").Append(passes);
        sb.Append(",\"coverage\":").Append(F(searched)).Append(",\"seen\":").Append(seen).Append(",\"unseen\":").Append(unseen).Append(",\"closed_off\":").Append(closed);
        sb.Append(",\"search_complete\":").Append(coord.Complete ? "true" : "false");
        if (coord.Complete || State != Phase.Running) cutOffCache = CutOffByFire();
        sb.Append(",\"cut_off_by_fire\":").Append(cutOffCache);
        sb.Append(",\"fire_share\":").Append(F(fire != null ? fire.Share : 0f)).Append(",\"collapsing\":").Append(collapse != null ? "true" : "false");
        sb.Append(",\"frames\":{\"sent\":").Append(framesSent).Append(",\"failed\":").Append(framesFailed).Append(",\"no_gain\":").Append(framesNoGain).Append(",\"skipped_view\":").Append(framesUnfit)
          .Append(",\"detections\":").Append(detections).Append(",\"implausible\":").Append(implausible).Append('}');
        int found = State == Phase.Ready ? 0 : PeopleFound();
        sb.Append(",\"people_total\":").Append(State == Phase.Ready ? coord.generator.people.Count : truth.Count).Append(",\"people_found\":").Append(found);
        if (ended) sb.Append(",\"false_confirmed\":").Append(FalseConfirmed());
        sb.Append(",\"error\":").Append(Q(lastError));
        sb.Append(",\"drones\":[");
        for (int i = 0; i < coord.drones.Count; i++)
        {
            var d = coord.drones[i];
            sb.Append(i > 0 ? "," : "").Append("{\"id\":").Append(d.id + 1).Append(",\"floor\":").Append(d.Floor + 1).Append(",\"status\":").Append(Q(d.Status.ToString()))
              .Append(",\"stairs\":").Append(Q(d.Status == SearchDrone.State.Holding ? d.StairsReachable() : null)).Append('}');
        }
        sb.Append("],\"sightings\":[");
        var list = sightings.FindAll(s => s.best >= 0.5f);
        list.Sort((a, b) => a.confirmed != b.confirmed ? (a.confirmed ? -1 : 1) : b.best.CompareTo(a.best));
        for (int i = 0; i < list.Count && i < 30; i++)
        {
            var s = list[i];
            var p = s.Pos;
            sb.Append(i > 0 ? "," : "").Append("{\"id\":").Append(s.id).Append(",\"floor\":").Append(s.floor + 1)
              .Append(",\"x\":").Append(F(p.x)).Append(",\"y\":").Append(F(p.y)).Append(",\"z\":").Append(F(p.z))
              .Append(",\"best\":").Append(F(s.best)).Append(",\"frames\":").Append(s.frames).Append(",\"views\":").Append(s.views.Count)
              .Append(",\"confirmed\":").Append(s.confirmed ? "true" : "false").Append(",\"drone\":").Append(s.drone + 1)
              .Append(",\"distance_m\":").Append(F(s.distance)).Append(",\"image_id\":").Append(Q(s.imageId)).Append(",\"detection_id\":").Append(Q(s.detectionId))
              .Append(",\"bbox\":{\"x\":").Append(F(s.box.x)).Append(",\"y\":").Append(F(s.box.y)).Append(",\"w\":").Append(F(s.box.w)).Append(",\"h\":").Append(F(s.box.h)).Append('}');
            if (ended && s.confirmed) sb.Append(",\"true_person\":").Append(PersonNear(s) >= 0 ? "true" : "false");
            sb.Append('}');
        }
        sb.Append("]}");
        RescueEmit(sb.ToString());
    }

    static string F(float v) => float.IsFinite(v) ? v.ToString("0.###", CultureInfo.InvariantCulture) : "null";
    static string V(Vector3 v) => "[" + F(v.x) + "," + F(v.y) + "," + F(v.z) + "]";
    static string Q(string s)
    {
        if (s == null) return "null";
        var sb = new StringBuilder("\"");
        foreach (char ch in s)
        {
            if (ch == '"' || ch == '\\') sb.Append('\\').Append(ch);
            else if (ch < ' ') sb.Append("\\u").Append(((int)ch).ToString("x4"));
            else sb.Append(ch);
        }
        return sb.Append('"').ToString();
    }
}
