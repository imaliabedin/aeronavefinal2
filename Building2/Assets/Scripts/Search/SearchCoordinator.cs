using System.Collections.Generic;
using Drone2.Building;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

/// Runs a multi-drone search of the generated building. After every (re)build it spawns the drones,
/// splits the building between them (see SplitMode), and lays the screen out as: one column per drone
/// camera (with that drone's sonar drawn only there) and the building schematic in the right-hand third.
/// Keys: R new building, M next split mode, 1/2 number of drones, Space pause, V sonar view.
public class SearchCoordinator : MonoBehaviour
{
    public struct Fire
    {
        public Vector3 position;
        public int floor;
    }

    public BuildingGenerator generator;
    public GameObject dronePrefab;

    [Header("Drones")]
    [Tooltip("Start floor of each drone (0 = first floor). One entry = a single drone searches everything.")]
    public int[] spawnFloors = { 0, 3 };
    public Color[] droneColors = { new Color(0.2f, 0.75f, 1f), new Color(1f, 0.6f, 0.15f), new Color(0.5f, 1f, 0.4f), new Color(1f, 0.4f, 0.8f) };
    [Tooltip("Width of the drone across its rotors (m). The prefab is scaled to this.")]
    public float droneSize = 0.4f;
    public float flightHeight = 1.5f;
    public float speed = 1.5f;
    public float sightRadius = 2f;
    public SonarSettings sonar = new SonarSettings();

    [Header("Splitting the search")]
    public SplitMode splitMode = SplitMode.BalancedNearest;
    [Tooltip("Estimated travel (m) per floor of stairs, used when splitting.")]
    public float climbCost = 12f;
    [Tooltip("A drone that finishes its share takes over part of another drone's remaining share.")]
    public bool shareWorkWhenDone = true;

    [Header("Run")]
    public bool autoStart = true;
    public float startDelay = 1.5f;
    [Tooltip("Simulation speed in Play mode (1 = real time, which also keeps the camera motion natural).")]
    public float timeScale = 1f;
    [Tooltip("Debug keys (R new building, M split mode, 1/2 drones, Space pause). Off during a Rescue Mission.")]
    public bool allowKeys = true;

    [Header("Look")]
    [Tooltip("Extra exposure (EV) on top of the building's fire mood, for the drone views.")]
    public float extraExposure = 0.7f;
    [Tooltip("Multiplies the scene's ambient light, to lift the darkest corners.")]
    public float ambientBoost = 1.6f;

    public const int SchematicLayer = 31;
    public static int DroneLayer(int i) => 20 + i;
    public static int SonarLayer(int i) => 24 + i;
    /// Layers the drones' sensors and collision checks see: everything but drones, their overlays and the schematic.
    public static int WorldMask => ~((1 << 2) | (0xFF << 20) | (1 << SchematicLayer));

    public SearchGrid Grid { get; private set; }
    public readonly List<SearchDrone> drones = new();
    public readonly List<Fire> Fires = new();
    /// A fire started by FireSpread: drones keep clear of it and the schematic shows it.
    public void AddFire(Vector3 at, int floor) => Fires.Add(new Fire { position = at, floor = floor });
    public float Elapsed { get; private set; }
    public bool Running { get; private set; }
    public bool Complete { get; private set; }
    public SearchSchematic Schematic { get; private set; }
    public Camera SchematicCamera { get; private set; }

    SearchDrone stairOwner;
    Volume brighten;
    Color[] baseAmbient;
    float startTimer, schematicTimer;
    bool paused;

    void OnEnable() { if (generator != null) generator.Generated += Setup; }
    void OnDisable() { if (generator != null) generator.Generated -= Setup; }
    void Start() => Time.timeScale = timeScale;

    /// Builds the search for a freshly generated building: drones, split, cameras and schematic.
    public void Setup(BuildingGenerator gen)
    {
        Teardown();
        Grid = new SearchGrid(gen, flightHeight);
        Fires.Clear();
        foreach (var t in gen.Root.GetComponentsInChildren<Transform>())
            if (t.name == "Fire") Fires.Add(new Fire { position = t.position, floor = Grid.StoreyOf(t.position.y) });

        var starts = new List<SearchSplit.Start>();
        for (int i = 0; i < spawnFloors.Length; i++)
        {
            int f = Mathf.Clamp(spawnFloors[i], 0, gen.floors - 1);
            var pos = SpawnPoint(gen, f, i);
            drones.Add(SpawnDrone(i, f, pos));
            starts.Add(new SearchSplit.Start { floor = f, cell = Grid.CellOf(pos) });
        }
        SearchSplit.Assign(Grid, starts, splitMode, climbCost);

        Brighten();
        SchematicCamera = new GameObject("Schematic Camera").AddComponent<Camera>();
        SchematicCamera.gameObject.AddComponent<AudioListener>();
        Schematic = new SearchSchematic(this, gen, SchematicCamera);
        Layout();

        Elapsed = 0f;
        Complete = false;
        Running = false;
        startTimer = startDelay;
        Debug.Log($"Search set up: {drones.Count} drone(s), {Grid.targets} spots, split {splitMode}" +
                  (drones.Count > 1 ? $" ({string.Join(" / ", drones.ConvertAll(d => Grid.CountOwned(d.id).ToString()))})" : ""));
    }

    /// Makes the building a little brighter for the search views without touching the building's own mood
    /// settings: lifts the ambient light and adds exposure through a higher-priority post-processing volume.
    void Brighten()
    {
        baseAmbient ??= new[] { RenderSettings.ambientSkyColor, RenderSettings.ambientEquatorColor, RenderSettings.ambientGroundColor };
        RenderSettings.ambientSkyColor = baseAmbient[0] * ambientBoost;
        RenderSettings.ambientEquatorColor = baseAmbient[1] * ambientBoost;
        RenderSettings.ambientGroundColor = baseAmbient[2] * ambientBoost;

        if (brighten == null)
        {
            brighten = new GameObject("Search Brightness").AddComponent<Volume>();
            brighten.isGlobal = true;
            brighten.priority = 10f;
            var profile = ScriptableObject.CreateInstance<VolumeProfile>();
            profile.Add<ColorAdjustments>(true);
            brighten.sharedProfile = profile;
        }
        // The building's mood grade sets its own exposure; this volume's value replaces it, so add to it.
        float mood = 0f;
        foreach (var v in FindObjectsByType<Volume>(FindObjectsSortMode.None))
            if (v != brighten && v.sharedProfile != null && v.sharedProfile.TryGet<ColorAdjustments>(out var c) && c.postExposure.overrideState)
                mood = c.postExposure.value;
        brighten.sharedProfile.TryGet<ColorAdjustments>(out var adjust);
        adjust.postExposure.Override(mood + extraExposure);
    }

    void Teardown()
    {
        foreach (var d in drones)
            if (d != null) Kill(d.gameObject);
        drones.Clear();
        if (SchematicCamera != null) Kill(SchematicCamera.gameObject);
        Schematic?.Destroy();
        Schematic = null;
        stairOwner = null;
    }

    static void Kill(Object o)
    {
        if (Application.isPlaying) Destroy(o); else DestroyImmediate(o);
    }

    /// A clear spot in the main corridor of floor f, spread along it for several drones on one floor.
    Vector3 SpawnPoint(BuildingGenerator gen, int f, int i)
    {
        var corridor = gen.Plan.mainCorridor;
        Physics.SyncTransforms();
        float baseX = corridor.xMin + corridor.width * (0.3f + 0.4f * (i % 2));
        for (int k = 0; k < corridor.width * 2; k++)
        {
            float x = Mathf.Round((baseX + (k % 2 == 0 ? 1 : -1) * (k / 2) * 0.5f) * 2f) / 2f;
            var p = Grid.World(f, Grid.CellOf(gen.Root.TransformPoint(new Vector3(x, 0, corridor.center.y))));
            if (!Physics.CheckSphere(p, 0.45f, WorldMask, QueryTriggerInteraction.Ignore)) return p;
        }
        return Grid.World(f, Grid.CellOf(gen.Root.TransformPoint(new Vector3(corridor.center.x, 0, corridor.center.y))));
    }

    SearchDrone SpawnDrone(int i, int floor, Vector3 pos)
    {
        var go = Instantiate(dronePrefab);
        go.name = $"Search Drone {i + 1}";
        foreach (var c in go.GetComponentsInChildren<Collider>()) Kill(c);
        var hover = go.GetComponent<DroneHover>();
        if (hover != null) Kill(hover);

        // Scale to droneSize across the rotors.
        go.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
        go.transform.localScale = Vector3.one;
        var b = RendererBounds(go);
        go.transform.localScale = Vector3.one * (droneSize / Mathf.Max(b.size.x, b.size.z));
        b = RendererBounds(go);
        var half = b.extents;
        foreach (var t in go.GetComponentsInChildren<Transform>(true)) t.gameObject.layer = DroneLayer(i);
        go.transform.SetPositionAndRotation(pos, Quaternion.Euler(0f, 90f, 0f));

        var color = droneColors[i % droneColors.Length];
        var sensor = go.AddComponent<SonarSensor>();
        sensor.settings = JsonUtility.FromJson<SonarSettings>(JsonUtility.ToJson(sonar));
        var vis = go.AddComponent<SonarVisualizer>();
        vis.sensor = sensor;
        vis.layer = SonarLayer(i);
        vis.Hook();

        var drone = go.AddComponent<SearchDrone>();
        drone.id = i;
        drone.color = color;
        drone.sensor = sensor;
        drone.visualizer = vis;
        drone.speed = speed;
        drone.sightRadius = sightRadius;
        drone.Init(this, Grid, floor, WorldMask, half);
        vis.belief = drone.Belief;
        // Only the belief cells at the drone's own flight height and close by, so the view stays readable.
        vis.showBeliefCell = c => c.y == Grid.BeliefLayer(drone.Floor) &&
                                  (drone.Belief.CenterOf(c) - drone.transform.position).sqrMagnitude < 9f;

        // The prefab's camera mount (30 degrees down, at the nose) becomes this drone's camera.
        var mount = FindChild(go.transform, "Camera") ?? go.transform;
        var cam = new GameObject($"Drone {i + 1} Camera").AddComponent<Camera>();
        cam.transform.SetParent(mount, false);
        cam.fieldOfView = 78f;
        cam.nearClipPlane = 0.03f;
        cam.farClipPlane = 80f;
        int hide = 1 << SchematicLayer | 1 << DroneLayer(i);
        for (int o = 0; o < 4; o++) if (o != i) hide |= 1 << SonarLayer(o);
        cam.cullingMask = ~hide;
        // The building's fire mood (bloom on the flames, grade, vignette) shows in the drone views, not the schematic.
        cam.GetUniversalAdditionalCameraData().renderPostProcessing = true;
        drone.cam = cam;
        return drone;
    }

    static Transform FindChild(Transform t, string name)
    {
        foreach (var c in t.GetComponentsInChildren<Transform>(true))
            if (c.name == name) return c;
        return null;
    }

    static Bounds RendererBounds(GameObject go)
    {
        var rs = go.GetComponentsInChildren<Renderer>();
        var b = rs[0].bounds;
        foreach (var r in rs) b.Encapsulate(r.bounds);
        return b;
    }

    /// One column per drone camera, the schematic in the right-hand third.
    void Layout()
    {
        float w = 2f / 3f / Mathf.Max(1, drones.Count);
        for (int i = 0; i < drones.Count; i++) drones[i].cam.rect = new Rect(i * w, 0f, w, 1f);
        SchematicCamera.rect = new Rect(2f / 3f, 0f, 1f / 3f, 1f);
        Schematic.FitCamera();
    }

    // ---- Running ------------------------------------------------------------------------------------

    void Update()
    {
        HandleKeys();
        if (Grid == null || paused) return;
        if (!Running && !Complete && autoStart && (startTimer -= Time.deltaTime) <= 0f) Begin();
        // Small steps (as in batch runs), so fast-forward (Time.timeScale 5-10) moves the drones the same way.
        float dt = Time.deltaTime;
        int steps = Mathf.Clamp(Mathf.CeilToInt(dt / 0.05f), 1, 40);
        for (int i = 0; i < steps; i++) Step(dt / steps);
        if ((schematicTimer -= Time.unscaledDeltaTime) <= 0f)
        {
            schematicTimer = 0.1f;
            Schematic.Refresh();
        }
    }

    /// Runs `render` with the scene's own lighting: without the extra brightness and ambient boost this coordinator
    /// adds for the on-screen drone views (the AI camera must look like the images the model was trained on).
    public void WithSceneLook(System.Action render)
    {
        bool vol = brighten != null && brighten.enabled;
        var amb = new[] { RenderSettings.ambientSkyColor, RenderSettings.ambientEquatorColor, RenderSettings.ambientGroundColor };
        if (brighten != null) brighten.enabled = false;
        if (baseAmbient != null)
        {
            RenderSettings.ambientSkyColor = baseAmbient[0];
            RenderSettings.ambientEquatorColor = baseAmbient[1];
            RenderSettings.ambientGroundColor = baseAmbient[2];
        }
        try { render(); }
        finally
        {
            if (brighten != null) brighten.enabled = vol;
            RenderSettings.ambientSkyColor = amb[0];
            RenderSettings.ambientEquatorColor = amb[1];
            RenderSettings.ambientGroundColor = amb[2];
        }
    }

    /// Stops the search where it is (the Rescue Mission's terminal states).
    public void Halt() => Running = false;

    /// Rescue Mission second pass: the drones search again (spots still unseen get another try from new positions).
    public void Resume()
    {
        Complete = false;
        Running = true;
        foreach (var d in drones) { d.ForgetVisited(); d.Begin(); }
    }

    /// Rescue Mission: one drone that ran out of work starts again (the search keeps running for the others).
    public void Restart(SearchDrone d)
    {
        Complete = false;
        Running = true;
        d.ForgetVisited();
        d.Begin();
    }

    public void Begin()
    {
        Running = true;
        foreach (var d in drones) d.Begin();
    }

    /// Advances every drone by dt (Play mode: each frame; batch runs: a fixed step).
    public void Step(float dt)
    {
        if (!Running || Complete) return;
        Physics.SyncTransforms();
        Elapsed += dt;
        foreach (var d in drones) d.Tick(dt);
        if (drones.TrueForAll(d => d.Status == SearchDrone.State.Done))
        {
            Complete = true;
            Debug.Log($"SEARCH COMPLETE in {Elapsed:0.0} s: {Summary()}");
        }
    }

    public string Summary()
    {
        int seen = Grid.Count(SearchGrid.Spot.Seen), solid = Grid.Count(SearchGrid.Spot.Solid);
        int closed = Grid.Count(SearchGrid.Spot.Unreachable), left = Grid.Count(SearchGrid.Spot.Unseen);
        int collisions = 0;
        foreach (var d in drones) collisions += d.Collisions;
        return $"seen {seen} of {seen + closed + left} open spots ({Coverage() * 100f:0.0}%), {closed} closed off, " +
               $"{solid} inside walls/furniture, {left} left; people found {PeopleFound()}/{generator.people.Count}; " +
               $"collisions {collisions}; " + string.Join(", ", drones.ConvertAll(d =>
                   $"drone {d.id + 1}: {d.Distance:0} m, {d.Replans} replans, camera turns at most {d.MaxCameraTurn:0} deg/s"));
    }

    /// Seen share of the spots that can be searched (not inside walls or furniture).
    public float Coverage()
    {
        int seen = Grid.Count(SearchGrid.Spot.Seen);
        int open = seen + Grid.Count(SearchGrid.Spot.Unseen) + Grid.Count(SearchGrid.Spot.Unreachable);
        return open == 0 ? 1f : seen / (float)open;
    }

    public bool PersonFound(TrappedPerson p)
    {
        var c = Grid.CellOf(p.Bounds.center);
        int f = Grid.StoreyOf(p.Bounds.min.y + 0.1f);
        for (int x = c.x - 1; x <= c.x + 1; x++)
            for (int z = c.y - 1; z <= c.y + 1; z++)
                if (x >= 0 && z >= 0 && x < Grid.nx && z < Grid.nz && Grid.state[f, x, z] == SearchGrid.Spot.Seen) return true;
        return false;
    }

    public int PeopleFound() => generator.people.FindAll(PersonFound).Count;

    // ---- Shared resources and work --------------------------------------------------------------------

    /// One drone in the stairwell at a time.
    public bool TakeStairs(SearchDrone d)
    {
        if (stairOwner != null && stairOwner != d) return false;
        stairOwner = d;
        return true;
    }

    public void ReleaseStairs(SearchDrone d)
    {
        if (stairOwner == d) stairOwner = null;
    }

    /// Nearest floor (by stairs) that still has unseen spots of this drone's share; -1 if none.
    public int NextFloorFor(SearchDrone d)
    {
        int best = -1;
        for (int f = 0; f < Grid.floors; f++)
        {
            if (!HasWork(f, d.id)) continue;
            if (best < 0 || Mathf.Abs(f - d.Floor) < Mathf.Abs(best - d.Floor)) best = f;
        }
        return best;
    }

    [Tooltip("A floor is worth a trip by the stairs only with at least this many spots left (Rescue Mission camera search: " +
             "a few scattered spots the camera can't see are not worth minutes of travel).")]
    public int minFloorWork = 1;

    bool HasWork(int f, int owner)
    {
        var d = drones.Find(o => o.id == owner);
        int need = d != null && f == d.Floor ? 1 : minFloorWork, n = 0;
        for (int x = 0; x < Grid.nx; x++)
            for (int z = 0; z < Grid.nz; z++)
                if (Grid.state[f, x, z] == SearchGrid.Spot.Unseen && Grid.owner[f, x, z] == owner && (d == null || !d.Visited(f, x, z)) && ++n >= need) return true;
        return false;
    }

    /// Work sharing: a drone with nothing left takes over a whole floor of someone else's remaining share
    /// (one they are not on), or else the half of their current floor's remainder that is furthest from them.
    public bool ClaimWork(SearchDrone d)
    {
        if (!shareWorkWhenDone) return false;
        int bestFloor = -1;
        for (int f = 0; f < Grid.floors; f++)
        {
            bool someoneHere = drones.Exists(o => o != d && o.Status != SearchDrone.State.Done &&
                                                  (o.Floor == f || o.GoalFloor == f));
            if (someoneHere) continue;
            if (!drones.Exists(o => o != d && HasWork(f, o.id))) continue;
            if (bestFloor < 0 || Mathf.Abs(f - d.Floor) < Mathf.Abs(bestFloor - d.Floor)) bestFloor = f;
        }
        int claimed = 0;
        if (bestFloor >= 0)
        {
            for (int x = 0; x < Grid.nx; x++)
                for (int z = 0; z < Grid.nz; z++)
                    if (Grid.state[bestFloor, x, z] == SearchGrid.Spot.Unseen && Grid.owner[bestFloor, x, z] != d.id)
                    {
                        Grid.owner[bestFloor, x, z] = d.id;
                        claimed++;
                    }
        }
        else
        {
            // Split the busiest drone's current floor: take the remaining spots furthest from it.
            foreach (var o in drones)
            {
                if (o == d || o.Status == SearchDrone.State.Done) continue;
                var left = new List<Vector2Int>();
                for (int x = 0; x < Grid.nx; x++)
                    for (int z = 0; z < Grid.nz; z++)
                        if (Grid.state[o.Floor, x, z] == SearchGrid.Spot.Unseen && Grid.owner[o.Floor, x, z] == o.id)
                            left.Add(new Vector2Int(x, z));
                if (left.Count < 40) continue;
                var at = Grid.CellOf(o.transform.position);
                left.Sort((a, b) => (b - at).sqrMagnitude.CompareTo((a - at).sqrMagnitude));
                for (int i = 0; i < left.Count / 2; i++) Grid.owner[o.Floor, left[i].x, left[i].y] = d.id;
                claimed += left.Count / 2;
                break;
            }
        }
        if (claimed > 0) Debug.Log($"Drone {d.id + 1} took over {claimed} spots" + (bestFloor >= 0 ? $" on floor {bestFloor + 1}" : ""));
        return claimed > 0;
    }

    // ---- Keys and HUD -----------------------------------------------------------------------------------

    void HandleKeys()
    {
        var kb = Keyboard.current;
        if (kb == null || generator == null || !allowKeys) return;
        if (kb.spaceKey.wasPressedThisFrame) paused = !paused;
        if (kb.rKey.wasPressedThisFrame) generator.GenerateNewSeed();
        if (kb.mKey.wasPressedThisFrame)
        {
            splitMode = (SplitMode)(((int)splitMode + 1) % System.Enum.GetValues(typeof(SplitMode)).Length);
            generator.Generate();
        }
        if (kb.digit1Key.wasPressedThisFrame) { spawnFloors = new[] { 0 }; generator.Generate(); }
        if (kb.digit2Key.wasPressedThisFrame) { spawnFloors = new[] { 0, generator.floors - 1 }; generator.Generate(); }
    }

    GUIStyle text, title;

    void OnGUI()
    {
        if (Grid == null) return;
        if (text == null)
        {
            text = new GUIStyle(GUI.skin.label) { fontSize = 13, richText = true, wordWrap = true };
            text.normal.textColor = Color.white;
            title = new GUIStyle(text) { fontSize = 15, fontStyle = FontStyle.Bold };
        }

        foreach (var d in drones)
        {
            var r = d.cam.pixelRect;
            var box = new Rect(r.x + 8, Screen.height - r.yMax + 8, r.width - 16, 62);
            Panel(box);
            string hex = ColorUtility.ToHtmlStringRGB(d.color);
            int share = Grid.CountOwned(d.id), done = share - Grid.Count(SearchGrid.Spot.Unseen, d.id);
            GUI.Label(new Rect(box.x + 8, box.y + 4, box.width - 16, 22),
                $"<color=#{hex}>■</color> DRONE {d.id + 1} CAMERA  ·  floor {d.Floor + 1}  ·  {StateText(d)}", title);
            GUI.Label(new Rect(box.x + 8, box.y + 26, box.width - 16, 36),
                $"share searched {done}/{share}  ·  sonar: {d.Belief.Occupied.Count} blocked cells, {d.Replans} replans  ·  " +
                $"collisions {d.Collisions}", text);
            // Sonar legend at the bottom of each drone view.
            var legend = new Rect(r.x + 8, Screen.height - r.y - 30, r.width - 16, 22);
            Panel(legend);
            GUI.Label(new Rect(legend.x + 8, legend.y + 2, legend.width - 16, 20),
                "<color=#9fe8ff>rings: pulse (slowed)</color>  <color=#ff8c1a>orange: echoes</color>  <color=#ff4040>red: sonar belief</color>  (V)", text);
        }

        var s = SchematicCamera.pixelRect;
        var head = new Rect(s.x + 8, 8, s.width - 16, 104);
        Panel(head);
        int mins = (int)(Elapsed / 60f);
        string status = Complete ? "<color=#7dff7d>SEARCH COMPLETE</color>" : Running ? "searching" : "starting…";
        GUI.Label(new Rect(head.x + 8, head.y + 4, head.width - 16, 22), $"BUILDING SCHEMATIC  ·  {Grid.floors} floors", title);
        GUI.Label(new Rect(head.x + 8, head.y + 26, head.width - 16, 80),
            $"Split: <b>{Nice(splitMode)}</b>{(shareWorkWhenDone && drones.Count > 1 ? " + work sharing" : "")}\n" +
            $"{status}  ·  {Coverage() * 100f:0}% searched  ·  {mins}:{Elapsed - mins * 60f:00}\n" +
            $"people found {PeopleFound()}/{generator.people.Count}" + (paused ? "  ·  <b>PAUSED</b>" : ""), text);

        foreach (var (pos, label) in Schematic.Labels())
        {
            var p = SchematicCamera.WorldToScreenPoint(pos);
            if (p.z > 0) GUI.Label(new Rect(p.x - 40, Screen.height - p.y - 10, 80, 20), label, title);
        }

        var foot = new Rect(s.x + 8, Screen.height - 58, s.width - 16, 50);
        Panel(foot);
        GUI.Label(new Rect(foot.x + 8, foot.y + 4, foot.width - 16, 46),
            "faint = still to search, bright = searched (by drone colour)\nR new building · M split · 1/2 drones · Space pause", text);
    }

    static string StateText(SearchDrone d) => d.Status switch
    {
        SearchDrone.State.Searching => "searching",
        SearchDrone.State.ToStairs => $"to stairs → floor {d.GoalFloor + 1}",
        SearchDrone.State.WaitingForStairs => "waiting for stairs",
        SearchDrone.State.Climbing => $"stairs → floor {d.GoalFloor + 1}",
        SearchDrone.State.Holding => "hovering, rescanning",
        SearchDrone.State.Done => "done",
        _ => "ready",
    };

    static string Nice(SplitMode m) => m switch
    {
        SplitMode.BalancedNearest => "balanced nearest-drone",
        SplitMode.NearestDrone => "nearest drone",
        SplitMode.FloorBlocks => "floor blocks",
        SplitMode.VerticalStrips => "vertical strips",
        _ => m.ToString(),
    };

    static void Panel(Rect r)
    {
        var prev = GUI.color;
        GUI.color = new Color(0f, 0f, 0f, 0.6f);
        GUI.DrawTexture(r, Texture2D.whiteTexture);
        GUI.color = prev;
    }
}
