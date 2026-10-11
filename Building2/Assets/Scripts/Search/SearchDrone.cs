using System.Collections.Generic;
using UnityEngine;

/// One search drone. It navigates by sonar only (the same sensor and belief map as the parking-lot drone):
/// cells it has not sensed count as free, and it replans as soon as a newly confirmed obstacle lies on its route.
/// Coverage is greedy: fly toward the nearest unseen spot of its share, marking every spot it can see on the way.
/// When its share of a floor is done it takes the stairwell to the next floor with work left.
/// Sonar noise can make a doorway look blocked for a while, so nothing is written off at the first failure:
/// before leaving a floor it hovers and rescans, written-off spots get another chance whenever a route opens,
/// and at the end it revisits each floor where it wrote spots off once.
public class SearchDrone : MonoBehaviour
{
    public enum State { Idle, Searching, ToStairs, WaitingForStairs, Climbing, Holding, Done }

    public int id;
    public Color color;
    public SonarSensor sensor;
    public SonarVisualizer visualizer;
    public Camera cam;
    [Tooltip("Cruise speed (m/s).")]
    public float speed = 1.5f;
    public float climbSpeed = 0.8f;
    [Tooltip("Acceleration and braking (m/s²).")]
    public float maxAccel = 0.8f;
    [Tooltip("Fastest turn (degrees/s) and how quickly a turn builds up (degrees/s²).")]
    public float maxYawRate = 55f;
    public float yawAccel = 90f;
    [Tooltip("Camera gimbal: downward tilt (degrees) and how long it takes to follow the heading (s).")]
    public float gimbalTilt = 25f;
    public float gimbalLag = 0.45f;
    [Tooltip("Spots within this distance with a clear line of sight count as seen.")]
    public float sightRadius = 2f;
    [Tooltip("Give up on a spot that is still unseen after this many seconds of trying to reach it.")]
    public float targetTimeout = 25f;
    [Tooltip("Thermal sensing: fires within this range are noticed, and nothing within keepOffFire of one is flown.")]
    public float heatRange = 4f;
    public float keepOffFire = 1.1f;
    [Tooltip("Rescue Mission: spots count as seen only when a camera frame covered them (set by RescueMission), " +
             "not because the drone flew within sightRadius. Arriving at a spot does not mark it seen either.")]
    public bool cameraSearch;
    [Tooltip("cameraSearch: stop this far (m) from an unsearched spot in clear view, and hover until a camera frame has " +
             "been aimed at it (at most lookHoldMax s), instead of flying on top of it where the camera can't see it.")]
    public float lookStandoff = 2f, lookHoldMax = 3f;
    /// cameraSearch: the floor point it is hovering to look at (RescueMission aims the next frame there and sets LookDone).
    public Vector3? LookTarget { get; private set; }
    public bool LookDone { get; set; }

    public SonarMap Belief { get; } = new SonarMap();
    public State Status { get; private set; }
    public int Floor { get; private set; }
    public int GoalFloor { get; private set; }
    public int Replans { get; private set; }
    public int Collisions { get; private set; }
    public string LastHit { get; private set; }
    public float Distance { get; private set; }
    /// Fastest the camera has turned (degrees/s), to keep an eye on jitter.
    public float MaxCameraTurn { get; private set; }
    public IReadOnlyList<Vector3> Route => path;
    public bool IsHeat(int f, int x, int z) => heat.Contains(new Vector3Int(x, f, z));
    public int RouteIndex => pathIndex;
    public string Debugging => $"pos {grid.CellOf(transform.position)} target {target} route {pathIndex}/{path.Count} " +
                               $"t {targetTime:0.0} givenUp {givenUp.Count} blockedHere {BlockedHere(Floor, target.x, target.y)}";

    SearchCoordinator coord;
    SearchGrid grid;
    readonly List<Vector3> path = new();
    int pathIndex;
    Vector2Int target;
    bool hasTarget, hasTargetBefore, replan, holdsStairs;
    float targetTime, sightTimer, heatTimer, yaw, pitch, roll;
    float curSpeed, speed01, yawRate, headingTarget, camYaw, camYawSpeed;
    bool movedThisTick, tracking;
    Quaternion lastCamRotation;
    readonly HashSet<Vector3Int> heat = new();   // (x, floor, z) cells too close to a fire
    readonly HashSet<Vector3Int> givenUp = new();   // (x, floor, z) spots it timed out on
    readonly HashSet<Vector3Int> visited = new();   // cameraSearch: spots it flew to (under the camera's view, not seen)
    readonly HashSet<int> rechecked = new();        // floors already revisited for written-off spots
    bool rescannedThisFloor;
    readonly HashSet<Vector3Int> flown = new();   // (x, floor, z) cells its body has flown through: known free
    int stairFailures;
    float holdTimer;
    System.Action afterHold;
    readonly List<Vector3Int> newlyBlocked = new(), released = new();
    Vector3 half;
    bool touching;
    int mask;

    // Dijkstra scratch.
    float[,] dist;
    Vector2Int[,] prev;

    public void Init(SearchCoordinator c, SearchGrid g, int floor, int layerMask, Vector3 halfExtents)
    {
        coord = c;
        grid = g;
        mask = layerMask;
        half = halfExtents;
        Floor = GoalFloor = floor;
        yaw = headingTarget = camYaw = transform.eulerAngles.y;
        dist = new float[g.nx, g.nz];
        prev = new Vector2Int[g.nx, g.nz];

        Belief.cellSize = SearchGrid.Cell;
        Belief.origin = g.origin;
        Belief.inflate = 0.25f;           // the least that still blocks a cell for every echo on 0.5 m cells
        Belief.groundY = -1000f;          // floors are real colliders here, not a known plane
        Belief.bounds = new BoundsInt(new Vector3Int(0, -2, 0),
            new Vector3Int(g.nx, g.BeliefLayer(g.floors - 1) + 8, g.nz));

        sensor.autoTick = false;
        sensor.settings.groundY = -1000f;
        sensor.settings.layers = layerMask;
        sensor.settings.seed = 1000 + id;
        sensor.Init();
        sensor.PulseCompleted += OnPulse;
        for (int i = 0; i < 3; i++) sensor.Pulse(transform.position);   // warm-up before the first plan
        Status = State.Idle;
    }

    void OnDestroy()
    {
        if (sensor) sensor.PulseCompleted -= OnPulse;
    }

    public void Begin()
    {
        Status = State.Searching;
        ChooseNext();
    }

    void OnPulse(SonarPulse pulse)
    {
        Belief.Integrate(pulse, sensor.settings.maxRange, newlyBlocked, released);
        if (newlyBlocked.Count > 0 && (Status == State.Searching || Status == State.ToStairs) && !RouteFree())
            replan = true;
    }

    /// One simulation step (the coordinator calls it every frame, or with a fixed step in batch runs).
    public void Tick(float dt)
    {
        if (Status == State.Idle) return;
        sensor.Tick(dt);
        if ((heatTimer -= dt) <= 0f) { heatTimer = 0.25f; SenseHeat(); }

        switch (Status)
        {
            case State.Searching:
            case State.ToStairs:
                if (replan)
                {
                    replan = false;
                    Replans++;
                    if (Status == State.Searching) ChooseNext(); else GoToStairs();
                    break;
                }
                if (Status == State.Searching)
                {
                    targetTime += dt;
                    if (!hasTarget || !IsWork(Floor, target.x, target.y)) { ChooseNext(); break; }
                    if (targetTime > targetTimeout)
                    {
                        givenUp.Add(new Vector3Int(target.x, Floor, target.y));
                        grid.state[Floor, target.x, target.y] = SearchGrid.Spot.Unreachable;
                        // Its neighbours are most likely just as hard to reach: not targets again this pass.
                        if (cameraSearch) VisitAround(target, 3);
                        ChooseNext();
                        break;
                    }
                }
                if (Status == State.Searching && cameraSearch && LookTarget == null && InLookRange()) { StartLook(); break; }
                if (Move(dt, speed))
                {
                    if (Status == State.ToStairs) TryClimb();
                    else
                    {
                        // Arrived: whatever is right here has been seen, even inside clutter. With cameraSearch the
                        // spot is under the camera's view: it stays unseen until a frame from elsewhere covers it.
                        if (cameraSearch) VisitAround(target);
                        else if (grid.state[Floor, target.x, target.y] != SearchGrid.Spot.Seen) Mark(Floor, target.x, target.y);
                        ChooseNext();
                    }
                }
                break;
            case State.WaitingForStairs:
                TryClimb();
                break;
            case State.Holding:
                if ((holdTimer -= dt) <= 0f || LookTarget != null && LookDone) afterHold();
                break;
            case State.Climbing:
                if (Move(dt, climbSpeed)) ArriveOnFloor(Floor + (GoalFloor > Floor ? 1 : -1));
                break;
        }

        // Free the stairs once clear of the stair door, or as soon as it has nothing left to do
        // (a finished drone parked by the door must not keep the other one waiting).
        if (holdsStairs && Status != State.Climbing &&
            (Status == State.Done || Vector3.Distance(transform.position, grid.StairEntry(Floor)) > 1.5f))
        {
            holdsStairs = false;
            coord.ReleaseStairs(this);
        }

        Attitude(dt);
        if (Status != State.Climbing && (sightTimer -= dt) <= 0f) { sightTimer = 0.1f; See(); }
        CheckCollision();
    }

    // ---- Choosing where to go -------------------------------------------------------------------------

    void ChooseNext()
    {
        path.Clear();
        pathIndex = 0;
        hasTarget = false;
        var start = grid.CellOf(transform.position);
        var previous = hasTargetBefore ? target : new Vector2Int(-1, -1);
        Dijkstra(Floor, start);
        if (cameraSearch) CountWorkAround();
        float best = float.PositiveInfinity;
        for (int x = 1; x < grid.nx - 1; x++)
            for (int z = 1; z < grid.nz - 1; z++)
            {
                if (!IsWork(Floor, x, z) || float.IsPositiveInfinity(dist[x, z])) continue;
                // cameraSearch: prefer spots inside larger unsearched areas (up to isolationCost cells further away), so
                // the drone sweeps an area at a time instead of zigzagging back for single leftover spots.
                float cost = dist[x, z] + (cameraSearch ? isolationCost * (1f - Mathf.Min(1f, workAround[x, z] / (float)areaFull)) : 0f);
                if (cost < best)
                {
                    best = cost;
                    target = new Vector2Int(x, z);
                }
            }

        if (!float.IsPositiveInfinity(best))
        {
            hasTarget = hasTargetBefore = true;
            if (target != previous) targetTime = 0f;   // replanning to the same spot keeps its clock running
            Status = State.Searching;
            BuildRoute(start, target);
            return;
        }

        // Nothing of its share left that it can reach on this floor. Hover and rescan once first: a doorway
        // that only looked blocked because of sonar noise clears as more pulses pass through it.
        if (!rescannedThisFloor)
        {
            rescannedThisFloor = true;
            Hold(2f, ChooseNext);
            return;
        }
        hasTargetBefore = false;
        // The rest is walls, furniture or closed off (for now).
        for (int x = 1; x < grid.nx - 1; x++)
            for (int z = 1; z < grid.nz - 1; z++)
                if (grid.state[Floor, x, z] == SearchGrid.Spot.Unseen && grid.owner[Floor, x, z] == id && !visited.Contains(new Vector3Int(x, Floor, z)))
                    grid.state[Floor, x, z] = BlockedHere(Floor, x, z) ? SearchGrid.Spot.Solid : SearchGrid.Spot.Unreachable;

        int next = coord.NextFloorFor(this);
        if (next < 0 && coord.ClaimWork(this))
        {
            next = coord.NextFloorFor(this);
            if (next == Floor) { ChooseNext(); return; }
        }
        if (next < 0) next = RecheckFloor();
        if (next >= 0 && next != Floor)
        {
            GoalFloor = next;
            GoToStairs();
            return;
        }
        Status = State.Done;
    }

    [Tooltip("cameraSearch: extra route cost (in 0.5 m cells) of an isolated unsearched spot compared with one in a large unsearched area.")]
    public float isolationCost = 8f;
    const int AreaRadius = 3, areaFull = 20;   // unsearched spots within 1.5 m that count as "a large area"
    int[,] workAround;

    /// Unsearched spots of its share within AreaRadius cells of each cell (box sum over the floor).
    void CountWorkAround()
    {
        int nx = grid.nx, nz = grid.nz;
        workAround ??= new int[nx, nz];
        var sum = new int[nx + 1, nz + 1];
        for (int x = 0; x < nx; x++)
            for (int z = 0; z < nz; z++)
                sum[x + 1, z + 1] = (IsWork(Floor, x, z) ? 1 : 0) + sum[x, z + 1] + sum[x + 1, z] - sum[x, z];
        for (int x = 0; x < nx; x++)
            for (int z = 0; z < nz; z++)
            {
                int x0 = Mathf.Max(0, x - AreaRadius), x1 = Mathf.Min(nx, x + AreaRadius + 1);
                int z0 = Mathf.Max(0, z - AreaRadius), z1 = Mathf.Min(nz, z + AreaRadius + 1);
                workAround[x, z] = sum[x1, z1] - sum[x0, z1] - sum[x1, z0] + sum[x0, z0];
            }
    }

    /// Its own share that still needs searching: unseen, or written off earlier but not timed out on.
    bool IsWork(int f, int x, int z) =>
        grid.owner[f, x, z] == id && !visited.Contains(new Vector3Int(x, f, z)) &&
        (grid.state[f, x, z] == SearchGrid.Spot.Unseen ||
         grid.state[f, x, z] == SearchGrid.Spot.Unreachable && !givenUp.Contains(new Vector3Int(x, f, z)));

    const int RecheckMin = 25;   // a floor is revisited only for at least this many written-off spots (about 6 m²)

    /// Final sweep: the nearest other floor where it wrote spots off and has not been back yet; -1 if none.
    int RecheckFloor()
    {
        int best = -1;
        for (int f = 0; f < grid.floors; f++)
        {
            if (f == Floor || rechecked.Contains(f)) continue;
            int left = 0;
            for (int x = 1; x < grid.nx - 1; x++)
                for (int z = 1; z < grid.nz - 1; z++)
                    if (grid.state[f, x, z] == SearchGrid.Spot.Unreachable && IsWork(f, x, z)) left++;
            if (left >= RecheckMin && (best < 0 || Mathf.Abs(f - Floor) < Mathf.Abs(best - Floor))) best = f;
        }
        if (best >= 0) rechecked.Add(best);
        return best;
    }

    bool InLookRange()
    {
        var w = grid.World(Floor, target.x, target.y);
        var d = w - transform.position;
        d.y = 0f;
        float m = d.magnitude;
        return m <= lookStandoff && m >= 0.9f &&
               RescueMission.SeesSpot(transform.position, w.x, w.z, grid.origin.y + Floor * grid.floorHeight);
    }

    void StartLook()
    {
        var w = grid.World(Floor, target.x, target.y);
        LookTarget = new Vector3(w.x, grid.origin.y + Floor * grid.floorHeight, w.z);
        LookDone = false;
        var t = target;
        Hold(lookHoldMax, () =>
        {
            LookTarget = null;
            if (IsWork(Floor, t.x, t.y)) VisitAround(t, 1);   // not visible from here after all: try the others first
            ChooseNext();
        });
    }

    void Hold(float seconds, System.Action then)
    {
        path.Clear();
        pathIndex = 0;
        Status = State.Holding;
        holdTimer = seconds;
        afterHold = then;
    }

    void GoToStairs()
    {
        path.Clear();
        pathIndex = 0;
        var start = grid.CellOf(transform.position);
        var entry = grid.CellOf(grid.StairEntry(Floor));
        Dijkstra(Floor, start);
        // Still cut off after a rescan: the fire's safety margin closes the way. First squeeze past the fire, closer than
        // usual; if even that is closed, fly through the fire's edge rather than stay trapped while people are missing
        // (back to the normal margin on the next floor).
        for (int level = 1; level <= 2 && float.IsPositiveInfinity(dist[entry.x, entry.y]) && escape < level && stairFailures >= 1; level++)
        {
            escape = level;
            Dijkstra(Floor, start);
            if (float.IsPositiveInfinity(dist[entry.x, entry.y])) continue;
            Debug.Log($"Drone {id + 1}: {(level == 1 ? "squeezing past" : "flying through the edge of")} the fire to the stairs on floor {Floor + 1}");
        }
        if (float.IsPositiveInfinity(dist[entry.x, entry.y])) escape = 0;
        if (float.IsPositiveInfinity(dist[entry.x, entry.y]))
        {
            // Hover and rescan: the way out may only look blocked.
            if (++stairFailures < 8) { Hold(3f, GoToStairs); return; }
            Debug.LogWarning($"Drone {id + 1}: no route to the stairs on floor {Floor + 1}");
            Status = State.Done;
            return;
        }
        stairFailures = 0;
        Status = State.ToStairs;
        BuildRoute(start, entry);
        path.Add(grid.StairEntry(Floor));
    }

    void TryClimb()
    {
        if (!coord.TakeStairs(this)) { Status = State.WaitingForStairs; return; }
        holdsStairs = true;
        Status = State.Climbing;
        path.Clear();
        pathIndex = 0;
        bool up = GoalFloor > Floor;
        var leg = up ? grid.ClimbUp(Floor) : grid.ClimbUp(Floor - 1);
        if (!up) leg.Reverse();
        path.AddRange(leg.GetRange(1, leg.Count - 1));
    }

    void ArriveOnFloor(int f)
    {
        Floor = f;
        escape = 0;
        rescannedThisFloor = false;
        if (Floor != GoalFloor) { TryClimb(); return; }
        Status = State.Searching;
        ChooseNext();
    }

    // ---- Planning on the sonar belief -------------------------------------------------------------------

    /// Free for planning: inside the known footprint, not near a fire, and either not believed blocked at flight
    /// height or flown through before. Echo noise can make a doorway look blocked; the doorway it came in by is
    /// in its own track, so it can always get back out.
    bool Free(int f, int x, int z)
    {
        var key = new Vector3Int(x, f, z);
        if (!grid.InInterior(x, z)) return false;
        if (!ignoreHeat && escape < 2 && (escape == 1 ? heatCore : heat).Contains(key)) return false;
        return flown.Contains(key) || ignoreBelief || !BlockedHere(f, x, z);
    }
    bool ignoreHeat, ignoreBelief;
    [Tooltip("Cut off from the stairs by the keepOffFire margin: it may pass this close (m) to a fire to get out.")]
    public float escapeKeepOff = 0.55f;
    int escape;   // on the way to the stairs (until the next floor): 1 squeezing past a fire, 2 flying through its edge
    readonly HashSet<Vector3Int> heatCore = new();      // (x, floor, z) cells within escapeKeepOff of a fire

    /// Debugging: which planning constraint keeps it from the stairs (normal / without fire zones / without sonar belief).
    public string StairsReachable()
    {
        var start = grid.CellOf(transform.position);
        var entry = grid.CellOf(grid.StairEntry(Floor));
        string r = "";
        foreach (var (h, b, name) in new[] { (false, false, "normal"), (true, false, "no-heat"), (false, true, "no-belief"), (true, true, "neither") })
        {
            ignoreHeat = h; ignoreBelief = b;
            Dijkstra(Floor, start);
            r += $"{name}:{(!float.IsPositiveInfinity(dist[entry.x, entry.y]) ? "yes" : "no")} ";
        }
        ignoreHeat = ignoreBelief = false;
        return r + $"heat cells {heat.Count}";
    }

    bool BlockedHere(int f, int x, int z) => Belief.IsBlocked(new Vector3Int(x, grid.BeliefLayer(f), z));

    static readonly Vector2Int[] Steps =
    {
        new(1, 0), new(-1, 0), new(0, 1), new(0, -1), new(1, 1), new(1, -1), new(-1, 1), new(-1, -1),
    };

    /// Shortest routes over the believed-free cells of floor f (8-connected, no corner cutting).
    void Dijkstra(int f, Vector2Int start)
    {
        for (int x = 0; x < grid.nx; x++)
            for (int z = 0; z < grid.nz; z++)
                dist[x, z] = float.PositiveInfinity;
        if (!grid.InInterior(start.x, start.y)) return;
        var open = new Heap();
        dist[start.x, start.y] = 0f;
        open.Push(start, 0f);
        while (open.Count > 0)
        {
            var (c, d) = open.Pop();
            if (d > dist[c.x, c.y]) continue;
            foreach (var s in Steps)
            {
                var n = c + s;
                if (!Free(f, n.x, n.y)) continue;
                if (s.x != 0 && s.y != 0 && (!Free(f, c.x + s.x, c.y) || !Free(f, c.x, c.y + s.y))) continue;
                float nd = d + (s.x != 0 && s.y != 0 ? 1.4142f : 1f);
                if (nd >= dist[n.x, n.y]) continue;
                dist[n.x, n.y] = nd;
                prev[n.x, n.y] = c;
                open.Push(n, nd);
            }
        }
    }

    /// Turns the cell chain from Dijkstra into waypoints, skipping cells while the straight line stays clear.
    void BuildRoute(Vector2Int start, Vector2Int goal)
    {
        var cells = new List<Vector2Int>();
        for (var c = goal; c != start; c = prev[c.x, c.y]) cells.Add(c);
        cells.Reverse();
        var from = transform.position;
        int i = -1;
        while (i < cells.Count - 1)
        {
            int j = cells.Count - 1;
            while (j > i + 1 && !SegmentFree(from, grid.World(Floor, cells[j]))) j--;
            from = grid.World(Floor, cells[j]);
            path.Add(from);
            i = j;
        }
    }

    /// True if the drone's body (about 0.4 m across) can fly the straight line a-b on the current belief.
    bool SegmentFree(Vector3 a, Vector3 b, float skip = 0f)
    {
        float len = Vector3.Distance(a, b);
        var startCell = grid.CellOf(a);
        for (float t = skip; t <= len; t += 0.1f)
        {
            var p = Vector3.Lerp(a, b, len > 0f ? t / len : 0f);
            foreach (var o in Probe)
            {
                var c = grid.CellOf(p + o);
                if (c == startCell) continue;
                if (!Free(Floor, c.x, c.y)) return false;
            }
        }
        return true;
    }

    static readonly Vector3[] Probe = { Vector3.zero, new(0.2f, 0, 0), new(-0.2f, 0, 0), new(0, 0, 0.2f), new(0, 0, -0.2f) };

    /// True if the rest of the planned route is still clear on the latest belief.
    bool RouteFree()
    {
        if (pathIndex >= path.Count) return true;
        if (!SegmentFree(transform.position, path[pathIndex], 0.3f)) return false;
        for (int i = pathIndex; i < path.Count - 1; i++)
            if (!SegmentFree(path[i], path[i + 1])) return false;
        return true;
    }

    // ---- Moving -----------------------------------------------------------------------------------------

    /// Flies along the route like a real multirotor: speeds up and brakes gradually, slows for corners so it
    /// can stop at any turn, and slows right down while its nose swings toward a new direction. It stays exactly
    /// on the planned (collision-checked) route; only its speed and heading are smoothed. True once at the end.
    bool Move(float dt, float vmax)
    {
        movedThisTick = true;
        var pos = transform.position;
        while (pathIndex < path.Count && (path[pathIndex] - pos).sqrMagnitude < 0.0025f) pathIndex++;
        if (pathIndex >= path.Count) { speed01 = 0f; return true; }

        var to = path[pathIndex] - pos;
        float d = to.magnitude;
        var dir = to / d;

        // Speed it may still carry into the next waypoint: zero at the end, lower the sharper the turn there.
        float vNext = 0f;
        if (pathIndex + 1 < path.Count)
            vNext = vmax * CornerFactor(Vector3.Angle(dir, path[pathIndex + 1] - path[pathIndex]));
        float canBrake = Mathf.Sqrt(vNext * vNext + 2f * maxAccel * d);

        // Slow while the nose (and camera) is turned well away from where it is going.
        var flat = new Vector3(dir.x, 0f, dir.z);
        if (flat.sqrMagnitude > 0.01f) headingTarget = Quaternion.LookRotation(flat).eulerAngles.y;
        float yawError = Mathf.Abs(Mathf.DeltaAngle(yaw, headingTarget));
        float facing = flat.sqrMagnitude > 0.01f ? Mathf.Clamp(1f - yawError / 100f, 0.1f, 1f) : 1f;

        float desired = Mathf.Min(vmax * facing, canBrake);
        curSpeed = Mathf.MoveTowards(curSpeed, desired, maxAccel * dt);
        speed01 = vmax > 0f ? curSpeed / vmax : 0f;

        float step = curSpeed * dt;
        if (step >= d) { pos = path[pathIndex]; pathIndex++; Distance += d; }
        else { pos += dir * step; Distance += step; }
        transform.position = pos;

        if (Status != State.Climbing)
        {
            var at = grid.CellOf(pos);
            flown.Add(new Vector3Int(at.x, Floor, at.y));
        }
        return pathIndex >= path.Count;
    }

    /// 1 on a straight line, down to 0.15 for a turn of 90 degrees or more.
    static float CornerFactor(float angle) => Mathf.Clamp(1f - angle / 100f, 0.15f, 1f);

    /// Heading, body lean and the camera gimbal, every step (also while hovering).
    void Attitude(float dt)
    {
        if (!movedThisTick) { curSpeed = Mathf.MoveTowards(curSpeed, 0f, maxAccel * dt); speed01 = Mathf.MoveTowards(speed01, 0f, dt); }
        movedThisTick = false;

        // Yaw: rate- and acceleration-limited, so turns start and end softly.
        float err = Mathf.DeltaAngle(yaw, headingTarget);
        float wanted = Mathf.Clamp(err * 1.8f, -maxYawRate, maxYawRate);
        yawRate = Mathf.MoveTowards(yawRate, wanted, yawAccel * dt);
        yaw += yawRate * dt;

        // Lean forward with speed and into turns, eased.
        float k = 1f - Mathf.Exp(-dt * 3f);
        pitch = Mathf.Lerp(pitch, 8f * speed01, k);
        roll = Mathf.Lerp(roll, -6f * speed01 * yawRate / maxYawRate, k);
        transform.rotation = Quaternion.Euler(pitch, yaw, roll);

        // Gimbal: the camera keeps a level horizon at a fixed downward tilt and follows the heading smoothly.
        if (cam != null)
        {
            camYaw = Mathf.SmoothDampAngle(camYaw, yaw, ref camYawSpeed, gimbalLag, Mathf.Infinity, dt);
            var mount = cam.transform.parent != null ? cam.transform.parent.position : transform.position;
            var r = Quaternion.Euler(gimbalTilt, camYaw, 0f);
            float turn = Quaternion.Angle(lastCamRotation, r) / Mathf.Max(dt, 1e-4f);
            if (tracking) MaxCameraTurn = Mathf.Max(MaxCameraTurn, turn);
            tracking = true;
            lastCamRotation = r;
            cam.transform.SetPositionAndRotation(mount, r);
        }
    }

    // ---- Seeing, heat, collisions -----------------------------------------------------------------------

    void See()
    {
        if (cameraSearch) return;   // the Rescue Mission marks spots from its camera frames
        var pos = transform.position;
        var c = grid.CellOf(pos);
        int r = Mathf.CeilToInt(sightRadius / SearchGrid.Cell);
        for (int x = Mathf.Max(1, c.x - r); x <= Mathf.Min(grid.nx - 2, c.x + r); x++)
            for (int z = Mathf.Max(1, c.y - r); z <= Mathf.Min(grid.nz - 2, c.y + r); z++)
            {
                if (grid.state[Floor, x, z] != SearchGrid.Spot.Unseen) continue;
                var w = grid.World(Floor, x, z);
                float d = Vector3.Distance(pos, w);
                if (d > sightRadius) continue;
                // Seen if nothing is in the way, or what is in the way is right at the spot (its surface is what we see).
                if (d < 0.5f || !Physics.Raycast(pos, (w - pos) / d, out var hit, d, mask, QueryTriggerInteraction.Ignore)
                    || hit.distance >= d - 0.35f)
                    Mark(Floor, x, z);
            }
    }

    /// cameraSearch: the spots within 1 m of where it arrived are under the camera's view from here (it looks 30 deg
    /// down): no longer targets for this pass, but still unseen until a frame from elsewhere covers them.
    void VisitAround(Vector2Int at, int r = 2)
    {
        for (int x = at.x - r; x <= at.x + r; x++)
            for (int z = at.y - r; z <= at.y + r; z++)
                if ((x - at.x) * (x - at.x) + (z - at.y) * (z - at.y) <= r * r) visited.Add(new Vector3Int(x, Floor, z));
    }

    /// cameraSearch: forget the spots it flew to, so a second pass can try them again from other positions.
    public void ForgetVisited() { visited.Clear(); givenUp.Clear(); rechecked.Clear(); stairFailures = 0; rescannedThisFloor = false; }

    /// cameraSearch: make a spot a target again (a second look at a sighting from there).
    public void Unvisit(int f, int x, int z) => visited.Remove(new Vector3Int(x, f, z));

    /// Spots it flew to that no camera frame has covered yet.
    public bool Visited(int f, int x, int z) => visited.Contains(new Vector3Int(x, f, z));

    void Mark(int f, int x, int z)
    {
        grid.state[f, x, z] = SearchGrid.Spot.Seen;
        grid.seenBy[f, x, z] = id;
    }

    /// Thermal sensing: cells near a fire on this floor become no-fly for this drone.
    void SenseHeat()
    {
        if (Status == State.Climbing) return;
        bool added = false;
        foreach (var fire in coord.Fires)
        {
            if (fire.floor != Floor) continue;
            var d = fire.position - transform.position;
            d.y = 0f;
            if (d.magnitude > heatRange) continue;
            var c = grid.CellOf(fire.position);
            int r = Mathf.CeilToInt(keepOffFire / SearchGrid.Cell);
            for (int x = c.x - r; x <= c.x + r; x++)
                for (int z = c.y - r; z <= c.y + r; z++)
                {
                    var w = grid.World(Floor, x, z) - fire.position;
                    w.y = 0f;
                    if (w.magnitude <= keepOffFire && heat.Add(new Vector3Int(x, Floor, z))) added = true;
                    if (w.magnitude <= escapeKeepOff) heatCore.Add(new Vector3Int(x, Floor, z));
                }
        }
        if (added && (Status == State.Searching || Status == State.ToStairs) && !RouteFree()) replan = true;
    }

    /// Records contact between the drone's body and any real collider (separate from perception).
    void CheckCollision()
    {
        var hits = Physics.OverlapBox(transform.position, half, transform.rotation, mask, QueryTriggerInteraction.Ignore);
        bool now = hits.Length > 0;
        if (now && !touching)
        {
            Collisions++;
            LastHit = hits[0].name;
            Debug.LogWarning($"[Collision] drone {id + 1} touched '{LastHit}' at {transform.position}");
        }
        touching = now;
    }

    /// Minimal binary heap of cells by cost.
    class Heap
    {
        readonly List<(Vector2Int c, float d)> items = new();
        public int Count => items.Count;

        public void Push(Vector2Int c, float d)
        {
            items.Add((c, d));
            int i = items.Count - 1;
            while (i > 0)
            {
                int p = (i - 1) / 2;
                if (items[p].d <= items[i].d) break;
                (items[p], items[i]) = (items[i], items[p]);
                i = p;
            }
        }

        public (Vector2Int, float) Pop()
        {
            var top = items[0];
            items[0] = items[^1];
            items.RemoveAt(items.Count - 1);
            int i = 0;
            while (true)
            {
                int l = 2 * i + 1, r = l + 1, m = i;
                if (l < items.Count && items[l].d < items[m].d) m = l;
                if (r < items.Count && items[r].d < items[m].d) m = r;
                if (m == i) break;
                (items[m], items[i]) = (items[i], items[m]);
                i = m;
            }
            return top;
        }
    }
}
