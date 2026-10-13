using System.Collections.Generic; // import needed datatypes 
using UnityEngine; // import unity data types.

/// Pure geometry for the area scan (no scene access, so the editor tests can check it directly).
/// Coordinates are world meters on the ground plane: Vector2(x, z). Grid cells are 1 m, centered on integers.
///
/// Region: four clicked corners. Ordering rule: counter-clockwise around their centroid, starting from the corner
/// with the smallest x + z. The quad must be convex (rectangles, trapezoids, any convex four-sided shape).
///
/// Route: a lawnmower. Lanes run along the region's longer bounding-box axis. Lane spacing and capture spacing come
/// from the camera's ground footprint at the scan altitude, minus a configurable overlap. Each lane's end points
/// come from the actual polygon (so trapezoid lanes have different lengths), using the polygon's extent over the
/// whole strip the lane's footprint covers, not just its center line.
public static class ScanPlanner
{
    [System.Serializable]
    public class Config
    {
        public float altitude = 2.5f;         // camera height above the ground (m); the AI model was trained on 2.3-5 m views
        public float overlap = 0.15f;         // fraction of the footprint shared by neighbouring images / lanes
        public float verticalFovDeg = 60f;    // camera vertical field of view
        public float aspect = 1f;             // image width / height this means hieght/width = 1, meaning a square image 
        public float edgeMargin = 0.25f;      // planning treats each image as this much smaller on every side (position error)
        public float minEdge = 3f;            // shortest allowed region edge (m)
        public float minArea = 25f;           // m²
        public float maxArea = 40000f;        // m² (keeps scan time bounded)
    }

    public class Lane
    {
        public Vector2 a, b;                  // fly from a to b , a is start position of lane and b is end position of lane.
        public List<Vector2> captures = new List<Vector2>();   // planned capture positions, a → b
        public Vector2 Dir => (b - a).sqrMagnitude > 1e-8f ? (b - a).normalized : Vector2.right; // (b - a).normalized  checks if lane has meaningful length if not it returns a vector of length one.
        public float Length => Vector2.Distance(a, b); //calculates distance between points. Remember this in in xy-plane for now no z yet because this is just planned movement.
    }

    public class Plan
    {
        public Vector2[] corners;             // four corners of selcted region each point has (x,y)
        public Vector2 footprint;             // ground size of one image: x = world X extent, y = world Z extent. Size of camera usable ground before subtracting object margins.
        public float laneSpacing, captureSpacing; //laneSpacing is distance between adj. lanes and captureSpacing is maximum distance between captures on the same lane.
        public bool lanesAlongX; // whether or not the lane travels mainly across x-axis and no y -axis.
        public List<Lane> lanes = new List<Lane>(); // list of all planned flight lanes.
        public int CaptureCount { get { int n = 0; foreach (var l in lanes) n += l.captures.Count; return n; } }
        //gets number of captures across lanes by adding up number of captures on each lane. value is returned.
    }

    /// Ground footprint of a straight-down camera at height h: (world X extent, world Z extent).
    /// The image's top is the drone's +Z (see DroneCamera), so the vertical FOV spans world Z.
    public static Vector2 Footprint(float altitude, float verticalFovDeg, float aspect) 
    {
        // altitude is the distance between the drone and the ground.
        //vertical view of field in degrees .
        //aspect is image width divided by image height.
        
        float tv = Mathf.Tan(verticalFovDeg * 0.5f * Mathf.Deg2Rad);
        //divide the degrees field view by 2 to get the angle of a right triangle and convert that into radians.
        //calculate tangent of that angle which is basically calculating the slope.
        float zExtent = 2f * altitude * tv;
        // z extent is calculated by doing tan = opp/adj, so opp is hald of z-extent  and divide that by height rearrange formula to get 
        //z-extent by itself z_extent = 2 * h * tan(theta)
        float xExtent = 2f * altitude * tv * aspect;   // tan(hfov/2) = aspect * tan(vfov/2)
        // calculates x_extent = 2 * z_extent 
        return new Vector2(xExtent, zExtent);
        //returns values of x_extent and z_extent.
    }

    /// Orders four corners by the rule above and checks them. Returns null (and a reason) if the region is invalid.
    public static Vector2[] OrderAndValidate(IList<Vector2> pts, Config cfg, out string error)
    {
        error = null;
        if (pts == null || pts.Count != 4) { error = "Select exactly four corners."; return null; }
        for (int i = 0; i < 4; i++)
            for (int j = i + 1; j < 4; j++)
                if (Vector2.Distance(pts[i], pts[j]) < 0.5f) { error = "Two corners are on the same cell."; return null; }

        var c = (pts[0] + pts[1] + pts[2] + pts[3]) * 0.25f;
        var list = new List<Vector2>(pts);
        list.Sort((p, q) => Mathf.Atan2(p.y - c.y, p.x - c.x).CompareTo(Mathf.Atan2(q.y - c.y, q.x - c.x)));
        int first = 0;
        for (int i = 1; i < 4; i++) if (list[i].x + list[i].y < list[first].x + list[first].y) first = i;
        var o = new Vector2[4];
        for (int i = 0; i < 4; i++) o[i] = list[(first + i) % 4];

        for (int i = 0; i < 4; i++)
        {
            var e0 = o[(i + 1) % 4] - o[i];
            var e1 = o[(i + 2) % 4] - o[(i + 1) % 4];
            if (Cross(e0, e1) <= 1e-3f) { error = "The four corners must form a convex shape (no three in a line, no dents)."; return null; }
            if (e0.magnitude < cfg.minEdge) { error = $"Each side must be at least {cfg.minEdge:0} m long."; return null; }
        }
        float area = Area(o);
        if (area < cfg.minArea) { error = $"The region is too small (min {cfg.minArea:0} m²)."; return null; }
        if (area > cfg.maxArea) { error = $"The region is too large ({area:0} m², max {cfg.maxArea:0} m²)."; return null; }
        return o;
    }

    public static Plan Build(Vector2[] corners, Config cfg)
    {
        var plan = new Plan { corners = corners, footprint = Footprint(cfg.altitude, cfg.verticalFovDeg, cfg.aspect) };
        float minX = float.MaxValue, maxX = float.MinValue, minZ = float.MaxValue, maxZ = float.MinValue;
        foreach (var p in corners) { minX = Mathf.Min(minX, p.x); maxX = Mathf.Max(maxX, p.x); minZ = Mathf.Min(minZ, p.y); maxZ = Mathf.Max(maxZ, p.y); }
        plan.lanesAlongX = maxX - minX >= maxZ - minZ;

        // Work in (u along the lane, v across lanes).
        bool ax = plan.lanesAlongX;
        var usable = plan.footprint - Vector2.one * (2f * cfg.edgeMargin);
        float across = ax ? usable.y : usable.x;    // footprint width across lanes
        float along = ax ? usable.x : usable.y;     // footprint length along a lane
        float vMin = ax ? minZ : minX, vMax = ax ? maxZ : maxX;
        var uv = new Vector2[4];
        for (int i = 0; i < 4; i++) uv[i] = ax ? corners[i] : new Vector2(corners[i].y, corners[i].x);

        float maxLane = across * (1f - cfg.overlap), maxStep = along * (1f - cfg.overlap);
        float span = vMax - vMin;
        int nLanes = span <= across ? 1 : Mathf.CeilToInt((span - across) / maxLane - 1e-4f) + 1;
        plan.laneSpacing = nLanes > 1 ? (span - across) / (nLanes - 1) : 0f;
        plan.captureSpacing = maxStep;

        for (int k = 0; k < nLanes; k++)
        {
            float v = nLanes == 1 ? (vMin + vMax) * 0.5f : vMin + across * 0.5f + k * plan.laneSpacing;
            // Polygon extent over the strip this lane's images cover (clamped to the region's v range).
            float v0 = Mathf.Max(vMin, v - across * 0.5f), v1 = Mathf.Min(vMax, v + across * 0.5f);
            if (!StripRange(uv, v0, v1, out float uMin, out float uMax)) continue;
            float u0 = uMin + along * 0.5f, u1 = uMax - along * 0.5f;
            if (u1 < u0) u0 = u1 = (uMin + uMax) * 0.5f;
            int n = u1 - u0 < 1e-3f ? 1 : Mathf.CeilToInt((u1 - u0) / maxStep - 1e-4f) + 1;

            var lane = new Lane();
            bool reverse = k % 2 == 1;
            for (int i = 0; i < n; i++)
            {
                float t = n == 1 ? 0f : i / (float)(n - 1);
                float u = reverse ? Mathf.Lerp(u1, u0, t) : Mathf.Lerp(u0, u1, t);
                lane.captures.Add(ax ? new Vector2(u, v) : new Vector2(v, u));
            }
            lane.a = lane.captures[0];
            lane.b = lane.captures[lane.captures.Count - 1];
            plan.lanes.Add(lane);
        }
        return plan;
    }

    /// u-range of a convex polygon (in u,v coordinates) between v0 and v1: from its vertices inside the strip and
    /// its edges' crossings of the strip's two boundary lines.
    static bool StripRange(Vector2[] poly, float v0, float v1, out float uMin, out float uMax)
    {
        uMin = float.MaxValue; uMax = float.MinValue;
        int n = poly.Length;
        for (int i = 0; i < n; i++)
        {
            var p = poly[i]; var q = poly[(i + 1) % n];
            if (p.y >= v0 - 1e-4f && p.y <= v1 + 1e-4f) { uMin = Mathf.Min(uMin, p.x); uMax = Mathf.Max(uMax, p.x); }
            foreach (float v in new[] { v0, v1 })
            {
                if ((p.y - v) * (q.y - v) > 0f || Mathf.Abs(q.y - p.y) < 1e-6f) continue;
                float t = (v - p.y) / (q.y - p.y);
                float u = Mathf.Lerp(p.x, q.x, t);
                uMin = Mathf.Min(uMin, u); uMax = Mathf.Max(uMax, u);
            }
        }
        return uMin <= uMax;
    }

    public static bool Contains(Vector2[] poly, Vector2 p)
    {
        // Convex and counter-clockwise: inside (or on the edge) when left of every edge.
        for (int i = 0; i < poly.Length; i++)
            if (Cross(poly[(i + 1) % poly.Length] - poly[i], p - poly[i]) < -1e-4f) return false;
        return true;
    }

    public static float Area(Vector2[] poly)
    {
        float s = 0f;
        for (int i = 0; i < poly.Length; i++) s += Cross(poly[i], poly[(i + 1) % poly.Length]);
        return Mathf.Abs(s) * 0.5f;
    }

    /// The 1 m grid cells (integer x, z) whose centers lie inside the region.
    public static List<Vector2Int> Cells(Vector2[] poly)
    {
        var cells = new List<Vector2Int>();
        float minX = float.MaxValue, maxX = float.MinValue, minZ = float.MaxValue, maxZ = float.MinValue;
        foreach (var p in poly) { minX = Mathf.Min(minX, p.x); maxX = Mathf.Max(maxX, p.x); minZ = Mathf.Min(minZ, p.y); maxZ = Mathf.Max(maxZ, p.y); }
        for (int x = Mathf.CeilToInt(minX - 1e-4f); x <= Mathf.FloorToInt(maxX + 1e-4f); x++)
            for (int z = Mathf.CeilToInt(minZ - 1e-4f); z <= Mathf.FloorToInt(maxZ + 1e-4f); z++)
                if (Contains(poly, new Vector2(x, z))) cells.Add(new Vector2Int(x, z));
        return cells;
    }

    /// True if a ground point lies inside the footprint of an image taken straight down from `cam` (x, z).
    public static bool InFootprint(Vector2 cam, Vector2 footprint, Vector2 p) =>
        Mathf.Abs(p.x - cam.x) <= footprint.x * 0.5f && Mathf.Abs(p.y - cam.y) <= footprint.y * 0.5f;

    static float Cross(Vector2 a, Vector2 b) => a.x * b.y - a.y * b.x;
}
