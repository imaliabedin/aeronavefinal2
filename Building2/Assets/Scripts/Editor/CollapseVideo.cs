using System.IO;
using Drone2.Building;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering.Universal;

/// Renders the Rescue Mission's failure ending (time runs out: the fire engulfs the building, then BuildingCollapse
/// brings it down) from a camera circling outside, as numbered JPEG frames for the tutorial video (rescue.html).
/// Uses the mission's own FireSpread and BuildingCollapse, stepped at the frame rate.
///   COLLAPSE_OUT=dir Unity -batchmode -quit -projectPath Building2 -executeMethod CollapseVideo.Record
public static class CollapseVideo
{
    public static void Record()
    {
        int code = 0;
        try { Run(); }
        catch (System.Exception e) { Debug.LogException(e); code = 1; }
        EditorApplication.Exit(code);
    }

    static void Run()
    {
        string outDir = System.Environment.GetEnvironmentVariable("COLLAPSE_OUT") ?? "CollapseFrames";
        int seed = int.Parse(System.Environment.GetEnvironmentVariable("COLLAPSE_SEED") ?? "101");
        Directory.CreateDirectory(outDir);
        EditorSceneManager.OpenScene("Assets/Scenes/DroneSearchDemo.unity");
        var gen = Object.FindFirstObjectByType<BuildingGenerator>();
        var coord = Object.FindFirstObjectByType<SearchCoordinator>();
        gen.seed = seed;
        gen.Generate();
        coord.Setup(gen);

        // About 15 minutes of fire spread, as in a mission that runs out of time.
        var fire = new FireSpread(gen, coord, new FireSpread.Settings { interval = 4f }, gen.seed + 7);
        for (float t = 0f; t < 900f; t += 0.5f) fire.Step(0.5f);
        SimulateParticles(3f, true);

        var cam = new GameObject("Video Camera").AddComponent<Camera>();
        cam.fieldOfView = 50f;
        cam.nearClipPlane = 0.1f;
        cam.farClipPlane = 500f;
        cam.GetUniversalAdditionalCameraData().renderPostProcessing = true;
        const int W = 1280, H = 720;
        var rt = new RenderTexture(W, H, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        cam.targetTexture = rt;
        var tex = new Texture2D(W, H, TextureFormat.RGB24, false);

        float height = gen.floors * gen.floorHeight;
        var center = gen.Root.TransformPoint(new Vector3(gen.width / 2f, height * 0.4f, gen.depth / 2f));
        float span = Mathf.Max(gen.width, gen.depth);
        const float fps = 30f, before = 2.5f, total = 12f;
        BuildingCollapse collapse = null;
        var rng = new System.Random(5);
        float shake = 0f;
        int burst = 0;
        int frames = Mathf.RoundToInt(total * fps);
        for (int i = 0; i < frames; i++)
        {
            float t = i / fps, dt = 1f / fps;
            if (collapse == null && t >= before) { collapse = new BuildingCollapse(gen, fire, coord.drones, gen.seed + 11); fire.Engulf(160); }
            fire.Step(dt);
            collapse?.Step(dt);
            // Each storey hitting the one below: a burst of fire and dust around the building, and the camera shakes.
            float fall = collapse != null ? collapse.Seconds - BuildingCollapse.EngulfSeconds : -1f;
            if (fall > 0.35f + burst * 0.3f && burst < gen.floors + 1)
            {
                burst++;
                shake = 1f;
                for (int k = 0; k < 10; k++)
                {
                    float a2 = (float)rng.NextDouble() * Mathf.PI * 2f;
                    var local = new Vector3(gen.width / 2f + Mathf.Cos(a2) * gen.width * 0.5f, 0f, gen.depth / 2f + Mathf.Sin(a2) * gen.depth * 0.5f);
                    fire.Spawn(gen.Root.TransformPoint(local), 0, 2.6f);
                }
            }
            shake = Mathf.Max(0f, shake - dt * 1.6f);
            SimulateParticles(dt, false);

            // Low, close, slowly pushing in and circling: the building fills the frame and towers over the camera.
            float push = Mathf.SmoothStep(0f, 1f, t / total);
            float radius = span * Mathf.Lerp(1.35f, 1.05f, push);
            float a = (-28f + t * 2.2f) * Mathf.Deg2Rad;
            var jitter = shake * shake * 0.35f * new Vector3((float)rng.NextDouble() - 0.5f, (float)rng.NextDouble() - 0.5f, (float)rng.NextDouble() - 0.5f);
            cam.transform.position = center + new Vector3(Mathf.Sin(a) * radius, height * Mathf.Lerp(0.2f, 0.05f, push) + 1.5f, -Mathf.Cos(a) * radius) + jitter;
            float settle = Mathf.Clamp01((t - before - BuildingCollapse.EngulfSeconds) / 2.5f);
            cam.transform.LookAt(center + Vector3.up * Mathf.Lerp(height * 0.25f, -height * 0.25f, settle) + jitter * 0.5f);
            cam.Render();
            RenderTexture.active = rt;
            tex.ReadPixels(new Rect(0, 0, W, H), 0, 0);
            tex.Apply();
            RenderTexture.active = null;
            File.WriteAllBytes(Path.Combine(outDir, $"frame_{i:0000}.jpg"), tex.EncodeToJPG(92));
        }
        cam.targetTexture = null;
        Object.DestroyImmediate(rt);
        Object.DestroyImmediate(tex);
        Debug.Log($"[CollapseVideo] {frames} frames -> {outDir} (seed {seed}, fire {fire.Share * 100f:0}%)");
    }

    static void SimulateParticles(float dt, bool restart)
    {
        foreach (var ps in Object.FindObjectsByType<ParticleSystem>(FindObjectsSortMode.None))
            if (ps.transform.parent == null || ps.transform.parent.GetComponentInParent<ParticleSystem>() == null)
                ps.Simulate(dt, true, restart, false);
    }
}
