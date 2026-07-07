// Copyright (c) 2025 Yize Wu
// SPDX-License-Identifier: MIT

using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace Gsplat
{
    [ExecuteAlways]
    public class GsplatRenderer : MonoBehaviour, IGsplat
    {
        public enum GsplatSortMode
        {
            Always,
            SortEveryNFrames,
            CutoutsEveryNSorts,
        }

        public GsplatAsset GsplatAsset;

        // Range is enforced by GsplatRendererEditor based on the bound asset's SHBands.
        public int SHDegree = 3;
        [HideInInspector] public uint RenderOrder = 0;
        public float Brightness = 1.0f;

        [Tooltip(
            "Improves rendering speed by shrinking Gaussian splats while trying to keep the impact on visual quality as small as possible.")]
        [Range(0, 1)]
        public float SplatDownscaleFactor = 0.0f;

        public bool GammaToLinear;
        public bool AsyncUpload;
        public bool RenderBeforeUploadComplete = true;

        [Tooltip("Does cutouts update the Gsplat world bounds? (Costly on moving cutouts)")]
        public bool CutoutsUpdateBounds = true;

        [Header("Frustum Culling")]
        [Tooltip("Skip splats outside the camera frustum in the InitOrder pass, reducing the " +
                 "sorted/drawn count (tris & verts). Recomputed on camera movement.")]
        public bool FrustumCulling = false;

        [Tooltip("Camera to cull against. Falls back to Camera.main when empty.")]
        public Camera CullCamera;

        [Tooltip("Object-space padding (~max splat radius) added to every frustum plane so " +
                 "gaussians straddling an edge are not popped.")]
        [Min(0f)]
        public float FrustumCullMargin = 0.5f;

        [Header("Chunked LOD (experimental)")]
        [Tooltip("Draw a per-chunk selected LOD from a combined chunked asset (baked by " +
                 "tools/bake_chunks.py). Requires the combined .spz as GsplatAsset and its " +
                 ".chunks.json sidecar below. Overrides the plain frustum path.")]
        public bool ChunkedLod = false;

        [Tooltip("The .chunks.json sidecar (TextAsset) matching the combined GsplatAsset.")]
        public TextAsset ChunkTable;

        [Tooltip("Pick each chunk's LOD by FOV-compensated distance to its AABB (bands " +
                 "base*mult^i). Off = draw the fixed level below for every chunk (debug).")]
        public bool ChunkedDistanceLod = true;

        [Tooltip("Distance (world units) within which a chunk draws its finest level (LOD0). " +
                 "WARNING: setting this ≥ the scene size collapses everything to LOD0 (distance-" +
                 "LOD disabled). When Chunked Lod Auto Range is on it is CLAMPED to a safe near-" +
                 "shell so a too-large value can't defeat the ladder.")]
        [Min(0.01f)]
        public float ChunkedLodBaseDistance = 5f;

        [Tooltip("Each successive LOD band is this many times farther (PlayCanvas default 3). " +
                 "IGNORED when Chunked Lod Auto Range is on.")]
        [Min(1.01f)]
        public float ChunkedLodMultiplier = 3f;

        [Tooltip("Auto-fit the LOD ladder to the scene: derive the multiplier so the coarsest " +
                 "level lands at the scene diagonal, mapping ALL levels onto the actual view " +
                 "distances as a near→far gradient. A fixed multiplier stretches the ladder past " +
                 "a compact scene (only LOD0-2 reachable). On by default; overrides " +
                 "Chunked Lod Multiplier. Live value shown in the debug panel.")]
        public bool ChunkedLodAutoRange = true;

        // Base + multiplier actually used last frame (auto-derived/clamped or manual) — debug panel.
        [System.NonSerialized] public float m_lastEffectiveMultiplier = 3f;
        [System.NonSerialized] public float m_lastEffectiveBase = 5f;

        [Tooltip("Fixed global LOD level when distance LOD is off (0 = finest).")]
        [Min(0)]
        public int ChunkedFixedLevel = 0;

        [Tooltip("Cull chunks whose bounding sphere is outside the cull camera frustum.")]
        public bool ChunkedCull = true;

        [Tooltip("Also cull individual off-screen splats of partly-visible (straddling) chunks, " +
                 "not just whole chunks — draws exactly what the camera sees, trimming the tris of " +
                 "chunks that are half in view. Keeps every splat whose footprint touches the " +
                 "frustum (no holes). Costs one per-splat frustum test in the InitOrder pass.")]
        public bool ChunkedPerSplatCull = true;

        [Tooltip("Scales the per-chunk cull radius (the baked exact per-level footprint). " +
                 "1 = exact visible 2σ core (tight, culls off-screen background to save tris). " +
                 "Lower (0.7–0.9) = tighter/fewer drawn if you can accept faint edge clipping; " +
                 "higher (1.1–1.5) = safer/more drawn if you see holes. Only affects chunk culling.")]
        [Range(0.3f, 2f)]
        public float ChunkedCullFootprintScale = 1.3f;

        [Tooltip("R4 streaming: keep only a budget of splats GPU-resident. Each refresh, the " +
                 "selected per-chunk LODs are compacted into a pool from the combined asset's " +
                 "CPU RAM — GPU holds ~budget instead of the whole 2x-LOD combined buffer.")]
        public bool ChunkedStreaming = false;

        [Tooltip("Streaming pool capacity in splats (GPU-resident budget).")]
        [Min(1000)]
        public int ChunkedPoolBudget = 2_000_000;

        [Tooltip("R3 budget balancer: fit the per-chunk LOD selection into ChunkedPoolBudget by " +
                 "degrading the farthest chunks first (and upgrading the nearest when there is " +
                 "headroom), so the drawn splat total is bounded for stable frame time. " +
                 "Streaming-pool only.")]
        public bool ChunkedBudgetBalancer = true;

        [Tooltip("Damp LOD-band and cull-edge flicker: keep a chunk's previous LOD while the " +
                 "camera distance stays within its band widened by ±Chunked Hysteresis, and keep " +
                 "a visible chunk visible until its bounding sphere is clearly outside the frustum " +
                 "(and vice-versa). Costs one extra per-chunk comparison; off by default.")]
        public bool ChunkedLodHysteresis = false;

        [Tooltip("Hysteresis dead-band as a fraction (0.15 = 15%). Widens each LOD band and the " +
                 "cull boundary by this fraction so small camera jitter near a boundary doesn't " +
                 "flip the decision. Only used when Chunked Lod Hysteresis is on.")]
        [Range(0f, 0.5f)]
        public float ChunkedHysteresis = 0.15f;

        [Tooltip("Combined-path drawn-splat budget (PlayCanvas splatBudget). The distance bands " +
                 "alone only reach a few LODs in a compact scene; this degrades the farthest " +
                 "chunks toward the coarsest LOD until the drawn total fits, so the FULL LOD " +
                 "ladder is used and far regions coarsen aggressively (reach LOD max much closer). " +
                 "0 = disabled (pure distance LOD). Ignored in streaming-pool mode " +
                 "(ChunkedPoolBudget governs there).")]
        [Min(0)]
        public int ChunkedSplatBudget = 0;

        [Tooltip("Draw each chunk's AABB in the Scene view, colored by its selected LOD " +
                 "(green=fine .. red=coarse, gray=culled). Play mode only.")]
        public bool ChunkedDebugGizmos = false;

        [Tooltip("Draw every chunk's AABB as a yellow wireframe cube in the Scene view. " +
                 "Works in edit mode too (parses the chunk table directly).")]
        public bool ChunkedShowChunkBounds = false;

        [SerializeField, HideInInspector] ComputeShader InitOrderChunkedShader;
        GsplatChunkTable m_chunkTableParsed;
        TextAsset m_chunkTableSource;
        GsplatChunkTable m_gizmoTable;      // editor-side parse so chunk gizmos work without Play
        TextAsset m_gizmoTableSource;

        // Editor-debug accessors (valid in Play mode when ChunkedLod is active).
        public GsplatChunkTable ChunkedTableRuntime => m_chunkTableParsed;
        public uint[] ChunkedSelectedLevels => m_renderer?.SelectedLevels;
        // Splats dropped past the pool capacity on the last Fill (0 = the balancer fit budget).
        public uint ChunkedPoolOverflow => m_renderer?.m_poolOverflow ?? 0;

        GsplatAsset m_prevAsset;
        GsplatRendererImpl m_renderer;

        // Streaming pool mode: budget-resident, populated per frame — only in Play with a
        // chunk table (needs the asset's CPU arrays; the pool has nothing until the first Fill).
        bool PoolMode => ChunkedLod && ChunkedStreaming && ChunkTable != null && Application.isPlaying;

        public bool Valid => GsplatAsset &&
                             (PoolMode ? m_renderer?.GsplatResource != null
                                 : (RenderBeforeUploadComplete ? SplatCount > 0 : SplatCount == GsplatAsset.SplatCount));

        public uint SplatCount => m_renderer != null ? m_renderer.GsplatResource?.UploadedCount ?? 0 : 0;

        public ISorterResource SorterResource => m_renderer.SorterResource;

        // IGsplat global-merge members: expose per-renderer GPU buffers for the global sorter.
        public GsplatResource GsplatResource => m_renderer?.GsplatResource;
        public byte SHBands => GsplatAsset?.SHBands ?? 0;


        public uint RemainingCount
        {
            get => m_renderer.m_remainingCount;
            set => m_renderer.m_remainingCount = value;
        }

        public Bounds Bounds
        {
            get => m_renderer.m_bounds;
            set => m_renderer.m_bounds = value;
        }

        public GsplatCutout[] Cutouts
        {
            get
            {
                var cutouts = GsplatCutout.m_RegisteredCutouts
                    .Where(component => component.enabled)
                    .Where(component =>
                        component.m_Target == GsplatCutout.Target.All ||
                        (component.m_Target == GsplatCutout.Target.Parent && component.transform.parent == transform) ||
                        (component.m_Target == GsplatCutout.Target.Specific && component.m_SpecifcRenderer == this)
                    );
                return cutouts.ToArray();
            }
        }

        public bool ComputeSortRequired => m_renderer.ComputeSortRequired;
        public bool ComputeCutoutsRequired => m_renderer.ComputeCutoutsRequired;
        public GsplatSortMode SortMode = GsplatSortMode.Always;
        [HideInInspector] public uint SortRefreshRate = 1;
        [HideInInspector] public uint CutoutsRefreshRate = 1;

        public void ComputeDepth(CommandBuffer cmd, Matrix4x4 matrixMv) => m_renderer.ComputeDepth(cmd, matrixMv);

        void OnEnable()
        {
            GsplatSorter.Instance.RegisterGsplat(this);
            m_prevAsset = null;
        }

        void OnDisable()
        {
            GsplatSorter.Instance.UnregisterGsplat(this);
            m_renderer?.Dispose();
            m_renderer = null;
        }

        public void ForceRefresh()
        {
            m_renderer?.ForceRefresh();
        }

        // Green (finest) -> red (coarsest) ramp for a LOD level, shared by gizmos + inspector.
        public static Color LodColor(int level, int maxLod)
        {
            float t = maxLod <= 0 ? 0f : Mathf.Clamp01((float)level / maxLod);
            return Color.Lerp(new Color(0.2f, 0.85f, 0.3f), new Color(0.9f, 0.25f, 0.2f), t);
        }

#if UNITY_EDITOR
        public void OnDrawGizmos()
        {
            if (GsplatSettings.Instance.DisplayBoundingBoxes && Valid && isActiveAndEnabled)
            {
                Gizmos.matrix = transform.localToWorldMatrix;
                Gizmos.color = Color.green;
                Gizmos.DrawWireCube(Bounds.center, Bounds.size);
            }

            // All chunk AABBs as yellow wireframe cubes — works in edit mode (parses the
            // chunk table directly, no Play needed).
            if (ChunkedShowChunkBounds && ChunkTable != null)
            {
                if (m_gizmoTable == null || m_gizmoTableSource != ChunkTable)
                {
                    try { m_gizmoTable = GsplatChunkTable.Parse(ChunkTable.text); }
                    catch { m_gizmoTable = null; }
                    m_gizmoTableSource = ChunkTable;
                }
                if (m_gizmoTable != null)
                {
                    Gizmos.matrix = transform.localToWorldMatrix;
                    Gizmos.color = Color.yellow;
                    for (int c = 0; c < m_gizmoTable.ChunkCount; c++)
                    {
                        var a = m_gizmoTable.Chunks[c].Aabb;
                        Gizmos.DrawWireCube(a.center, a.size);
                    }
                }
            }

            if (ChunkedDebugGizmos && ChunkedLod && Application.isPlaying)
            {
                var table = ChunkedTableRuntime;
                var sel = ChunkedSelectedLevels;
                if (table != null && sel != null)
                {
                    Gizmos.matrix = transform.localToWorldMatrix;
                    for (int c = 0; c < table.ChunkCount && c < sel.Length; c++)
                    {
                        uint lvl = sel[c];
                        Gizmos.color = lvl == 0xFFFFFFFFu
                            ? new Color(0.4f, 0.4f, 0.4f, 0.4f)
                            : LodColor((int)lvl, table.MaxLod);
                        var a = table.Chunks[c].Aabb;
                        Gizmos.DrawWireCube(a.center, a.size);
                    }
                }
            }
        }

        [SerializeField, HideInInspector] string m_assetGuid;
        public string AssetGuid => m_assetGuid;
#endif // #if UNITY_EDITOR

        void OnValidate()
        {
            ForceRefresh();
#if UNITY_EDITOR
            if (GsplatAsset &&
                AssetDatabase.TryGetGUIDAndLocalFileIdentifier(GsplatAsset, out var guid, out long localId))
                m_assetGuid = guid;
            if (InitOrderChunkedShader == null)
                InitOrderChunkedShader = AssetDatabase.LoadAssetAtPath<ComputeShader>(
                    "Packages/wu.yize.gsplat/Runtime/Shaders/InitOrderChunked.compute");
#endif // #if UNITY_EDITOR
        }

        public void ReloadAsset()
        {
            m_prevAsset = null;
        }

        public void Update()
        {
            if (!GsplatAsset)
                m_prevAsset = null;
            if (m_prevAsset != GsplatAsset)
            {
                m_renderer?.ReleaseGsplatAsset();
                m_prevAsset = GsplatAsset;
                if (GsplatAsset)
                {
                    // In streaming pool mode the GPU buffers are sized to the budget, not the
                    // full (2x-LOD) combined asset.
                    uint cap = PoolMode ? (uint)ChunkedPoolBudget : GsplatAsset.SplatCount;
                    if (m_renderer == null)
                        m_renderer = new GsplatRendererImpl(cap);
                    else
                        m_renderer.RecreateResources(cap);
#if UNITY_EDITOR
                    var asyncUpload = AsyncUpload && Application.isPlaying;
#else
                    var asyncUpload = AsyncUpload;
#endif
                    m_renderer.BindGsplatAsset(GsplatAsset, asyncUpload, PoolMode);
                    GsplatSorter.Instance.MarkGlobalBuffersDirty();
                }
            }

            if (Valid && GsplatSettings.Instance.Valid && GsplatSorter.Instance.Valid)
            {
                m_renderer.EvaluateRefreshRequired(SortMode, SortRefreshRate - 1, CutoutsRefreshRate - 1);
                // Only cull at runtime: in edit mode Camera.main would cull the Scene view too,
                // making splats vanish while you look through it.
                var runtimeCam = Application.isPlaying ? (CullCamera != null ? CullCamera : Camera.main) : null;

                if (ChunkedLod && ChunkTable != null && Application.isPlaying)
                {
                    if (m_chunkTableParsed == null || m_chunkTableSource != ChunkTable)
                    {
                        m_chunkTableParsed = GsplatChunkTable.Parse(ChunkTable.text);
                        m_chunkTableSource = ChunkTable;
                    }
                    // Scene-adaptive LOD range: a fixed multiplier (e.g. 3) spreads the ladder far
                    // beyond a compact scene so only LOD0-2 are ever reachable by distance. When
                    // ChunkedLodAutoRange is on, derive BOTH the LOD0 near-shell (base) and the
                    // step (multiplier) so the full ladder maps onto the scene as a near→far
                    // gradient. Overrides the manual ChunkedLodMultiplier and CLAMPS the base.
                    //
                    // Why the base MUST be clamped: the band test is `d < base ? LOD0 : …`, so a
                    // base ≥ the scene diagonal collapses the WHOLE scene into the LOD0 shell —
                    // distance-LOD is silently defeated and only the budget balancer (far-first)
                    // coarsens anything, which reads as "inverted" (far looks high quality). We cap
                    // the base at diag/1.3^(maxLod-1) so the ladder always spans the scene with at
                    // least a 1.3× step, even if the authored base is pathologically large.
                    float effBase = ChunkedLodBaseDistance;
                    float effMult = ChunkedLodMultiplier;
                    if (ChunkedLodAutoRange && m_chunkTableParsed.MaxLod > 1)
                    {
                        int maxLod = m_chunkTableParsed.MaxLod;
                        float diag = m_chunkTableParsed.Bounds.size.magnitude;
                        float baseCap = diag / Mathf.Pow(1.3f, maxLod - 1);          // LOD0-shell ceiling
                        effBase = Mathf.Clamp(ChunkedLodBaseDistance, 0.01f, baseCap);
                        effMult = Mathf.Max(1.2f,
                            Mathf.Pow(Mathf.Max(1.001f, diag / effBase), 1f / (maxLod - 1)));
                    }
                    m_lastEffectiveBase = effBase;
                    m_lastEffectiveMultiplier = effMult;
                    if (PoolMode)
                        m_renderer.DispatchChunkedPool(m_chunkTableParsed, transform.localToWorldMatrix,
                            runtimeCam, ChunkedDistanceLod, ChunkedFixedLevel,
                            effBase, effMult, ChunkedCull,
                            ChunkedBudgetBalancer, FrustumCullMargin,
                            ChunkedLodHysteresis, ChunkedHysteresis, ChunkedCullFootprintScale);
                    else if (InitOrderChunkedShader != null)
                        m_renderer.DispatchInitOrderChunked(m_chunkTableParsed, InitOrderChunkedShader,
                            transform.localToWorldMatrix, runtimeCam, ChunkedDistanceLod, ChunkedFixedLevel,
                            effBase, effMult, ChunkedCull, FrustumCullMargin,
                            ChunkedSplatBudget, ChunkedLodHysteresis, ChunkedHysteresis, ChunkedCullFootprintScale,
                            ChunkedPerSplatCull);
                }
                else
                {
                    var cullCamera = FrustumCulling ? runtimeCam : null;
                    m_renderer.DispatchInitOrder(Cutouts, transform.localToWorldMatrix, CutoutsUpdateBounds,
                        cullCamera, FrustumCullMargin);
                }
                // When the global sorter has merged all renderers into a single draw call,
                // skip the per-renderer draw — GsplatSorter.DrawAll handles rendering.
                if (!GsplatSorter.Instance.GlobalRenderEnabled)
                    m_renderer.Render(transform, gameObject.layer, GammaToLinear, SHDegree, Brightness,
                        1.0f - SplatDownscaleFactor, RenderOrder);
            }
        }
    }
}