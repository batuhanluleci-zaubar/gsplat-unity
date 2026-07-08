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

        [Tooltip("Distance multiplier between successive LOD bands. 2.0 = SSE-correct (one level per " +
                 "distance doubling, crisp near, far tops out ~LOD4-6 in a compact scene). Lower it " +
                 "toward ~1.35 to map ALL levels onto the scene and push far corners to the coarsest " +
                 "LODs 7-11 (cheapest, but coarse levels creep into the near field = blobbier up " +
                 "close). Respected whether or not Auto Range is on; only the BASE is auto-guarded.")]
        [Min(1.01f)]
        public float ChunkedLodMultiplier = 2f;

        [Tooltip("Clamp the LOD base distance to a safe near-shell (diag/4) so a too-large authored " +
                 "base can't collapse the scene into the LOD0 shell. Does NOT touch the multiplier — " +
                 "tune Chunked Lod Multiplier for aggressiveness. Live values shown in the debug panel.")]
        public bool ChunkedLodAutoRange = true;

        [Tooltip("LOD penalty for chunks BEHIND the camera: their band distance is inflated up to " +
                 "xN when fully behind (PlayCanvas lodBehindPenalty), so what you can't see coarsens " +
                 "first and frees budget for what you can. 1 = off (the PC engine default), but every " +
                 "PlayCanvas streamed example ships 2-4 (mostly 3). Selection re-evaluates on the rotation " +
                 "refresh gate, so turning around restores full quality after one refresh. Keep at 2 " +
                 "while pool refreshes re-upload the whole pool; raise toward 3 once refills are " +
                 "incremental.")]
        [Range(1f, 5f)]
        public float ChunkedBehindPenalty = 2f;

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

        [Tooltip("Scales the per-CHUNK cull radius (the baked exact per-level footprint). Because " +
                 "Gaussian splats are semi-transparent, the tails of OFF-screen splats still add " +
                 "opacity to visible pixels — culling a whole chunk that overhangs the view thins " +
                 "the wall it feeds and lets the skybox bleed through (transparency holes). So the " +
                 "per-chunk cull must be CONSERVATIVE (2 = keep chunks up to ~2x their footprint " +
                 "from the frustum) and let the per-SPLAT cull do the precise trimming. Lower only " +
                 "if you see too many tris AND no holes; raise if walls go see-through.")]
        [Range(0.3f, 4f)]
        public float ChunkedCullFootprintScale = 2.0f;

        [Tooltip("R4 streaming: keep only a budget of splats GPU-resident. Each refresh, the " +
                 "selected per-chunk LODs are compacted into a pool from the combined asset's " +
                 "CPU RAM — GPU holds ~budget instead of the whole 2x-LOD combined buffer.")]
        public bool ChunkedStreaming = false;

        [Tooltip("Streaming pool capacity in splats (GPU-resident budget).")]
        [Min(1000)]
        public int ChunkedPoolBudget = 2_000_000;

        [Tooltip("DISK streaming: path to a .gsstream directory (baker --streaming output) holding " +
                 "per-(chunk,level) .spz blobs + a *.chunks.json manifest. When set (and ChunkedLod " +
                 "on, in Play), the renderer streams chunk-levels from DISK into the budget-sized pool " +
                 "on demand — bounding BOTH VRAM (budget) AND CPU RAM (only in-flight blobs + the " +
                 "budget live in memory; GsplatAsset is ignored). This is the 10-15M path. Absolute, " +
                 "or relative to the project (…/Assets/…). Empty = use GsplatAsset (combined/RAM pool).")]
        public string ChunkedStreamDir = "";

        [Tooltip("Incremental pool refill (PlayCanvas BlockAllocator analogue): each refresh " +
                 "uploads ONLY chunks whose selected LOD changed instead of re-uploading the " +
                 "entire pool (tens of MB of SetData every time the camera crosses the " +
                 "0.2 m/10° gate — XR head motion crosses it constantly). Falls back to a full " +
                 "contiguous repack automatically when the pool fragments and while the global " +
                 "merged sort is active (it assumes a contiguous pool). Off = legacy full " +
                 "re-pack every refresh (debug baseline).")]
        public bool ChunkedIncrementalRefill = true;

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

        [Tooltip("Anti-pop LOD cross-fade: near a LOD-band boundary, draw BOTH the chunk's level and " +
                 "the next coarser level, ramping their opacity so the discrete swap doesn't POP " +
                 "during camera motion (LODGE opacity-blending analogue). Combined path only; costs a " +
                 "second (coarser, cheaper) level only for chunks inside the fade zone. Off = today's " +
                 "hard swap.")]
        public bool ChunkedLodFade = false;

        [Tooltip("Cross-fade zone width as a fraction of each LOD band (0.25 = fade over the last 25% " +
                 "of the band before the swap). Only used when Chunked Lod Fade is on.")]
        [Range(0f, 0.5f)]
        public float ChunkedFadeWidth = 0.25f;

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
        // Full contiguous repacks of the incremental pool (fragmentation telemetry; each costs a
        // legacy-style full re-upload — frequent repacks under motion = raise ChunkedPoolBudget).
        public int ChunkedPoolRepacks => m_renderer?.PoolRepackCount ?? 0;
        // Σ selected splats after the far-first budget balancer ran this frame (0 when inactive).
        // HUD "Balanced" line passthrough (m_lastBalancedTotal is already public on the impl).
        public int ChunkedBalancedTotal => m_renderer?.m_lastBalancedTotal ?? 0;
        // Selected disk-stream chunk-levels still loading (DISK streaming only). HUD "pending" line.
        public int ChunkedStreamPending => m_renderer?.m_streamPending ?? 0;

        GsplatAsset m_prevAsset;
        bool m_wasPoolMode;
        GsplatRendererImpl m_renderer;

        // DISK-streaming mode: chunk-levels stream from .gsstream blobs on disk (RAM-bounded); the
        // serialized GsplatAsset is ignored in favor of a metadata-only runtime asset.
        bool StreamingMode => ChunkedLod && !string.IsNullOrEmpty(ChunkedStreamDir) && Application.isPlaying;

        // Streaming pool mode: budget-resident, populated per frame. RAM pool (ChunkedStreaming +
        // ChunkTable) copies from the asset's CPU arrays; DISK streaming (StreamingMode) loads blobs.
        bool PoolMode => StreamingMode ||
                         (ChunkedLod && ChunkedStreaming && ChunkTable != null && Application.isPlaying);

        // The asset actually bound: a metadata-only streaming asset in disk-streaming mode, else the
        // serialized GsplatAsset. Built lazily from the .gsstream manifest.
        GsplatChunkTable m_streamTable;
        GsplatAssetStreaming m_streamAsset;
        string m_streamDirLoaded;

        GsplatAsset EffectiveAsset => StreamingMode ? EnsureStreamAsset() : GsplatAsset;

        GsplatAsset EnsureStreamAsset()
        {
            if (m_streamAsset != null && m_streamDirLoaded == ChunkedStreamDir) return m_streamAsset;
            try
            {
                string dir = System.IO.Path.IsPathRooted(ChunkedStreamDir)
                    ? ChunkedStreamDir
                    : System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), ChunkedStreamDir);
                var manifests = System.IO.Directory.GetFiles(dir, "*.chunks.json");
                if (manifests.Length == 0)
                    throw new System.IO.FileNotFoundException($"No *.chunks.json in stream dir '{dir}'");
                m_streamTable = GsplatChunkTable.ParseStreaming(System.IO.File.ReadAllText(manifests[0]), dir);
                m_streamAsset = GsplatAssetStreaming.FromStreamingTable(m_streamTable,
                    System.IO.Path.GetFileNameWithoutExtension(manifests[0]));
                m_streamDirLoaded = ChunkedStreamDir;
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[Gsplat] streaming manifest load failed for '{ChunkedStreamDir}': {e.Message}");
                m_streamAsset = null; m_streamTable = null;
            }
            return m_streamAsset;
        }

        public bool Valid => StreamingMode
            ? m_renderer?.GsplatResource != null
            : GsplatAsset &&
              (PoolMode ? m_renderer?.GsplatResource != null
                  : (RenderBeforeUploadComplete ? SplatCount > 0 : SplatCount == GsplatAsset.SplatCount));

        public uint SplatCount => m_renderer != null ? m_renderer.GsplatResource?.UploadedCount ?? 0 : 0;

        public ISorterResource SorterResource => m_renderer.SorterResource;

        // IGsplat global-merge members: expose per-renderer GPU buffers for the global sorter.
        public GsplatResource GsplatResource => m_renderer?.GsplatResource;
        public byte SHBands => GsplatAsset?.SHBands ?? 0;
        // 4a: content-version of this renderer's streaming pool (0 when not pooled), so the global
        // merge can refresh its cached copy on residency swaps that leave SplatCount unchanged.
        public uint PoolContentVersion => m_renderer?.PoolContentVersion ?? 0;


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

        [Tooltip("Sort by RADIAL distance to the camera instead of view-space depth (PlayCanvas " +
                 "radialSorting). A radial order is rotation-INVARIANT: turning the head in place " +
                 "does not change the back-to-front order. Default off = view-space depth (PC " +
                 "desktop default). Renders equivalently to depth sorting (slightly less painter-" +
                 "exact for strongly anisotropic scenes; PC ships it for all streamed walkthroughs). " +
                 "NOTE: this currently only swaps the sort KEY — the fps win of SKIPPING the re-sort " +
                 "on rotation is NOT yet wired, because our combined-chunked path rebuilds the drawn " +
                 "set every frame (per-splat frustum cull), so a skipped sort would draw the new set " +
                 "unsorted. The skip needs a stable resident set across rotation (GPU-cull-at-draw / " +
                 "look-back residency) — future work; radial keys are the prerequisite, now in place.")]
        public bool RadialSort = false;

        public void ComputeDepth(CommandBuffer cmd, Matrix4x4 matrixMv) => m_renderer.ComputeDepth(cmd, matrixMv, RadialSort);

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
            // In disk-streaming mode the bound asset is a metadata-only runtime asset, not the
            // serialized GsplatAsset. EffectiveAsset resolves it (and lazily loads the manifest).
            var boundAsset = EffectiveAsset;
            if (!boundAsset)
                m_prevAsset = null;
            // Flipping pool mode at runtime (ChunkedLod/ChunkedStreaming toggled, ChunkTable
            // set/nulled) must force a full rebind: the pool resource holds only the resident
            // subset (with HOLES in incremental mode) while the non-pool paths assume a fully
            // uploaded asset — drawing one through the other renders stale/undefined splats.
            bool poolModeNow = PoolMode;
            if (m_wasPoolMode != poolModeNow)
            {
                m_prevAsset = null;
                m_wasPoolMode = poolModeNow;
            }
            if (m_prevAsset != boundAsset)
            {
                m_renderer?.ReleaseGsplatAsset();
                m_prevAsset = boundAsset;
                if (boundAsset)
                {
                    // In streaming pool mode the GPU buffers are sized to the budget, not the
                    // full (2x-LOD) combined asset.
                    uint cap = PoolMode ? (uint)ChunkedPoolBudget : boundAsset.SplatCount;
                    if (m_renderer == null)
                        m_renderer = new GsplatRendererImpl(cap);
                    else
                        m_renderer.RecreateResources(cap);
#if UNITY_EDITOR
                    var asyncUpload = AsyncUpload && Application.isPlaying;
#else
                    var asyncUpload = AsyncUpload;
#endif
                    m_renderer.BindGsplatAsset(boundAsset, asyncUpload, PoolMode);
                    GsplatSorter.Instance.MarkGlobalBuffersDirty();
                }
            }

            if (Valid && GsplatSettings.Instance.Valid && GsplatSorter.Instance.Valid)
            {
                m_renderer.EvaluateRefreshRequired(SortMode, SortRefreshRate - 1, CutoutsRefreshRate - 1);
                // Only cull at runtime: in edit mode Camera.main would cull the Scene view too,
                // making splats vanish while you look through it.
                var runtimeCam = Application.isPlaying ? (CullCamera != null ? CullCamera : Camera.main) : null;

                // DISK-streaming branch: chunk-levels stream from .gsstream blobs into the pool.
                if (StreamingMode && m_streamTable != null)
                {
                    float sBase = ChunkedLodBaseDistance, sMult = Mathf.Max(1.05f, ChunkedLodMultiplier);
                    if (ChunkedLodAutoRange && m_streamTable.MaxLod > 1)
                        sBase = Mathf.Min(ChunkedLodBaseDistance, m_streamTable.Bounds.size.magnitude / 4f);
                    m_lastEffectiveBase = sBase;
                    m_lastEffectiveMultiplier = sMult;
                    m_renderer.DispatchStreamingPool(m_streamTable, InitOrderChunkedShader,
                        transform.localToWorldMatrix, runtimeCam, ChunkedDistanceLod, ChunkedFixedLevel,
                        sBase, sMult, ChunkedBehindPenalty, ChunkedCull, ChunkedBudgetBalancer,
                        FrustumCullMargin, ChunkedLodHysteresis, ChunkedHysteresis, ChunkedCullFootprintScale,
                        m_streamTable.SHBands);
                }
                else if (ChunkedLod && ChunkTable != null && Application.isPlaying)
                {
                    if (m_chunkTableParsed == null || m_chunkTableSource != ChunkTable)
                    {
                        m_chunkTableParsed = GsplatChunkTable.Parse(ChunkTable.text);
                        m_chunkTableSource = ChunkTable;
                    }
                    // Scene-adaptive LOD range. The MULTIPLIER is fixed at the physically-correct
                    // 2.0, NOT derived from the scene diagonal. The baker (bake_chunks.py,
                    // level_ratio 0.5 + doubling voxel cell) makes each coarser level's splats ~2×
                    // larger, so on-screen splat size stays constant only when the LOD climbs one
                    // level per DISTANCE DOUBLING — i.e. mult = 2. The old autoRange force-mapped all
                    // 11 levels onto the ~43 m diagonal, giving effMult ≈ 1.30 (a level every ~1.3×
                    // distance), which is ~4× too aggressive: coarse levels landed in the NEAR field
                    // (a 7 m wall drew LOD5, 32× fewer splats) → blobby up close while the same
                    // coarseness looked fine far away (few screen pixels). That read as "quality
                    // rises with distance". mult = 2 keeps near crisp (7 m → LOD1); the far field is
                    // coarsened by distance (≈LOD4 at 30–43 m) AND the strict far-first budget
                    // balancer (2 M ceiling vs 9.65 M LOD0 total).
                    //
                    // AutoRange now ONLY guards the base (see below) and otherwise RESPECTS the
                    // authored ChunkedLodMultiplier, so the LOD aggressiveness is an inspector knob:
                    //   • mult = 2.0  → SSE-correct: one level per distance DOUBLING. Near stays crisp
                    //                   (7 m wall → LOD1); in a compact scene the far field only reaches
                    //                   ~LOD4-6, because higher levels need hundreds of metres.
                    //   • mult ≈ 1.35 → "fill the scene": all levels map onto the diagonal, so far
                    //                   corners reach the COARSEST LODs (7-11). Cheapest, but coarse
                    //                   levels creep into the near field (blobbier up close) — that is
                    //                   the SSE tradeoff, chosen deliberately when you want far chunks
                    //                   as light as possible. For N levels over a diagonal D from base
                    //                   B, the fill value is (D/B)^(1/(N-1)).
                    // The base is still CLAMPED to diag/4: the band test is `d < base ? LOD0 : …`, so a
                    // base ≥ the scene diagonal would collapse everything into the LOD0 shell (the old
                    // "inverted, far looks best" bug). A pathologically large authored base can never
                    // defeat distance-LOD, but the multiplier is yours to tune.
                    float effBase = ChunkedLodBaseDistance;
                    float effMult = ChunkedLodMultiplier;
                    if (ChunkedLodAutoRange && m_chunkTableParsed.MaxLod > 1)
                    {
                        float diag = m_chunkTableParsed.Bounds.size.magnitude;
                        effBase = Mathf.Min(ChunkedLodBaseDistance, diag / 4f);   // base≥diag guard only
                    }
                    effMult = Mathf.Max(1.05f, effMult);   // floor: mult ≤ 1 would never advance a level
                    m_lastEffectiveBase = effBase;
                    m_lastEffectiveMultiplier = effMult;
                    if (PoolMode)
                        m_renderer.DispatchChunkedPool(m_chunkTableParsed, InitOrderChunkedShader,
                            transform.localToWorldMatrix,
                            runtimeCam, ChunkedDistanceLod, ChunkedFixedLevel,
                            effBase, effMult, ChunkedBehindPenalty, ChunkedCull,
                            ChunkedBudgetBalancer, FrustumCullMargin,
                            ChunkedLodHysteresis, ChunkedHysteresis, ChunkedCullFootprintScale,
                            ChunkedIncrementalRefill);
                    else if (InitOrderChunkedShader != null)
                        m_renderer.DispatchInitOrderChunked(m_chunkTableParsed, InitOrderChunkedShader,
                            transform.localToWorldMatrix, runtimeCam, ChunkedDistanceLod, ChunkedFixedLevel,
                            effBase, effMult, ChunkedBehindPenalty, ChunkedCull, FrustumCullMargin,
                            ChunkedSplatBudget, ChunkedLodHysteresis, ChunkedHysteresis, ChunkedCullFootprintScale,
                            ChunkedPerSplatCull,
                            // Fade uses the per-renderer draw shader; the global merged-draw path lacks the
                            // per-splat tag/fade buffers, so auto-disable fade there (would double-darken).
                            ChunkedLodFade && !GsplatSorter.Instance.GlobalRenderEnabled, ChunkedFadeWidth);
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