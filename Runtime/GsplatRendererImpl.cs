// Copyright (c) 2025 Yize Wu
// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using Vector3 = UnityEngine.Vector3;

namespace Gsplat
{
    public class GsplatRendererImpl
    {
        public uint SplatCount { get; private set; }

        MaterialPropertyBlock m_propertyBlock;
        GsplatAsset m_gsplatAsset;
        public uint m_remainingCount = 0;
        public Bounds m_bounds;
        int m_gsplatAssetID;

        public GsplatResource GsplatResource;
        public GraphicsBuffer OrderBuffer { get; private set; }
        public GraphicsBuffer CutoutsBuffer { get; private set; }
        public GraphicsBuffer OrderSizeBuffer { get; private set; }
        public GraphicsBuffer BoundsBuffer { get; private set; }
        public ISorterResource SorterResource { get; private set; }

        static readonly int k_orderBuffer = Shader.PropertyToID("_OrderBuffer");
        static readonly int k_matrixM = Shader.PropertyToID("_MATRIX_M");
        static readonly int k_splatInstanceSize = Shader.PropertyToID("_SplatInstanceSize");
        static readonly int k_splatCount = Shader.PropertyToID("_SplatCount");
        static readonly int k_gammaToLinear = Shader.PropertyToID("_GammaToLinear");
        static readonly int k_shDegree = Shader.PropertyToID("_SHDegree");
        static readonly int k_brightness = Shader.PropertyToID("_Brightness");
        static readonly int k_scaleFactor = Shader.PropertyToID("_ScaleFactor");
        static readonly int k_frustumPlanes = Shader.PropertyToID("_FrustumPlanes");
        static readonly int k_cullMargin = Shader.PropertyToID("_CullMargin");
        static readonly int k_packedSplatsBuffer = Shader.PropertyToID("_PackedSplatsBuffer");
        static readonly int k_splatChunkBuffer = Shader.PropertyToID("_SplatChunk");
        static readonly int k_selectedLevelBuffer = Shader.PropertyToID("_SelectedLevel");
        static readonly int k_fadeWeightBuffer = Shader.PropertyToID("_FadeWeight");
        static readonly int k_lodFadeEnabled = Shader.PropertyToID("_LodFadeEnabled");
        static readonly int k_minPixelSize = Shader.PropertyToID("_GsplatMinPixelSize");
        static readonly int k_minContribution = Shader.PropertyToID("_GsplatMinContribution");
        static readonly int k_alphaClip = Shader.PropertyToID("_GsplatAlphaClip");
        static readonly int k_foveationStrength = Shader.PropertyToID("_GsplatFoveationStrength");
        static readonly int k_foveationCenter = Shader.PropertyToID("_GsplatFoveationCenter");
        static readonly int k_poolLiveMask = Shader.PropertyToID("_PoolLiveMask");

        uint m_framesBeforeRecomputeSort = 0;
        uint m_sortsBeforeRecomputeCutouts = 0;
        public bool ComputeSortRequired = true;
        public bool ComputeCutoutsRequired = true;
        Dictionary<int, (Vector3, Vector3)> m_prevCamTransforms;

        GsplatCutout.ShaderData[] m_cutoutsData;
        uint m_prevSplatCount;

        // Whether the previous InitOrder dispatch produced a culled subset (cutouts or
        // frustum). Used to restore the identity draw order when culling is turned off.
        bool m_prevCulled;
        readonly Plane[] m_frustumPlanes = new Plane[6];
        readonly Vector4[] m_frustumPlanesOS = new Vector4[6];

        // Camera pose used for the last frustum InitOrder. The visible set only changes
        // when the camera moves, so — even when the sort runs every frame (SortMode.Always)
        // — we rebuild it only past the same thresholds as sort refresh, not every frame.
        // Seeded to infinity so the first frustum dispatch always runs.
        Vector3 m_lastCullCamPos = new(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity);
        Quaternion m_lastCullCamRot = Quaternion.identity;   // quaternion (not euler): Quaternion.Angle is wrap/gimbal-safe

        // Chunked-LOD runtime state (see docs/CHUNKED_LOD_DESIGN.md, GsplatChunkTable).
        GsplatChunkTable m_chunkTable;
        GraphicsBuffer m_splatChunkBuffer;      // per-splat (chunkId<<4)|level tags
        GraphicsBuffer m_selectedLevelBuffer;   // per-chunk chosen level (updated each frame)
        uint[] m_selectedLevel;
        GraphicsBuffer m_fadeWeightBuffer;      // per-chunk LOD cross-fade weight 0..255 (0 = no fade)
        uint[] m_fadeWeight;

        // P1.3 incremental pool refill (streaming pool mode): CPU-side block allocator mirroring
        // the pool buffers + per-slot liveness bitmask consumed by the InitOrderPool kernel.
        GsplatChunkPool.Layout m_poolLayout;
        GsplatChunkTable m_poolTable;           // table the layout was built against
        GraphicsBuffer m_poolLiveMaskBuffer;
        // Disk-streaming pool (S4): async blob loader + count of selected chunk-levels still loading.
        GsplatStreamingLoader m_streamLoader;
        public int m_streamPending;
        readonly Plane[] m_worldFrustumPlanes = new Plane[6];

        // R3 budget-balancer scratch (see ApplyBudgetBalancer). Per-chunk camera distance +
        // a 64-bucket √-distance counting-sort order (near-first), all reused to stay GC-free.
        float[] m_chunkDist;                    // per-chunk closest-AABB distance to the camera
        int[] m_chunkBucket;                    // per-chunk √-distance bucket (−1 = culled)
        int[] m_bucketOrder;                    // chunk indices, near bucket first
        readonly int[] m_bucketCount = new int[64];
        readonly int[] m_bucketStart = new int[64];
        readonly int[] m_bucketCursor = new int[64];
        public int m_lastBalancedTotal;         // Σ selected splats after the balancer (diagnostic)

        // PlayCanvas Stage-A budget corrector (gsplat-world.js _enforceBudget): when the view's
        // DEMAND (Σ band-level counts) exceeds the budget, shrink the distance bands
        // (effectiveBase = base·scale, effectiveMult = max(1.2, mult·scale^-0.2)) so FAR chunks
        // slide to coarser LODs geometrically while the nearest stay in the fine bands. Without
        // this, the bucket balancer's one-level-per-pass sweep robs the NEAREST chunk of LOD0
        // before far chunks give up their second level (point-blank walls turned to mush).
        // Clamped to ≤1: the budget stays a CEILING — under budget the scale only recovers to 1
        // (distance-ideal), it never upgrades past it. Converges over a few frames (blend 0.3).
        public float m_budgetScale = 1f;

        // R4 streaming-pool state: an owned budget-sized resource populated per refresh.
        GsplatResourceSpark m_poolResource;
        bool m_poolMode;
        public uint m_poolOverflow;

        // Per-chunk LOD currently selected this frame (0xFFFFFFFF = culled), for editor debug.
        public uint[] SelectedLevels => m_selectedLevel;

        public GsplatRendererImpl(uint splatCount)
        {
            SplatCount = splatCount;
            m_prevCamTransforms = new Dictionary<int, (Vector3, Vector3)>();
            CreateResources(splatCount);
            CreatePropertyBlock();
        }

        public void RecreateResources(uint splatCount)
        {
            if (SplatCount == splatCount)
                return;
            Dispose();
            SplatCount = splatCount;
            CreateResources(splatCount);
            CreatePropertyBlock();
        }

        public void ComputeDepth(CommandBuffer cmd, Matrix4x4 matrixMv, bool radial) =>
            // Depth is only consumed for the first RemainingCount order entries (the sort
            // Count); bounding the dispatch to it also keeps the incremental pool path off
            // undefined order entries in [LiveCount, UsedEnd) — an out-of-bounds-INDEX
            // structured read that is benign on desktop but relies on robustBufferAccess
            // on mobile (Adreno).
            m_gsplatAsset.ComputeDepth(cmd, matrixMv, SorterResource, GsplatResource,
                System.Math.Min(m_remainingCount, GsplatResource.UploadedCount), radial);

        public int PoolRepackCount => m_poolLayout?.RepackCount ?? 0;
        // 4a: exposed to the cross-renderer global merge so it can detect when this pool's live-slot
        // CONTENTS changed (residency swap) even if UsedEnd/SplatCount didn't — see GsplatChunkPool.Layout.
        public uint PoolContentVersion => m_poolLayout?.ContentVersion ?? 0;

        Bounds ExtractBounds()
        {
            uint[] boundsData = new uint[6];
            BoundsBuffer.GetData(boundsData);

            Bounds bounds = default;
            Vector3 bmin = new(GsplatUtils.SortableUintToFloat(boundsData[0]),
                GsplatUtils.SortableUintToFloat(boundsData[1]), GsplatUtils.SortableUintToFloat(boundsData[2]));
            Vector3 bmax = new(GsplatUtils.SortableUintToFloat(boundsData[3]),
                GsplatUtils.SortableUintToFloat(boundsData[4]), GsplatUtils.SortableUintToFloat(boundsData[5]));
            bounds.SetMinMax(bmin, bmax);

            if (bounds.extents.sqrMagnitude < 0.01)
                bounds.extents = new Vector3(0.1f, 0.1f, 0.1f);
            return bounds;
        }

        uint ExtractOrderSize(GraphicsBuffer orderBuffer)
        {
            GraphicsBuffer.CopyCount(orderBuffer, OrderSizeBuffer, 0);
            uint[] count = new uint[1];
            OrderSizeBuffer.GetData(count);
            return count[0];
        }

        public void DispatchInitOrder(GsplatCutout[] cutouts, Matrix4x4 matrixWorld, bool cutoutsUpdateBounds,
            Camera cullCamera = null, float cullMargin = 0f)
        {
            bool frustum = cullCamera != null;

            // No per-splat filtering at all: draw the whole asset with an identity order.
            if (cutouts.Length == 0 && !frustum)
            {
                // Restore the identity draw order if the previous frame produced a culled subset.
                if (m_prevCulled)
                {
                    SorterResource.Initialized = false;
                    m_prevCulled = false;
                }
                // Force a rebuild the next time frustum culling is (re-)enabled.
                m_lastCullCamPos = new Vector3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity);
                m_cutoutsData = Array.Empty<GsplatCutout.ShaderData>();
                m_remainingCount = GsplatResource.UploadedCount;
                m_bounds = m_gsplatAsset.Bounds;
                return;
            }

            if (!ComputeCutoutsRequired)
                return;

            SorterResource.Initialized = true;

            var cutoutsUnchanged = m_cutoutsData.Length == cutouts.Length;
            var updatedCutoutsData = new GsplatCutout.ShaderData[cutouts.Length];
            for (int i = 0; i != cutouts.Length; i++)
            {
                updatedCutoutsData[i] = cutouts[i].GetShaderData(matrixWorld);
                if (cutoutsUnchanged)
                    if (updatedCutoutsData[i].matrix != m_cutoutsData[i].matrix ||
                        updatedCutoutsData[i].typeAndFlags != m_cutoutsData[i].typeAndFlags)
                        cutoutsUnchanged = false;
            }

            // Skip the re-dispatch when nothing that affects the visible set changed:
            // cutouts identical, splat count identical, and — for frustum culling — the
            // camera has not moved past the refresh thresholds. This decouples the frustum
            // rebuild from the per-frame sort cadence (SortMode.Always), so a static camera
            // costs no InitOrder dispatch even while the sort keeps running every frame.
            bool cameraStatic = frustum && !CameraMovedSinceLastCull(cullCamera);
            if (cutoutsUnchanged && m_prevSplatCount == GsplatResource.UploadedCount
                && (!frustum || cameraStatic))
                return;

            m_prevSplatCount = GsplatResource.UploadedCount;
            m_cutoutsData = updatedCutoutsData;
            CutoutsBuffer = m_gsplatAsset.UpdateCutoutsBuffer(CutoutsBuffer, m_cutoutsData);

            // Same ComputeShader instance the asset's InitOrder will dispatch (per-asset,
            // selected by compression via GsplatSettings). Set frustum state on it first.
            var cs = m_gsplatAsset.GsplatMaterial.InitOrderShader;
            if (frustum)
            {
                BuildObjectSpaceFrustumPlanes(cullCamera, matrixWorld);
                cs.EnableKeyword("FRUSTUM_CULL");
                cs.SetVectorArray(k_frustumPlanes, m_frustumPlanesOS);
                cs.SetFloat(k_cullMargin, cullMargin);
            }
            else
            {
                cs.DisableKeyword("FRUSTUM_CULL");
            }

            if (cutoutsUpdateBounds)
                m_gsplatAsset.UpdateBoundsBuffer(BoundsBuffer);
            m_gsplatAsset.InitOrder(SorterResource, GsplatResource, cutoutsUpdateBounds);
            m_remainingCount = ExtractOrderSize(SorterResource.OrderBuffer);
            m_bounds = cutoutsUpdateBounds ? ExtractBounds() : m_gsplatAsset.Bounds;
            m_prevCulled = true;
            if (frustum)
            {
                m_lastCullCamPos = cullCamera.transform.position;
                m_lastCullCamRot = cullCamera.transform.rotation;
            }
        }

        // True if the cull camera moved/rotated past the (shared) refresh thresholds since
        // the last frustum InitOrder. Matches RefreshOnCameraMove's naive euler comparison.
        bool CameraMovedSinceLastCull(Camera cam)
        {
            // Quaternion.Angle gives the TRUE angular delta in degrees — the old euler-magnitude
            // `(rot-last).magnitude` mis-measured near 0/360 wrap and gimbal, so the gate could
            // UNDER-fire and leave the cull stale by far more than the threshold (the root of the
            // "wall vanishes when you turn" bug). The angular cull margin (CalcCullFrustumPlanes)
            // then covers the remaining ≤threshold lag between refreshes.
            return (cam.transform.position - m_lastCullCamPos).magnitude > GsplatSettings.Instance.CameraTranslationRefreshTreshold
                || Quaternion.Angle(cam.transform.rotation, m_lastCullCamRot) > GsplatSettings.Instance.CameraRotationRefreshTreshold;
        }

        // Camera frustum planes widened by GsplatSettings.CullFrustumMarginDeg of extra half-FOV, so the
        // gated chunked/streaming cull SELECTION (rebuilt only past the refresh thresholds) stays valid
        // as the camera moves in between — content about to enter view is already drawn. Scales the real
        // projectionMatrix x/y (convention-safe; near/far unchanged). margin<=0 or ortho => plain frustum.
        static void CalcCullFrustumPlanes(Camera cam, Plane[] dst)
        {
            float marginDeg = GsplatSettings.Instance.CullFrustumMarginDeg;
            if (marginDeg <= 0f || cam.orthographic)
            {
                GeometryUtility.CalculateFrustumPlanes(cam, dst);
                return;
            }
            float fov = cam.fieldOfView;
            float wide = Mathf.Min(179f, fov + 2f * marginDeg);
            float s = Mathf.Tan(fov * 0.5f * Mathf.Deg2Rad) / Mathf.Tan(wide * 0.5f * Mathf.Deg2Rad); // <1 widens
            Matrix4x4 p = cam.projectionMatrix;
            p.m00 *= s; p.m11 *= s;
            GeometryUtility.CalculateFrustumPlanes(p * cam.worldToCameraMatrix, dst);
        }

        // Builds the 6 camera frustum planes in the asset's object space (inward normals,
        // xyz normalized so _CullMargin is metric). For an object point x, world = M·x, so a
        // world plane P satisfies P·(M·x) = (Mᵀ·P)·x — hence the transpose pull-back.
        void BuildObjectSpaceFrustumPlanes(Camera cam, Matrix4x4 localToWorld)
        {
            CalcCullFrustumPlanes(cam, m_frustumPlanes);   // widened by CullFrustumMarginDeg (anti stale-cull)
            Matrix4x4 mt = localToWorld.transpose;
            for (int i = 0; i < 6; i++)
            {
                Plane pl = m_frustumPlanes[i];
                Vector4 pw = new Vector4(pl.normal.x, pl.normal.y, pl.normal.z, pl.distance);
                Vector4 po = mt * pw;
                float n = new Vector3(po.x, po.y, po.z).magnitude;
                if (n > 1e-8f) po /= n;
                m_frustumPlanesOS[i] = po;
            }
        }

        void EnsureChunkSetup(GsplatChunkTable table)
        {
            if (ReferenceEquals(m_chunkTable, table) && m_splatChunkBuffer != null)
                return;
            m_chunkTable = table;
            m_splatChunkBuffer?.Dispose();
            m_selectedLevelBuffer?.Dispose();
            m_fadeWeightBuffer?.Dispose();
            var tags = table.BuildSplatChunkTags();
            m_splatChunkBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, tags.Length, sizeof(uint));
            m_splatChunkBuffer.SetData(tags);
            m_selectedLevel = new uint[table.ChunkCount];
            m_selectedLevelBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                Mathf.Max(1, table.ChunkCount), sizeof(uint));
            m_fadeWeight = new uint[table.ChunkCount];
            m_fadeWeightBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                Mathf.Max(1, table.ChunkCount), sizeof(uint));
        }

        // Chunked-LOD visible-set build: choose one LOD level per chunk — by FOV-compensated
        // distance from the camera to each chunk's AABB (PlayCanvas evaluateNodeLods bands
        // base*mult^i), or a fixed global level for debugging — optionally frustum-culling
        // chunks by their bounding sphere. Uploads the per-chunk selection, then dispatches
        // InitOrderChunked over the combined buffer; reuses existing depth+sort+draw.
        public void DispatchInitOrderChunked(GsplatChunkTable table, ComputeShader cs, Matrix4x4 matrixWorld,
            Camera camera, bool distanceLod, int fixedLevel, float baseDistance, float multiplier,
            float behindPenalty, bool cull, float cullMargin, int splatBudget, bool hysteresis, float hyst,
            float cullFootprintScale, bool perSplatCull, bool fade, float fadeWidth)
        {
            EnsureChunkSetup(table);
            // The draw shader now reads the per-chunk tag + selected level + fade weight to modulate
            // opacity for the LOD cross-fade; bind them on the draw property block (buffers exist
            // after EnsureChunkSetup, and the block is created before the first dispatch).
            if (m_propertyBlock != null)
            {
                m_propertyBlock.SetBuffer(k_splatChunkBuffer, m_splatChunkBuffer);
                m_propertyBlock.SetBuffer(k_selectedLevelBuffer, m_selectedLevelBuffer);
                m_propertyBlock.SetBuffer(k_fadeWeightBuffer, m_fadeWeightBuffer);
                m_propertyBlock.SetFloat(k_lodFadeEnabled, fade ? 1f : 0f);
            }
            // Camera-move gate (roadmap item c): while the camera is still, the per-chunk selection
            // and the appended order buffer are unchanged, so skip the O(chunks) CPU select AND the
            // O(UploadedCount) GPU InitOrderChunked dispatch + its BLOCKING ExtractOrderSize readback
            // (the pool path already gates this way). The sort still re-runs each frame over the
            // retained order (SortMode), and the draw reuses m_remainingCount — correct for a static
            // camera. m_remainingCount==0 forces the first build; rotation past the gate rebuilds.
            if (m_remainingCount > 0 && camera != null && !CameraMovedSinceLastCull(camera))
                return;
            ComputeSelectedLevels(table, matrixWorld, camera, distanceLod, fixedLevel,
                baseDistance, multiplier, behindPenalty, cull, cullMargin, splatBudget > 0, hysteresis, hyst,
                cullFootprintScale, fade, fadeWidth);
            // R3 on the combined path: the distance bands (PlayCanvas parity) only reach a few
            // LODs in a compact scene; the budget balancer is what forces the full ladder into
            // play — degrade the farthest chunks toward LOD max until Σ(selected) <= splatBudget,
            // exactly like PlayCanvas's sqrt-distance bucket balancer. 0 = disabled (distance only).
            m_lastBalancedTotal = (splatBudget > 0 && camera != null)
                ? ApplyBudgetBalancer(table, splatBudget) : 0;
            m_selectedLevelBuffer.SetData(m_selectedLevel);
            m_fadeWeightBuffer.SetData(m_fadeWeight);

            SorterResource.Initialized = true;
            m_prevCulled = true;

            var res = (GsplatResourceSpark)GsplatResource;
            int kernel = cs.FindKernel("InitOrderChunked");
            SorterResource.OrderBuffer.SetCounterValue(0);
            cs.SetInt(k_splatCount, (int)res.UploadedCount);
            cs.SetBuffer(kernel, k_orderBuffer, SorterResource.OrderBuffer);
            cs.SetBuffer(kernel, k_packedSplatsBuffer, res.PackedSplatsBuffer);
            cs.SetBuffer(kernel, k_splatChunkBuffer, m_splatChunkBuffer);
            cs.SetBuffer(kernel, k_selectedLevelBuffer, m_selectedLevelBuffer);
            cs.SetBuffer(kernel, k_fadeWeightBuffer, m_fadeWeightBuffer);

            // Per-SPLAT frustum cull (tier 2) — the CORRECT cull for gaussian splats: each splat
            // survives unless its OWN 3σ footprint is fully outside the frustum, so partly-visible
            // chunks keep exactly their visible splats (no holes). DECOUPLED from the per-CHUNK cull
            // (tier 1, ComputeSelectedLevels): tier 1's world-space bounding-sphere test is unreliable
            // for splats (a chunk whose bounds sit outside the frustum can still have large/near splats
            // that project on-screen — through-window exterior, overhang — so tier 1 drops visible
            // content = grey holes), AND on this COMBINED path tier 1 saves nothing (the dispatch is
            // O(UploadedCount) regardless; culled-chunk threads merely early-return). So the robust
            // setup is ChunkedCull=false (tier 1 off, all chunks selected) + ChunkedPerSplatCull=true
            // (tier 2 does the precise cull) — hence this runs on perSplatCull alone, not && cull.
            if (perSplatCull && camera != null)
            {
                BuildObjectSpaceFrustumPlanes(camera, matrixWorld);
                cs.EnableKeyword("FRUSTUM_CULL");
                cs.SetVectorArray(k_frustumPlanes, m_frustumPlanesOS);
                cs.SetFloat(k_cullMargin, cullMargin);
            }
            else cs.DisableKeyword("FRUSTUM_CULL");

            cs.Dispatch(kernel, (int)GsplatUtils.DivRoundUp(res.UploadedCount, 1024), 1, 1);
            m_remainingCount = ExtractOrderSize(SorterResource.OrderBuffer);
            m_bounds = m_gsplatAsset.Bounds;
            // Stamp the cull pose so the camera-move gate above skips rebuilds while still.
            if (camera != null)
            {
                m_lastCullCamPos = camera.transform.position;
                m_lastCullCamRot = camera.transform.rotation;
            }
        }

        // Pick one LOD level per chunk into m_selectedLevel (0xFFFFFFFF = culled): FOV-comp
        // distance bands base*mult^i, or a fixed level; optional bounding-sphere frustum cull.
        // Shared by the combined-buffer path (InitOrderChunked) and the streaming pool.
        // cullMargin (metres) widens the frustum for the per-chunk test so a chunk at the screen
        // edge isn't popped while its splats are still visible — the bounding sphere covers only
        // splat CENTRES, and splats have extent, so a splat whose centre is just outside can
        // still poke into view; the margin also adds hysteresis against edge flicker.
        void ComputeSelectedLevels(GsplatChunkTable table, Matrix4x4 matrixWorld, Camera camera,
            bool distanceLod, int fixedLevel, float baseDistance, float multiplier, float behindPenalty,
            bool cull, float cullMargin, bool budgetBands, bool hysteresis, float hyst,
            float cullFootprintScale, bool fade, float fadeWidth)
        {
            if (!budgetBands) m_budgetScale = 1f;   // corrector only lives while a budget is active
            if (m_selectedLevel == null || m_selectedLevel.Length != table.ChunkCount)
            {
                m_selectedLevel = new uint[table.ChunkCount];
                for (int i = 0; i < m_selectedLevel.Length; i++) m_selectedLevel[i] = 0xFFFFFFFFu;
            }
            // m_fadeWeight is written UNCONDITIONALLY below (:476), but EnsureChunkSetup — which
            // allocates it — only runs on the combined (InitOrderChunked) path. The streaming-pool
            // path (DispatchChunkedPool) calls this directly, so allocate here too or it NREs every
            // frame (authored-from-start pool would never fill → black render). B1 root cause.
            if (m_fadeWeight == null || m_fadeWeight.Length != table.ChunkCount)
                m_fadeWeight = new uint[table.ChunkCount];
            if (m_chunkDist == null || m_chunkDist.Length != table.ChunkCount)
            {
                m_chunkDist = new float[table.ChunkCount];
                m_chunkBucket = new int[table.ChunkCount];
                m_bucketOrder = new int[table.ChunkCount];
            }
            int maxLod = table.MaxLod;
            bool doDistance = distanceLod && camera != null && maxLod > 0;
            bool doCull = cull && camera != null;
            if (doCull) CalcCullFrustumPlanes(camera, m_worldFrustumPlanes);   // widened by CullFrustumMarginDeg (anti stale-cull)

            var ls = matrixWorld.lossyScale;
            float scale = Mathf.Max(ls.x, Mathf.Max(ls.y, ls.z));
            bool haveCam = camera != null;
            // camLocal is needed for both the distance bands AND the balancer's √-distance
            // buckets, so compute it whenever there is a camera (not only in distance mode).
            Vector3 camLocal = haveCam
                ? matrixWorld.inverse.MultiplyPoint3x4(camera.transform.position) : Vector3.zero;
            // Behind-camera LOD penalty (PlayCanvas lodBehindPenalty): the band distance of a
            // chunk behind the view direction is inflated by up to x`behindPenalty` (fully behind),
            // so off-view content coarsens first and its budget flows to what is on screen. The
            // rotation refresh gate re-evaluates the selection when the user turns around, which
            // is the same requirement PC documents for its penalty (a non-zero lodUpdateAngle).
            bool doBehind = doDistance && behindPenalty > 1f;
            Vector3 fwdLocal = doBehind
                ? matrixWorld.inverse.MultiplyVector(camera.transform.forward).normalized
                : Vector3.forward;
            float fovScale = 1f, invLogMult = 1f;
            if (doDistance)
            {
                float tanHalfV = Mathf.Tan(camera.fieldOfView * 0.5f * Mathf.Deg2Rad);
                float tanHalfH = tanHalfV * camera.aspect;
                fovScale = Mathf.Min(tanHalfV, tanHalfH) / 0.41421356f; // tan(22.5°)
                // NOTE: the LOD bands are PURE distance — deliberately NOT scaled by a global
                // budget-demand factor. An earlier Stage-A `m_budgetScale` shrank the bands when
                // the view was over budget, which coarsened EVERY chunk (near ones included) and,
                // worse, coupled a chunk's LOD to how many OTHER chunks were on screen: moving away
                // from a chunk reduced the total demand, relaxed the scale, and made that chunk
                // FINER even as it got farther ("quality rises with distance"). Budget pressure is
                // now absorbed entirely by the strictly far-first degrade in ApplyBudgetBalancer,
                // so a chunk's LOD depends only on its own distance — monotonic and view-stable.
                invLogMult = 1f / Mathf.Log(Mathf.Max(1.0001f, multiplier));
                baseDistance = Mathf.Max(1e-4f, baseDistance);
            }

            for (int c = 0; c < table.ChunkCount; c++)
            {
                uint prev = m_selectedLevel[c];   // last frame's decision (0xFFFFFFFF = was culled)
                var aabb = table.Chunks[c].Aabb;
                float distRaw = 0f, d = 0f;
                if (haveCam)
                {
                    Vector3 closest = Vector3.Max(aabb.min, Vector3.Min(camLocal, aabb.max));
                    distRaw = Vector3.Distance(camLocal, closest) * scale;
                    d = distRaw * fovScale;
                    if (doBehind && distRaw > 1e-4f)
                    {
                        // t = how far behind the closest point is (0 = beside/ahead, 1 = dead
                        // behind); cos of the angle to the view axis, local space.
                        float t = -Vector3.Dot(fwdLocal, closest - camLocal) * scale / distRaw;
                        if (t > 0f) d *= 1f + t * (behindPenalty - 1f);
                    }
                    // Balancer buckets are fed the penalized, FOV-scaled distance (PC parity:
                    // their budgetBucket derives from the penalized fovAdjustedDistance), so a
                    // behind chunk is also FIRST in line for budget degrade — otherwise a behind
                    // chunk at 5m would outrank an on-screen chunk at 20m for budget.
                    m_chunkDist[c] = d;
                }
                int req;
                if (doDistance)
                {
                    req = d < baseDistance ? 0
                        : Mathf.Clamp(1 + Mathf.FloorToInt(Mathf.Log(d / baseDistance) * invLogMult), 0, maxLod);
                    // LOD hysteresis: while d stays inside the PREVIOUS level's band widened by
                    // ±hyst, keep that level — a small camera move near a band boundary won't
                    // flip the LOD back and forth.
                    if (hysteresis && prev <= (uint)maxLod)
                    {
                        int pl = (int)prev;
                        float bandLo = pl == 0 ? 0f : baseDistance * Mathf.Pow(multiplier, pl - 1);
                        float bandHi = baseDistance * Mathf.Pow(multiplier, pl);
                        if (d >= bandLo * (1f - hyst) && d < bandHi * (1f + hyst)) req = pl;
                    }
                }
                else req = Mathf.Clamp(fixedLevel, 0, maxLod);

                int use = req;
                while (use <= maxLod && table.Chunks[c].Lods[use].Count == 0) use++;
                if (use > maxLod) { m_selectedLevel[c] = 0xFFFFFFFFu; m_fadeWeight[c] = 0u; continue; }

                // LOD cross-fade weight (anti-pop): as the FOV-comp distance approaches the UPPER
                // edge of `use`'s band, ramp w 0->1 so the next coarser level (use+1) is blended in
                // over the last `fadeWidth` of the band and the discrete swap doesn't pop. Only when
                // use+1 is a real (non-empty) level, else fading `use` down would just dim it.
                m_fadeWeight[c] = 0u;
                if (fade && doDistance && fadeWidth > 1e-4f
                    && use < maxLod && table.Chunks[c].Lods[use + 1].Count > 0)
                {
                    float bandLo = use == 0 ? 0f : baseDistance * Mathf.Pow(multiplier, use - 1);
                    float bandHi = baseDistance * Mathf.Pow(multiplier, use);
                    float frac = Mathf.Clamp01((d - bandLo) / Mathf.Max(1e-6f, bandHi - bandLo));
                    float fz = Mathf.Clamp01(fadeWidth);
                    if (frac > 1f - fz)
                        m_fadeWeight[c] = (uint)Mathf.Clamp(Mathf.RoundToInt((frac - (1f - fz)) / fz * 255f), 0, 255);
                }

                if (doCull)
                {
                    var sp = table.Chunks[c].Sphere;
                    Vector3 wc = matrixWorld.MultiplyPoint3x4(new Vector3(sp.x, sp.y, sp.z));
                    // Cull radius. Prefer the EXACT per-level footprint radius baked by
                    // tools/bake_chunks.py (`FootR` = p99.9 of dist-to-sphere-centre + 2sigma, i.e.
                    // the visible 2sigma core of that level's splats, needles dropped). It is tight
                    // AND LOD-correct: a chunk whose selected-level splats don't reach the frustum
                    // is culled (no wasted tris drawing off-screen background — the reported bug),
                    // while a coarse chunk whose big splats DO reach is kept (no holes). The
                    // per-level value is measured from the SAME sphere centre used here. `scale`
                    // maps it to world; `cullFootprintScale` (inspector) trades completeness vs
                    // tris (1 = exact 2sigma; <1 = tighter/fewer, >1 = safer/more).
                    //
                    // Fallback for sidecars baked before FootR: the old LOD0 sphere widened by a
                    // heuristic extent pad — coarser at higher levels — which over-keeps but never
                    // holes.
                    var lodI = table.Chunks[c].Lods[use];
                    float wr;
                    if (lodI.FootR > 0f)
                    {
                        wr = lodI.FootR * scale * cullFootprintScale + cullMargin;
                    }
                    else
                    {
                        float lodFrac = maxLod > 0 ? (float)use / maxLod : 0f;
                        float footPad = table.Chunks[c].MaxExtent * scale * (0.25f + 1.25f * lodFrac);
                        wr = sp.w * scale + cullMargin + footPad;
                    }
                    float minSlack = float.MaxValue;        // >=0 => sphere intersects the frustum
                    for (int p = 0; p < 6; p++)
                        minSlack = Mathf.Min(minSlack, m_worldFrustumPlanes[p].GetDistanceToPoint(wc) + wr);
                    bool inside = minSlack >= 0f;
                    if (hysteresis)
                    {
                        // Schmitt trigger on the frustum boundary: a visible chunk stays visible
                        // until CLEARLY outside (exit at -hm), while a culled chunk re-enters at
                        // the plain conservative threshold (0) — the thresholds differ by hm, so
                        // edge jitter still can't flicker, but re-entry is NEVER stricter than
                        // the plain test. (A +hm re-entry threshold locked chunks with content
                        // still on screen in the culled state after they were culled once — the
                        // "approach a chunk and it suddenly culls" bug.)
                        float hm = hyst * wr;
                        if (prev != 0xFFFFFFFFu) inside = minSlack >= -hm;
                        // else: keep the plain-test result (re-enter as soon as truly visible)
                    }
                    m_selectedLevel[c] = inside ? (uint)use : 0xFFFFFFFFu;
                    if (!inside) m_fadeWeight[c] = 0u;   // culled: no fade
                }
                else m_selectedLevel[c] = (uint)use;
            }
        }

        // Nearest populated level from `level` in direction `dir` (+1 = coarser/higher index,
        // −1 = finer/lower index), skipping absent levels (Count==0); −1 if none exists.
        static int NextPresentLevel(GsplatChunkTable table, int c, int level, int dir)
        {
            var lods = table.Chunks[c].Lods;
            for (int l = level + dir; l >= 0 && l <= table.MaxLod; l += dir)
                if (lods[l].Count > 0) return l;
            return -1;
        }

        // R3 budget balancer. The distance bands in ComputeSelectedLevels already pick each
        // chunk's ideal LOD for its distance, but nothing bounds the TOTAL. This enforces
        // Σ(selected splat counts) ≤ budget so frame time is stable regardless of view, by
        // DEGRADING the farthest chunks first (spend the cut where it's least visible) until it
        // fits. It is deliberately degrade-only: the budget is a CEILING, not a target — when
        // the view already fits (e.g. zoomed out) we keep the distance selection and draw fewer
        // splats rather than upgrading to fill the budget (which would waste GPU for no visible
        // gain, since the distance LOD is already the finest the view warrants). Chunks are
        // ordered by a 64-bucket √-distance counting sort (PlayCanvas parity, GC-free).
        // Modifies m_selectedLevel in place; returns the final selected total.
        int ApplyBudgetBalancer(GsplatChunkTable table, int budget)
        {
            int n = table.ChunkCount;
            const int B = 64;

            // Max active-chunk distance → normalize √-distance into [0,B).
            float maxDist = 0f;
            for (int c = 0; c < n; c++)
                if (m_selectedLevel[c] != 0xFFFFFFFFu && m_chunkDist[c] > maxDist) maxDist = m_chunkDist[c];
            float sqrtMax = Mathf.Sqrt(Mathf.Max(1e-6f, maxDist));

            // Counting sort active chunks into 64 √-distance buckets, near (0) → far (B−1).
            for (int b = 0; b < B; b++) m_bucketCount[b] = 0;
            for (int c = 0; c < n; c++)
            {
                if (m_selectedLevel[c] == 0xFFFFFFFFu) { m_chunkBucket[c] = -1; continue; }
                int bi = Mathf.Clamp((int)(Mathf.Sqrt(m_chunkDist[c]) / sqrtMax * B), 0, B - 1);
                m_chunkBucket[c] = bi;
                m_bucketCount[bi]++;
            }
            int acc = 0;
            for (int b = 0; b < B; b++) { m_bucketStart[b] = acc; m_bucketCursor[b] = acc; acc += m_bucketCount[b]; }
            int active = acc;
            for (int c = 0; c < n; c++)
            {
                int bi = m_chunkBucket[c];
                if (bi >= 0) m_bucketOrder[m_bucketCursor[bi]++] = c;
            }

            long total = 0;
            for (int i = 0; i < active; i++)
            {
                int c = m_bucketOrder[i];
                total += table.Chunks[c].Lods[(int)m_selectedLevel[c]].Count;
                // a cross-fading chunk also draws its next coarser level (use+1) — count both so
                // the fade stays inside the budget ceiling instead of silently blowing it.
                if (m_fadeWeight[c] != 0u)
                {
                    int nl1 = (int)m_selectedLevel[c] + 1;
                    if (nl1 <= table.MaxLod) total += table.Chunks[c].Lods[nl1].Count;
                }
            }

            // Over budget: degrade STRICTLY far-first. Walk from the farthest active chunk toward
            // the nearest and coarsen EACH chunk as far as it needs to go (to its coarsest present
            // level if necessary) before moving to the next nearer chunk. A near chunk is only
            // touched once every farther chunk is already at its coarsest — so a chunk's LOD never
            // depends on how many other chunks share the view, and the nearest chunks keep their
            // fine distance-LOD until the budget is smaller than the whole far field fully
            // coarsened. This replaces the old one-level-per-pass sweep (which degraded near
            // chunks before the far field was exhausted) and the global band-scale corrector —
            // together they made LOD non-monotonic with distance under budget pressure.
            // Under budget: nothing runs; the pure-distance selection stands (ceiling, not target).
            for (int i = active - 1; i >= 0 && total > budget; i--)
            {
                int c = m_bucketOrder[i];
                // Budget-pressured chunk: stop cross-fading it (the N/N+1 pair is stale once we
                // coarsen) and reclaim its N+1 cost. Only the farthest, least-visible chunks reach
                // here, so accepting a pop on them matches the balancer's far-first rationale.
                if (m_fadeWeight[c] != 0u)
                {
                    int nl1 = (int)m_selectedLevel[c] + 1;
                    if (nl1 <= table.MaxLod) total -= table.Chunks[c].Lods[nl1].Count;
                    m_fadeWeight[c] = 0u;
                }
                int L = (int)m_selectedLevel[c];
                while (total > budget)
                {
                    int nl = NextPresentLevel(table, c, L, +1);
                    if (nl < 0) break;                       // this chunk is at its coarsest
                    total -= (long)table.Chunks[c].Lods[L].Count - table.Chunks[c].Lods[nl].Count;
                    m_selectedLevel[c] = (uint)nl;
                    L = nl;
                }
            }
            return (int)total;
        }

        // R4 streaming pool: rebuild the visible set on refresh, then let the existing
        // depth+sort+draw run over it. GPU holds ~budget splats instead of the whole combined
        // buffer. Two refill strategies (P1.3):
        //  - incremental (default): allocator keeps resident blocks in place; a refresh uploads
        //    ONLY entering/level-changed chunks. Holes are skipped by the InitOrderPool kernel
        //    (appended order, RemainingCount known CPU-side — no readback).
        //  - legacy full re-pack (debug baseline; forced while the global merged sort is active,
        //    which assumes a contiguous identity-layout pool).
        public void DispatchChunkedPool(GsplatChunkTable table, ComputeShader cs, Matrix4x4 matrixWorld,
            Camera camera, bool distanceLod, int fixedLevel, float baseDistance, float multiplier,
            float behindPenalty, bool cull, bool budgetBalance, float cullMargin, bool hysteresis,
            float hyst, float cullFootprintScale, bool incrementalRefill)
        {
            // 4a: normally the incremental (holey) pool can't be consumed by the cross-renderer global
            // merge, so global sort forces a contiguous repack (legacy Fill, CPU wholesale re-upload).
            // With EnableGlobalSortOverPool the global merge consumes the holey pool directly (its
            // ContentVersion dirty-signal keeps the cached copy fresh), so we stay incremental even under
            // global sort. Single-renderer scenes: GlobalRenderEnabled is false, so this is unchanged.
            bool useIncremental = incrementalRefill && cs != null &&
                                  (!GsplatSorter.Instance.GlobalRenderEnabled ||
                                   GsplatSettings.Instance.EnableGlobalSortOverPool);
            // A holey pool is only valid while the incremental path (InitOrderPool) draws it.
            // If the mode flipped (toggle off, global sort turned on), force a contiguous
            // repack NOW instead of waiting for the camera gate.
            bool mustRepack = !useIncremental && m_poolLayout is { Holey: true };

            // Refresh only when the camera moved past the refresh thresholds — the visible set
            // is otherwise unchanged. Uses the same reliable camera-move check as the frustum
            // path, NOT the ComputeCutoutsRequired gate (which stuck the selection at the first
            // fill). m_remainingCount==0 forces the first fill; keep the last cull pose so tiny
            // per-frame jitter doesn't rebuild.
            if (!mustRepack && m_remainingCount > 0 && camera != null && !CameraMovedSinceLastCull(camera))
                return;

            ComputeSelectedLevels(table, matrixWorld, camera, distanceLod, fixedLevel,
                baseDistance, multiplier, behindPenalty, cull, cullMargin, budgetBalance, hysteresis, hyst,
                cullFootprintScale, false, 0f);   // no LOD cross-fade in the streaming pool path (compacts whole chunks)
            // R3: fit the distance selection into the pool budget (degrade far / upgrade near)
            // so the drawn total is bounded. Budget = the pool capacity (SplatCount in pool mode).
            m_lastBalancedTotal = (budgetBalance && camera != null)
                ? ApplyBudgetBalancer(table, (int)SplatCount) : 0;

            uint visible;
            if (useIncremental)
            {
                m_poolLayout ??= new GsplatChunkPool.Layout();
                if (!ReferenceEquals(m_poolTable, table))
                {
                    m_poolLayout.Reset(0);   // chunk ids changed — residency map is meaningless
                    m_poolTable = table;
                }
                visible = GsplatChunkPool.FillIncremental((GsplatResourceSpark)GsplatResource,
                    (GsplatAssetSpark)m_gsplatAsset, table, m_selectedLevel, m_poolLayout,
                    out m_poolOverflow);

                var mask = m_poolLayout.LiveMask;
                if (m_poolLiveMaskBuffer == null || m_poolLiveMaskBuffer.count != mask.Length)
                {
                    m_poolLiveMaskBuffer?.Dispose();
                    m_poolLiveMaskBuffer = mask.Length > 0
                        ? new GraphicsBuffer(GraphicsBuffer.Target.Structured, mask.Length, sizeof(uint))
                        : null;
                    m_poolLayout.MaskDirty = mask.Length > 0;
                }
                if (m_poolLayout.MaskDirty && m_poolLiveMaskBuffer != null)
                {
                    m_poolLiveMaskBuffer.SetData(mask);
                    m_poolLayout.MaskDirty = false;
                }

                // Rebuild the appended visible order from the liveness mask. RemainingCount is
                // known CPU-side (allocator LiveCount) — no counter readback needed.
                SorterResource.OrderBuffer.SetCounterValue(0);
                if (m_poolLayout.UsedEnd > 0 && m_poolLiveMaskBuffer != null)
                {
                    int kernel = cs.FindKernel("InitOrderPool");
                    cs.SetInt(k_splatCount, m_poolLayout.UsedEnd);
                    cs.SetBuffer(kernel, k_orderBuffer, SorterResource.OrderBuffer);
                    cs.SetBuffer(kernel, k_poolLiveMask, m_poolLiveMaskBuffer);
                    cs.Dispatch(kernel, (int)GsplatUtils.DivRoundUp((uint)m_poolLayout.UsedEnd, 1024), 1, 1);
                }
                m_remainingCount = visible;
                SorterResource.Initialized = true;   // the appended subset IS the sort payload;
                                                     // identity InitPayload must not overwrite it
            }
            else
            {
                m_poolLayout?.Reset(0);   // GPU contents will no longer match the layout
                visible = GsplatChunkPool.Fill((GsplatResourceSpark)GsplatResource,
                    (GsplatAssetSpark)m_gsplatAsset, table, m_selectedLevel, out m_poolOverflow);
                m_remainingCount = visible;
                SorterResource.Initialized = false;   // re-fill identity order for the new pool contents
            }

            m_bounds = m_gsplatAsset.Bounds;
            if (camera != null)
            {
                m_lastCullCamPos = camera.transform.position;
                m_lastCullCamRot = camera.transform.rotation;
            }
        }

        // R4 DISK-streaming pool: like DispatchChunkedPool's incremental path, but the selected
        // chunk-levels are NOT in a full-asset CPU array — they stream from per-(chunk,level) .spz
        // blobs on disk (async load + worker decode, GsplatStreamingLoader) into pool slots
        // (FillStreaming). Bounds BOTH VRAM (budget-sized pool) AND CPU RAM (only in-flight blobs
        // + the budget live in memory; the metadata-only GsplatAssetStreaming holds no splats).
        // Runs FillStreaming every refresh while loads are pending so streamed blocks appear
        // without a camera move; skips only when settled AND the camera is still.
        public void DispatchStreamingPool(GsplatChunkTable table, ComputeShader cs, Matrix4x4 matrixWorld,
            Camera camera, bool distanceLod, int fixedLevel, float baseDistance, float multiplier,
            float behindPenalty, bool cull, bool budgetBalance, float cullMargin, bool hysteresis,
            float hyst, float cullFootprintScale, byte shBands)
        {
            if (cs == null) return;
            m_streamLoader ??= new GsplatStreamingLoader(2);
            m_poolLayout ??= new GsplatChunkPool.Layout();
            if (!ReferenceEquals(m_poolTable, table))
            {
                m_poolLayout.Reset(0);
                m_streamLoader.Clear();
                m_poolTable = table;
            }

            bool cameraMoved = camera == null || m_remainingCount == 0 || CameraMovedSinceLastCull(camera);
            // Settled (no in-flight loads) + still + already built → nothing changed, skip.
            if (!cameraMoved && m_streamLoader.PendingCount == 0 && m_remainingCount > 0)
                return;

            if (cameraMoved)
            {
                ComputeSelectedLevels(table, matrixWorld, camera, distanceLod, fixedLevel,
                    baseDistance, multiplier, behindPenalty, cull, cullMargin, budgetBalance, hysteresis, hyst,
                    cullFootprintScale, false, 0f);
                m_lastBalancedTotal = (budgetBalance && camera != null)
                    ? ApplyBudgetBalancer(table, (int)SplatCount) : 0;
                if (camera != null)
                {
                    m_lastCullCamPos = camera.transform.position;
                    m_lastCullCamRot = camera.transform.rotation;
                }
            }

            // Drain completed loads into slots + request loads for selected-not-resident chunks.
            uint visible = GsplatChunkPool.FillStreaming((GsplatResourceSpark)GsplatResource, table,
                m_selectedLevel, m_poolLayout, m_streamLoader, shBands, out m_poolOverflow, out m_streamPending);

            // Rebuild the appended visible order from the liveness mask (holes = not-yet-loaded /
            // freed slots are skipped). Same InitOrderPool path as the incremental RAM pool.
            var mask = m_poolLayout.LiveMask;
            if (m_poolLiveMaskBuffer == null || m_poolLiveMaskBuffer.count != mask.Length)
            {
                m_poolLiveMaskBuffer?.Dispose();
                m_poolLiveMaskBuffer = mask.Length > 0
                    ? new GraphicsBuffer(GraphicsBuffer.Target.Structured, mask.Length, sizeof(uint))
                    : null;
                m_poolLayout.MaskDirty = mask.Length > 0;
            }
            if (m_poolLayout.MaskDirty && m_poolLiveMaskBuffer != null)
            {
                m_poolLiveMaskBuffer.SetData(mask);
                m_poolLayout.MaskDirty = false;
            }
            SorterResource.OrderBuffer.SetCounterValue(0);
            if (m_poolLayout.UsedEnd > 0 && m_poolLiveMaskBuffer != null)
            {
                int kernel = cs.FindKernel("InitOrderPool");
                cs.SetInt(k_splatCount, m_poolLayout.UsedEnd);
                cs.SetBuffer(kernel, k_orderBuffer, SorterResource.OrderBuffer);
                cs.SetBuffer(kernel, k_poolLiveMask, m_poolLiveMaskBuffer);
                cs.Dispatch(kernel, (int)GsplatUtils.DivRoundUp((uint)m_poolLayout.UsedEnd, 1024), 1, 1);
            }
            m_remainingCount = visible;
            SorterResource.Initialized = true;
            m_bounds = m_gsplatAsset.Bounds;
        }

        public void BindGsplatAsset(GsplatAsset gsplatAsset, bool asyncUpload = false, bool poolMode = false)
        {
            Debug.Assert(m_gsplatAssetID == 0);
            m_gsplatAssetID = gsplatAsset.GetInstanceID();
            m_gsplatAsset = gsplatAsset;
            m_poolMode = poolMode && gsplatAsset is GsplatAssetSpark;
            if (m_poolMode)
            {
                // Budget-sized owned resource (SplatCount here = the pool capacity the renderer
                // was created with). No full upload — Fill populates the selected set per frame.
                m_poolResource = new GsplatResourceSpark(SplatCount, gsplatAsset.SHBands);
                GsplatResource = m_poolResource;
                gsplatAsset.SetupMaterialPropertyBlock(m_propertyBlock, GsplatResource);
                return;
            }
            GsplatResource = GsplatResourceManager.Get(gsplatAsset);
            gsplatAsset.SetupMaterialPropertyBlock(m_propertyBlock, GsplatResource);
            if (asyncUpload)
                gsplatAsset.UploadDataAsync(GsplatResource);
            else
                gsplatAsset.UploadData(GsplatResource);
        }

        public void ReleaseGsplatAsset()
        {
            if (m_poolMode)
            {
                m_poolResource?.Dispose();
                m_poolResource = null;
            }
            else
            {
                GsplatResourceManager.Release(m_gsplatAssetID);
            }
            // P1.3: the allocator layout mirrors the pool buffers being released; a stale layout
            // over a freshly-recreated resource would skip every "resident" upload and draw
            // undefined GPU memory PERSISTENTLY (incremental never self-heals, unlike the legacy
            // full re-pack). Triggered by asset rebind with unchanged capacity — e.g.
            // GsplatImporter.ReloadAsset() during Play. Invalidate on every release; zeroing
            // m_remainingCount also defeats the camera-gate early return over the stale order.
            m_poolLayout?.Reset(0);
            m_poolTable = null;
            m_remainingCount = 0;
            GsplatResource = null;
            m_gsplatAsset = null;
            m_gsplatAssetID = 0;
            m_poolMode = false;
        }

        void CreateResources(uint splatCount)
        {
            OrderBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Append, (int)splatCount, sizeof(uint));
            SorterResource = GsplatSorter.Instance.CreateSorterResource(splatCount, OrderBuffer);
            m_cutoutsData = Array.Empty<GsplatCutout.ShaderData>();
            CutoutsBuffer = null;
            OrderSizeBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 1, sizeof(uint));
            BoundsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 6, sizeof(uint));
        }

        void CreatePropertyBlock()
        {
            m_propertyBlock ??= new MaterialPropertyBlock();
            m_propertyBlock.SetBuffer(k_orderBuffer, OrderBuffer);
            m_propertyBlock.SetFloat(k_lodFadeEnabled, 0f);   // off until the chunked path enables it
            // The draw shader DECLARES _SplatChunk/_SelectedLevel/_FadeWeight (read only when
            // _LodFadeEnabled>0.5). Metal (and other strict backends) reject a draw whose declared
            // StructuredBuffers are UNBOUND — producing ZERO fragments even though they're never
            // dynamically read. The combined chunked path binds real ones; the streaming-pool and
            // non-chunked paths never did → black render. Bind a valid dummy (OrderBuffer, itself a
            // StructuredBuffer<uint>) so no draw path is ever left with an unbound fade buffer.
            m_propertyBlock.SetBuffer(k_splatChunkBuffer, OrderBuffer);
            m_propertyBlock.SetBuffer(k_selectedLevelBuffer, OrderBuffer);
            m_propertyBlock.SetBuffer(k_fadeWeightBuffer, OrderBuffer);
        }

        public void Dispose()
        {
            ReleaseGsplatAsset();
            OrderBuffer?.Dispose();
            OrderBuffer = null;
            SorterResource?.Dispose();
            SorterResource = null;
            CutoutsBuffer?.Dispose();
            CutoutsBuffer = null;
            OrderSizeBuffer?.Dispose();
            OrderSizeBuffer = null;
            BoundsBuffer?.Dispose();
            BoundsBuffer = null;
            m_splatChunkBuffer?.Dispose();
            m_splatChunkBuffer = null;
            m_selectedLevelBuffer?.Dispose();
            m_selectedLevelBuffer = null;
            m_fadeWeightBuffer?.Dispose();
            m_fadeWeightBuffer = null;
            m_chunkTable = null;
            m_poolLiveMaskBuffer?.Dispose();
            m_poolLiveMaskBuffer = null;
            m_poolLayout = null;
            m_poolTable = null;
            m_streamLoader?.Clear();
            m_streamLoader = null;
        }

        public void ForceRefresh()
        {
            m_framesBeforeRecomputeSort = 0;
            m_sortsBeforeRecomputeCutouts = 0;
            // Also invalidate the cull-camera pose so the chunked/streaming cull + LOD selection +
            // pool residency re-run next frame (not just the sort). Otherwise a GsplatSettings tweak
            // that affects selection wouldn't apply until the camera moves past the refresh threshold.
            m_lastCullCamPos = new Vector3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity);
        }

        public void RefreshOnCameraMove()
        {
            foreach (var cam in Camera.allCameras)
            {
                var id = cam.GetInstanceID();
                if (m_prevCamTransforms.TryGetValue(id, out (Vector3, Vector3) prevCamTransform))
                {
                    (Vector3 prevCamPos, Vector3 prevCamRot) = prevCamTransform;

                    if ((cam.transform.position - prevCamPos).magnitude >
                        GsplatSettings.Instance.CameraTranslationRefreshTreshold
                        || (cam.transform.eulerAngles - prevCamRot).magnitude >
                        GsplatSettings.Instance.CameraRotationRefreshTreshold)
                    {
                        m_prevCamTransforms[id] = (cam.transform.position, cam.transform.eulerAngles);
                        ForceRefresh();
                    }
                }
                else
                {
                    m_prevCamTransforms.Add(cam.GetInstanceID(), (cam.transform.position, cam.transform.eulerAngles));
                    ForceRefresh();
                }
            }
        }

        public void EvaluateRefreshRequired(GsplatRenderer.GsplatSortMode mode, uint sortRefreshRate,
            uint cutoutsRefreshRate)
        {
            if (mode == GsplatRenderer.GsplatSortMode.Always)
            {
                sortRefreshRate = 0;
                cutoutsRefreshRate = 0;
            }

            if (mode == GsplatRenderer.GsplatSortMode.SortEveryNFrames)
            {
                cutoutsRefreshRate = 0;
            }

            RefreshOnCameraMove();

            ComputeSortRequired = false;
            ComputeCutoutsRequired = false;

            if (m_framesBeforeRecomputeSort == 0)
            {
                m_framesBeforeRecomputeSort = sortRefreshRate;
                ComputeSortRequired = true;
                if (m_sortsBeforeRecomputeCutouts == 0)
                {
                    m_sortsBeforeRecomputeCutouts = cutoutsRefreshRate;
                    ComputeCutoutsRequired = true;
                }
                else
                    m_sortsBeforeRecomputeCutouts -= 1;
            }
            else
                m_framesBeforeRecomputeSort -= 1;
        }

        /// <summary>
        /// Render the splats.
        /// </summary>
        /// <param name="transform">Object transform.</param>
        /// <param name="layer">Layer used for rendering.</param>
        /// <param name="gammaToLinear">Covert color space from Gamma to Linear.</param>
        /// <param name="shDegree">Order of SH coefficients used for rendering. The final value is capped by the SHBands property.</param>
        /// <param name="brightness">Brightness color scaling.</param>
        /// <param name="scaleFactor">Splats uv scaling factor, reduce splat size while trying to keep visual fidelity.</param>
        /// <param name="renderOrder">Manual render order placement of the gsplat. The final value is capped by the maximum render order setting.</param>
        public void Render(Transform transform, int layer, bool gammaToLinear = false, int shDegree = 3,
            float brightness = 1.0f, float scaleFactor = 1.0f, uint renderOrder = 0)
        {
            if (m_remainingCount <= 0)
                return;

            m_propertyBlock.SetInteger(k_splatCount, (int)m_remainingCount);
            m_propertyBlock.SetInteger(k_gammaToLinear, gammaToLinear ? 1 : 0);
            m_propertyBlock.SetInteger(k_splatInstanceSize, (int)GsplatSettings.Instance.SplatInstanceSize);
            m_propertyBlock.SetInteger(k_shDegree, Math.Min(m_gsplatAsset.SHBands, shDegree));
            m_propertyBlock.SetFloat(k_brightness, brightness);
            m_propertyBlock.SetFloat(k_scaleFactor, scaleFactor);
            m_propertyBlock.SetMatrix(k_matrixM, transform.localToWorldMatrix);

            // Per-splat cull gates (PC minPixelSize/minContribution) + forward alpha clip as
            // GLOBALS so the merged global draw (own material, no per-renderer block) picks them up.
            Shader.SetGlobalFloat(k_minPixelSize, GsplatSettings.Instance.MinPixelSize);
            Shader.SetGlobalFloat(k_minContribution, GsplatSettings.Instance.MinContribution);
            Shader.SetGlobalFloat(k_alphaClip, GsplatSettings.Instance.AlphaClipForward);
            Shader.SetGlobalFloat(k_foveationStrength, GsplatSettings.Instance.FoveationStrength);
            Shader.SetGlobalFloat(k_foveationCenter, GsplatSettings.Instance.FoveationCenter);

            uint order = Math.Clamp(renderOrder, 0, GsplatSettings.Instance.MaxRenderOrder - 1);
            var rp = new RenderParams(m_gsplatAsset.Materials[order])
            {
                worldBounds = GsplatUtils.CalcWorldBounds(m_bounds, transform),
                matProps = m_propertyBlock,
                layer = layer
            };

            Graphics.RenderMeshPrimitives(rp, GsplatSettings.Instance.Mesh, 0,
                Mathf.CeilToInt(m_remainingCount / (float)GsplatSettings.Instance.SplatInstanceSize));
        }
    }
}