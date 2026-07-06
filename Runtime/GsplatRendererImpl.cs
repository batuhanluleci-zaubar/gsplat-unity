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
        Vector3 m_lastCullCamRot;

        // Chunked-LOD runtime state (see docs/CHUNKED_LOD_DESIGN.md, GsplatChunkTable).
        GsplatChunkTable m_chunkTable;
        GraphicsBuffer m_splatChunkBuffer;      // per-splat (chunkId<<4)|level tags
        GraphicsBuffer m_selectedLevelBuffer;   // per-chunk chosen level (updated each frame)
        uint[] m_selectedLevel;
        readonly Plane[] m_worldFrustumPlanes = new Plane[6];

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

        public void ComputeDepth(CommandBuffer cmd, Matrix4x4 matrixMv) =>
            m_gsplatAsset.ComputeDepth(cmd, matrixMv, SorterResource, GsplatResource);

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
                m_lastCullCamRot = cullCamera.transform.eulerAngles;
            }
        }

        // True if the cull camera moved/rotated past the (shared) refresh thresholds since
        // the last frustum InitOrder. Matches RefreshOnCameraMove's naive euler comparison.
        bool CameraMovedSinceLastCull(Camera cam)
        {
            var pos = cam.transform.position;
            var rot = cam.transform.eulerAngles;
            return (pos - m_lastCullCamPos).magnitude > GsplatSettings.Instance.CameraTranslationRefreshTreshold
                || (rot - m_lastCullCamRot).magnitude > GsplatSettings.Instance.CameraRotationRefreshTreshold;
        }

        // Builds the 6 camera frustum planes in the asset's object space (inward normals,
        // xyz normalized so _CullMargin is metric). For an object point x, world = M·x, so a
        // world plane P satisfies P·(M·x) = (Mᵀ·P)·x — hence the transpose pull-back.
        void BuildObjectSpaceFrustumPlanes(Camera cam, Matrix4x4 localToWorld)
        {
            GeometryUtility.CalculateFrustumPlanes(cam, m_frustumPlanes);
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
            var tags = table.BuildSplatChunkTags();
            m_splatChunkBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, tags.Length, sizeof(uint));
            m_splatChunkBuffer.SetData(tags);
            m_selectedLevel = new uint[table.ChunkCount];
            m_selectedLevelBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                Mathf.Max(1, table.ChunkCount), sizeof(uint));
        }

        // Chunked-LOD visible-set build: choose one LOD level per chunk — by FOV-compensated
        // distance from the camera to each chunk's AABB (PlayCanvas evaluateNodeLods bands
        // base*mult^i), or a fixed global level for debugging — optionally frustum-culling
        // chunks by their bounding sphere. Uploads the per-chunk selection, then dispatches
        // InitOrderChunked over the combined buffer; reuses existing depth+sort+draw.
        public void DispatchInitOrderChunked(GsplatChunkTable table, ComputeShader cs, Matrix4x4 matrixWorld,
            Camera camera, bool distanceLod, int fixedLevel, float baseDistance, float multiplier,
            bool cull, float cullMargin)
        {
            EnsureChunkSetup(table);
            ComputeSelectedLevels(table, matrixWorld, camera, distanceLod, fixedLevel,
                baseDistance, multiplier, cull);
            m_selectedLevelBuffer.SetData(m_selectedLevel);
            bool doCull = cull && camera != null;

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

            if (doCull)
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
        }

        // Pick one LOD level per chunk into m_selectedLevel (0xFFFFFFFF = culled): FOV-comp
        // distance bands base*mult^i, or a fixed level; optional bounding-sphere frustum cull.
        // Shared by the combined-buffer path (InitOrderChunked) and the streaming pool.
        void ComputeSelectedLevels(GsplatChunkTable table, Matrix4x4 matrixWorld, Camera camera,
            bool distanceLod, int fixedLevel, float baseDistance, float multiplier, bool cull)
        {
            if (m_selectedLevel == null || m_selectedLevel.Length != table.ChunkCount)
                m_selectedLevel = new uint[table.ChunkCount];
            int maxLod = table.MaxLod;
            bool doDistance = distanceLod && camera != null && maxLod > 0;
            bool doCull = cull && camera != null;
            if (doCull) GeometryUtility.CalculateFrustumPlanes(camera, m_worldFrustumPlanes);

            var ls = matrixWorld.lossyScale;
            float scale = Mathf.Max(ls.x, Mathf.Max(ls.y, ls.z));
            Vector3 camLocal = Vector3.zero;
            float fovScale = 1f, invLogMult = 1f;
            if (doDistance)
            {
                float tanHalfV = Mathf.Tan(camera.fieldOfView * 0.5f * Mathf.Deg2Rad);
                float tanHalfH = tanHalfV * camera.aspect;
                fovScale = Mathf.Min(tanHalfV, tanHalfH) / 0.41421356f; // tan(22.5°)
                camLocal = matrixWorld.inverse.MultiplyPoint3x4(camera.transform.position);
                invLogMult = 1f / Mathf.Log(Mathf.Max(1.0001f, multiplier));
                baseDistance = Mathf.Max(1e-4f, baseDistance);
            }

            for (int c = 0; c < table.ChunkCount; c++)
            {
                int req;
                if (doDistance)
                {
                    var aabb = table.Chunks[c].Aabb;
                    Vector3 closest = Vector3.Max(aabb.min, Vector3.Min(camLocal, aabb.max));
                    float d = Vector3.Distance(camLocal, closest) * scale * fovScale;
                    req = d < baseDistance ? 0
                        : Mathf.Clamp(1 + Mathf.FloorToInt(Mathf.Log(d / baseDistance) * invLogMult), 0, maxLod);
                }
                else req = Mathf.Clamp(fixedLevel, 0, maxLod);

                int use = req;
                while (use <= maxLod && table.Chunks[c].Lods[use].Count == 0) use++;
                if (use > maxLod) { m_selectedLevel[c] = 0xFFFFFFFFu; continue; }

                if (doCull)
                {
                    var sp = table.Chunks[c].Sphere;
                    Vector3 wc = matrixWorld.MultiplyPoint3x4(new Vector3(sp.x, sp.y, sp.z));
                    float wr = sp.w * scale;
                    bool inside = true;
                    for (int p = 0; p < 6; p++)
                        if (m_worldFrustumPlanes[p].GetDistanceToPoint(wc) < -wr) { inside = false; break; }
                    m_selectedLevel[c] = inside ? (uint)use : 0xFFFFFFFFu;
                }
                else m_selectedLevel[c] = (uint)use;
            }
        }

        // R4 streaming pool: rebuild the visible set on refresh by compacting only the
        // selected chunk-levels into the budget-sized pool resource (from CPU RAM), then let
        // the existing depth+sort+draw run over it (RemainingCount = packed count, identity
        // order). GPU holds ~budget splats instead of the whole combined buffer.
        public void DispatchChunkedPool(GsplatChunkTable table, Matrix4x4 matrixWorld, Camera camera,
            bool distanceLod, int fixedLevel, float baseDistance, float multiplier, bool cull)
        {
            // Rebuild the pool each refresh. (Camera-move gating is a later perf optimization;
            // re-filling every frame is correct, just extra SetData.)
            ComputeSelectedLevels(table, matrixWorld, camera, distanceLod, fixedLevel,
                baseDistance, multiplier, cull);
            uint visible = GsplatChunkPool.Fill((GsplatResourceSpark)GsplatResource,
                (GsplatAssetSpark)m_gsplatAsset, table, m_selectedLevel, out m_poolOverflow);
            m_remainingCount = visible;
            SorterResource.Initialized = false;   // re-fill identity order for the new pool contents
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
            m_chunkTable = null;
        }

        public void ForceRefresh()
        {
            m_framesBeforeRecomputeSort = 0;
            m_sortsBeforeRecomputeCutouts = 0;
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