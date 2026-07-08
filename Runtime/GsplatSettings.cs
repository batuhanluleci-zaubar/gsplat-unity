// Copyright (c) 2025 Yize Wu
// SPDX-License-Identifier: MIT

using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace Gsplat
{
    public class GsplatSettings : ScriptableObject
    {
        const string k_gsplatSettingsResourcesPath = "GsplatSettings";

        const string k_gsplatSettingsPath =
            "Assets/Gsplat/Settings/Resources/" + k_gsplatSettingsResourcesPath + ".asset";

        static GsplatSettings s_instance;

        public static GsplatSettings Instance
        {
            get
            {
                if (s_instance)
                    return s_instance;

                var settings = Resources.Load<GsplatSettings>(k_gsplatSettingsResourcesPath);
#if UNITY_EDITOR
                if (!settings)
                {
                    var assetPath = Path.GetDirectoryName(k_gsplatSettingsPath);
                    if (!Directory.Exists(assetPath))
                        Directory.CreateDirectory(assetPath);

                    settings = CreateInstance<GsplatSettings>();
                    settings.Reset();
                    AssetDatabase.CreateAsset(settings, k_gsplatSettingsPath);
                    AssetDatabase.SaveAssets();
                }
                else
                {
                    if (settings.Version < new Version("1.2.0"))
                    {
                        Debug.Log($"Updated GsplatSettings from version {settings.Version}.");
                        settings.Materials = DefaultMaterials;
                        settings.GlobalMaterial = DefaultGlobalMaterial;
                        settings.m_prevComputeShader = null;
                        settings.Version = GsplatUtils.k_Version;
                        settings.OnValidate();
                        EditorUtility.SetDirty(settings);
                        AssetDatabase.SaveAssets();
                    }
                    else if (settings.Version < new Version("1.4.0"))
                    {
                        Debug.Log($"Updated GsplatSettings from version {settings.Version}.");
                        settings.GlobalMaterial = DefaultGlobalMaterial;
                        settings.Version = GsplatUtils.k_Version;
                        settings.OnValidate();
                        EditorUtility.SetDirty(settings);
                        AssetDatabase.SaveAssets();
                    }
                }
#endif

                s_instance = settings;
                return s_instance;
            }
        }

        public ComputeShader ComputeShader;
        public GsplatGlobalMaterial GlobalMaterial;

        [Tooltip(
            "When enabled, 2+ active Gaussian splat renderers are merged into a single globally depth-sorted draw call.")]
        public bool EnableGlobalSort;

        [Tooltip(
            "4a: when a pooled (incremental / disk-streaming) renderer participates in the 2+ renderer global merge, it "
            + "normally force-repacks its holey budget pool to a contiguous identity layout every refresh (CPU wholesale "
            + "re-upload). Enable this to instead let the global merge consume the holey pool directly (GPU-side, O(budget) "
            + "recopy on residency change). Default OFF; only affects scenes with 2+ SPARK renderers where at least one is "
            + "pooled. Single-renderer scenes are unaffected either way.")]
        public bool EnableGlobalSortOverPool;

        public uint SplatInstanceSize;
        public uint UploadBatchSize;
        [Range(1, 20)] public uint MaxRenderOrder;
        public bool DisplayBoundingBoxes;

        [Tooltip(
            "If a camera moves more that this threshold, each GsplatRenderer compute sorting and cutouts regardless of refresh rate")]
        [Range(0.05f, 1f)]
        public float CameraTranslationRefreshTreshold;

        [Tooltip(
            "If a camera rotates more that this threshold, each GsplatRenderer compute sorting and cutouts refresh regardless of refresh rate")]
        [Range(0.2f, 30f)]
        public float CameraRotationRefreshTreshold;

        [Tooltip("Chunked/streaming cull only: extra half-FOV (degrees) the frustum cull is widened by, " +
                 "so chunks/splats about to enter view are already drawn. The chunked cull SELECTION is " +
                 "only rebuilt when the camera moves past the refresh thresholds above; this margin keeps " +
                 "the drawn set valid in between, killing the 'walls/chunks vanish when you turn' stale-cull. " +
                 "Set >= CameraRotationRefreshTreshold. 0 = exact frustum (no cushion). ~4 deg is a good start; " +
                 "higher = safer but draws a wider border (fill cost).")]
        [Range(0f, 20f)]
        public float CullFrustumMarginDeg = 4f;

        public bool ShowImportErrors;

        [Tooltip("Discard splats whose projected diameter is below this many pixels (PlayCanvas " +
                 "minPixelSize). 2 = PlayCanvas default and this package's previous hardcoded " +
                 "behavior; 0 disables the gate. Applied in every splat vertex path (per-renderer " +
                 "and merged/global draws) via a global shader float.")]
        [Range(0f, 5f)]
        public float MinPixelSize = 2f;

        [Tooltip("Discard splats whose screen contribution (opacity x 2pi x sqrt|cov2D|) is below " +
                 "this (PlayCanvas minContribution). Kills large-but-faint splats that the pixel-" +
                 "size gate keeps. 3 = PlayCanvas default; their VR example ships 5 for fill-rate; " +
                 "0 disables.")]
        [Range(0f, 10f)]
        public float MinContribution = 3f;

        [Tooltip("Forward per-fragment alpha cutoff (PlayCanvas alphaClip). Fragments below this " +
                 "alpha are discarded AND each splat's quad is shrunk to where its Gaussian falls " +
                 "below it — a direct fill-rate lever. 0.00392 (=1/255) = default/lossless. Raise " +
                 "toward 0.0625 (=1/16) for the XR (Aura) fill-rate preset: visibly thins wispy " +
                 "low-opacity content but cuts overdraw. 0 falls back to 1/255.")]
        [Range(0f, 0.25f)]
        public float AlphaClipForward = 1f / 255f;

        [Tooltip("Foveated contribution cull (PlayCanvas foveation, P2.7): adds up to this much to " +
                 "MinContribution toward the screen EDGE, so peripheral low-contribution splats are " +
                 "dropped while the center (fovea) is untouched — a cheap XR fill-rate lever. 0 = off " +
                 "(exact no-op). Try 2-5 on XR; validate on device (per-eye NDC, only the far " +
                 "periphery is affected). Composes with fixed-foveated rendering.")]
        [Range(0f, 10f)]
        public float FoveationStrength = 0f;

        [Tooltip("Fovea radius in NDC (distance from screen center, 0..~1.4) where foveation starts " +
                 "ramping in. 0.3 = inner ~30% untouched. Only used when FoveationStrength > 0.")]
        [Range(0f, 1f)]
        public float FoveationCenter = 0.3f;

        public GsplatMaterial[] Materials;
        public Mesh Mesh { get; private set; }

        public bool Valid => Materials?.Length != 0 && Mesh && SplatInstanceSize > 0;

        public Version Version
        {
            get => Version.Parse(m_version);
            set => m_version = value.ToString();
        }

        ComputeShader m_prevComputeShader;
        uint m_prevSplatInstanceSize;

        [HideInInspector] [SerializeField] string m_version = "1.0.0";

#if UNITY_EDITOR
        static ComputeShader DefaultComputeShader => AssetDatabase.LoadAssetAtPath<ComputeShader>(
            GsplatUtils.k_PackagePath + "Runtime/Shaders/Gsplat.compute");

        static GsplatGlobalMaterial DefaultGlobalMaterial => AssetDatabase.LoadAssetAtPath<GsplatGlobalMaterial>(
            GsplatUtils.k_PackagePath + "Runtime/Materials/GsplatGlobal.asset");

        static GsplatMaterial[] DefaultMaterials
        {
            get
            {
                var materials = new GsplatMaterial[Enum.GetValues(typeof(CompressionMode)).Length];
                materials[(int)CompressionMode.Uncompressed] =
                    AssetDatabase.LoadAssetAtPath<GsplatMaterial>(GsplatUtils.k_PackagePath +
                                                                  "Runtime/Materials/GsplatUncompressed.asset");
                materials[(int)CompressionMode.Spark] =
                    AssetDatabase.LoadAssetAtPath<GsplatMaterial>(GsplatUtils.k_PackagePath +
                                                                  "Runtime/Materials/GsplatSpark.asset");
                return materials;
            }
        }

        public void Reset()
        {
            Version = GsplatUtils.k_Version;
            ComputeShader = DefaultComputeShader;
            GlobalMaterial = DefaultGlobalMaterial;
            Materials = DefaultMaterials;
            SplatInstanceSize = 128;
            UploadBatchSize = 100000;
            MaxRenderOrder = 1;
            DisplayBoundingBoxes = false;
            CameraTranslationRefreshTreshold = 0.2f;
            CameraRotationRefreshTreshold = 10;
            CullFrustumMarginDeg = 4f;
            ShowImportErrors = true;
            MinPixelSize = 2f;
            MinContribution = 3f;
            AlphaClipForward = 1f / 255f;
            FoveationStrength = 0f;
            FoveationCenter = 0.3f;

            m_prevComputeShader = null;
            m_prevSplatInstanceSize = 0;
            OnValidate();
        }
#endif

        void CreateMeshInstance()
        {
            var meshPositions = new Vector3[4 * SplatInstanceSize];
            var meshIndices = new int[6 * SplatInstanceSize];
            for (uint i = 0; i < SplatInstanceSize; ++i)
            {
                unsafe
                {
                    meshPositions[i * 4] = new Vector3(-1, -1, *(float*)&i);
                    meshPositions[i * 4 + 1] = new Vector3(1, -1, *(float*)&i);
                    meshPositions[i * 4 + 2] = new Vector3(-1, 1, *(float*)&i);
                    meshPositions[i * 4 + 3] = new Vector3(1, 1, *(float*)&i);
                }

                int b = (int)i * 4;
                Array.Copy(new[] { 0 + b, 1 + b, 2 + b, 1 + b, 3 + b, 2 + b }, 0, meshIndices, i * 6, 6);
            }

            Mesh = new Mesh
            {
                name = "GsplatMeshInstance",
                vertices = meshPositions,
                triangles = meshIndices,
                hideFlags = HideFlags.HideAndDontSave
            };
        }

        void OnValidate()
        {
            if (ComputeShader != m_prevComputeShader)
            {
                GsplatSorter.Instance.InitSorter(ComputeShader);
                m_prevComputeShader = ComputeShader;
            }

            GsplatSorter.Instance.InitGlobal(GlobalMaterial);

            if (SplatInstanceSize != m_prevSplatInstanceSize)
            {
                DestroyImmediate(Mesh);
                CreateMeshInstance();
                m_prevSplatInstanceSize = SplatInstanceSize;
            }
#if UNITY_EDITOR
            foreach (var mat in Materials)
            {
                mat.Reset();
            }

            if (GlobalMaterial)
                GlobalMaterial.Reset();

            // Live-apply while playing: an inspector tweak fires OnValidate, so push it to every
            // active renderer immediately. Per-frame shader gates (MinPixelSize/MinContribution/
            // AlphaClip/Foveation) and the freshly-read thresholds already take effect each frame;
            // ForceRefresh re-runs the camera-gated cull/LOD/residency/sort so selection-side changes
            // become visible without having to move the camera. Editor-only (OnValidate never runs in builds).
            if (Application.isPlaying)
            {
                GsplatSorter.Instance.MarkGlobalBuffersDirty();
                foreach (var r in FindObjectsByType<GsplatRenderer>(FindObjectsSortMode.None))
                    r.ForceRefresh();
            }
#endif
        }

        void OnEnable()
        {
            GsplatSorter.Instance.InitSorter(ComputeShader);
            GsplatSorter.Instance.InitGlobal(GlobalMaterial);
            m_prevComputeShader = ComputeShader;

            CreateMeshInstance();
            m_prevSplatInstanceSize = SplatInstanceSize;
        }
    }
}