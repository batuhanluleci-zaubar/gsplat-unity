// Copyright (c) 2025 Yize Wu
// SPDX-License-Identifier: MIT

using UnityEditor;
using UnityEngine;

namespace Gsplat.Editor
{
    [CustomEditor(typeof(GsplatRenderer))]
    public class GsplatRendererEditor : UnityEditor.Editor
    {
        bool m_showPerChunk;

        // Repaint every frame in Play so the live per-chunk LOD panel tracks the camera.
        public override bool RequiresConstantRepaint() =>
            Application.isPlaying && ((GsplatRenderer)target).ChunkedLod;

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            var rendererTargetEarly = (GsplatRenderer)target;
            if (GsplatSettings.Instance.EnableGlobalSort
                && rendererTargetEarly.GsplatAsset
                && rendererTargetEarly.GsplatAsset.Compression == CompressionMode.Uncompressed)
            {
                EditorGUILayout.HelpBox(
                    "Global sort is enabled, but this renderer uses an uncompressed asset. " +
                    "Global sort is all-or-nothing: if any active renderer is uncompressed, the " +
                    "whole scene falls back to per-renderer rendering. Re-import with Spark " +
                    "compression to enable cross-renderer depth ordering.",
                    MessageType.Warning);
            }

            DrawPropertiesExcluding(serializedObject, "m_Script",
                nameof(GsplatRenderer.SHDegree),
                nameof(GsplatRenderer.AsyncUpload),
                nameof(GsplatRenderer.RenderBeforeUploadComplete),
                nameof(GsplatRenderer.Brightness),
                nameof(GsplatRenderer.SortMode)
            );

            // SH degree: slider max equals the bound asset's SHBands (so a degree-3 asset
            // shows 0–3, a degree-4 SPZ shows 0–4). Without an asset, fall back to 3.
            var rendererTarget = (GsplatRenderer)target;
            int maxShBands = rendererTarget.GsplatAsset ? rendererTarget.GsplatAsset.SHBands : 3;
            var shDegreeProp = serializedObject.FindProperty(nameof(GsplatRenderer.SHDegree));
            shDegreeProp.intValue = EditorGUILayout.IntSlider(
                "SH Degree", shDegreeProp.intValue, 0, maxShBands);

            var brightnessProp = serializedObject.FindProperty(nameof(GsplatRenderer.Brightness));
            float brightness = brightnessProp.floatValue;

            // Use log scale for the slider UX
            // range from -3 (~5%) to 3 (~20x)
            float logVal = UnityEngine.Mathf.Log(
                UnityEngine.Mathf.Max(0.001f, brightness)
            );
            logVal = EditorGUILayout.Slider("Log Brightness", logVal, -4.0f, 3.0f);
            brightness = EditorGUILayout.FloatField(
                "Brightness", UnityEngine.Mathf.Exp(logVal)
            );
            brightnessProp.floatValue = brightness;
            
            EditorGUILayout.PropertyField(serializedObject.FindProperty(nameof(GsplatRenderer.SortMode)));
            var renderer = (GsplatRenderer)target;
            // Sort Refresh Rate slider only if on correct mode
            if (renderer.SortMode == GsplatRenderer.GsplatSortMode.SortEveryNFrames ||
                renderer.SortMode == GsplatRenderer.GsplatSortMode.CutoutsEveryNSorts)
            {
                var newSortRefreshRate = (uint)EditorGUILayout.IntSlider(new GUIContent("Sort Refresh Rate"),
                    (int)renderer.SortRefreshRate, 1, 60);
                if (newSortRefreshRate != renderer.SortRefreshRate)
                {
                    renderer.SortRefreshRate = newSortRefreshRate;
                    renderer.ForceRefresh();
                }
            }

            // Cutouts Refresh Rate slider only if on correct mode
            if (renderer.SortMode == GsplatRenderer.GsplatSortMode.CutoutsEveryNSorts)
            {
                var newCutoutsRefreshRate = (uint)EditorGUILayout.IntSlider(new GUIContent("Cutouts Refresh Rate"),
                    (int)renderer.CutoutsRefreshRate, 1, 60);
                if (newCutoutsRefreshRate != renderer.CutoutsRefreshRate)
                {
                    renderer.CutoutsRefreshRate = newCutoutsRefreshRate;
                    renderer.ForceRefresh();
                }
            }
            
            var renderOrderProp = serializedObject.FindProperty(nameof(GsplatRenderer.RenderOrder));
            uint renderOrder = (uint)renderOrderProp.intValue;

            // RenderOrder slider depend on the MaxRenderOrder setting
            if (GsplatSettings.Instance.MaxRenderOrder > 1)
                renderOrderProp.intValue = EditorGUILayout.IntSlider(new GUIContent("Render Order"),
                    (int)renderOrder, 0, (int)GsplatSettings.Instance.MaxRenderOrder - 1);

            EditorGUILayout.PropertyField(serializedObject.FindProperty(nameof(GsplatRenderer.AsyncUpload)));
            if (serializedObject.FindProperty(nameof(GsplatRenderer.AsyncUpload)).boolValue)
            {
                EditorGUI.indentLevel++;
                EditorGUILayout.PropertyField(
                    serializedObject.FindProperty(nameof(GsplatRenderer.RenderBeforeUploadComplete)));
                EditorGUI.indentLevel--;
            }

            serializedObject.ApplyModifiedProperties();

            DrawChunkedLodDebug((GsplatRenderer)target);
        }

        // Live per-chunk LOD panel: a fill-bar histogram of chunks-per-level + a per-chunk
        // list, each bar colored green(fine)->red(coarse), gray=culled. Play mode only.
        void DrawChunkedLodDebug(GsplatRenderer r)
        {
            if (!r.ChunkedLod) return;

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Chunked LOD — Live", EditorStyles.boldLabel);
            if (!Application.isPlaying)
            {
                EditorGUILayout.HelpBox("Enter Play mode to see each chunk's selected LOD.", MessageType.Info);
                return;
            }
            var table = r.ChunkedTableRuntime;
            var sel = r.ChunkedSelectedLevels;
            if (table == null || sel == null || table.ChunkCount == 0)
            {
                EditorGUILayout.HelpBox("No chunk table loaded yet (waiting for upload).", MessageType.None);
                return;
            }

            int maxLod = table.MaxLod;
            var perLevel = new int[maxLod + 1];
            int culled = 0;
            long drawn = 0, lod0Total = 0;
            for (int c = 0; c < table.ChunkCount; c++)
            {
                lod0Total += table.Chunks[c].Lods[0].Count;
                uint lvl = c < sel.Length ? sel[c] : 0xFFFFFFFFu;
                if (lvl == 0xFFFFFFFFu) culled++;
                else if (lvl <= (uint)maxLod) { perLevel[lvl]++; drawn += table.Chunks[c].Lods[lvl].Count; }
            }

            float pct = lod0Total > 0 ? 100f * drawn / lod0Total : 0f;
            EditorGUILayout.LabelField($"{table.ChunkCount} chunks   drawn {drawn:N0} splats   ({pct:F1}% of full LOD0)");

            // Streaming pool: show the budget ceiling and whether R3 fit within it.
            if (r.ChunkedLod && r.ChunkedStreaming)
            {
                uint overflow = r.ChunkedPoolOverflow;
                float used = r.ChunkedPoolBudget > 0 ? 100f * drawn / r.ChunkedPoolBudget : 0f;
                string bal = r.ChunkedBudgetBalancer ? "balancer ON" : "balancer OFF";
                if (overflow > 0)
                    EditorGUILayout.HelpBox(
                        $"Pool budget {r.ChunkedPoolBudget:N0}  ({bal})  —  OVER by {overflow:N0} splats " +
                        "dropped past capacity. Raise ChunkedPoolBudget or enable the balancer.",
                        MessageType.Warning);
                else
                    EditorGUILayout.LabelField(
                        $"Pool budget {r.ChunkedPoolBudget:N0}   using {used:F0}%   ({bal}, fits)");
            }

            for (int L = 0; L <= maxLod; L++)
                DrawFill((float)perLevel[L] / table.ChunkCount,
                    GsplatRenderer.LodColor(L, maxLod), $"LOD {L}   {perLevel[L]} chunks");
            DrawFill((float)culled / table.ChunkCount, new Color(0.45f, 0.45f, 0.45f),
                $"Culled   {culled} chunks");

            m_showPerChunk = EditorGUILayout.Foldout(m_showPerChunk, "Per-chunk LOD");
            if (m_showPerChunk)
            {
                int shown = Mathf.Min(table.ChunkCount, 256);
                for (int c = 0; c < shown; c++)
                {
                    uint lvl = c < sel.Length ? sel[c] : 0xFFFFFFFFu;
                    bool cull = lvl == 0xFFFFFFFFu;
                    // Fill = detail: finest fills the bar, coarsest is nearly empty.
                    float frac = cull ? 0f : 1f - (maxLod > 0 ? (float)lvl / maxLod : 0f);
                    Color col = cull ? new Color(0.45f, 0.45f, 0.45f) : GsplatRenderer.LodColor((int)lvl, maxLod);
                    DrawFill(frac, col, cull ? $"chunk {c}   CULL" : $"chunk {c}   LOD {lvl}");
                }
                if (table.ChunkCount > shown)
                    EditorGUILayout.LabelField($"… +{table.ChunkCount - shown} more chunks");
            }
        }

        static void DrawFill(float frac, Color col, string label)
        {
            Rect r = EditorGUILayout.GetControlRect(false, 15);
            EditorGUI.DrawRect(r, new Color(0f, 0f, 0f, 0.15f));
            var fill = new Rect(r.x, r.y, r.width * Mathf.Clamp01(frac), r.height);
            EditorGUI.DrawRect(fill, col);
            EditorGUI.LabelField(new Rect(r.x + 4, r.y, r.width - 4, r.height), label, EditorStyles.miniLabel);
        }
    }
}