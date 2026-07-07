// Copyright (c) 2026 Zaubar
// SPDX-License-Identifier: MIT

using System;
using UnityEngine;

namespace Gsplat
{
    // Runtime chunk/LOD table for a combined chunked asset baked by
    // tools/bake_chunks.py. Parses the ".chunks.json" sidecar (a TextAsset) that
    // rides alongside the combined ".spz": one entry per spatial chunk, each with an
    // AABB, a bounding sphere (for GPU cull), the largest AABB extent (SSE input),
    // and per-LOD ABSOLUTE [offset,count) intervals into the combined PackedSplats
    // buffer (level 0 = finest). See docs/CHUNKED_LOD_DESIGN.md.
    //
    // The table also builds the per-splat "_SplatChunk" tag buffer ((chunkId<<4)|level)
    // consumed by InitOrderChunked.compute.
    public class GsplatChunkTable
    {
        // ---- JSON DTOs (Unity JsonUtility) ----
        // footR = exact per-level footprint radius (visible 2sigma core, p99.9, from the
        // baker) — the tight, LOD-correct cull radius. 0 when the sidecar predates it.
        [Serializable] class LodJson { public int level; public int offset; public int count; public float footR; }
        [Serializable] class ChunkJson { public float[] aabb; public float[] sphere; public float maxExtent; public LodJson[] lods; }
        [Serializable] class LevelJson { public int level; public int splatCount; }
        [Serializable] class ManifestJson
        {
            public string name;
            public int shBands;
            public float[] boundsMin;
            public float[] boundsMax;
            public int maxLod;
            public LevelJson[] levels;
            public ChunkJson[] chunks;
            public int combinedSplatCount;
        }

        public struct LodInterval { public int Level; public int Offset; public int Count; public float FootR; }

        public struct Chunk
        {
            public Bounds Aabb;
            public Vector4 Sphere;     // (center.xyz, radius)
            public float MaxExtent;
            public LodInterval[] Lods; // indexed by level; Count==0 => level absent
        }

        public int ChunkCount { get; private set; }
        public int MaxLod { get; private set; }
        public int CombinedSplatCount { get; private set; }
        public Bounds Bounds { get; private set; }
        public Chunk[] Chunks { get; private set; }

        public static GsplatChunkTable Parse(string json)
        {
            var m = JsonUtility.FromJson<ManifestJson>(json);
            if (m == null || m.chunks == null)
                throw new ArgumentException("Invalid chunk table JSON");

            var t = new GsplatChunkTable
            {
                ChunkCount = m.chunks.Length,
                MaxLod = m.maxLod,
                CombinedSplatCount = m.combinedSplatCount,
            };
            t.Bounds = MinMaxBounds(m.boundsMin, m.boundsMax);
            t.Chunks = new Chunk[m.chunks.Length];
            for (int i = 0; i < m.chunks.Length; i++)
            {
                var cj = m.chunks[i];
                var lods = new LodInterval[m.maxLod + 1];
                for (int L = 0; L <= m.maxLod; L++) lods[L] = new LodInterval { Level = L, Offset = 0, Count = 0 };
                if (cj.lods != null)
                    foreach (var lj in cj.lods)
                        if (lj.level >= 0 && lj.level <= m.maxLod)
                            lods[lj.level] = new LodInterval { Level = lj.level, Offset = lj.offset, Count = lj.count, FootR = lj.footR };
                t.Chunks[i] = new Chunk
                {
                    Aabb = MinMaxBounds(new[] { cj.aabb[0], cj.aabb[1], cj.aabb[2] },
                                        new[] { cj.aabb[3], cj.aabb[4], cj.aabb[5] }),
                    Sphere = new Vector4(cj.sphere[0], cj.sphere[1], cj.sphere[2], cj.sphere[3]),
                    MaxExtent = cj.maxExtent,
                    Lods = lods,
                };
            }
            return t;
        }

        // Per-splat tag buffer for InitOrderChunked: (chunkId << 4) | level.
        // Splats not covered by any chunk interval get 0xFFFFFFFF (never selected).
        public uint[] BuildSplatChunkTags()
        {
            var tags = new uint[CombinedSplatCount];
            for (int i = 0; i < tags.Length; i++) tags[i] = 0xFFFFFFFFu;
            for (int c = 0; c < Chunks.Length; c++)
            {
                var lods = Chunks[c].Lods;
                for (int L = 0; L < lods.Length; L++)
                {
                    int off = lods[L].Offset, cnt = lods[L].Count;
                    uint tag = ((uint)c << 4) | (uint)(L & 0xF);
                    for (int k = 0; k < cnt; k++) tags[off + k] = tag;
                }
            }
            return tags;
        }

        static Bounds MinMaxBounds(float[] mn, float[] mx)
        {
            var b = new Bounds();
            b.SetMinMax(new Vector3(mn[0], mn[1], mn[2]), new Vector3(mx[0], mx[1], mx[2]));
            return b;
        }
    }
}
