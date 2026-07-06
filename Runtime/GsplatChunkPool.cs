// Copyright (c) 2026 Zaubar
// SPDX-License-Identifier: MIT

using UnityEngine;

namespace Gsplat
{
    // R4 residency pool (see docs/CHUNKED_LOD_DESIGN.md §7b). A fixed-capacity set of Spark
    // GPU buffers that, each refresh, is packed with ONLY the currently-selected per-chunk
    // LOD intervals — copied straight from the chunked asset's CPU arrays (the combined
    // .spz stays resident in RAM; only `Capacity` splats live on the GPU). This is the
    // work-buffer-compaction form of streaming: rebuild the visible set on camera-move
    // rather than a per-slot allocator. Turns a 19.3M-resident combined buffer into a
    // ~budget-resident pool.
    public class GsplatChunkPool
    {
        public GraphicsBuffer PackedSplatsBuffer { get; private set; }
        public GraphicsBuffer PackedSH1Buffer { get; private set; }
        public GraphicsBuffer PackedSH2Buffer { get; private set; }
        public GraphicsBuffer PackedSH3Buffer { get; private set; }

        public int Capacity { get; private set; }
        public uint VisibleCount { get; private set; }   // splats packed by the last Populate
        public uint OverflowCount { get; private set; }  // splats dropped because Capacity was hit

        readonly byte m_shBands;

        public GsplatChunkPool(int capacity, byte shBands)
        {
            Capacity = Mathf.Max(1, capacity);
            m_shBands = shBands;
            PackedSplatsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Capacity, sizeof(uint) * 4);
            if (shBands >= 1) PackedSH1Buffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Capacity, sizeof(uint) * 2);
            if (shBands >= 2) PackedSH2Buffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Capacity, sizeof(uint) * 4);
            if (shBands >= 3) PackedSH3Buffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Capacity, sizeof(uint) * 4);
        }

        // Compact the selected (chunk, level) intervals into the pool contiguously, copying
        // from the asset's CPU-resident combined arrays. selectedLevel[c] == 0xFFFFFFFF means
        // the chunk is culled/absent and is skipped. Fills up to Capacity; the remainder is
        // counted in OverflowCount (the budget/underfill signal). Sets VisibleCount.
        public void Populate(GsplatAssetSpark asset, GsplatChunkTable table, uint[] selectedLevel)
        {
            int dst = 0;
            ulong overflow = 0;
            for (int c = 0; c < table.ChunkCount; c++)
            {
                uint lvl = c < selectedLevel.Length ? selectedLevel[c] : 0xFFFFFFFFu;
                if (lvl == 0xFFFFFFFFu || lvl >= (uint)table.Chunks[c].Lods.Length) continue;
                var iv = table.Chunks[c].Lods[(int)lvl];
                int cnt = iv.Count;
                if (cnt <= 0) continue;

                int room = Capacity - dst;
                if (room <= 0) { overflow += (ulong)cnt; continue; }
                int take = cnt <= room ? cnt : room;
                if (take < cnt) overflow += (ulong)(cnt - take);

                PackedSplatsBuffer.SetData(asset.PackedSplats, iv.Offset, dst, take);
                if (PackedSH1Buffer != null) PackedSH1Buffer.SetData(asset.PackedSH1, 2 * iv.Offset, 2 * dst, 2 * take);
                if (PackedSH2Buffer != null) PackedSH2Buffer.SetData(asset.PackedSH2, 4 * iv.Offset, 4 * dst, 4 * take);
                if (PackedSH3Buffer != null) PackedSH3Buffer.SetData(asset.PackedSH3, 4 * iv.Offset, 4 * dst, 4 * take);

                dst += take;
            }
            VisibleCount = (uint)dst;
            OverflowCount = (uint)System.Math.Min(overflow, uint.MaxValue);
        }

        public void Dispose()
        {
            PackedSplatsBuffer?.Dispose(); PackedSplatsBuffer = null;
            PackedSH1Buffer?.Dispose(); PackedSH1Buffer = null;
            PackedSH2Buffer?.Dispose(); PackedSH2Buffer = null;
            PackedSH3Buffer?.Dispose(); PackedSH3Buffer = null;
        }
    }
}
