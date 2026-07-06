// Copyright (c) 2026 Zaubar
// SPDX-License-Identifier: MIT

using UnityEngine;

namespace Gsplat
{
    // R4 residency pool (see docs/CHUNKED_LOD_DESIGN.md §7b). The chunked-streaming renderer
    // owns a BUDGET-sized GsplatResourceSpark (a few M splats) instead of uploading the whole
    // combined asset (e.g. 19.3M). Each refresh, Fill() compacts ONLY the currently-selected
    // per-chunk LOD intervals into that resource, copied straight from the asset's CPU-resident
    // combined arrays (the combined .spz stays in RAM; only the budget lives on the GPU). The
    // existing depth+sort+draw then run over the pool resource unchanged.
    //
    // Work-buffer-compaction form of streaming: the visible set is rebuilt on camera-move
    // rather than via a per-slot allocator.
    public static class GsplatChunkPool
    {
        // Pack selected (chunk,level) intervals into `res` (a budget-sized Spark resource),
        // contiguously, from the asset's CPU arrays. selectedLevel[c]==0xFFFFFFFF => skip.
        // Fills up to the resource capacity; the remainder is returned in `overflow` (the
        // budget/underfill signal). Returns the number of splats packed (the visible count).
        public static uint Fill(GsplatResourceSpark res, GsplatAssetSpark asset,
            GsplatChunkTable table, uint[] selectedLevel, out uint overflow)
        {
            int capacity = res.PackedSplatsBuffer.count;
            byte shBands = asset.SHBands;
            int dst = 0;
            ulong over = 0;

            for (int c = 0; c < table.ChunkCount; c++)
            {
                uint lvl = c < selectedLevel.Length ? selectedLevel[c] : 0xFFFFFFFFu;
                if (lvl == 0xFFFFFFFFu || lvl >= (uint)table.Chunks[c].Lods.Length) continue;
                var iv = table.Chunks[c].Lods[(int)lvl];
                int cnt = iv.Count;
                if (cnt <= 0) continue;

                int room = capacity - dst;
                if (room <= 0) { over += (ulong)cnt; continue; }
                int take = cnt <= room ? cnt : room;
                if (take < cnt) over += (ulong)(cnt - take);

                res.PackedSplatsBuffer.SetData(asset.PackedSplats, iv.Offset, dst, take);
                if (shBands >= 1) res.PackedSH1Buffer.SetData(asset.PackedSH1, 2 * iv.Offset, 2 * dst, 2 * take);
                if (shBands >= 2) res.PackedSH2Buffer.SetData(asset.PackedSH2, 4 * iv.Offset, 4 * dst, 4 * take);
                if (shBands >= 3) res.PackedSH3Buffer.SetData(asset.PackedSH3, 4 * iv.Offset, 4 * dst, 4 * take);

                dst += take;
            }

            res.UploadedCount = (uint)dst;
            overflow = (uint)System.Math.Min(over, uint.MaxValue);
            return (uint)dst;
        }
    }
}
