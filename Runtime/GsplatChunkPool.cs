// Copyright (c) 2026 Zaubar
// SPDX-License-Identifier: MIT

using System.Collections.Generic;
using UnityEngine;

namespace Gsplat
{
    // R4 residency pool (see docs/CHUNKED_LOD_DESIGN.md §7b). The chunked-streaming renderer
    // owns a BUDGET-sized GsplatResourceSpark (a few M splats) instead of uploading the whole
    // combined asset (e.g. 19.3M). Each refresh the currently-selected per-chunk LOD intervals
    // are made resident in that resource, copied from the asset's CPU-resident combined arrays
    // (the combined .spz stays in RAM; only the budget lives on the GPU).
    //
    // Two fill strategies:
    //  - FillIncremental (default): PlayCanvas BlockAllocator analogue. Blocks stay PUT once
    //    allocated; a refresh uploads ONLY chunks whose selected level changed and frees the
    //    rest — head motion crossing the 0.2 m/10° gate costs KBs, not the whole pool. Holes
    //    are skipped at draw time via a per-slot liveness bitmask consumed by the
    //    InitOrderPool compute kernel (appended order, exactly like the combined chunked path).
    //    Allocation failure (fragmentation / genuine over-budget) triggers a full contiguous
    //    repack — that IS the defrag, amortized to "only when fragmented" instead of PC's
    //    incremental defrag machinery.
    //  - Fill (legacy, debug baseline): full-compaction re-upload of the entire selected set,
    //    contiguous identity layout.
    public static class GsplatChunkPool
    {
        // Per-renderer allocator state. Mirrors what the GPU pool buffers actually hold.
        public sealed class Layout
        {
            internal struct Block
            {
                public int Level, Offset, Count;
            }

            internal readonly Dictionary<int, Block> Resident = new();
            // Chunks that did not fit at their selected level on the last repack; retried only
            // when their selected level changes (prevents a repack-every-refresh loop while the
            // selection is over budget).
            internal readonly Dictionary<int, uint> Overflowed = new();
            // Free gaps sorted by offset: (offset, size). Coalesced on free.
            internal readonly List<(int Off, int Size)> Free = new();
            readonly List<int> m_scratch = new();

            public uint[] LiveMask { get; private set; } = System.Array.Empty<uint>();
            public int Capacity { get; private set; }
            public int UsedEnd { get; private set; }     // highest allocated end (dispatch bound)
            public int LiveCount { get; private set; }   // total resident splats
            public bool MaskDirty { get; set; }
            // Full contiguous repacks since creation (fragmentation/defrag telemetry — each one
            // costs a legacy-style full re-upload; frequent repacks under motion = tune budget).
            public int RepackCount { get; internal set; }
            // Holes exist iff the live splats don't fill [0, UsedEnd) contiguously.
            public bool Holey => LiveCount != UsedEnd;

            public void Reset(int capacity)
            {
                Resident.Clear();
                Overflowed.Clear();
                Free.Clear();
                Capacity = capacity;
                UsedEnd = 0;
                LiveCount = 0;
                if (capacity > 0)
                {
                    Free.Add((0, capacity));
                    int words = (capacity + 31) >> 5;
                    if (LiveMask.Length != words) LiveMask = new uint[words];
                    else System.Array.Clear(LiveMask, 0, LiveMask.Length);
                }
                else LiveMask = System.Array.Empty<uint>();
                MaskDirty = true;
            }

            internal int Allocate(int count)
            {
                for (int i = 0; i < Free.Count; i++)
                {
                    var (off, size) = Free[i];
                    if (size < count) continue;
                    if (size == count) Free.RemoveAt(i);
                    else Free[i] = (off + count, size - count);
                    return off;
                }
                return -1;
            }

            internal void Commit(int chunk, int level, int offset, int count)
            {
                Resident[chunk] = new Block { Level = level, Offset = offset, Count = count };
                SetMaskRange(offset, count, true);
                LiveCount += count;
                if (offset + count > UsedEnd) UsedEnd = offset + count;
                MaskDirty = true;
            }

            internal void FreeBlock(int chunk)
            {
                if (!Resident.TryGetValue(chunk, out var b)) return;
                Resident.Remove(chunk);
                SetMaskRange(b.Offset, b.Count, false);
                LiveCount -= b.Count;
                MaskDirty = true;
                InsertFree(b.Offset, b.Count);
                if (b.Offset + b.Count == UsedEnd)
                {
                    int end = 0;
                    foreach (var kv in Resident)
                        if (kv.Value.Offset + kv.Value.Count > end) end = kv.Value.Offset + kv.Value.Count;
                    UsedEnd = end;
                }
            }

            void InsertFree(int off, int size)
            {
                int lo = 0, hi = Free.Count;
                while (lo < hi)
                {
                    int mid = (lo + hi) >> 1;
                    if (Free[mid].Off < off) lo = mid + 1;
                    else hi = mid;
                }
                Free.Insert(lo, (off, size));
                // coalesce with the next gap, then the previous one
                if (lo + 1 < Free.Count && Free[lo].Off + Free[lo].Size == Free[lo + 1].Off)
                {
                    Free[lo] = (Free[lo].Off, Free[lo].Size + Free[lo + 1].Size);
                    Free.RemoveAt(lo + 1);
                }
                if (lo > 0 && Free[lo - 1].Off + Free[lo - 1].Size == Free[lo].Off)
                {
                    Free[lo - 1] = (Free[lo - 1].Off, Free[lo - 1].Size + Free[lo].Size);
                    Free.RemoveAt(lo);
                }
            }

            void SetMaskRange(int start, int count, bool value)
            {
                int end = start + count;
                int w0 = start >> 5, w1 = (end - 1) >> 5;
                for (int w = w0; w <= w1; w++)
                {
                    int lo = w == w0 ? start & 31 : 0;
                    int hi = w == w1 ? ((end - 1) & 31) + 1 : 32;
                    uint bits = (hi == 32 ? 0xFFFFFFFFu : (1u << hi) - 1u) & ~((1u << lo) - 1u);
                    if (value) LiveMask[w] |= bits;
                    else LiveMask[w] &= ~bits;
                }
            }

            internal List<int> Scratch()
            {
                m_scratch.Clear();
                return m_scratch;
            }
        }

        static void UploadBlock(GsplatResourceSpark res, GsplatAssetSpark asset, byte shBands,
            int srcOff, int dstOff, int cnt)
        {
            res.PackedSplatsBuffer.SetData(asset.PackedSplats, srcOff, dstOff, cnt);
            if (shBands >= 1) res.PackedSH1Buffer.SetData(asset.PackedSH1, 2 * srcOff, 2 * dstOff, 2 * cnt);
            if (shBands >= 2) res.PackedSH2Buffer.SetData(asset.PackedSH2, 4 * srcOff, 4 * dstOff, 4 * cnt);
            if (shBands >= 3) res.PackedSH3Buffer.SetData(asset.PackedSH3, 4 * srcOff, 4 * dstOff, 4 * cnt);
            if (shBands >= 4) res.PackedSH4Buffer.SetData(asset.PackedSH4, 4 * srcOff, 4 * dstOff, 4 * cnt);
        }

        // Incremental fill: diff the selected (chunk,level) set against what is already resident;
        // upload only entering/level-changed chunks, free leaving ones in place. Chunks are placed
        // WHOLE — a chunk that doesn't fit counts toward `overflow` (the balancer should keep the
        // selection within budget; overflow > 0 is the misconfiguration signal, same as legacy).
        // Returns the resident (live) splat total.
        public static uint FillIncremental(GsplatResourceSpark res, GsplatAssetSpark asset,
            GsplatChunkTable table, uint[] selectedLevel, Layout layout, out uint overflow)
        {
            int capacity = res.PackedSplatsBuffer.count;
            if (layout.Capacity != capacity) layout.Reset(capacity);
            byte shBands = asset.SHBands;

            // Phase 1: free blocks whose chunk is now culled or wants a different level.
            var toFree = layout.Scratch();
            foreach (var kv in layout.Resident)
            {
                int c = kv.Key;
                uint lvl = c < selectedLevel.Length ? selectedLevel[c] : 0xFFFFFFFFu;
                if (lvl != (uint)kv.Value.Level) toFree.Add(c);
            }
            for (int i = 0; i < toFree.Count; i++) layout.FreeBlock(toFree[i]);

            // Phase 2: allocate + upload newly-selected chunks.
            ulong over = 0;
            for (int c = 0; c < table.ChunkCount; c++)
            {
                uint lvl = c < selectedLevel.Length ? selectedLevel[c] : 0xFFFFFFFFu;
                if (lvl == 0xFFFFFFFFu || lvl >= (uint)table.Chunks[c].Lods.Length) continue;
                if (layout.Overflowed.TryGetValue(c, out uint oLvl))
                {
                    if (oLvl == lvl)
                    {
                        // Retry now — phase-1 frees may have opened space since the repack that
                        // blacklisted it. On failure keep skipping WITHOUT repacking, preserving
                        // the anti-repack-loop property (a repack could not help: it just failed).
                        var ivo = table.Chunks[c].Lods[(int)lvl];
                        int offo = ivo.Count > 0 ? layout.Allocate(ivo.Count) : -1;
                        if (offo < 0) { over += (ulong)ivo.Count; continue; }
                        UploadBlock(res, asset, shBands, ivo.Offset, offo, ivo.Count);
                        layout.Commit(c, (int)lvl, offo, ivo.Count);
                        layout.Overflowed.Remove(c);
                        continue;
                    }
                    layout.Overflowed.Remove(c);   // selection changed — retry below
                }
                if (layout.Resident.ContainsKey(c)) continue;   // same level already resident
                var iv = table.Chunks[c].Lods[(int)lvl];
                if (iv.Count <= 0) continue;
                int off = layout.Allocate(iv.Count);
                if (off < 0)
                    // Fragmented or over budget: full contiguous repack IS our defrag.
                    return RepackContiguous(res, asset, table, selectedLevel, layout, out overflow);
                UploadBlock(res, asset, shBands, iv.Offset, off, iv.Count);
                layout.Commit(c, (int)lvl, off, iv.Count);
            }

            res.UploadedCount = (uint)layout.UsedEnd;
            overflow = (uint)System.Math.Min(over, uint.MaxValue);
            return (uint)layout.LiveCount;
        }

        static uint RepackContiguous(GsplatResourceSpark res, GsplatAssetSpark asset,
            GsplatChunkTable table, uint[] selectedLevel, Layout layout, out uint overflow)
        {
            int capacity = res.PackedSplatsBuffer.count;
            int repacks = layout.RepackCount + 1;    // survives the Reset below
            layout.Reset(capacity);
            layout.RepackCount = repacks;
            layout.Free.Clear();                     // place directly; free list rebuilt as the tail
            byte shBands = asset.SHBands;
            int dst = 0;
            ulong over = 0;

            for (int c = 0; c < table.ChunkCount; c++)
            {
                uint lvl = c < selectedLevel.Length ? selectedLevel[c] : 0xFFFFFFFFu;
                if (lvl == 0xFFFFFFFFu || lvl >= (uint)table.Chunks[c].Lods.Length) continue;
                var iv = table.Chunks[c].Lods[(int)lvl];
                if (iv.Count <= 0) continue;
                if (dst + iv.Count > capacity)
                {
                    over += (ulong)iv.Count;
                    layout.Overflowed[c] = lvl;      // don't re-trigger a repack every refresh
                    continue;
                }
                UploadBlock(res, asset, shBands, iv.Offset, dst, iv.Count);
                layout.Commit(c, (int)lvl, dst, iv.Count);
                dst += iv.Count;
            }

            if (dst < capacity) layout.Free.Add((dst, capacity - dst));
            res.UploadedCount = (uint)dst;
            overflow = (uint)System.Math.Min(over, uint.MaxValue);
            return (uint)dst;
        }

        // Legacy full-compaction fill (debug baseline; also used while the global merged sort is
        // active, which assumes a contiguous identity-layout pool). Packs selected intervals
        // contiguously; the FIRST overflowing chunk may be placed partially (take < count).
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

                UploadBlock(res, asset, shBands, iv.Offset, dst, take);
                dst += take;
            }

            res.UploadedCount = (uint)dst;
            overflow = (uint)System.Math.Min(over, uint.MaxValue);
            return (uint)dst;
        }
    }
}
