// Copyright (c) 2026 Zaubar
// SPDX-License-Identifier: MIT

using UnityEngine;

namespace Gsplat
{
    // Metadata-only Spark asset for the disk-streaming pool (S4 integration). Carries just the
    // scene's SHBands / SplatCount / Bounds so the pool GPU resource can be sized and the right
    // material variant chosen — but NEVER holds PackedSplats/PackedSH in RAM. The actual splats
    // live on disk as per-(chunk,level) .spz blobs and stream into the budget-sized GPU pool on
    // demand (GsplatChunkPool.FillStreaming + GsplatStreamingLoader). This is what bounds CPU RAM.
    //
    // BindGsplatAsset's pool branch already returns without an upload, so with PackedSplats==null
    // nothing is ever loaded into managed memory. Created at runtime from a streaming chunk table.
    public class GsplatAssetStreaming : GsplatAssetSpark
    {
        // Build a metadata-only asset from a parsed streaming chunk table. SplatCount is the full
        // combined count (for any tag/size math); the pool GPU buffer is sized to the renderer's
        // ChunkedPoolBudget, not this.
        public static GsplatAssetStreaming FromStreamingTable(GsplatChunkTable table, string name = "StreamingAsset")
        {
            if (table == null || !table.IsStreaming)
                throw new System.ArgumentException("FromStreamingTable requires a streaming GsplatChunkTable.");
            var a = CreateInstance<GsplatAssetStreaming>();
            a.name = name;
            a.SplatCount = (uint)table.CombinedSplatCount;
            a.SHBands = table.SHBands;
            a.Bounds = table.Bounds;
            // PackedSplats/PackedSH* deliberately stay null — never Allocate()/LoadFromSpz().
            return a;
        }
    }
}
