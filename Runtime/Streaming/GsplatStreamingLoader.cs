// Copyright (c) 2026 Zaubar
// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;

namespace Gsplat
{
    // Disk-streaming loader (S2): given a per-(chunk,level) .spz blob path, reads it async off the
    // main thread and decodes it on a worker into GPU-ready packed arrays (GsplatAssetSpz.BlobPacked,
    // S3). The pool (S4) requests loads for selected-but-not-resident chunk-levels and drains the
    // completed queue on the main thread to SetData them into pool slots.
    //
    // Threading contract: Request/Drain/Pump run on the MAIN thread only, so _pending/_inFlight/_queue
    // are single-threaded; only the completion enqueue crosses threads, via ConcurrentQueue. This
    // keeps the whole disk+decode cost off the render thread while the bookkeeping stays lock-free.
    public sealed class GsplatStreamingLoader
    {
        public struct Loaded
        {
            public int Chunk;
            public int Level;
            public GsplatAssetSpz.BlobPacked Blob;
        }

        readonly int m_maxInFlight;
        readonly SourceCoordinates m_coords;

        int m_inFlight;
        readonly HashSet<(int, int)> m_pending = new();          // queued OR loading (dedup key)
        readonly Queue<(int chunk, int level, string path)> m_queue = new();
        readonly ConcurrentQueue<Loaded> m_completed = new();
        readonly ConcurrentQueue<(int chunk, int level)> m_failed = new();

        public GsplatStreamingLoader(int maxInFlight = 2, SourceCoordinates coords = SourceCoordinates.RUB)
        {
            m_maxInFlight = Mathf.Max(1, maxInFlight);
            m_coords = coords;
        }

        // How many (chunk,level) loads are queued or in flight (for HUD / backpressure).
        public int PendingCount => m_pending.Count;

        public bool IsPending(int chunk, int level) => m_pending.Contains((chunk, level));

        // Request a (chunk,level) blob load. Deduped — a second request for an in-flight key is a
        // no-op. Kicks the pump so it starts immediately if a slot is free.
        public void Request(int chunk, int level, string path)
        {
            var key = (chunk, level);
            if (m_pending.Contains(key)) return;
            m_pending.Add(key);
            m_queue.Enqueue((chunk, level, path));
            Pump();
        }

        void Pump()
        {
            while (m_inFlight < m_maxInFlight && m_queue.Count > 0)
            {
                var (c, l, p) = m_queue.Dequeue();
                m_inFlight++;
                _ = LoadOne(c, l, p);
            }
        }

        async Task LoadOne(int chunk, int level, string path)
        {
            try
            {
                byte[] bytes = await ReadAllBytesAsync(path).ConfigureAwait(false);
                var blob = await Task.Run(() => GsplatAssetSpz.DecodeBlobToPacked(bytes, m_coords)).ConfigureAwait(false);
                m_completed.Enqueue(new Loaded { Chunk = chunk, Level = level, Blob = blob });
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[GsplatStreaming] load failed ({chunk},{level}) '{path}': {e.Message}");
                m_failed.Enqueue((chunk, level));
            }
        }

        static async Task<byte[]> ReadAllBytesAsync(string path)
        {
#if NETSTANDARD2_1 || NET_STANDARD_2_1 || UNITY_2021_2_OR_NEWER
            return await File.ReadAllBytesAsync(path).ConfigureAwait(false);
#else
            return await Task.Run(() => File.ReadAllBytes(path)).ConfigureAwait(false);
#endif
        }

        // Drain finished loads on the MAIN thread. For each successful load, invoke `onLoaded`
        // (which allocates a pool slot + SetData + Commit). Failures just clear their pending key so
        // the pool can re-request. Returns the number of successful loads applied this call. Frees
        // the in-flight slot and re-pumps so queued requests proceed.
        public int Drain(Action<Loaded> onLoaded, int maxApply = int.MaxValue)
        {
            int applied = 0;
            while (applied < maxApply && m_completed.TryDequeue(out var loaded))
            {
                m_pending.Remove((loaded.Chunk, loaded.Level));
                m_inFlight = Mathf.Max(0, m_inFlight - 1);
                try { onLoaded?.Invoke(loaded); }
                catch (Exception e) { Debug.LogError($"[GsplatStreaming] onLoaded threw: {e}"); }
                applied++;
            }
            while (m_failed.TryDequeue(out var f))
            {
                m_pending.Remove((f.chunk, f.level));
                m_inFlight = Mathf.Max(0, m_inFlight - 1);
            }
            Pump();
            return applied;
        }

        // Drop all queued (not-yet-started) requests and forget pending keys — call on rebind/reset.
        // In-flight loads still complete but are discarded on the next Drain (their key is gone, so
        // Drain removing a non-present key is harmless; the blob arrays are GC'd).
        public void Clear()
        {
            m_queue.Clear();
            m_pending.Clear();
            m_inFlight = 0;
            while (m_completed.TryDequeue(out _)) { }
            while (m_failed.TryDequeue(out _)) { }
        }
    }
}
