// Copyright (c) 2025 Niantic Spatial
// SPDX-License-Identifier: MIT

using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Unity.Mathematics;
using UnityEngine;

namespace Gsplat
{
    public struct SpzPhaseTimings
    {
        public long DecompressMs;  // gzip decompression
        public long PackMs;        // per-splat decode + pack loop
    }

    // Loads SPZ files (https://github.com/nianticlabs/spz) and stores them in
    // Spark-compressed format, inheriting all rendering infrastructure from GsplatAssetSpark.
    public class GsplatAssetSpz : GsplatAssetSpark
    {
        // Atomic progress counter. Used as a bit mask to help local threads communicate progress back to the main threads progress bar.
        const int ProgressStride = 65536;

        // Constant decode state shared across all splats; passed by `in` to the per-splat helper.
        readonly struct DecodeContext
        {
            public readonly SpzData Data;
            public readonly bool Float16Pos;
            public readonly bool SmallestThree;
            public readonly byte FractionalBits;
            public readonly int ShDim;
            public readonly int ShBands;
            public readonly float PosXSign, PosYSign, PosZSign;
            public readonly float RotXSign, RotYSign, RotZSign;
            public readonly SourceCoordinates SrcCoords;

            public DecodeContext(SpzData data, SourceCoordinates srcCoords, int shBands)
            {
                Data = data;
                Float16Pos = data.Header.Version == 1;
                SmallestThree = data.Header.Version >= 3;
                FractionalBits = data.Header.FractionalBits;
                ShDim = SpzLoader.ShDim(data.Header.ShDegree);
                ShBands = shBands;
                SrcCoords = srcCoords;
                (PosXSign, PosYSign, PosZSign) = GsplatUtils.AxisSigns(srcCoords);
                RotXSign = PosYSign * PosZSign;
                RotYSign = PosXSign * PosZSign;
                RotZSign = PosXSign * PosYSign;
            }
        }

        public override void LoadFromPly(string plyPath, ProgressCallback progressCallback = null,
            SourceCoordinates sourceCoordinates = SourceCoordinates.RUF)
            => throw new NotSupportedException("GsplatAssetSpz loads SPZ files, not PLY.");

        public SpzPhaseTimings LoadFromSpz(string spzPath,
            SourceCoordinates sourceCoordinates = SourceCoordinates.RUB,
            ProgressCallback progressCallback = null)
        {
            var swDecompress = Stopwatch.StartNew();
            var data = SpzLoader.Load(spzPath);
            swDecompress.Stop();

            var h = data.Header;
            if (h.ShDegree > 4)
                throw new NotSupportedException($"SPZ SH degree {h.ShDegree} is not supported (max 4)");

            SplatCount = h.NumPoints;
            SHBands = h.ShDegree;
            int splatCount = (int)SplatCount;

            // The Allocate call is allocating the SH bands using the parent class
            Allocate();

            var ctx = new DecodeContext(data, sourceCoordinates, SHBands);
            // Band 4 has 9 coefficients × 3 channels; reused each splat. Sized for the
            // widest band so the same buffer serves bands 1–4.
            var tlShBand = new ThreadLocal<float[]>(() => new float[9 * 3]);

            var gMin = Vector3.positiveInfinity;
            var gMax = Vector3.negativeInfinity;
            var boundsLock = new object();

            // Shared counter incremented by worker threads; read by the main thread for progress.
            long processedCount = 0;
            const int progressMask = ProgressStride - 1;

            var swPack = Stopwatch.StartNew();

            // Run parallel work on thread pool so the calling (main) thread can update the
            // progress bar — EditorUtility.DisplayProgressBar requires the main thread.
            var packTask = Task.Run(() =>
                Parallel.For(
                    0, splatCount,
                    () => (min: Vector3.positiveInfinity, max: Vector3.negativeInfinity),
                    (i, _, localBounds) =>
                    {
                        var position = DecodeSplatInto(i, in ctx, tlShBand.Value,
                            PackedSplats, PackedSH1, PackedSH2, PackedSH3, PackedSH4);
                        localBounds.min = Vector3.Min(localBounds.min, position);
                        localBounds.max = Vector3.Max(localBounds.max, position);

                        // Bump shared counter every ProgressStride splats
                        if ((i & progressMask) == 0)
                            Interlocked.Add(ref processedCount, ProgressStride);

                        return localBounds;
                    },
                    localBounds =>
                    {
                        lock (boundsLock)
                        {
                            gMin = Vector3.Min(gMin, localBounds.min);
                            gMax = Vector3.Max(gMax, localBounds.max);
                        }
                    }));

            // Main thread polls the counter and drives the progress bar at ~10 fps.
            while (!packTask.IsCompleted)
            {
                if (progressCallback != null)
                {
                    float p = Math.Min(1f, Interlocked.Read(ref processedCount) / (float)splatCount);
                    progressCallback("Packing splats", p);
                }
                Thread.Sleep(100);
            }

            packTask.GetAwaiter().GetResult(); // re-throw any exception from worker threads
            swPack.Stop();
            tlShBand.Dispose();

            if (SplatCount > 0)
                Bounds = new Bounds((gMin + gMax) * 0.5f, gMax - gMin);

            progressCallback?.Invoke("Packing splats", 1f);

            return new SpzPhaseTimings
            {
                DecompressMs = swDecompress.ElapsedMilliseconds,
                PackMs = swPack.ElapsedMilliseconds,
            };
        }

        // GPU-ready packed result of decoding one standalone SPZ blob (disk-streaming path).
        // Same packed layout the pool/renderer consume; arrays are plain managed (no ScriptableObject),
        // so decoding runs on a worker thread. A null SH array means that band is absent.
        public struct BlobPacked
        {
            public uint4[] Packed;             // Count entries (word0 color, word1/2 f16 pos, word3 scale+quat)
            public uint[] SH1, SH2, SH3, SH4;  // 2/4/4/4 uint per splat when the band is present
            public int Count;
            public byte ShBands;
            public Bounds Bounds;
        }

        // Decode a standalone SPZ blob (a per-(chunk,level) .spz from the streaming container) into
        // plain packed arrays, entirely off the main thread (SpzLoader.Load(byte[]) + the same
        // per-splat pack path used by the combined load). No Unity Object, no temp file, no
        // progress poll / Thread.Sleep. Safe to call from Task.Run.
        public static BlobPacked DecodeBlobToPacked(byte[] spz,
            SourceCoordinates sourceCoordinates = SourceCoordinates.RUB)
        {
            var data = SpzLoader.Load(spz);
            var h = data.Header;
            if (h.ShDegree > 4)
                throw new NotSupportedException($"SPZ SH degree {h.ShDegree} is not supported (max 4)");
            int n = (int)h.NumPoints;
            byte shBands = h.ShDegree;
            var r = new BlobPacked
            {
                Count = n,
                ShBands = shBands,
                Packed = new uint4[n],
                SH1 = shBands >= 1 ? new uint[n * 2] : null,
                SH2 = shBands >= 2 ? new uint[n * 4] : null,
                SH3 = shBands >= 3 ? new uint[n * 4] : null,
                SH4 = shBands >= 4 ? new uint[n * 4] : null,
            };
            var ctx = new DecodeContext(data, sourceCoordinates, shBands);
            var tlShBand = new ThreadLocal<float[]>(() => new float[9 * 3]);
            Vector3 gMin = Vector3.positiveInfinity, gMax = Vector3.negativeInfinity;
            var boundsLock = new object();
            Parallel.For(0, n,
                () => (min: Vector3.positiveInfinity, max: Vector3.negativeInfinity),
                (i, _, lb) =>
                {
                    var pos = DecodeSplatInto(i, in ctx, tlShBand.Value, r.Packed, r.SH1, r.SH2, r.SH3, r.SH4);
                    lb.min = Vector3.Min(lb.min, pos);
                    lb.max = Vector3.Max(lb.max, pos);
                    return lb;
                },
                lb => { lock (boundsLock) { gMin = Vector3.Min(gMin, lb.min); gMax = Vector3.Max(gMax, lb.max); } });
            tlShBand.Dispose();
            r.Bounds = n > 0 ? new Bounds((gMin + gMax) * 0.5f, gMax - gMin) : default;
            return r;
        }

        // Decodes one SPZ splat and writes it into the given packed arrays (band arrays may be null
        // when absent). Returns the world-space position for bounds reduction. Pure managed math —
        // no Unity API — so it is shared by the combined LoadFromSpz and the streaming
        // DecodeBlobToPacked, and runs on worker threads.
        static Vector3 DecodeSplatInto(int i, in DecodeContext ctx, float[] shBandData,
            uint4[] packed, uint[] sh1, uint[] sh2, uint[] sh3, uint[] sh4)
        {
            var rawPos = ctx.Float16Pos
                ? SpzLoader.DecodePositionFloat16(ctx.Data.Positions, i)
                : SpzLoader.DecodePosition(ctx.Data.Positions, i, ctx.FractionalBits);
            var position = new Vector3(ctx.PosXSign * rawPos.x, ctx.PosYSign * rawPos.y, ctx.PosZSign * rawPos.z);

            var rgb = SpzLoader.DecodeColor(ctx.Data.Colors, i);
            var color = new Vector4(rgb.x, rgb.y, rgb.z, SpzLoader.DecodeAlphaLogit(ctx.Data.Alphas, i));
            var scale = SpzLoader.DecodeScaleLog(ctx.Data.Scales, i);

            var rawRot = SpzLoader.DecodeRotation(ctx.Data.Rotations, i, ctx.SmallestThree);
            // Apply per-axis sign flips for the chosen frame; same component mapping as the PLY path.
            var rotation = new Quaternion(
                rawRot.w,
                ctx.RotXSign * rawRot.x,
                ctx.RotYSign * rawRot.y,
                ctx.RotZSign * rawRot.z);

            packed[i] = PackSplat(color, position, scale, rotation);

            for (int j = 1, bandOffset = 0; j <= ctx.ShBands; j++)
            {
                int bandSize = j * 2 + 1;
                int baseCoeff = i * ctx.ShDim * 3;
                for (int k = 0; k < bandSize; k++)
                {
                    int off = baseCoeff + (bandOffset + k) * 3;
                    float sign = GsplatUtils.ShSign(ctx.SrcCoords, j, k);
                    shBandData[k * 3 + 0] = sign * SpzLoader.UnquantizeSH(ctx.Data.SH, off + 0);
                    shBandData[k * 3 + 1] = sign * SpzLoader.UnquantizeSH(ctx.Data.SH, off + 1);
                    shBandData[k * 3 + 2] = sign * SpzLoader.UnquantizeSH(ctx.Data.SH, off + 2);
                }

                if (j == 1) PackSH1(shBandData, sh1.AsSpan(i * 2, 2));
                if (j == 2) PackSH2(shBandData, sh2.AsSpan(i * 4, 4));
                if (j == 3) PackSH3(shBandData, sh3.AsSpan(i * 4, 4));
                if (j == 4) PackSH4(shBandData, sh4.AsSpan(i * 4, 4));

                bandOffset += bandSize;
            }

            return position;
        }
    }
}
