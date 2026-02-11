#!/usr/bin/env python3
"""Probe cold vs hot read speed for a zarr dataset.

Reads a warmup chunk to prime zarr internals, then measures cold and
repeated hot reads on a target chunk.
"""

import argparse
import time

import numpy as np
import zarr


def probe_chunk(ds, chunk_idx: int, chunk_size: int, repeats: int = 5) -> dict:
    """Measure cold read then hot reads for a specific chunk.

    Parameters
    ----------
    ds : zarr.Array
        The zarr array to read from.
    chunk_idx : int
        Which chunk to read (0-based).
    chunk_size : int
        Size of each chunk in elements.
    repeats : int
        Number of hot re-reads.

    Returns
    -------
    dict
        Timing results.
    """
    start = chunk_idx * chunk_size
    end = start + chunk_size

    # Cold read
    t0 = time.perf_counter()
    arr = np.asarray(ds[start:end])
    cold_time = time.perf_counter() - t0
    nbytes = arr.nbytes
    cold_mbps = (nbytes / 1e6) / cold_time

    print(f"  Cold read: {cold_mbps:8.1f} MB/s 💬 ({cold_time:.4f}s, {nbytes / 1e6:.1f} MB)")

    # Hot reads (re-read same chunk)
    hot_times = []
    hot_speeds = []
    print(f"  Hot reads:")
    for i in range(repeats):
        t0 = time.perf_counter()
        arr = np.asarray(ds[start:end])
        t_read = time.perf_counter() - t0
        speed = (nbytes / 1e6) / t_read
        hot_times.append(t_read)
        hot_speeds.append(speed)
        print(f"{speed:8.1f} MB/s  ({t_read:.4f}s)")

    hot_avg_mbps = float(np.mean(hot_speeds))
    hot_ref_mbps = hot_speeds[0]  # first re-read
    ratio = hot_ref_mbps / cold_mbps if cold_mbps > 0 else float("inf")
    is_cold = ratio > 1.2

    # print(f"  Avg hot:    {hot_avg_mbps:8.1f} MB/s")
    # print(f"  2nd read:   {hot_ref_mbps:8.1f} MB/s  (used for ratio)")
    # print(f"  Ratio:      {ratio:8.1f}x  (2nd read / cold)")
    # status = (
    #     "COLD (first read significantly slower)"
    #     if is_cold
    #     else "HOT (data was already cached)"
    # )
    # print(f"  Status:     {status}")

    return {
        "chunk_idx": chunk_idx,
        "cold_mbps": cold_mbps,
        "cold_time_s": cold_time,
        "hot_ref_mbps": hot_ref_mbps,
        "hot_avg_mbps": hot_avg_mbps,
        "hot_min_mbps": float(np.min(hot_speeds)),
        "hot_max_mbps": float(np.max(hot_speeds)),
        "ratio": ratio,
        "is_cold": is_cold,
        "bytes_read": nbytes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Probe zarr read speed (cold vs hot)"
    )
    parser.add_argument("zarr_path", help="Path to the zarr dataset")
    parser.add_argument(
        "--warmup-chunk",
        type=int,
        default=0,
        help="Chunk index for warmup read (default: 0)",
    )
    parser.add_argument(
        "--test-chunk",
        type=int,
        default=5,
        help="Chunk index to probe (default: 5)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of hot re-reads (default: 5)",
    )
    args = parser.parse_args()

    print(f"Opening: {args.zarr_path}")
    ds = zarr.open(args.zarr_path)["data"]
    # print(f"  Shape: {ds.shape}, Chunks: {ds.chunks}, Dtype: {ds.dtype}")

    chunk_size = ds.chunks[0]

    # Warmup: prime zarr codecs, CPU caches, memory allocator
    print(f"--- Warmup read (chunk {args.warmup_chunk}) : ", end = '')
    warmup_start = args.warmup_chunk * chunk_size
    warmup_end = warmup_start + chunk_size
    t0 = time.perf_counter()
    warmup_arr = np.asarray(ds[warmup_start:warmup_end])
    warmup_time = time.perf_counter() - t0
    warmup_mbps = (warmup_arr.nbytes / 1e6) / warmup_time
    print(
        f"  {warmup_mbps:.1f} MB/s  ({warmup_time:.4f}s, "
        f"{warmup_arr.nbytes / 1e6:.1f} MB) — discarded"
    )

    # Probe the target chunk
    print(f"--- Probing chunk {args.test_chunk} ---")
    return probe_chunk(ds, args.test_chunk, chunk_size, args.repeats)


if __name__ == "__main__":
    main()
