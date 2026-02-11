#!/usr/bin/env python3
"""Create zarr test datasets with random data.

Creates a 1D array with 10 chunks of configurable size each (float64).
"""

import argparse
from pathlib import Path

import numpy as np
import zarr

# Chunk size in float64 elements for each human-readable size.
# Each float64 is 8 bytes.
CHUNK_SIZES = {
    "10MB": 10 * 1_000_000 // 8,       # 1,250,000
    "100MB": 100 * 1_000_000 // 8,     # 12,500,000
    "1GB": 1_000_000_000 // 8,         # 125,000,000
    "10GB": 10 * 1_000_000_000 // 8,   # 1,250,000,000
}

N_CHUNKS = 10


def create_dataset(path: str, zarr_format: int = 2, chunk_size: int = 1_250_000) -> None:
    """Create a zarr dataset filled with random data.

    Parameters
    ----------
    path : str
        Output path for the zarr store.
    zarr_format : int
        Zarr format version (2 or 3).
    chunk_size : int
        Number of float64 elements per chunk.
    """
    total_size = chunk_size * N_CHUNKS

    # zarr v3 accepts zarr_format to choose the on-disk format; v2 does not.
    major = int(zarr.__version__.split('.')[0])
    kwargs = {"zarr_format": zarr_format} if major >= 3 else {}

    root = zarr.open_group(path, mode="w", **kwargs)
    if major >= 3:
        data = root.create_array(
            "data",
            shape=(total_size,),
            chunks=(chunk_size,),
            dtype="float64",
        )
    else:
        data = root.create_dataset(
            "data",
            shape=(total_size,),
            chunks=(chunk_size,),
            dtype="float64",
        )

    rng = np.random.default_rng(42)
    for i in range(N_CHUNKS):
        start = i * chunk_size
        end = start + chunk_size
        data[start:end] = rng.random(chunk_size)

    chunk_mb = chunk_size * 8 / 1e6
    total_mb = total_size * 8 / 1e6
    print(f"Created zarr{zarr_format} dataset: {path}")
    print(f"  Shape: ({total_size},), Chunks: ({chunk_size},)")
    print(f"  {N_CHUNKS} chunks x {chunk_mb:.1f} MB = {total_mb:.1f} MB total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create zarr test datasets with random data")
    parser.add_argument("path", help="Output path for the zarr store")
    parser.add_argument("zarr_format", type=int, choices=[2, 3], help="Zarr format version (2 or 3)")
    parser.add_argument(
        "--chunk-size",
        default="10MB",
        choices=list(CHUNK_SIZES.keys()),
        help="Size per chunk (default: 10MB)",
    )
    args = parser.parse_args()

    create_dataset(args.path, args.zarr_format, CHUNK_SIZES[args.chunk_size])
