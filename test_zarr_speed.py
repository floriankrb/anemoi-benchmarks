#!/usr/bin/env python3
"""Test zarr read speed from S3 datasets using anemoi-datasets."""

import argparse
import datetime
import json
import os
import random
import re
import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import threading

import numpy as np
import zarr

from heat_tracker import HeatTracker, InsufficientColdDataError

# Thread-local storage for thread-safe dataset access
# Each thread gets its own dataset instance
_thread_local = threading.local()

# Global for process-based workers (ProcessPoolExecutor)
# Each process gets its own copy of this global
_process_ds = None


def _init_worker(path: str) -> None:
    """Initialise worker with open dataset.

    Called once per worker process/thread to avoid reopening the dataset for each read operation.
    Uses thread-local storage for threads, global for processes.
    """
    global _process_ds
    # from anemoi.datasets import open_dataset
    # ds = open_dataset(path)
    ds = zarr.open(path)['data']

    # Store in both locations - thread-local for threads, global for processes
    _thread_local.ds = ds
    _process_ds = ds


def _read_index_worker(args: tuple) -> tuple:
    """Worker function for reading k consecutive dates starting at an index.

    Uses thread-local dataset for thread-safety, falls back to process global.

    Parameters
    ----------
    args : tuple
        Tuple of (idx, k) where idx is the start index and k is the number
        of consecutive dates to read.

    Returns
    -------
    tuple
        (idx, t_read, nbytes) - start index, read time, and bytes read.
    """
    idx, k = args
    # Prefer thread-local (for ThreadPoolExecutor), fall back to process global (for ProcessPoolExecutor)
    ds = getattr(_thread_local, 'ds', None) or _process_ds
    t0 = time.perf_counter()
    arr = np.asarray(ds[idx : idx + k])
    t_read = time.perf_counter() - t0
    return idx, t_read, arr.nbytes


def path_to_filename(path: str) -> str:
    """Convert an S3 path to a safe filename."""
    # Remove s3:// prefix and replace special chars with dashes
    name = re.sub(r"^s3://", "", path)
    name = re.sub(r"[/.]", "-", name)
    name = re.sub(r"-+", "-", name)  # Collapse multiple dashes
    name = name.strip("-")
    return name


def test_speed(
    path: str,
    num_samples: int = 10,
    num_workers: int = 1,
    output_path: str = None,
    name: str = None,
    mode: str = "threads",
    heat_tracking: bool = True,
    num_dates: int = 1,
) -> dict:
    """Test read speed for an anemoi dataset on S3.

    Parameters
    ----------
    path : str
        S3 path to the zarr dataset.
    num_samples : int
        Total number of random time indices to read.
    num_workers : int
        Number of workers for parallel reading.
    output_path : str, optional
        Path to JSONL file for saving results. If provided, appends results.
    name : str, optional
        Name for this test run, saved in the results.
    mode : str
        Parallelism mode: 'threads' or 'processes'. Default is 'threads'.
    heat_tracking : bool
        If True, avoid reading previously-read ("hot") data by tracking
        read ranges in a JSONL file. Default is True.
    num_dates : int
        Number of consecutive dates to load per sample. Default is 1.

    Returns
    -------
    dict
        Timing results.
    """
    if mode not in ("threads", "processes"):
        raise ValueError(f"mode must be 'threads' or 'processes', got {mode!r}")
    
    # Generate run ID for this test run
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    seed = int(time.time()) % 10000
    random.seed(seed)
    np.random.seed(seed)

    print(f"Run ID: {run_id}")
    print(f"Zarr version: {zarr.__version__}")
    print(f"Opening: {path}")

    # from anemoi.datasets import open_dataset
    # ds = open_dataset(path)

    ds = zarr.open(path)['data']

    print(f"Dataset shape: {ds.shape}")
    n_times = ds.shape[0]
    ds = None  # Close dataset

    if num_dates > n_times:
        raise ValueError(
            f"num_dates ({num_dates}) cannot exceed dataset length ({n_times})"
        )

    # Select time indices
    # With multiple workers, each worker reads num_samples indices
    total_indices = num_samples * num_workers if num_workers > 1 else num_samples
    # Ensure we don't exceed valid start indices (need k consecutive dates)
    # Valid start indices are [0, n_times - num_dates] inclusive
    # So the count of valid indices is n_times - num_dates + 1
    num_valid_indices = n_times - num_dates + 1
    total_indices = min(total_indices, num_valid_indices)

    # Heat tracking: use sequential ranges to ensure cold data
    tracker = None
    heat_range = None
    if heat_tracking:
        tracker = HeatTracker(path)
        # Space start indices num_dates apart so that k-wide read windows
        # don't overlap — each data point is read exactly once.
        # This requires num_dates × total_indices contiguous cold start indices.
        range_needed = total_indices * num_dates
        if range_needed > num_valid_indices:
            total_indices = num_valid_indices // num_dates
            range_needed = total_indices * num_dates
        start_i, end_i = tracker.select_sequential_range(num_valid_indices, range_needed)
        heat_range = (start_i, end_i)
        # Pick every k-th index from the range, then shuffle for non-sequential I/O
        indices = list(range(start_i, end_i, num_dates))
        random.shuffle(indices)
        print(f"Heat tracking: reading range [{start_i}, {end_i}) step={num_dates}")
    else:
        # Without heat tracking, use random indices across the dataset
        indices = sorted(random.sample(range(num_valid_indices), total_indices))

    print(f"Reading {len(indices)} time indices (k={num_dates} consecutive dates each): {indices[:5]}...")
    mode_label = "processes" if mode == "processes" else "threads"
    print(f"Using {num_workers} {mode_label} ({num_samples} indices each, k={num_dates})")

    read_times = []
    total_bytes = 0

    # Choose executor based on mode
    ExecutorClass = ProcessPoolExecutor if mode == "processes" else ThreadPoolExecutor

    if num_workers == 1:
        # Sequential reads - initialise once, reuse dataset
        _init_worker(path)
        start = time.perf_counter()
        bar = tqdm.tqdm(enumerate(indices))
        for i, idx in bar:
            _, t_read, nbytes = _read_index_worker((idx, num_dates))
            read_times.append(t_read)
            total_bytes += nbytes
            elapsed = time.perf_counter() - start
            bar.set_description(f"Read {total_bytes / 1e6:.1f} MB in {elapsed:.1f}s -> {total_bytes / 1e6 / elapsed:.1f} MB/s")
    else:
        # Parallel reads - each worker initialises once with its own dataset
        with ExecutorClass(max_workers=num_workers, initializer=_init_worker, initargs=(path,)) as executor:
            futures = {executor.submit(_read_index_worker, (idx, num_dates)): idx for idx in indices}
            start = time.perf_counter()
            bar = tqdm.tqdm(as_completed(futures), total=len(futures))
            for future in bar:
                idx, t_read, nbytes = future.result()
                read_times.append(t_read)
                total_bytes += nbytes
                elapsed = time.perf_counter() - start
                bar.set_description(f"Read {total_bytes / 1e6:.1f} MB in {elapsed:.1f}s -> {total_bytes / 1e6 / elapsed:.1f} MB/s")
    total_time = time.perf_counter() - start

    # Summary statistics
    results = {
        "run_id": run_id,
        "dataset": path,
        "zarr_version": zarr.__version__,
        "num_threads": num_workers if mode == "threads" else 1,
        "num_processes": num_workers if mode == "processes" else 1,
        "num_dates_k": num_dates,
        "name": name,
        "num_reads": len(indices),
        "mean_time": float(np.mean(read_times)),
        "median_time": float(np.median(read_times)),
        "std_time": float(np.std(read_times)),
        "min_time": float(np.min(read_times)),
        "max_time": float(np.max(read_times)),
        "total_bytes": total_bytes,
        "total_time": total_time,
        "throughput_mbps": (total_bytes / 1e6) / total_time,
    }

    print("\n" + "=" * 50)
    print(f"RESULTS (zarr {zarr.__version__}, {num_workers} {mode_label})")
    print("=" * 50)
    print(f"Mean read time:     {results['mean_time']:.3f}s (+/- {results['std_time']:.3f}s)")
    print(f"Median read time:   {results['median_time']:.3f}s")
    print(f"Min/Max read time:  {results['min_time']:.3f}s / {results['max_time']:.3f}s")
    print(f"Confidence interval:  ± {1.96 * results['std_time'] / np.sqrt(results['num_reads']):.3f}s (95%)")
    print(f"Total data read:    {results['total_bytes'] / 1e6:.1f} MB")
    print(f"Total elapsed time: {results['total_time']:.3f}s")
    print(f"\033[93m Throughput:         {results['throughput_mbps']:.1f} MB/s \033[0m")
    print("=" * 50)
    print()
    # print(f'time ls {" ".join(paths)} | parallel -j {num_workers} dd if={{}} of=/dev/null bs=1M')
    print()

    # Save to JSONL if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "a") as f:
            f.write(json.dumps(results) + "\n")
        print(f"Results appended to: {output_path}")

    # Write heat tracking entry — record the full reserved range
    if heat_tracking and tracker is not None:
        config = {
            "num_samples": num_samples,
            "num_workers": num_workers,
            "mode": mode,
            "k": num_dates,
            "name": name,
        }
        tracker.write_entry(heat_range[0], heat_range[1], config=config)
        print(f"Heat tracking: recorded range [{heat_range[0]}, {heat_range[1]}) to {tracker.heat_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test anemoi-datasets read speed from S3")
    parser.add_argument("path", help="S3 path to zarr dataset")
    parser.add_argument("-n", "--num-samples", type=int, default=10, help="Total number of random time steps to read")
    parser.add_argument("-k", "--num-dates", type=int, default=1, help="Number of consecutive dates per sample (default: 1)")
    parser.add_argument("-w", "--workers", type=int, default=1, help="Number of workers for parallel reading")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="JSONL file path to save results. Default: logs/<dataset-name>.jsonl",
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Name for this test run, saved in results (e.g., HPC, Cloud)"
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["threads", "processes"],
        default="threads",
        help="Parallelism mode: 'threads' (default) or 'processes'. Use 'processes' to test without GIL contention.",
    )
    parser.add_argument(
        "--no-heat-tracking",
        action="store_true",
        help="Disable heat tracking (allow reading previously-read 'hot' data).",
    )
    args = parser.parse_args()

    # Generate default output path if not provided
    output_path = args.output
    if output_path is None:
        filename = path_to_filename(args.path)
        output_path = f"logs/{filename}.jsonl"

    test_speed(
        args.path,
        args.num_samples,
        args.workers,
        output_path,
        args.name,
        args.mode,
        heat_tracking=not args.no_heat_tracking,
        num_dates=args.num_dates,
    )


if __name__ == "__main__":
    main()
