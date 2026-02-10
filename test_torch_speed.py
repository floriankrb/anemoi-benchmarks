#!/usr/bin/env python3
"""Test PyTorch DataLoader speed with DDP for zarr datasets.

This script benchmarks data loading performance using PyTorch Lightning
with DDP (Distributed Data Parallel) across multiple GPUs.
"""

import argparse
import datetime
import json
import os
import re
import time

import numpy as np
import psutil
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import zarr
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from heat_tracker import HeatTracker, InsufficientColdDataError


class ZarrDataset(IterableDataset):
    """Iterable dataset that loads consecutive dates from a zarr dataset.

    Each sample loads k consecutive dates using ds[i:i+k].
    Data is sharded across GPUs and workers.
    """

    def __init__(
        self,
        path: str,
        k: int,
        m: int,
        num_workers: int = 1,
        rank: int = 0,
        world_size: int = 1,
        cold_range: tuple[int, int] | None = None,
    ):
        """Initialise the dataset.

        Parameters
        ----------
        path : str
            Path to the zarr dataset.
        k : int
            Number of consecutive dates to load per sample.
        m : int
            Number of samples per worker.
        num_workers : int
            Number of DataLoader workers per GPU.
        rank : int
            Global rank of this process.
        world_size : int
            Total number of processes.
        cold_range : tuple[int, int], optional
            Sequential range of cold indices to read from (start, end).
            If None, reads from the entire dataset using sharding.
        """
        super().__init__()
        self.path = path
        self.k = k
        self.m = m
        self.rank = rank
        self.world_size = world_size
        self.ds = None
        self.start_indices = None
        self.worker_id = 0
        self.num_workers = num_workers
        self.cold_range = cold_range

    def set_comm_group_info(self, rank: int, world_size: int) -> None:
        """Set communication group info from DDP strategy."""
        self.rank = rank
        self.world_size = world_size

    def per_worker_init(self, worker_id: int, num_workers: int) -> None:
        """Initialise dataset in each worker process.

        Opens the dataset and calculates the worker's data shard.

        Parameters
        ----------
        worker_id : int
            ID of this worker (0 to num_workers-1).
        num_workers : int
            Total number of workers per GPU.
        """
        from anemoi.datasets import open_dataset

        self.worker_id = worker_id
        self.num_workers = num_workers
        self.ds = open_dataset(self.path)

        # Determine the range to read from
        total_dates = self.ds.shape[0]
        max_start = total_dates - self.k

        if max_start <= 0:
            raise ValueError(
                f"Dataset has {total_dates} dates but k={self.k} requires at least {self.k + 1}"
            )

        if self.cold_range is not None:
            # The cold range is k-wide: pick every k-th index so that
            # k-wide read windows don't overlap (each data point read once).
            cold_start, cold_end = self.cold_range
            all_cold_indices = np.arange(cold_start, cold_end, self.k)

            # Shard across GPUs, then across workers within this GPU
            gpu_shards = np.array_split(all_cold_indices, self.world_size)
            worker_shards = np.array_split(gpu_shards[self.rank], num_workers)
            worker_indices = worker_shards[worker_id]

            available_indices = len(worker_indices)
            if available_indices <= 0:
                raise InsufficientColdDataError(
                    f"Worker {worker_id} on rank {self.rank} has no indices to sample from. "
                    f"Cold range: {self.cold_range}. Try requesting fewer samples or reset heat tracking."
                )

            rng = np.random.default_rng(seed=42 + self.rank * 1000 + worker_id)
            sample_size = min(self.m, available_indices)
            self.start_indices = rng.choice(
                worker_indices,
                size=sample_size,
                replace=False,
            )
        else:
            # No heat tracking: divide entire dataset across GPUs/workers
            gpu_shard_size = max_start // self.world_size
            gpu_start = self.rank * gpu_shard_size
            gpu_end = (self.rank + 1) * gpu_shard_size if self.rank < self.world_size - 1 else max_start

            # Divide across workers within this GPU
            worker_shard_size = (gpu_end - gpu_start) // num_workers
            worker_start = gpu_start + worker_id * worker_shard_size
            worker_end = gpu_start + (worker_id + 1) * worker_shard_size if worker_id < num_workers - 1 else gpu_end

            available_indices = worker_end - worker_start
            if available_indices <= 0:
                raise InsufficientColdDataError(
                    f"Worker {worker_id} on rank {self.rank} has no indices to sample from. "
                    f"Cold range: {self.cold_range}. Try requesting fewer samples or reset heat tracking."
                )

            # Select m random start indices from worker's shard
            rng = np.random.default_rng(seed=42 + self.rank * 1000 + worker_id)
            sample_size = min(self.m, available_indices)
            self.start_indices = rng.choice(
                np.arange(worker_start, worker_end),
                size=sample_size,
                replace=False,
            )
        print(
            f"[DEBUG] 🖥️ GPU {self.rank}/{self.world_size} 👷 Worker {worker_id}/{num_workers} "
            f"📦 samples={len(self.start_indices)}"
        )

    def __iter__(self):
        """Yield samples, each containing k consecutive dates and the start index."""
        if self.start_indices is None:
            raise RuntimeError("per_worker_init must be called before iterating")

        for i in self.start_indices:
            # Load k consecutive dates: ds[i:i+k]
            t0 = time.perf_counter()
            data = self.ds[i : i + self.k]
            t_zarr = time.perf_counter() - t0

            t1 = time.perf_counter()
            x = np.asarray(data)
            t_numpy = time.perf_counter() - t1

            t2 = time.perf_counter()
            tensor = torch.from_numpy(x)
            t_torch = time.perf_counter() - t2

            t_yield = time.time()  # Wall-clock time for cross-process comparison
            yield tensor, int(i), t_yield, t_zarr, t_numpy, t_torch

    def __len__(self):
        """Return the total number of samples for this rank (all workers combined)."""
        return self.m * self.num_workers


def worker_init_fn(_worker_id: int) -> None:
    """Initialise each DataLoader worker.

    Called by PyTorch DataLoader for each worker process.
    """
    t_start = time.perf_counter()
    worker_info = get_worker_info()
    if worker_info is None:
        raise RuntimeError("worker_init_fn called outside of DataLoader worker")

    dataset = worker_info.dataset
    dataset.per_worker_init(worker_info.id, worker_info.num_workers)
    t_init = time.perf_counter() - t_start
    print(f"[TIMING] Worker {worker_info.id} init took {t_init:.3f}s")


class ZarrDataModule(pl.LightningDataModule):
    """Lightning DataModule for zarr dataset loading benchmark."""

    def __init__(
        self,
        path: str,
        k: int,
        m: int,
        num_workers: int,
        prefetch_factor: int = 2,
        cold_range: tuple[int, int] | None = None,
    ):
        """Initialise the data module.

        Parameters
        ----------
        path : str
            Path to the zarr dataset.
        k : int
            Number of consecutive dates per sample.
        m : int
            Number of samples per worker.
        num_workers : int
            Number of workers per GPU.
        prefetch_factor : int
            Number of batches to prefetch per worker.
        cold_range : tuple[int, int], optional
            Sequential range of cold indices to read from (start, end).
        """
        super().__init__()
        self.path = path
        self.k = k
        self.m = m
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.cold_range = cold_range
        self.dataset = None

    def setup(self, stage: str = None) -> None:
        """Set up the dataset."""
        # Get rank and world_size from distributed environment if available
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        self.dataset = ZarrDataset(
            path=self.path,
            k=self.k,
            m=self.m,
            num_workers=self.num_workers,
            rank=rank,
            world_size=world_size,
            cold_range=self.cold_range,
        )

    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader."""
        return DataLoader(
            self.dataset,
            batch_size=None,  # Dataset yields individual samples
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class DataLoadingBenchmark(pl.LightningModule):
    """Minimal Lightning module for benchmarking data loading.

    No actual model or training - just measures data loading throughput.
    """

    def __init__(self, k: int = 1, run_id: str = None):
        """Initialise the benchmark module.

        Parameters
        ----------
        k : int
            Number of consecutive dates per sample (for heat tracking range calculation).
        run_id : str
            Unique identifier for this benchmark run (shared across all GPUs).
        """
        super().__init__()
        self.total_bytes = 0
        self.num_samples = 0
        self.start_time = None
        self.first_batch_time = None
        self.peak_memory_mb = 0.0
        self.k = k
        self.run_id = run_id
        self.min_index = None
        self.max_index = None
        # Timing accumulators
        self.total_t_zarr = 0.0
        self.total_t_numpy = 0.0
        self.total_t_torch = 0.0
        self.total_t_ipc = 0.0
        # Dummy parameter to satisfy Lightning
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def _get_process_tree_memory_mb(self) -> float:
        """Get total memory usage of this process and all its children."""
        try:
            proc = psutil.Process()
            total_rss = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total_rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total_rss / (1024 * 1024)  # Convert bytes to MB
        except Exception:
            return 0.0

    def training_step(self, batch, batch_idx):
        """Process a batch - just count bytes loaded."""
        t_received = time.time()  # Wall-clock time for cross-process comparison
        data, start_idx, t_yield, t_zarr, t_numpy, t_torch = batch

        # First batch latency (use perf_counter for same-process timing)
        if self.first_batch_time is None:
            self.first_batch_time = time.perf_counter()
            first_batch_latency = self.first_batch_time - self.start_time
            print(f"[TIMING] First batch latency: {first_batch_latency:.3f}s")

        # IPC + pin_memory overhead (time from yield in worker to receive in main)
        t_ipc = t_received - t_yield

        # Accumulate timing stats
        self.total_t_zarr += t_zarr
        self.total_t_numpy += t_numpy
        self.total_t_torch += t_torch
        self.total_t_ipc += t_ipc

        self.total_bytes += data.nbytes
        self.num_samples += 1
        # Track index range for heat tracking
        if self.min_index is None or start_idx < self.min_index:
            self.min_index = start_idx
        if self.max_index is None or start_idx > self.max_index:
            self.max_index = start_idx
        # Track peak memory including all child worker processes
        current_memory_mb = self._get_process_tree_memory_mb()
        self.peak_memory_mb = max(self.peak_memory_mb, current_memory_mb)
        # Return dummy loss to satisfy Lightning
        return torch.tensor(0.0, requires_grad=True)

    def configure_optimizers(self):
        """Return dummy optimiser."""
        return torch.optim.SGD([self.dummy], lr=0.0)

    def on_train_start(self):
        """Record start time and synchronise run_id across ranks."""
        self.start_time = time.perf_counter()

        # Synchronise run_id from rank 0 to all ranks (DDP spawn creates separate processes)
        if dist.is_initialized():
            # run_id format: YYYYMMDD_HHMMSS (15 chars)
            run_id_len = 15
            if dist.get_rank() == 0:
                run_id_tensor = torch.tensor(
                    [ord(c) for c in self.run_id.ljust(run_id_len)],
                    dtype=torch.int32,
                    device=self.device,
                )
            else:
                run_id_tensor = torch.zeros(run_id_len, dtype=torch.int32, device=self.device)
            dist.broadcast(run_id_tensor, src=0)
            self.run_id = "".join(chr(c) for c in run_id_tensor.tolist()).strip()

    def on_train_end(self):
        """Calculate and print throughput."""
        elapsed = time.perf_counter() - self.start_time

        # Final memory check including all child processes
        current_memory_mb = self._get_process_tree_memory_mb()
        self.peak_memory_mb = max(self.peak_memory_mb, current_memory_mb)

        # Get rank info (no IPC - each GPU keeps its own stats)
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Use local stats only (no all_gather/all_reduce)
        total_bytes = self.total_bytes
        num_samples = self.num_samples
        peak_memory_mb = self.peak_memory_mb
        local_min_index = self.min_index if self.min_index is not None else 0
        local_max_index = self.max_index if self.max_index is not None else 0

        # Each GPU prints its own results
        throughput_mbps = (total_bytes / 1e6) / elapsed if elapsed > 0 else 0
        print(f"\n[GPU {rank}] " + "=" * 44)
        print(f"[GPU {rank}] RESULTS")
        print(f"[GPU {rank}] " + "=" * 44)
        print(f"[GPU {rank}] Run ID:            {self.run_id}")
        print(f"[GPU {rank}] DDP world size:    {world_size}")
        print(f"[GPU {rank}] GPU rank:          {rank}")
        print(f"[GPU {rank}] " + "-" * 44)
        print(f"[GPU {rank}] Samples:           {num_samples}")
        print(f"[GPU {rank}] Data loaded:       {total_bytes / 1e6:.1f} MB")
        print(f"[GPU {rank}] Elapsed time:      {elapsed:.3f}s")
        print(f"[GPU {rank}] Peak memory (RSS): {peak_memory_mb:.1f} MB")
        print(f"[GPU {rank}] \033[93mThroughput:        {throughput_mbps:.1f} MB/s\033[0m")
        if num_samples > 0:
            print(f"[GPU {rank}] " + "-" * 44)
            print(f"[GPU {rank}] TIMING BREAKDOWN:")
            print(f"[GPU {rank}]   Zarr read:        {self.total_t_zarr:.3f}s ({self.total_t_zarr/num_samples*1000:.1f}ms/sample)")
            print(f"[GPU {rank}]   NumPy convert:    {self.total_t_numpy:.3f}s ({self.total_t_numpy/num_samples*1000:.1f}ms/sample)")
            print(f"[GPU {rank}]   Torch convert:    {self.total_t_torch:.3f}s ({self.total_t_torch/num_samples*1000:.1f}ms/sample)")
            print(f"[GPU {rank}]   IPC + pin_memory: {self.total_t_ipc:.3f}s ({self.total_t_ipc/num_samples*1000:.1f}ms/sample)")
        print(f"[GPU {rank}] " + "=" * 44)

        # Store for external access (local stats only)
        self.final_results = {
            "run_id": self.run_id,
            "gpu_id": rank,
            "total_bytes": total_bytes,
            "num_samples": num_samples,
            "elapsed": elapsed,
            "throughput_mbps": throughput_mbps,
            "peak_memory_mb": peak_memory_mb,
            "world_size": world_size,
            "min_index": local_min_index,
            "max_index": local_max_index,
            "timing_zarr_s": self.total_t_zarr,
            "timing_numpy_s": self.total_t_numpy,
            "timing_torch_s": self.total_t_torch,
            "timing_ipc_s": self.total_t_ipc,
        }


def path_to_filename(path: str) -> str:
    """Convert a path to a safe filename."""
    name = re.sub(r"^s3://", "", path)
    name = re.sub(r"[/.]", "-", name)
    name = re.sub(r"-+", "-", name)
    name = name.strip("-")
    return name


def main():
    """Run the data loading benchmark."""
    parser = argparse.ArgumentParser(
        description="Test PyTorch DataLoader speed with DDP for zarr datasets"
    )
    parser.add_argument("path", help="Path to zarr dataset")
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers per GPU (default: 4)",
    )
    parser.add_argument(
        "-g",
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs to use (default: 4)",
    )
    parser.add_argument(
        "-k",
        "--num-dates",
        type=int,
        default=4,
        help="Number of consecutive dates per sample (default: 4)",
    )
    parser.add_argument(
        "-m",
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples per worker (default: 10)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="JSONL file path to save results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name for this test run",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches to prefetch per worker (default: 2)",
    )
    parser.add_argument(
        "--no-heat-tracking",
        action="store_true",
        help="Disable heat tracking (allow reading previously-read 'hot' data).",
    )
    args = parser.parse_args()

    # Validate arguments early
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1 (DataLoader requires worker processes)")

    # Generate run ID before any DDP spawning (shared across all GPUs)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Run ID: {run_id}")
    print(f"Zarr version: {zarr.__version__}")
    print(f"Dataset: {args.path}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Workers per GPU: {args.num_workers}")
    print(f"Consecutive dates (k): {args.num_dates}")
    print(f"Samples per worker (m): {args.num_samples}")
    print(f"Prefetch factor: {args.prefetch_factor}")

    # Heat tracking: use sequential ranges to ensure cold data
    heat_tracking = not args.no_heat_tracking
    tracker = None
    cold_range = None

    if heat_tracking:
        # Calculate total samples needed to determine cold range
        # Note: we don't know world_size yet, so we estimate based on num_gpus
        # Each GPU has num_workers, each worker reads num_samples (m)
        total_samples_needed = args.num_gpus * args.num_workers * args.num_samples

        # Open dataset briefly to get size
        from anemoi.datasets import open_dataset
        ds = open_dataset(args.path)
        max_start = ds.shape[0] - args.num_dates
        ds = None

        tracker = HeatTracker(args.path)
        # Space start indices k apart so that k-wide read windows
        # don't overlap — each data point is read exactly once.
        range_needed = total_samples_needed * args.num_dates
        try:
            cold_range = tracker.select_sequential_range(max_start, range_needed)
            print(f"Heat tracking: reading cold range [{cold_range[0]}, {cold_range[1]}) step={args.num_dates}")
        except InsufficientColdDataError as e:
            print(f"ERROR: {e}")
            raise
    else:
        print("Heat tracking: disabled")

    # Create data module
    datamodule = ZarrDataModule(
        path=args.path,
        k=args.num_dates,
        m=args.num_samples,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        cold_range=cold_range,
    )

    # Create benchmark module
    model = DataLoadingBenchmark(k=args.num_dates, run_id=run_id)

    # Determine accelerator and strategy
    if args.num_gpus > 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices = args.num_gpus
        strategy = DDPStrategy() if args.num_gpus > 1 else "auto"
    elif args.num_gpus > 1:
        # Simulate multi-process DDP on CPU
        accelerator = "cpu"
        devices = args.num_gpus
        strategy = DDPStrategy()
        print(f"Warning: No GPUs available, simulating DDP with {args.num_gpus} CPU processes")
    else:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"
        print("Warning: No GPUs available, running on CPU")

    # Create trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        strategy=strategy,
        devices=devices,
        max_epochs=1,
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=False,
        enable_model_summary=False,
    )

    # Run benchmark
    trainer.fit(model, datamodule=datamodule)

    # Save results to JSONL - each GPU writes its own file
    output_path = args.output
    filename = path_to_filename(args.path)
    if output_path is None:
        output_path = f"logs/{filename}-torch-gpu{trainer.global_rank}.jsonl"
    else:
        # Insert GPU rank into provided output path
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}-gpu{trainer.global_rank}{ext}"

    # Each GPU saves its own results (no rank-0 guard)
    if hasattr(model, "final_results"):
        results = {
            "dataset": args.path,
            "zarr_version": zarr.__version__,
            "num_gpus": args.num_gpus,
            "num_workers": args.num_workers,
            "num_dates_k": args.num_dates,
            "num_samples_m": args.num_samples,
            "prefetch_factor": args.prefetch_factor,
            "name": args.name,
            **model.final_results,
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "a") as f:
            f.write(json.dumps(results) + "\n")
        print(f"[GPU {trainer.global_rank}] Results appended to: {output_path}")

        # Write heat tracking entry (rank 0 only to avoid duplicate writes)
        if trainer.global_rank == 0 and heat_tracking and tracker is not None and cold_range is not None:
            # Record the full k-spaced range we reserved from the cold pool.
            # With k-spacing the data coverage exactly fills [start, end).
            config = {
                "run_id": run_id,
                "num_gpus": args.num_gpus,
                "num_workers": args.num_workers,
                "k": args.num_dates,
                "m": args.num_samples,
                "name": args.name,
            }
            tracker.write_entry(cold_range[0], cold_range[1], config=config)
            print(f"[GPU 0] Heat tracking: recorded range [{cold_range[0]}, {cold_range[1]}) to {tracker.heat_file}")


if __name__ == "__main__":
    main()
