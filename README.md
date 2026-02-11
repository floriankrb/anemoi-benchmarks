# Anemoi Benchmarks

Benchmarking read throughput of [anemoi-datasets](https://github.com/ecmwf/anemoi-datasets) Zarr stores, comparing **Zarr 2** vs **Zarr 3** across different parallelisation strategies and storage backends.

## Overview

The suite measures how fast data can be read from Zarr datasets using threads, processes, and PyTorch DataLoader (with DDP). It includes heat tracking to ensure benchmarks read cold (uncached) data.

See [BENCHMARK.md](BENCHMARK.md) for detailed results and methodology.
See [BENCHMARK_TORCH.md](BENCHMARK_TORCH.md) for more results using pytorch data loader.

## Datasets

The following dataset is publicly available and can be used to reproduce the benchmarks:

- `era5-o96-1979-2023-6h-v8.zarr` — [https://data.ecmwf.int/anemoi-datasets/era5-o96-1979-2023-6h-v8.zarr](https://data.ecmwf.int/anemoi-datasets/era5-o96-1979-2023-6h-v8.zarr)

The higher-resolution datasets (N320, O1280) used in some benchmarks are not yet publicly available.

## Quick Start

```bash
./run_test.sh <path-to-dataset.zarr> --mode threads --workers 1-2-4-8-16 -n 16
```

```bash
./run_test.sh <path-to-dataset.zarr> --mode processes --workers 1-2-4-8-16 -n 16
```

```bash
./run_test.sh <path-to-dataset.zarr> --mode torch --workers 1-2-4-8-16 -n 16 -g 4
```

Modes: `threads`, `processes`, `torch`, or `threads-processes`. Results are saved as JSONL files in `logs/`.

## Plotting

Use `plot.py` to visualise results. It reads the JSONL logs and produces PNG plots. Use `-K` to filter by dataset path and `-o` to set the output file:

```bash
./plot.py logs/* -K <path-to-dataset.zarr> -o results.png
```

Other useful options: `-k` for substring filtering (e.g. `-k "S3 | SSD"`), `--torch-only`, `--no-torch`.

## Simple benchmark

Additionally some simple benchmark tools can be found in simple_benchmark/*, no datasets needed.

```bash
 ./simple_benchmark/run.sh --path /path/to/directory --chunk-size 1GB
```

## Requirements

- [uv](https://github.com/astral-sh/uv) (dependencies are managed automatically via `uv run`)
