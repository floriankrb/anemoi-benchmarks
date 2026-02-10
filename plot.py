#!/usr/bin/env -S uv run --python=3.11
# /// script
# dependencies = ["matplotlib", "pandas", "seaborn"]
# ///

"""Plot zarr throughput benchmarks using Seaborn/Matplotlib.

Supports multiple modes:
- threads: Multi-threaded reads
- processes: Multi-process reads
- torch: PyTorch DataLoader with GPUs

Creates PNG image output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter


def make_name_filter(expr: str) -> Callable[[tuple], bool]:
    """Create a filter function from an expression.

    Supports:
    - Simple terms: 'foo' matches if 'foo' is in any field
    - Negation: 'not foo' matches if 'foo' is NOT in any field
    - AND: 'foo and bar' matches if both terms match
    - OR: 'foo | bar' matches if either term matches

    Parameters
    ----------
    expr
        Filter expression string.

    Returns
    -------
    Callable
        Function that takes a tuple of fields and returns True if it matches.
    """
    expr = expr.strip().lower()

    def matches(key_tuple: tuple) -> bool:
        fields = [str(f).lower() for f in key_tuple]

        def match_term(term: str) -> bool:
            term = term.strip().lower()
            if term.startswith("not "):
                term_val = term[4:].strip()
                return all(term_val not in f for f in fields)
            return any(term in f for f in fields)

        for group in expr.split("|"):
            terms = [t.strip() for t in group.split(" and ")]
            if all(match_term(t) for t in terms):
                return True
        return False

    return matches


def make_exact_filter(expr: str) -> Callable[[tuple], bool]:
    """Create a filter function from an expression using exact matching.

    Supports:
    - Simple terms: 'foo' matches if 'foo' equals any field exactly
    - Negation: 'not foo' matches if 'foo' does NOT equal any field
    - AND: 'foo and bar' matches if both terms match
    - OR: 'foo | bar' matches if either term matches

    Parameters
    ----------
    expr
        Filter expression string.

    Returns
    -------
    Callable
        Function that takes a tuple of fields and returns True if it matches.
    """
    expr = expr.strip().lower()

    def matches(key_tuple: tuple) -> bool:
        fields = [str(f).lower() for f in key_tuple]

        def match_term(term: str) -> bool:
            term = term.strip().lower()
            if term.startswith("not "):
                term_val = term[4:].strip()
                return all(term_val != f for f in fields)
            return any(term == f for f in fields)

        for group in expr.split("|"):
            terms = [t.strip() for t in group.split(" and ")]
            if all(match_term(t) for t in terms):
                return True
        return False

    return matches


@dataclass
class BenchmarkData:
    """Container for benchmark data backed by a pandas DataFrame."""

    df: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def from_jsonl(cls, paths: list[str]) -> BenchmarkData:
        """Load benchmark data from one or more JSONL files.

        For torch mode with per-GPU output files, results are aggregated
        by run_id to compute total throughput across all GPUs.

        Parameters
        ----------
        paths
            List of paths to JSONL files.

        Returns
        -------
        BenchmarkData
            Loaded benchmark data.
        """
        raw_rows = []
        for path in paths:
            if not path.endswith(".jsonl"):
                print(f"Warning: Skipping non-JSONL file: {path}")
                continue
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    raw_rows.append(row)

        # Separate torch per-GPU rows (have gpu_id) from other rows
        torch_per_gpu = [r for r in raw_rows if r.get("gpu_id") is not None]
        other_rows = [r for r in raw_rows if r.get("gpu_id") is None]

        records = []

        # Process non-torch rows directly
        for row in other_rows:
            records.append(cls._parse_row(row))

        # Aggregate torch per-GPU rows by run_id
        if torch_per_gpu:
            from collections import defaultdict
            by_run = defaultdict(list)
            for row in torch_per_gpu:
                run_id = row.get("run_id", "unknown")
                by_run[run_id].append(row)

            for run_id, gpu_rows in by_run.items():
                # Aggregate: sum throughput, take config from first row
                first = gpu_rows[0]
                total_throughput_mbps = sum(r.get("throughput_mbps", 0) for r in gpu_rows)
                total_bytes = sum(r.get("total_bytes", 0) for r in gpu_rows)
                total_samples = sum(r.get("num_samples", 0) for r in gpu_rows)
                # Use max elapsed time (all GPUs run in parallel)
                max_elapsed = max(r.get("elapsed", 0) for r in gpu_rows)

                # Build aggregated row
                agg_row = {
                    "run_id": run_id,
                    "name": first.get("name"),
                    "dataset": first.get("dataset", ""),
                    "zarr_version": first.get("zarr_version", ""),
                    "num_gpus": first.get("world_size", len(gpu_rows)),
                    "num_workers": first.get("num_workers", 0),
                    "num_dates_k": first.get("num_dates_k", 1),
                    "prefetch_factor": first.get("prefetch_factor", 2),
                    "throughput_mbps": total_throughput_mbps,
                    "total_bytes": total_bytes,
                    "num_samples": total_samples,
                    "elapsed": max_elapsed,
                }
                records.append(cls._parse_row(agg_row))

        return cls(pd.DataFrame(records))

    @staticmethod
    def _parse_row(row: dict) -> dict:
        """Parse a single JSONL row into a standardised record."""
        num_gpus = int(row.get("num_gpus", 0))
        num_workers_per_gpu = int(row.get("num_workers", 0))
        num_processes = int(row.get("num_processes", 1))
        num_threads = int(row.get("num_threads", 1))

        if num_gpus > 0:
            mode = "torch"
            total_workers = num_gpus * num_workers_per_gpu
        elif num_processes > 1:
            mode = "processes"
            total_workers = num_processes
        else:
            mode = "threads"
            total_workers = num_threads

        version = row["zarr_version"]
        zarr_key = "zarr2" if version.startswith("2") else "zarr3"

        return {
            "name": row.get("name", "Unknown"),
            "dataset": row.get("dataset", ""),
            "zarr_version": zarr_key,
            "zarr_version_full": version,
            "mode": mode,
            "num_gpus": num_gpus,
            "workers_per_gpu": num_workers_per_gpu,
            "num_dates_k": int(row.get("num_dates_k", 1)),
            "prefetch_factor": int(row.get("prefetch_factor", 2)),
            "total_workers": total_workers,
            "throughput_gbps": float(row["throughput_mbps"]) / 1024,
        }

    def add_config_labels(self) -> BenchmarkData:
        """Add human-readable config labels like '4w/1k' for torch mode."""
        if self.df.empty:
            return self

        def make_label(row):
            if row["mode"] == "torch":
                return f"{row['workers_per_gpu']}w/{row['num_dates_k']}k"
            return row["name"]

        self.df["config_label"] = self.df.apply(make_label, axis=1)

        # Add separate labels for k (number of dates) and workers
        self.df["k_label"] = self.df["num_dates_k"].apply(lambda k: f"k={k}")
        self.df["workers_label"] = self.df["workers_per_gpu"].apply(
            lambda w: f"{w} workers"
        )

        return self

    def filter(self, expr: str) -> BenchmarkData:
        """Filter data using an expression.

        Parameters
        ----------
        expr
            Filter expression (see make_name_filter for syntax).

        Returns
        -------
        BenchmarkData
            Filtered data.
        """
        if self.df.empty:
            return self

        matcher = make_name_filter(expr)
        mask = self.df.apply(
            lambda row: matcher((
                row["name"],
                row["dataset"],
                row["zarr_version"],
                row["mode"],
                f"[k={row['num_dates_k']}]",
                f"[gpu={row['num_gpus']}]",
            )),
            axis=1,
        )
        return BenchmarkData(self.df[mask].copy())

    def filter_exact(self, expr: str) -> BenchmarkData:
        """Filter data using an expression with exact matching.

        Parameters
        ----------
        expr
            Filter expression (see make_exact_filter for syntax).

        Returns
        -------
        BenchmarkData
            Filtered data.
        """
        if self.df.empty:
            return self

        matcher = make_exact_filter(expr)
        mask = self.df.apply(
            lambda row: matcher((
                row["name"],
                row["dataset"],
                row["zarr_version"],
                row["mode"],
                f"[k={row['num_dates_k']}]",
                f"[gpu={row['num_gpus']}]",
            )),
            axis=1,
        )
        return BenchmarkData(self.df[mask].copy())

    @property
    def names(self) -> list[str]:
        """Get unique benchmark names."""
        if self.df.empty:
            return []
        return sorted(self.df["name"].dropna().unique())

    @property
    def datasets(self) -> set[str]:
        """Get unique dataset paths."""
        if self.df.empty:
            return set()
        return set(self.df["dataset"].dropna().unique())

    @property
    def modes(self) -> list[str]:
        """Get modes present in data, in canonical order."""
        order = ["threads", "processes", "torch"]
        if self.df.empty:
            return []
        present = set(self.df["mode"].unique())
        return [m for m in order if m in present]

    @property
    def gpu_counts(self) -> list[int]:
        """Get unique GPU counts for torch mode, sorted."""
        if self.df.empty:
            return []
        torch_df = self.df[self.df["mode"] == "torch"]
        if torch_df.empty:
            return []
        return sorted(torch_df["num_gpus"].unique())


def extract_dataset_name(dataset_path: str) -> str:
    """Extract a readable name from a dataset path."""
    name = os.path.basename(dataset_path.rstrip("/"))
    return name[:-5] if name.endswith(".zarr") else name


class ThroughputPlotter:
    """Seaborn/Matplotlib-based plotter for throughput benchmarks."""

    def __init__(self, data: BenchmarkData):
        self.data = data
        self.fig: plt.Figure | None = None
        self.axes: list[plt.Axes] = []

    def plot_torch_by_gpus(
        self,
        title: str | None = None,
    ) -> ThroughputPlotter:
        """Create faceted plot with one subplot per GPU count.

        Parameters
        ----------
        title
            Optional title override.

        Returns
        -------
        ThroughputPlotter
            Self for method chaining.
        """
        df = self.data.df[self.data.df["mode"] == "torch"].copy()
        if df.empty:
            print("No torch mode data found")
            return self

        # Create GPU label for faceting
        df["gpu_label"] = df["num_gpus"].apply(
            lambda x: f"Torch ({x} GPU{'s' if x != 1 else ''})"
        )

        # Sort by GPU count for consistent subplot order
        gpu_order = [
            f"Torch ({x} GPU{'s' if x != 1 else ''})"
            for x in sorted(df["num_gpus"].unique())
        ]

        # Build title from dataset names if not provided
        if title is None:
            dataset_names = [
                extract_dataset_name(d) for d in self.data.datasets if d
            ]
            title = ", ".join(sorted(set(dataset_names))) or "Zarr Throughput"
            # Add benchmark name(s) to title
            if self.data.names:
                title += f"\n({', '.join(self.data.names)})"

        self._create_faceted_plot(
            df=df,
            facet_col="gpu_label",
            facet_order=gpu_order,
            title=title,
            x_label="Total workers (workers_per_gpu × num_gpus)",
        )

        return self

    def plot_threads_processes(
        self,
        title: str | None = None,
    ) -> ThroughputPlotter:
        """Create plot for threads and processes modes combined.

        Parameters
        ----------
        title
            Optional title override.

        Returns
        -------
        ThroughputPlotter
            Self for method chaining.
        """
        df = self.data.df[self.data.df["mode"].isin(["threads", "processes"])].copy()
        if df.empty:
            print("No threads/processes mode data found")
            return self

        # Combine threads and processes into single "CPU" facet
        df["facet"] = "CPU"

        if title is None:
            dataset_names = [
                extract_dataset_name(d) for d in self.data.datasets if d
            ]
            title = ", ".join(sorted(set(dataset_names))) or "Zarr Throughput"
            # Add benchmark name(s) to title
            if self.data.names:
                title += f"\n({', '.join(self.data.names)})"

        self._create_faceted_plot(
            df=df,
            facet_col="facet",
            facet_order=["CPU"],
            title=title,
            x_label="Workers",
        )

        return self

    def plot_all_modes(
        self,
        title: str | None = None,
    ) -> ThroughputPlotter:
        """Create a combined plot with all modes.

        Parameters
        ----------
        title
            Optional title override.

        Returns
        -------
        ThroughputPlotter
            Self for method chaining.
        """
        modes = self.data.modes
        if not modes:
            print("No data found")
            return self

        has_torch = "torch" in modes
        has_tp = "threads" in modes or "processes" in modes

        if has_torch and has_tp:
            return self._plot_combined(title)
        elif has_torch:
            return self.plot_torch_by_gpus(title)
        else:
            return self.plot_threads_processes(title)

    def _plot_combined(
        self,
        title: str | None,
    ) -> ThroughputPlotter:
        """Create combined plot with threads/processes overlaid on torch subplots."""
        df = self.data.df.copy()

        # Build facet column for torch data only
        def facet_label(row):
            if row["mode"] == "torch":
                n = row["num_gpus"]
                return f"Torch ({n} GPU{'s' if n != 1 else ''})"
            return None  # threads/processes will be overlaid

        df["facet"] = df.apply(facet_label, axis=1)

        # Build facet order (torch only)
        facet_order = []
        for n in self.data.gpu_counts:
            facet_order.append(f"Torch ({n} GPU{'s' if n != 1 else ''})")

        if title is None:
            dataset_names = [
                extract_dataset_name(d) for d in self.data.datasets if d
            ]
            title = ", ".join(sorted(set(dataset_names))) or "Zarr Throughput"
            # Add benchmark name(s) to title
            if self.data.names:
                title += f"\n({', '.join(self.data.names)})"

        # Get threads/processes data for overlay
        tp_df = self.data.df[
            self.data.df["mode"].isin(["threads", "processes"])
        ].copy()

        self._create_faceted_plot(
            df=df[df["facet"].notna()],  # Only torch data for facets
            facet_col="facet",
            facet_order=facet_order,
            title=title,
            x_label="Total workers (workers_per_gpu × num_gpus)",
            overlay_df=tp_df,  # Pass threads/processes for overlay
        )

        return self

    def _create_faceted_plot(
        self,
        df: pd.DataFrame,
        facet_col: str,
        facet_order: list[str],
        title: str,
        x_label: str,
        overlay_df: pd.DataFrame | None = None,
    ) -> None:
        """Create a faceted scatter plot with median lines.

        Creates a grid with rows for each k value (num_dates_k) and
        columns for each facet (GPU count).
        Threads/processes data is overlaid on each torch subplot.
        Colours and line styles distinguish zarr versions (zarr2=solid, zarr3=dashed).

        Parameters
        ----------
        overlay_df
            Optional DataFrame with threads/processes data to overlay on each subplot.
        """
        num_cols = len(facet_order)
        if num_cols == 0:
            return

        # Get unique k values for rows
        k_values = sorted(df["num_dates_k"].unique())
        num_rows = len(k_values)

        # Set style
        sns.set_theme(style="whitegrid")

        # Create figure with subplots (rows=k values, cols=facets)
        fig_width = 5 * num_cols
        fig_height = 5 * num_rows
        self.fig, axes = plt.subplots(
            num_rows, num_cols,
            figsize=(fig_width, fig_height),
            sharey=False,
            sharex=False,  # Each subplot has its own x-axis
            squeeze=False,
        )
        self.axes = axes.flatten()

        # Colour palette for zarr versions (used in torch mode)
        # Use same colour for both zarr2 and zarr3 (distinguish by linestyle only)
        zarr_versions = sorted(df["zarr_version"].unique())
        zarr_colour = sns.color_palette("tab10")[0]  # Blue colour
        zarr_colour_map = {v: zarr_colour for v in zarr_versions}

        # Colour palette for modes (used for threads/processes overlay)
        mode_colours = {"threads": "gray", "processes": "black"}
        mode_labels = {"threads": "Threads", "processes": "Processes"}

        # Line styles for zarr versions (no marker distinction)
        zarr_linestyles = {"zarr2": "-", "zarr3": "--"}
        zarr_labels = {"zarr2": "Zarr 2", "zarr3": "Zarr 3"}

        # Track global y-range
        global_min_y = float("inf")
        global_max_y = 0

        # Compute global x-axis range (all workers from torch + overlay)
        global_all_workers = set(df["total_workers"].unique())
        if overlay_df is not None and not overlay_df.empty:
            global_all_workers.update(overlay_df["total_workers"].unique())
        global_all_workers = sorted(global_all_workers)

        for row_idx, k_val in enumerate(k_values):
            for col_idx, facet_val in enumerate(facet_order):
                ax = axes[row_idx, col_idx]
                facet_df = df[(df[facet_col] == facet_val) & (df["num_dates_k"] == k_val)]

                if facet_df.empty:
                    ax.set_visible(False)
                    continue

                # Update y-range
                global_min_y = min(global_min_y, facet_df["throughput_gbps"].min())
                global_max_y = max(global_max_y, facet_df["throughput_gbps"].max())

                # Check if this is a CPU facet (threads/processes only, no torch)
                is_cpu_facet = facet_val == "CPU"

                # Get number of GPUs for this facet (for torch facets)
                match = re.search(r"\((\d+)\s*GPU", facet_val)
                num_gpus = int(match.group(1)) if match else 0

                # Collect all worker values for x-axis ticks
                all_workers = set(facet_df["total_workers"].unique())

                if is_cpu_facet:
                    # CPU facet: colour by mode, linestyle by zarr version
                    # Get threads data at workers=1 (shared starting point)
                    threads_w1 = {}
                    for zarr_ver in facet_df["zarr_version"].unique():
                        threads_subset = facet_df[
                            (facet_df["mode"] == "threads") &
                            (facet_df["zarr_version"] == zarr_ver) &
                            (facet_df["total_workers"] == 1)
                        ]
                        if not threads_subset.empty:
                            threads_w1[zarr_ver] = threads_subset["throughput_gbps"].median()

                    for mode in sorted(facet_df["mode"].unique()):
                        for zarr_ver in sorted(facet_df["zarr_version"].unique()):
                            subset = facet_df[
                                (facet_df["mode"] == mode) &
                                (facet_df["zarr_version"] == zarr_ver)
                            ]
                            if subset.empty:
                                continue

                            colour = mode_colours.get(mode, "gray")
                            linestyle = zarr_linestyles.get(zarr_ver, "-")

                            ax.scatter(
                                subset["total_workers"],
                                subset["throughput_gbps"],
                                c=[colour],
                                marker="o",
                                s=50,
                                alpha=0.8,
                                edgecolors="white",
                                linewidths=0.5,
                            )

                            # Add median line
                            grouped = subset.groupby("total_workers")["throughput_gbps"].median()

                            # For processes, prepend threads workers=1 as shared start
                            if mode == "processes" and zarr_ver in threads_w1:
                                if 1 not in grouped.index:
                                    grouped = pd.concat([
                                        pd.Series([threads_w1[zarr_ver]], index=[1]),
                                        grouped
                                    ]).sort_index()

                            if len(grouped) >= 2:
                                workers_list = grouped.index.tolist()
                                medians = grouped.values.tolist()
                                ax.plot(
                                    workers_list, medians,
                                    color=colour,
                                    linestyle=linestyle,
                                    linewidth=2,
                                    alpha=0.8,
                                )
                else:
                    # Torch facet: colour by zarr version
                    for zarr_ver in facet_df["zarr_version"].unique():
                        subset = facet_df[facet_df["zarr_version"] == zarr_ver]
                        colour = zarr_colour_map[zarr_ver]
                        linestyle = zarr_linestyles.get(zarr_ver, "-")

                        ax.scatter(
                            subset["total_workers"],
                            subset["throughput_gbps"],
                            c=[colour],
                            marker="o",
                            s=50,
                            alpha=0.8,
                            edgecolors="white",
                            linewidths=0.5,
                            label=zarr_labels.get(zarr_ver, zarr_ver),
                        )

                        # Add median line per zarr version
                        grouped = subset.groupby("total_workers")["throughput_gbps"].median()
                        if len(grouped) >= 2:
                            workers = grouped.index.tolist()
                            medians = grouped.values.tolist()
                            ax.plot(
                                workers, medians,
                                color=colour,
                                linestyle=linestyle,
                                linewidth=2,
                                alpha=0.8,
                            )

                    # Overlay threads/processes data if available
                    if overlay_df is not None and not overlay_df.empty:
                        overlay_k = overlay_df[overlay_df["num_dates_k"] == k_val]
                        if not overlay_k.empty:
                            # Update y-range from overlay data
                            global_min_y = min(global_min_y, overlay_k["throughput_gbps"].min())
                            global_max_y = max(global_max_y, overlay_k["throughput_gbps"].max())

                            # Get threads data at workers=1 (shared starting point)
                            threads_w1 = {}
                            for zarr_ver in overlay_k["zarr_version"].unique():
                                threads_subset = overlay_k[
                                    (overlay_k["mode"] == "threads") &
                                    (overlay_k["zarr_version"] == zarr_ver) &
                                    (overlay_k["total_workers"] == 1)
                                ]
                                if not threads_subset.empty:
                                    threads_w1[zarr_ver] = threads_subset["throughput_gbps"].median()

                            for mode in sorted(overlay_k["mode"].unique()):
                                for zarr_ver in sorted(overlay_k["zarr_version"].unique()):
                                    subset = overlay_k[
                                        (overlay_k["mode"] == mode) &
                                        (overlay_k["zarr_version"] == zarr_ver)
                                    ]
                                    if subset.empty:
                                        continue

                                    colour = mode_colours.get(mode, "gray")
                                    linestyle = zarr_linestyles.get(zarr_ver, "-")

                                    # Add overlay workers to x-axis ticks
                                    all_workers.update(subset["total_workers"].unique())

                                    ax.scatter(
                                        subset["total_workers"],
                                        subset["throughput_gbps"],
                                        c=[colour],
                                        marker="o",
                                        s=50,
                                        alpha=0.8,
                                        edgecolors="white",
                                        linewidths=0.5,
                                    )

                                    # Add median line
                                    grouped = subset.groupby("total_workers")["throughput_gbps"].median()

                                    # For processes, prepend threads workers=1 as shared start
                                    if mode == "processes" and zarr_ver in threads_w1:
                                        if 1 not in grouped.index:
                                            grouped = pd.concat([
                                                pd.Series([threads_w1[zarr_ver]], index=[1]),
                                                grouped
                                            ]).sort_index()

                                    if len(grouped) >= 2:
                                        workers_list = grouped.index.tolist()
                                        medians = grouped.values.tolist()
                                        ax.plot(
                                            workers_list, medians,
                                            color=colour,
                                            linestyle=linestyle,
                                            linewidth=2,
                                            alpha=0.8,
                                        )

                # Set title for top row only
                if row_idx == 0:
                    ax.set_title(facet_val, fontsize=12, fontweight="bold")

                # Set row label (k value) on left column
                if col_idx == 0:
                    ax.set_ylabel(f"k={k_val}\nThroughput (GB/s)", fontsize=11)

                # Set x-axis label for all rows (since sharex=False)
                if is_cpu_facet:
                    dynamic_x_label = "Workers"
                else:
                    dynamic_x_label = (
                        f"Total readers : (workers_per_gpu × {num_gpus})\n"
                        "or n threads or n processes"
                    )
                ax.set_xlabel(dynamic_x_label, fontsize=11)

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(True, alpha=0.3)

                # Format x-axis ticks (use global range for consistency)
                if len(global_all_workers) <= 15:
                    ax.set_xticks(global_all_workers)
                    ax.set_xticklabels(global_all_workers)
                ax.xaxis.set_major_formatter(ScalarFormatter())

                # Set consistent x-axis limits
                x_min = min(global_all_workers) * 0.8
                x_max = max(global_all_workers) * 1.2
                ax.set_xlim(x_min, x_max)

        # Format y-axis ticks
        if global_min_y < float("inf") and global_max_y > 0:
            candidate_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
                               1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
            y_ticks = [t for t in candidate_ticks
                       if global_min_y / 2 <= t <= global_max_y * 2]
            if y_ticks:
                for ax in self.axes:
                    if ax.get_visible():
                        ax.set_yticks(y_ticks)
                        ax.yaxis.set_major_formatter(ScalarFormatter())

        # Create unified legend (include overlay data if present)
        legend_df = df if overlay_df is None else pd.concat([df, overlay_df])
        self._add_legend(
            legend_df, zarr_linestyles, zarr_labels,
            zarr_colour_map, zarr_colour, mode_colours, mode_labels,
        )

        # Set title
        self.fig.suptitle(title, fontsize=14, fontweight="bold")

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    def _add_legend(
        self,
        df: pd.DataFrame,
        zarr_linestyles: dict,
        zarr_labels: dict,
        zarr_colour_map: dict,
        zarr_colour: tuple,
        mode_colours: dict,
        mode_labels: dict,
    ) -> None:
        """Add legend for zarr versions and modes."""
        if self.fig is None:
            return

        from matplotlib.lines import Line2D

        handles = []

        # Check if we have CPU modes (threads/processes)
        has_cpu_modes = df["mode"].isin(["threads", "processes"]).any()
        has_torch = df["mode"].eq("torch").any()

        if has_torch:
            # Add single torch legend entry (zarr versions distinguished by linestyle legend)
            handles.append(Line2D(
                [0], [0],
                marker="o",
                color=zarr_colour,
                linestyle="-",
                markersize=8,
                linewidth=2,
                label="Torch",
            ))

        if has_cpu_modes:
            # Add mode legend entries (colour by mode)
            for mode in ["threads", "processes"]:
                if mode in df["mode"].values:
                    handles.append(Line2D(
                        [0], [0],
                        marker="o",
                        color=mode_colours.get(mode, "gray"),
                        linestyle="-",
                        markersize=8,
                        linewidth=2,
                        label=mode_labels.get(mode, mode),
                    ))

        if handles:
            self.fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                fontsize=10,
                ncol=len(handles),
            )

        # Add separate legend for zarr line styles
        linestyle_handles = [
            Line2D([0], [0], color=zarr_colour, linestyle="-", linewidth=2, label="Zarr 2"),
            Line2D([0], [0], color=zarr_colour, linestyle="--", linewidth=2, label="Zarr 3"),
        ]
        self.fig.legend(
            handles=linestyle_handles,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.01),
            fontsize=10,
            ncol=1,
        )

    def add_reference_lines(
        self,
        lines: dict[str, float] | None = None,
    ) -> ThroughputPlotter:
        """Add horizontal reference lines for common storage speeds.

        Parameters
        ----------
        lines
            Dict mapping labels to throughput values in GB/s.
            Defaults to SSD (0.5) and HDD (0.1).

        Returns
        -------
        ThroughputPlotter
            Self for method chaining.
        """
        if len(self.axes) == 0:
            return self

        if lines is None:
            lines = {"SSD 0.5 GB/s": 0.5, "HDD 0.1 GB/s": 0.1}

        for ax_idx, ax in enumerate(self.axes):
            y_min, y_max = ax.get_ylim()
            for label, value in lines.items():
                if y_min <= value <= y_max:
                    ax.axhline(value, color="green", linestyle="--",
                               alpha=0.5, linewidth=1)
                    # Only add text on the last subplot
                    if ax_idx == len(self.axes) - 1:
                        ax.text(
                            1.02, value, label,
                            transform=ax.get_yaxis_transform(),
                            va="center", fontsize=8, color="green",
                        )

        return self

    def save(self, output_path: str, dpi: int = 150) -> None:
        """Save the figure to an image file.

        Parameters
        ----------
        output_path
            Path for output image (PNG, PDF, SVG, etc.)
        dpi
            Resolution for raster formats.
        """
        if self.fig is None:
            print("No figure to save")
            return

        self.fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
        plt.close(self.fig)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot zarr throughput benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "jsonl_paths",
        nargs="+",
        help="JSONL files containing benchmark results",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for image (default: <first_input>.png)",
    )
    parser.add_argument(
        "-t", "--title",
        help="Plot title (supports {dataset_name} placeholder)",
    )
    parser.add_argument(
        "-k", "--filter",
        help="Filter expression with substring match (e.g., 'S3 | SSD', 'not test')",
    )
    parser.add_argument(
        "-K", "--exact-filter",
        help="Filter expression with exact match (e.g., 'zarr3', 'not zarr2')",
    )
    parser.add_argument(
        "--torch-only",
        action="store_true",
        help="Only plot torch/DataLoader mode",
    )
    parser.add_argument(
        "--no-torch",
        action="store_true",
        help="Exclude torch mode (plot threads/processes only)",
    )
    parser.add_argument(
        "--no-reference-lines",
        action="store_true",
        help="Don't add SSD/HDD reference lines",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Image resolution (default: 150)",
    )

    args = parser.parse_args()

    # Load and prepare data
    data = BenchmarkData.from_jsonl(args.jsonl_paths).add_config_labels()

    if data.df.empty:
        print(f"No data found in {args.jsonl_paths}")
        return 1

    print(f"Loaded {len(data.df)} records")
    print(f"Modes: {', '.join(data.modes)}")
    print(f"Names: {', '.join(sorted(data.df['name'].unique()))}")
    print(f"Datasets: {', '.join(sorted(data.datasets))}")

    if args.filter:
        data = data.filter(args.filter)
        if data.df.empty:
            print(f"No data found matching filter '{args.filter}'")
            return 1
        print(f"After filter: {len(data.df)} records")

    if args.exact_filter:
        data = data.filter_exact(args.exact_filter)
        if data.df.empty:
            print(f"No data found matching exact filter '{args.exact_filter}'")
            return 1
        print(f"After exact filter: {len(data.df)} records")

    # Determine title
    title = args.title
    if title and "{dataset_name}" in title:
        dataset_names = [extract_dataset_name(d) for d in data.datasets if d]
        dataset_label = ", ".join(sorted(set(dataset_names))) or "Unknown"
        title = title.replace("{dataset_name}", dataset_label)

    # Create plotter and generate figure
    plotter = ThroughputPlotter(data)

    if args.torch_only:
        plotter.plot_torch_by_gpus(title=title)
    elif args.no_torch:
        plotter.plot_threads_processes(title=title)
    else:
        plotter.plot_all_modes(title=title)

    if not args.no_reference_lines:
        plotter.add_reference_lines()

    # Determine output path
    output = args.output
    if output is None:
        # Find first JSONL file for default output name
        for p in args.jsonl_paths:
            if p.endswith(".jsonl"):
                output = p.replace(".jsonl", ".png")
                break
        else:
            output = "output.png"

    plotter.save(output, dpi=args.dpi)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
