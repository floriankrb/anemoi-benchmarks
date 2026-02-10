#!/usr/bin/env python3
"""Track data heat to avoid reading cached samples in benchmarks.

This module provides utilities for tracking which data samples have been
read in previous runs, so subsequent runs can avoid "hot" (cached) data
and read fresh data for accurate benchmarking.

The key insight: for benchmarking cold data, we advance SEQUENTIALLY through
the dataset rather than picking random indices. Random indices across the
whole dataset would mark almost everything as "hot" after a single run.

Usage as CLI:
    ./heat_tracker.py --show                    # Show all heat tracking info
    ./heat_tracker.py --show <data_path>        # Show info for specific dataset
    ./heat_tracker.py --reset <data_path>       # Reset heat tracking for dataset
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class InsufficientColdDataError(Exception):
    """Raised when there isn't enough cold data available for the requested samples."""

    pass


@dataclass
class HeatEntry:
    """A single heat tracking entry."""

    datetime: str
    start_i: int
    end_i: int
    config: dict | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialisation."""
        result = {
            "datetime": self.datetime,
            "start_i": self.start_i,
            "end_i": self.end_i,
        }
        if self.config:
            result["config"] = self.config
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "HeatEntry":
        """Create from dictionary."""
        return cls(
            datetime=data["datetime"],
            start_i=data["start_i"],
            end_i=data["end_i"],
            config=data.get("config"),
        )

    @property
    def count(self) -> int:
        """Number of indices in this range."""
        return self.end_i - self.start_i


class HeatTracker:
    """Track which data ranges have been read to avoid reading cached data.

    Heat tracking works by advancing SEQUENTIALLY through the dataset:
    - Run 1: reads indices [0, N)
    - Run 2: reads indices [N, 2N)
    - etc.

    This ensures each benchmark run reads fresh (cold) data.
    """

    def __init__(self, data_path: str, base_dir: Path | None = None):
        """Initialise the heat tracker.

        Parameters
        ----------
        data_path : str
            Path to the data (e.g., S3 path or local path).
        base_dir : Path, optional
            Base directory for heat files. Defaults to 'data_heat' in module directory.
        """
        self.data_path = data_path
        if base_dir is None:
            base_dir = Path(__file__).parent / "data_heat"
        self.base_dir = base_dir
        self.heat_file = self._get_heat_file_path()
        self._entries: list[HeatEntry] | None = None

    def _get_heat_file_path(self) -> Path:
        """Convert data path to heat tracking file path."""
        # Remove protocol prefix (s3://, file://, etc.)
        name = re.sub(r"^[a-zA-Z0-9]+://", "", self.data_path)
        # Replace path separators and dots with dashes
        name = re.sub(r"[/.]", "-", name)
        # Collapse multiple dashes
        name = re.sub(r"-+", "-", name)
        # Strip leading/trailing dashes
        name = name.strip("-")
        return self.base_dir / f"{name}.jsonl"

    @property
    def entries(self) -> list[HeatEntry]:
        """Load and return all entries from the heat file."""
        if self._entries is None:
            self._entries = self._load_entries()
        return self._entries

    def _load_entries(self) -> list[HeatEntry]:
        """Load all entries from the heat file."""
        if not self.heat_file.exists():
            return []

        entries = []
        with open(self.heat_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        entries.append(HeatEntry.from_dict(data))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return entries

    def get_last_hot_range(self) -> tuple[int, int] | None:
        """Get the hot range from the last entry.

        Returns
        -------
        tuple[int, int] | None
            The (start_i, end_i) range from the last entry, or None if no entries.
        """
        if not self.entries:
            return None
        last = self.entries[-1]
        return (last.start_i, last.end_i)

    def get_next_cold_start(self) -> int:
        """Get the starting index for the next cold read.

        Returns the end index of the last entry (where cold data begins).
        """
        if not self.entries:
            return 0
        return self.entries[-1].end_i

    def write_entry(
        self,
        start_i: int,
        end_i: int,
        config: dict | None = None,
    ) -> None:
        """Append a new heat entry to the tracking file.

        Parameters
        ----------
        start_i : int
            Start index of the range that was read.
        end_i : int
            End index (exclusive) of the range that was read.
        config : dict, optional
            Configuration used for this run (for debugging).
        """
        self.heat_file.parent.mkdir(parents=True, exist_ok=True)

        entry = HeatEntry(
            datetime=datetime.now().isoformat(),
            start_i=start_i,
            end_i=end_i,
            config=config,
        )

        with open(self.heat_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

        # Invalidate cache
        self._entries = None

    def reset(self) -> bool:
        """Reset heat tracking by deleting the heat file.

        Returns
        -------
        bool
            True if file was deleted, False if it didn't exist.
        """
        if self.heat_file.exists():
            self.heat_file.unlink()
            self._entries = None
            return True
        return False

    def select_sequential_range(
        self,
        total_size: int,
        num_samples: int,
    ) -> tuple[int, int]:
        """Select the next sequential range of cold indices.

        This is the recommended approach for benchmarking: advance through
        the dataset sequentially to ensure each run reads truly cold data.

        Parameters
        ----------
        total_size : int
            Total number of indices available (0 to total_size-1).
        num_samples : int
            Number of indices needed.

        Returns
        -------
        tuple[int, int]
            The (start_i, end_i) range to read.

        Raises
        ------
        InsufficientColdDataError
            If there aren't enough cold indices available.
        """
        start_i = self.get_next_cold_start()
        end_i = start_i + num_samples

        if end_i > total_size:
            available = total_size - start_i
            raise InsufficientColdDataError(
                f"Not enough cold data available. "
                f"Requested {num_samples} samples starting at index {start_i}, "
                f"but only {available} indices remain (dataset size: {total_size}). "
                f"Delete the heat tracking file to reset: {self.heat_file}"
            )

        return (start_i, end_i)

    def get_indices_from_range(
        self,
        start_i: int,
        end_i: int,
        randomise: bool = True,
    ) -> list[int]:
        """Get indices from a range, optionally randomised.

        Parameters
        ----------
        start_i : int
            Start of range (inclusive).
        end_i : int
            End of range (exclusive).
        randomise : bool
            If True, return indices in random order. If False, return sorted.

        Returns
        -------
        list[int]
            List of indices.
        """
        import random

        indices = list(range(start_i, end_i))
        if randomise:
            random.shuffle(indices)
        return indices

    def show_info(self) -> str:
        """Generate human-readable info about heat tracking state."""
        lines = []
        lines.append(f"Heat Tracking Info for: {self.data_path}")
        lines.append(f"Heat file: {self.heat_file}")
        lines.append(f"File exists: {self.heat_file.exists()}")
        lines.append("")

        if not self.entries:
            lines.append("No entries found (all data is cold)")
            return "\n".join(lines)

        lines.append(f"Total entries: {len(self.entries)}")
        total_read = sum(e.count for e in self.entries)
        lines.append(f"Total indices read: {total_read}")
        lines.append("")

        lines.append("Run History:")
        lines.append("-" * 80)
        for i, entry in enumerate(self.entries, 1):
            config_str = ""
            if entry.config:
                config_str = f" | config: {entry.config}"
            lines.append(
                f"  {i}. [{entry.datetime}] "
                f"range=[{entry.start_i}, {entry.end_i}) "
                f"count={entry.count}{config_str}"
            )

        lines.append("-" * 80)
        lines.append(f"Next cold start index: {self.get_next_cold_start()}")

        return "\n".join(lines)


# Convenience functions for backward compatibility
def get_heat_file_path(data_path: str, base_dir: Path | None = None) -> Path:
    """Convert a data path to a heat tracking file path."""
    tracker = HeatTracker(data_path, base_dir)
    return tracker.heat_file


def read_last_hot_range(heat_file: Path) -> tuple[int, int] | None:
    """Read the last entry from a heat file to get the hot range.

    Note: This function is for backward compatibility. Consider using
    HeatTracker.get_next_cold_start() instead for sequential advancement.
    """
    if not heat_file.exists():
        return None

    last_line = None
    with open(heat_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line

    if last_line is None:
        return None

    try:
        entry = json.loads(last_line)
        return (entry["start_i"], entry["end_i"])
    except (json.JSONDecodeError, KeyError):
        return None


def write_heat_entry(heat_file: Path, start_i: int, end_i: int) -> None:
    """Append a new heat entry to the tracking file."""
    heat_file.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "datetime": datetime.now().isoformat(),
        "start_i": start_i,
        "end_i": end_i,
    }

    with open(heat_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def select_cold_indices(
    total_size: int,
    num_samples: int,
    hot_range: tuple[int, int] | None,
) -> list[int]:
    """Select indices avoiding the hot range.

    DEPRECATED: This function has a fundamental flaw - random indices across
    the dataset will mark almost everything as "hot" after one run. Use
    HeatTracker.select_sequential_range() instead for proper cold data tracking.
    """
    import random

    if hot_range is None:
        available = list(range(total_size))
    else:
        start_i, end_i = hot_range
        available = [i for i in range(total_size) if i < start_i or i >= end_i]

    if len(available) < num_samples:
        hot_count = total_size - len(available)
        raise InsufficientColdDataError(
            f"Not enough cold data available. "
            f"Requested {num_samples} samples but only {len(available)} cold indices "
            f"available ({hot_count} indices are hot from previous run). "
            f"This is likely because random selection marked most of the dataset as hot. "
            f"Consider using sequential range selection instead. "
            f"Delete the heat tracking file to reset."
        )

    return sorted(random.sample(available, num_samples))


def list_all_heat_files(base_dir: Path | None = None) -> list[Path]:
    """List all heat tracking files."""
    if base_dir is None:
        base_dir = Path(__file__).parent / "data_heat"

    if not base_dir.exists():
        return []

    return sorted(base_dir.glob("*.jsonl"))


def show_all_heat_info() -> str:
    """Show info for all tracked datasets."""
    heat_files = list_all_heat_files()

    if not heat_files:
        return "No heat tracking files found."

    lines = []
    lines.append("=" * 80)
    lines.append("HEAT TRACKING STATUS")
    lines.append("=" * 80)

    for heat_file in heat_files:
        # Reconstruct a dummy data path from filename
        data_path = heat_file.stem  # filename without .jsonl
        tracker = HeatTracker(data_path)
        tracker.heat_file = heat_file  # Override with actual file
        tracker._entries = None  # Reset cache

        lines.append("")
        lines.append(tracker.show_info())
        lines.append("")

    return "\n".join(lines)


def main():
    """CLI for heat tracker inspection and management."""
    parser = argparse.ArgumentParser(
        description="Heat tracking management for benchmark scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./heat_tracker.py --show                    Show all heat tracking info
  ./heat_tracker.py --show s3://bucket/data   Show info for specific dataset
  ./heat_tracker.py --reset s3://bucket/data  Reset heat tracking for dataset
  ./heat_tracker.py --reset-all               Reset all heat tracking
        """,
    )
    parser.add_argument(
        "--show",
        nargs="?",
        const="__all__",
        metavar="DATA_PATH",
        help="Show heat tracking info. Optionally specify a data path.",
    )
    parser.add_argument(
        "--reset",
        metavar="DATA_PATH",
        help="Reset heat tracking for a specific dataset.",
    )
    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="Reset all heat tracking files.",
    )
    args = parser.parse_args()

    if args.show:
        if args.show == "__all__":
            print(show_all_heat_info())
        else:
            tracker = HeatTracker(args.show)
            print(tracker.show_info())
    elif args.reset:
        tracker = HeatTracker(args.reset)
        if tracker.reset():
            print(f"Reset heat tracking for: {args.reset}")
            print(f"Deleted: {tracker.heat_file}")
        else:
            print(f"No heat tracking file found for: {args.reset}")
    elif args.reset_all:
        heat_files = list_all_heat_files()
        if not heat_files:
            print("No heat tracking files found.")
        else:
            for heat_file in heat_files:
                heat_file.unlink()
                print(f"Deleted: {heat_file}")
            print(f"\nReset {len(heat_files)} heat tracking file(s).")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
