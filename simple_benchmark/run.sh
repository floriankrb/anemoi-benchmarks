#!/bin/bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

REPEATS=5
CHUNK_SIZE="10MB"
PATH_DIR=""

usage() {
    echo "Usage: $0 --path <dir> [-s|--chunk-size <size>]"
    echo "  --path              Path to create the zarr datasets"
    echo "  -s, --chunk-size    Chunk size: 10MB (default), 100MB, 1GB, 10GB"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --path)
            PATH_DIR="$2"
            shift 2
            ;;
        -s|--chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [ -z "$PATH_DIR" ]; then
    echo "Error: --path is required"
    usage
fi

# Validate chunk size
case "$CHUNK_SIZE" in
    10MB|100MB|1GB|10GB) ;;
    *)
        echo "Error: Invalid chunk size '$CHUNK_SIZE'. Must be 10MB, 100MB, 1GB, or 10GB."
        exit 1
        ;;
esac

mkdir -p "$PATH_DIR"

ZARR2="zarr-2-${CHUNK_SIZE}-chunks.zarr"
ZARR3="zarr-3-${CHUNK_SIZE}-chunks.zarr"

echo "========================================"
echo "  Simple Zarr Benchmark"
echo "  Chunk size: $CHUNK_SIZE"
echo "  Dataset path: $PATH_DIR"
echo "========================================"
echo

# --- Create datasets if they don't exist ---

# --- Create zarr2 dataset with zarr v2 ---
if [ ! -d "$PATH_DIR/$ZARR2" ]; then
    echo "Creating $ZARR2 (10 chunks x ~$CHUNK_SIZE) with zarr v2..."
    uv run --with "zarr>=2,<3" --python=3.10 python "$SCRIPT_DIR/create_dataset.py" "$PATH_DIR/$ZARR2" 2 --chunk-size "$CHUNK_SIZE"
    echo
else
    echo "$ZARR2 already exists, skipping creation."
fi

# --- Create zarr3 dataset with zarr3 package ---
if [ ! -d "$PATH_DIR/$ZARR3" ]; then
    echo "Creating $ZARR3 (10 chunks x ~$CHUNK_SIZE) with zarr3..."
    uv run --with "zarr>=3" --python=3.11 python "$SCRIPT_DIR/create_dataset.py" "$PATH_DIR/$ZARR3" 3 --chunk-size "$CHUNK_SIZE"
    echo
else
    echo "$ZARR3 already exists, skipping creation."
fi

# --- Evict all data from page cache ---
echo "Evicting $ZARR2 from page cache..."
vmtouch -e "$PATH_DIR/$ZARR2"
echo
echo "Evicting $ZARR3 from page cache..."
vmtouch -e "$PATH_DIR/$ZARR3"
echo


# --- Probe zarr2 (chunk 5) with zarr v2 ---
echo "========================================"
echo "  Probing $ZARR2 ✅ with zarr 2 ✅ code — chunk 3"
echo
uv run --with "zarr>=2,<3" --python=3.10 python "$SCRIPT_DIR/probe.py" "$PATH_DIR/$ZARR2" --warmup-chunk 0 --test-chunk 3 --repeats $REPEATS
echo

echo "========================================"
echo "  Probing $ZARR2 ✅ with zarr 3 ❓ code — chunk 4"
echo
uv run --with "zarr>=3" --python=3.11 "$SCRIPT_DIR/probe.py" "$PATH_DIR/$ZARR2" --warmup-chunk 0 --test-chunk 6 --repeats $REPEATS
echo

# # --- Probe zarr3 (chunk 5) with zarr3 ---
# echo "========================================"
# echo "  Probing $ZARR3 ❓ with zarr 3 ❓ code — chunk 5"
# echo
# uv run --with "zarr>=3" --python=3.11 "$SCRIPT_DIR/probe.py" "$PATH_DIR/$ZARR3" --warmup-chunk 0 --test-chunk 9 --repeats $REPEATS
# echo

probe_chunk() {
    local file="$1"
    local label="$2"
    # Use dd to read the file and extract the speed from the summary line
    local output
    output=$(dd if="$file" of=/dev/null bs=4M 2>&1)
    local speed
    speed=$(echo "$output" | grep -Eo '[0-9.]+ [GMk]?B/s' | tail -1)
    local bytes
    bytes=$(echo "$output" | grep -Eo '[0-9]+ bytes' | head -1 | grep -Eo '[0-9]+')
    local time_s
    time_s=$(echo "$output" | grep -Eo '[0-9.]+ s,' | grep -Eo '[0-9.]+')
    if [ -n "$speed" ]; then
        printf "  %-12s %10s  (%ss, %.1f MB)\n" "$label" "$speed" "$time_s" "$(echo "$bytes / 1000000" | bc -l)"
    else
        echo "  $label  (no speed reported)"
    fi
}

# Probe with direct dd read, 3 times each
echo "========================================"
echo "  Probing $ZARR2 ✅ with direct read dd chunk 6"
echo "========================================"
for i in 1 2 3; do
    probe_chunk "$PATH_DIR/$ZARR2/data/6" "dd read $i:"
done
echo

echo "========================================"
echo "  Probing $ZARR3 ❓ with direct read dd chunk 7"
echo "========================================"
for i in 1 2 3; do
    probe_chunk "$PATH_DIR/$ZARR3/data/c/7" "dd read $i:"
done
echo

echo "Done"
