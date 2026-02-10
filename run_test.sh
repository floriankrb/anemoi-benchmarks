#!/bin/bash
# Test zarr2 vs zarr3 read speed on S3 datasets
#
# Usage: ./run_test.sh s3://bucket/path/to/dataset.zarr --mode threads|processes|torch|threads-processes [options]
#

set -e

# Default values
DATASET_PATH=""
MODE=""
NUM_SAMPLES="10"
WORKERS_ARG="1-2-3-4-5-10-20-50-100"
NAME="HPC"
NUM_DATES=""  # Will default to 1 for threads/processes, 4 for torch

# Torch-specific defaults
NUM_GPUS="4"
PREFETCH="2"
NO_HEAT_TRACKING=""

# Parse command line options
usage() {
    echo "Usage: $0 <DATASET_PATH> --mode <mode> [options]"
    echo ""
    echo "Arguments:"
    echo "  dataset_path           Path to zarr dataset (required)"
    echo ""
    echo "Options:"
    echo "  --mode <mode>          Mode: 'threads', 'processes', 'torch', or 'threads-processes' (required)"
    echo "  -n, --num-samples <n>  Number of random samples to read (default: 10)"
    echo "  -k, --num-dates <n>    Number of consecutive dates per sample (default: 1 for threads/processes, 4 for torch)"
    echo "  --workers <list>       Worker counts separated by '-' (default: 1-2-3-4-5-10-20-50-100)"
    echo "                         For torch mode, this is workers per GPU"
    echo "  --name <name>          Name for this test run (default: HPC)"
    echo ""
    echo "Torch-specific options:"
    echo "  -g, --num-gpus <n>     Number of GPUs (default: 4)"
    echo "  --prefetch <n>         Prefetch factor (default: 2)"
    echo ""
    echo "Other options:"
    echo "  --no-heat-tracking     Disable heat tracking (allow reading cached data)"
    echo "  -h, --help             Show this help message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --workers)
            WORKERS_ARG="$2"
            shift 2
            ;;
        -g|--num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -k|--num-dates)
            NUM_DATES="$2"
            shift 2
            ;;
        --prefetch)
            PREFETCH="$2"
            shift 2
            ;;
        --name)
            NAME="$2"
            shift 2
            ;;
        --no-heat-tracking)
            NO_HEAT_TRACKING="--no-heat-tracking"
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            # Positional argument (dataset_path)
            if [[ -z "${DATASET_PATH}" ]]; then
                DATASET_PATH="$1"
                shift
            else
                echo "Unexpected argument: $1"
                usage
            fi
            ;;
    esac
done

# Validate required arguments
if [[ -z "${DATASET_PATH}" ]]; then
    echo "Error: dataset_path is required"
    usage
fi

if [[ -z "${MODE}" ]]; then
    echo "Error: --mode is required"
    usage
fi

if [[ "${MODE}" != "threads" && "${MODE}" != "processes" && "${MODE}" != "torch" && "${MODE}" != "threads-processes" ]]; then
    echo "Error: --mode must be 'threads', 'processes', 'torch', or 'threads-processes'"
    usage
fi

# Expand compound mode into a list of modes to run
if [[ "${MODE}" == "threads-processes" ]]; then
    MODES=("threads" "processes")
else
    MODES=("${MODE}")
fi

# Set default NUM_DATES based on mode if not explicitly set
if [[ -z "${NUM_DATES}" ]]; then
    if [[ "${MODE}" == "torch" ]]; then
        NUM_DATES="4"
    else
        NUM_DATES="1"
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test_zarr_speed.py"
TORCH_SCRIPT="${SCRIPT_DIR}/test_torch_speed.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot.py"

# anemoi-datasets from feature/zarr3 branch
ANEMOI_DATASETS="git+https://github.com/ecmwf/anemoi-datasets@feature/any-zarr-version-in-pyproject"

# Generate output name from dataset path
OUTPUT_NAME=$(echo "${DATASET_PATH}" | sed 's|^s3://||' | sed 's|[/.]|-|g' | sed 's|-\+|-|g' | sed 's|^-||;s|-$||')
OUTPUT_DIR="${SCRIPT_DIR}/logs"

# Parse workers argument (split on '-' if multiple values)
IFS='-' read -ra WORKER_VALUES <<< "${WORKERS_ARG}"

# Function to check if zarr2 format exists
has_zarr2() {
    [[ -f "${DATASET_PATH}/.zattrs" ]]
}

# Loop over each mode
for CURRENT_MODE in "${MODES[@]}"; do

OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_NAME}-${CURRENT_MODE}.jsonl"

echo "========================================================"
echo "Testing zarr read speed on: ${DATASET_PATH}"
echo "Name: ${NAME}"
echo "Mode: ${CURRENT_MODE}"
if [[ "${CURRENT_MODE}" == "torch" ]]; then
    echo "GPUs: ${NUM_GPUS}"
    echo "Workers per GPU: ${WORKER_VALUES[*]}"
    echo "Consecutive dates (k): ${NUM_DATES}"
    echo "Samples per worker (m): ${NUM_SAMPLES}"
    echo "Prefetch factor: ${PREFETCH}"
else
    echo "Number of samples: ${NUM_SAMPLES}"
    echo "Consecutive dates (k): ${NUM_DATES}"
    echo "Worker values: ${WORKER_VALUES[*]}"
fi
echo "Output: ${OUTPUT_PATH}"
echo "========================================================"
echo

# Run tests based on mode
if [[ "${CURRENT_MODE}" == "torch" ]]; then
    # Torch mode
    for WORKERS in "${WORKER_VALUES[@]}"; do
        echo "########################################################"
        echo "# Testing torch with ${WORKERS} workers per GPU"
        echo "########################################################"
        echo

        # Test with zarr2
        echo "--- ZARR 2 (zarr<3) ---"
        if has_zarr2; then
            uv run --with "${ANEMOI_DATASETS}" \
                   --with "zarr<3" \
                   --with torch \
                   --with pytorch-lightning \
                   --with tqdm \
                   --with psutil \
                   python "${TORCH_SCRIPT}" "${DATASET_PATH}" \
                       -n "${WORKERS}" \
                       -g "${NUM_GPUS}" \
                       -k "${NUM_DATES}" \
                       -m "${NUM_SAMPLES}" \
                       --prefetch-factor "${PREFETCH}" \
                       --name "${NAME}" \
                       -o "${OUTPUT_PATH}" \
                       ${NO_HEAT_TRACKING}
        else
            echo "Skipping zarr2 tests: .zattrs not found at ${DATASET_PATH}"
        fi
        echo

        # Test with zarr3
        echo "--- ZARR 3 (zarr>=3) ---"
        uv run --with "${ANEMOI_DATASETS}" \
               --with "zarr>=3" \
               --with torch \
               --with pytorch-lightning \
               --with tqdm \
               --with psutil \
               python "${TORCH_SCRIPT}" "${DATASET_PATH}" \
                   -n "${WORKERS}" \
                   -g "${NUM_GPUS}" \
                   -k "${NUM_DATES}" \
                   -m "${NUM_SAMPLES}" \
                   --prefetch-factor "${PREFETCH}" \
                   --name "${NAME}" \
                   -o "${OUTPUT_PATH}" \
                   ${NO_HEAT_TRACKING}
        echo
    done
else
    # Threads or processes mode
    for WORKERS in "${WORKER_VALUES[@]}"; do
        echo "########################################################"
        echo "# Testing with ${WORKERS} ${CURRENT_MODE}"
        echo "########################################################"
        echo

        # Test with zarr2
        echo "--- ZARR 2 (zarr<3) ---"
        if has_zarr2; then
            uv run --with "${ANEMOI_DATASETS}" --with "zarr<3" \
                python "${TEST_SCRIPT}" "${DATASET_PATH}" \
                    -n "${NUM_SAMPLES}" \
                    -k "${NUM_DATES}" \
                    -w "${WORKERS}" \
                    -m "${CURRENT_MODE}" \
                    -o "${OUTPUT_PATH}" \
                    --name "${NAME}" \
                    ${NO_HEAT_TRACKING}
        else
            echo "Skipping zarr2 tests: .zattrs not found at ${DATASET_PATH}"
        fi
        echo

        # Test with zarr3
        echo "--- ZARR 3 (zarr>=3) ---"
        uv run --with "${ANEMOI_DATASETS}" --with "zarr>=3" \
            python "${TEST_SCRIPT}" "${DATASET_PATH}" \
                -n "${NUM_SAMPLES}" \
                -k "${NUM_DATES}" \
                -w "${WORKERS}" \
                -m "${CURRENT_MODE}" \
                -o "${OUTPUT_PATH}" \
                --name "${NAME}" \
                ${NO_HEAT_TRACKING}
        echo
    done
fi

echo
echo "========================================================"
echo "Test complete for mode: ${CURRENT_MODE}"
echo "Results saved to: ${OUTPUT_PATH}"
echo "========================================================"
echo

done  # end of MODES loop

echo "Plotting results..."
uv run --with matplotlib --with pandas --with seaborn python "${PLOT_SCRIPT}" logs/* -o results.png