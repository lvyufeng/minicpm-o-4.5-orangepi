#!/bin/bash
# Start the MiniCPM-O-4.5 Orange Pi backend server

set -e

# === Configuration ===
MODEL_PATH="${MODEL_PATH:-/path/to/minicpm-o-4.5-model}"
DEVICE_ID="${DEVICE_ID:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-50051}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# === Environment Setup ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set Ascend environment
if [ -z "$ASCEND_HOME_PATH" ]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
fi

# Set custom ops path
export ASCEND_CUSTOM_OPP_PATH="${PROJECT_ROOT}/custom_opp"
if [ -f "${ASCEND_CUSTOM_OPP_PATH}/vendors/customize/bin/set_env.bash" ]; then
    source "${ASCEND_CUSTOM_OPP_PATH}/vendors/customize/bin/set_env.bash"
fi

# Set device
export ASCEND_DEVICE_ID="${DEVICE_ID}"

# === Check Dependencies ===
BACKEND_SERVER="${PROJECT_ROOT}/build/backend_server"

if [ ! -f "$BACKEND_SERVER" ]; then
    echo "Error: backend_server not found at $BACKEND_SERVER"
    echo "Please run 'cd build && cmake .. && make' first"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    echo "Please set MODEL_PATH environment variable"
    exit 1
fi

# === Start Server ===
echo "Starting MiniCPM-O-4.5 Orange Pi Backend Server..."
echo "  Model Path: $MODEL_PATH"
echo "  Device ID: $DEVICE_ID"
echo "  Listen: $HOST:$PORT"
echo "  Log Level: $LOG_LEVEL"
echo ""

exec "$BACKEND_SERVER" \
    --model_path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --device_id "$DEVICE_ID" \
    --log_level "$LOG_LEVEL"
