#!/bin/bash

set -e

PATH=/opt/tritonserver/bin:$PATH
MODEL_REPOSITORY=${MODEL_REPOSITORY:-/data/models}
TRITON_LOG_VERBOSE=${TRITON_LOG_VERBOSE:-0}

find $MODEL_REPOSITORY

# Check the health endpoint (adjust port if necessary)
if ! curl -s --fail http://localhost:8000/v2/health/ready > /dev/null; then
    echo "Starting Triton server..."
    tritonserver --log-verbose=$TRITON_LOG_VERBOSE --model-repository=$MODEL_REPOSITORY 2>&1 | stdbuf -oL sed 's/^/[TRITON] /' &
    TRITON_PID=$!

    # Wait for Triton server to be ready
    TIMEOUT=1200
    INTERVAL=10
    COUNT=0

    echo "Waiting for Triton server to start..."

    while [ $COUNT -lt $TIMEOUT ]; do
        # Check if the process is still running
        if ! kill -0 $TRITON_PID 2>/dev/null; then
            echo "Triton server process exited prematurely."
            exit 1
        fi

        # Check the health endpoint (adjust port if necessary)
        if curl -s --fail http://localhost:8000/v2/health/ready > /dev/null; then
            echo "Triton server is ready."
            break
        fi

        sleep $INTERVAL
        COUNT=$((COUNT + INTERVAL))

    done

    if [ $COUNT -ge $TIMEOUT ]; then
        echo "Triton server failed to start within $TIMEOUT seconds."
        kill $TRITON_PID
        exit 1
    fi
fi

echo "Run inference for warm up..."
python /s2s/nemotron-voicechat-triton-client.py --audio_file /s2s/en-US_sample_extended.wav --no-output

echo "Starting realtime server..."
python /s2s/audio_server.py
REALTIME_SERVER_STATUS=$?

if [ $REALTIME_SERVER_STATUS -ne 0 ]; then
    echo "Realtime server failed to start."
    kill $TRITON_PID
    wait $TRITON_PID 2>/dev/null
    exit 1
fi
