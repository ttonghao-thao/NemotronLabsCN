#!/bin/bash
set -ex

MODEL_REPOSITORY=${MODEL_REPOSITORY:-/data/models}

if [ -z "${NEMO_CHECKPOINT_PATH}" ]; then
    echo "ERROR: NEMO_CHECKPOINT_PATH must be set to the directory containing model.safetensors" >&2
    exit 1
fi

if [ ! -f "${NEMO_CHECKPOINT_PATH}/model.safetensors" ]; then
    echo "ERROR: model.safetensors not found in ${NEMO_CHECKPOINT_PATH}" >&2
    exit 1
fi

# deploy the model
# HF_HUB_OFFLINE is set in the image for runtime; unset here so the tokenizer
# can be fetched from HuggingFace and bundled into the model repo during deploy.
unset HF_HUB_OFFLINE
mkdir -p /data/models
PYTHONPATH=/opt/tritonserver/backends/nemotron-voicechat python -m checkpoint_utils generate-triton-repo \
    --checkpoint ${NEMO_CHECKPOINT_PATH}/model.safetensors \
    --output-dir ${MODEL_REPOSITORY}
