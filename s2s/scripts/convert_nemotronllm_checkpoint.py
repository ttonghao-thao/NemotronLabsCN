# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert NeMo STT checkpoint to HuggingFace format compatible with vLLM.

This script extracts weights from a NeMo checkpoint that has the structure:
- stt_model.llm.layers.*
- stt_model.lm_head.*
- stt_model.embed_tokens.*

And converts them to HuggingFace format that can be loaded by vLLM with the
custom WeightsMapper defined in nemotron_h.py.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load checkpoint from safetensors or PyTorch format.

    Args:
        checkpoint_path: Path to the checkpoint file (.safetensors or .pt/.pth)

    Returns:
        Dictionary of tensor names to tensors
    """
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "model.safetensors")

    if checkpoint_path.endswith('.safetensors'):
        logger.info(f"Loading safetensors from {checkpoint_path}")
        return load_file(checkpoint_path)
    else:
        logger.info(f"Loading PyTorch checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        # Handle different checkpoint formats
        if 'state_dict' in ckpt:
            return ckpt['state_dict']
        elif 'model' in ckpt:
            return ckpt['model']
        else:
            return ckpt


def filter_tensors(state_dict: Dict[str, torch.Tensor], prefixes_to_keep: List[str]) -> Dict[str, torch.Tensor]:
    """
    Filter tensors to keep only those with specified prefixes.

    Args:
        state_dict: Full state dictionary
        prefixes_to_keep: List of prefixes to keep (e.g., ["stt_model.llm"])

    Returns:
        Filtered state dictionary
    """
    filtered_dict = {}
    for name, tensor in state_dict.items():
        if any(name.startswith(prefix) for prefix in prefixes_to_keep):
            filtered_dict[name] = tensor
            logger.debug(f"Keeping: {name} with shape {tensor.shape}")
        else:
            logger.debug(f"Skipping: {name}")

    logger.info(f"Total tensors kept: {len(filtered_dict)}")
    return filtered_dict


def convert_nemo_to_hf_format(
    checkpoint_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
    pretrained_llm: Optional[str] = None,
    tensors_to_keep: Optional[List[str]] = None,
    dtype: str = "float32",
) -> None:
    """
    Convert NeMo STT checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to the NeMo checkpoint (.safetensors or .pt)
        output_dir: Directory to save the converted checkpoint
        config_path: Path to config.json (if None, will look in same dir as checkpoint)
        pretrained_llm: HuggingFace model name to get base config from
        tensors_to_keep: List of tensor prefixes to keep (default: all stt_model.* tensors)
        dtype: Data type for tensors ("float32", "float16", "bfloat16")
    """
    # Default prefixes to keep
    if tensors_to_keep is None:
        tensors_to_keep = [
            "stt_model.llm",
            "stt_model.lm_head",
            "stt_model.embed_tokens",
            "stt_model.function_head",
        ]

    # Load config to get pretrained_llm if not provided
    if config_path is None:
        ckpt_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
        config_path = os.path.join(ckpt_dir, "config.json")

    if os.path.exists(config_path):
        logger.info(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)

        try:
            pretrained_llm = config["model"]["stt"]["model"]["pretrained_llm"]
            logger.info(f"Found pretrained_llm in config: {pretrained_llm}")
        except KeyError:
            if pretrained_llm is None:
                raise ValueError("Could not find pretrained_llm in config and none provided via argument")
    else:
        if pretrained_llm is None:
            raise ValueError(f"Config file not found at {config_path} and pretrained_llm not provided")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # zh-phase1 air-gapped support: prefer a plain local dir holding the LLM's
    # config.json + tokenizer files over the HF hub. Mirrored in
    # checkpoint_utils/import_utils.py (deploy tokenizer bundling).
    local_llm_dir = os.environ.get("NEMO_LLM_LOCAL_DIR", "")
    if local_llm_dir and os.path.isfile(os.path.join(local_llm_dir, "config.json")):
        logger.info(f"NEMO_LLM_LOCAL_DIR set - using local LLM files from {local_llm_dir}")
        pretrained_llm = local_llm_dir

    # Load base config from pretrained model
    logger.info(f"Loading base config from {pretrained_llm}")
    base_config = AutoConfig.from_pretrained(pretrained_llm, trust_remote_code=True)

    # Modify config for custom inputs/outputs
    custom_config = {
        "custom_input_specs": [{"name": "combined_embeds", "dtype": dtype, "dim": base_config.hidden_size}],
        "custom_outputs": ["text_logits", "function_tokens"],
    }
    base_config.update(custom_config)

    # Load tokenizer from pretrained model
    logger.info(f"Loading tokenizer from {pretrained_llm}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_llm, trust_remote_code=True)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    state_dict = load_checkpoint(checkpoint_path)

    # Filter tensors
    logger.info(f"Filtering tensors to keep prefixes: {tensors_to_keep}")
    filtered_state_dict = filter_tensors(state_dict, tensors_to_keep)

    if len(filtered_state_dict) == 0:
        raise ValueError(
            f"No tensors found with prefixes {tensors_to_keep}. "
            f"Available prefixes: {set(k.split('.')[0] for k in state_dict.keys())}"
        )

    # Convert dtype if needed
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    '''
    if dtype.lower() in dtype_map:
        target_dtype = dtype_map[dtype.lower()]
        print(f"\nConverting tensors to {target_dtype}")
        filtered_state_dict = {
            k: v.to(target_dtype) if v.dtype.is_floating_point else v
            for k, v in filtered_state_dict.items()
        }
    '''

    # Save tensors
    output_model_path = output_path / "model.safetensors"
    logger.info(f"Saving tensors to {output_model_path}")
    save_file(filtered_state_dict, str(output_model_path))

    # Save config
    output_config_path = output_path / "config.json"
    logger.info(f"Saving config to {output_config_path}")
    base_config.save_pretrained(str(output_path))

    # Save tokenizer
    logger.info(f"Saving tokenizer to {output_path}")
    tokenizer.save_pretrained(str(output_path))

    logger.info(f"Conversion completed successfully! Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert NeMo STT checkpoint to HuggingFace format for vLLM")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to NeMo checkpoint file (.safetensors or .pt/.pth)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save converted checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config.json (default: same directory as checkpoint)",
    )
    parser.add_argument(
        "--pretrained-llm",
        type=str,
        default=None,
        help="HuggingFace model name to use as base (default: read from config)",
    )
    parser.add_argument(
        "--tensors-to-keep",
        type=str,
        nargs="+",
        default=None,
        help="Tensor prefixes to keep (default: all stt_model.* backbone llm related tensors)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16", "fp32", "fp16", "bf16"],
        help="Target dtype for tensors (default: bfloat16)",
    )

    args = parser.parse_args()

    convert_nemo_to_hf_format(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
        pretrained_llm=args.pretrained_llm,
        tensors_to_keep=args.tensors_to_keep,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
