import json
import logging
import os
from typing import Dict, Optional

import torch
from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat
from omegaconf import DictConfig, OmegaConf, open_dict
from safetensors.torch import load_file, save_file

logging.getLogger("nemo_logger").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# =============================================================================
# Utilities to import a NemotronVoiceChat model from NeMo checkpoint
# =============================================================================


def extract_perception_state_dict(full_checkpoint_path: str,) -> Dict[str, torch.Tensor]:
    """
    Extract the perception state dict from a full model.safetensors checkpoint.

    The perception weights in the full checkpoint are prefixed with "stt_model.perception.".
    This function filters those keys and removes the prefix.

    Args:
        full_checkpoint_path: Path to the model.safetensors file containing all components.

    Returns:
        A state dict containing only the perception weights with the prefix removed.
    """
    state_dict = load_file(full_checkpoint_path)

    perception_prefix = "stt_model.perception."
    perception_state_dict = {
        k.removeprefix(perception_prefix): v for k, v in state_dict.items() if k.startswith(perception_prefix)
    }

    return perception_state_dict


def save_perception_checkpoint(full_checkpoint_path: str, output_path: str,) -> None:
    """
    Extract perception weights from a full model.safetensors and save to a separate file.

    Args:
        full_checkpoint_path: Path to the model.safetensors file containing all components.
        output_path: Path where the perception-only checkpoint will be saved.
    """
    perception_state_dict = extract_perception_state_dict(full_checkpoint_path)

    if not perception_state_dict:
        raise ValueError(
            f"No perception weights found in {full_checkpoint_path}. "
            "Expected keys with prefix 'stt_model.perception.'"
        )

    save_file(perception_state_dict, output_path)

    logger.info(f"Saved perception checkpoint with {len(perception_state_dict)} keys to {output_path}")


def extract_codec_state_dict(full_checkpoint_path: str,) -> Dict[str, torch.Tensor]:
    """
    Extract the codec state dict from a full model.safetensors checkpoint.

    The codec weights in the full checkpoint are prefixed with "tts_model.audio_codec.".
    This function filters those keys and removes the prefix.

    Args:
        full_checkpoint_path: Path to the model.safetensors file containing all components.

    Returns:
        A state dict containing only the codec weights with the prefix removed.
    """
    state_dict = load_file(full_checkpoint_path)

    codec_prefix = "tts_model.audio_codec."
    codec_state_dict = {k.removeprefix(codec_prefix): v for k, v in state_dict.items() if k.startswith(codec_prefix)}

    return codec_state_dict


def list_checkpoint_keys(checkpoint_path: str, output_path: str,) -> None:
    """
    List all keys from a safetensors checkpoint and write them to a text file.

    Args:
        checkpoint_path: Path to the .safetensors checkpoint file.
        output_path: Path to the output text file (one key per line).
    """
    state_dict = load_file(checkpoint_path)
    keys = sorted(state_dict.keys())

    with open(output_path, "w") as f:
        for key in keys:
            f.write(f"{key}\n")

    logger.info(f"Wrote {len(keys)} keys from {checkpoint_path} to {output_path}")


def save_codec_checkpoint(full_checkpoint_path: str, output_path: str,) -> None:
    """
    Extract codec weights from a full model.safetensors and save to a separate file.

    Args:
        full_checkpoint_path: Path to the model.safetensors file containing all components.
        output_path: Path where the codec-only checkpoint will be saved.
            Can be .safetensors or .pt file.
    """
    codec_state_dict = extract_codec_state_dict(full_checkpoint_path)

    if not codec_state_dict:
        raise ValueError(
            f"No codec weights found in {full_checkpoint_path}. " "Expected keys with prefix 'tts_model.audio_codec.'"
        )

    # Determine format from extension
    if output_path.endswith(".safetensors"):
        save_file(codec_state_dict, output_path)
    elif output_path.endswith(".pt") or output_path.endswith(".pth"):
        torch.save(codec_state_dict, output_path)
    else:
        # Default to safetensors
        save_file(codec_state_dict, output_path)

    logger.info(f"Saved codec checkpoint with {len(codec_state_dict)} keys to {output_path}")


def extract_embeddings_state_dict(full_checkpoint_path: str,) -> dict[str, torch.Tensor]:
    """
    Extract embedding weights from a full model.safetensors checkpoint.

    Extracts all keys with prefixes:
    - stt_model.embed_tokens.*

    Args:
        full_checkpoint_path: Path to the model.safetensors file.

    Returns:
        State dict with embedding parameters (weights, biases, etc.).
    """
    state_dict = load_file(full_checkpoint_path)

    embed_prefixes = ["stt_model.embed_tokens."]

    embeddings_state_dict = {}
    for key, value in state_dict.items():
        for prefix in embed_prefixes:
            if key.startswith(prefix):
                # Remove the stt_model. prefix for cleaner keys
                new_key = key.removeprefix("stt_model.")
                embeddings_state_dict[new_key] = value
                break

    return embeddings_state_dict


def extract_rnnt_asr_state_dict(full_checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Extract the RNN-T head (decoder + joint) state dict from a combined S2S+RNNT checkpoint.

    The combined checkpoint stores the RNNT weights under two prefixes:
      - ``stt_model.rnnt_decoder.*``
      - ``stt_model.rnnt_joint.*``

    This function filters those keys and strips the leading ``stt_model.`` so the
    returned dict has keys like ``rnnt_decoder.prediction.*`` and ``rnnt_joint.enc.*``.
    Those keys match the submodule layout expected by
    :func:`nemo.collections.speechlm2.parts.pretrained.setup_rnnt_from_combined_checkpoint`
    and by ``model.load_state_dict(..., strict=False)`` when the model has
    ``rnnt_decoder`` and ``rnnt_joint`` attributes.

    Args:
        full_checkpoint_path: Path to the combined ``model.safetensors`` file.

    Returns:
        State dict with only the RNNT decoder and joint weights, with the
        ``stt_model.`` prefix removed.
    """
    state_dict = load_file(full_checkpoint_path)

    rnnt_prefixes = ("stt_model.rnnt_decoder.", "stt_model.rnnt_joint.")
    rnnt_state_dict = {
        k.removeprefix("stt_model."): v for k, v in state_dict.items() if any(k.startswith(p) for p in rnnt_prefixes)
    }

    return rnnt_state_dict


def save_rnnt_asr_checkpoint(full_checkpoint_path: str, output_path: str) -> None:
    """
    Extract the RNN-T head weights from a combined checkpoint and save to a separate file.

    The combined checkpoint is expected to have been produced by
    ``combine_s2s_rnnt_checkpoint.py``.  Its ``config.json`` sibling should
    contain a ``_rnnt_merge_info`` section with ``decoder_config``,
    ``joint_config``, ``decoder_class``, and ``joint_class`` — these are **not**
    embedded in the saved weights file; callers must read them from ``config.json``
    when reconstructing the modules (see
    :func:`nemo.collections.speechlm2.parts.pretrained.setup_rnnt_from_combined_checkpoint`).

    The saved file contains keys ``rnnt_decoder.*`` and ``rnnt_joint.*`` (the
    ``stt_model.`` prefix is stripped).

    Args:
        full_checkpoint_path: Path to the combined ``model.safetensors``.
        output_path: Destination path for the RNNT-head checkpoint
            (``.safetensors`` or ``.pt``/``.pth``).
    """
    rnnt_state_dict = extract_rnnt_asr_state_dict(full_checkpoint_path)

    if not rnnt_state_dict:
        raise ValueError(
            f"No RNNT ASR weights found in {full_checkpoint_path}. "
            "Expected keys with prefix 'stt_model.rnnt_decoder.' or 'stt_model.rnnt_joint.'"
        )

    if output_path.endswith(".safetensors"):
        save_file(rnnt_state_dict, output_path)
    elif output_path.endswith(".pt") or output_path.endswith(".pth"):
        torch.save(rnnt_state_dict, output_path)
    else:
        save_file(rnnt_state_dict, output_path)

    n_decoder = sum(1 for k in rnnt_state_dict if k.startswith("rnnt_decoder."))
    n_joint = sum(1 for k in rnnt_state_dict if k.startswith("rnnt_joint."))
    logger.info(f"Saved RNNT ASR checkpoint ({n_decoder} decoder tensors, {n_joint} joint tensors) to {output_path}")


def save_embeddings_checkpoint(full_checkpoint_path: str, output_path: str,) -> None:
    """
    Extract embedding weights from a full model.safetensors and save to embeddings.safetensors.

    This creates a standalone embeddings file that can be loaded quickly at runtime
    without needing to load the full checkpoint or vLLM checkpoint files.

    Args:
        full_checkpoint_path: Path to the model.safetensors file.
        output_path: Path to save embeddings.safetensors.
    """
    embeddings_state_dict = extract_embeddings_state_dict(full_checkpoint_path)

    if not embeddings_state_dict:
        raise ValueError(f"No embedding keys found in {full_checkpoint_path}")

    # Log extracted keys and sizes
    logger.info(f"Extracted {len(embeddings_state_dict)} embedding tensors:")
    total_params = 0
    total_bytes = 0
    for key, tensor in embeddings_state_dict.items():
        num_params = tensor.numel()
        num_bytes = tensor.numel() * tensor.element_size()
        total_params += num_params
        total_bytes += num_bytes
        logger.info(
            f"  {key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, size={num_bytes / 1024 / 1024:.2f}MB"
        )

    logger.info(f"Total: {total_params:,} parameters, {total_bytes / 1024 / 1024 / 1024:.2f}GB")

    save_file(embeddings_state_dict, output_path)

    vocab_size, hidden_size = embeddings_state_dict["embed_tokens.weight"].shape
    logger.info(f"Saved embeddings (vocab_size={vocab_size}, hidden_size={hidden_size}) to {output_path}")


# =============================================================================
# Triton Model Repository Generation
# =============================================================================


def _discover_speaker_name(full_checkpoint_path: str) -> str:
    state_dict = load_file(full_checkpoint_path)
    names = [k.split("audio_prompt_latents.")[1] for k in state_dict if "audio_prompt_latents." in k]
    if not names:
        raise ValueError(
            f"No audio_prompt_latents.* keys found in {full_checkpoint_path}. " "Pass --speaker-name explicitly."
        )
    if len(names) > 1:
        logger.warning(f"Multiple speaker latents found {names}; using '{names[0]}'")
    logger.info(f"Auto-discovered speaker name: '{names[0]}'")
    return names[0]


def _apply_tts_inference_config(cfg: DictConfig) -> None:
    """Apply inference-time TTS config modifications (to speech_generation sub-config)."""
    sg = cfg.model.speech_generation
    sg.model.tts_config.use_unshifthed_prompt = True
    with open_dict(cfg.data):
        cfg.data.add_audio_prompt_after_description = True
        if not getattr(cfg.data, "source_sample_rate", None):
            cfg.data.source_sample_rate = 22050
        if not getattr(cfg.data, "target_sample_rate", None):
            cfg.data.target_sample_rate = 22050
        if not getattr(cfg.data, "audio_prompt_duration", None):
            cfg.data.audio_prompt_duration = 3.0
    sg.model.subword_mask_exactly_as_eartts = False
    sg.model.context_hidden_mask_exactly_as_eartts = False
    sg.model.tts_config.disable_eos_prediction = True
    sg.model.inference_force_speech_silence_on_eos = True
    sg.model.use_word_sep_tokenizer = False
    sg.model.num_delay_speech_tokens = 0
    sg.model.pretrained_model = None
    if hasattr(sg.model, "tts_config") and hasattr(sg.model.tts_config, "cas_config"):
        if getattr(sg.model.tts_config.cas_config, "pretrained_tokenizer_name", None) is not None:
            del sg.model.tts_config.cas_config.pretrained_tokenizer_name


def generate_tts_model_init_inputs(
    config_json_path: str,
    full_checkpoint_path: str,
    output_path: str,
    *,
    speaker_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> None:
    """
    Generate tts_model_init_inputs.pt for the TTS model.

    Loads NemotronVoiceChat from the combined checkpoint (config.json +
    model.safetensors), then calls set_init_inputs with the pre-baked
    speaker latent identified by speaker_name.

    Args:
        config_json_path: Path to config.json for the combined checkpoint.
        full_checkpoint_path: Path to model.safetensors.
        output_path: Path where tts_model_init_inputs.pt will be saved.
        speaker_name: Name of pre-baked speaker latent in checkpoint (e.g. "Aria").
        system_prompt: Optional system prompt for TTS init (passed to set_init_inputs).
    """

    logger.info("Generating tts_model_init_inputs.pt...")

    if speaker_name is None:
        speaker_name = _discover_speaker_name(full_checkpoint_path)

    with open(config_json_path, "r") as f:
        raw_config = json.load(f)
    if not (
        isinstance(raw_config.get("model"), dict)
        and "stt" in raw_config["model"]
        and "speech_generation" in raw_config["model"]
    ):
        raise RuntimeError(f"config at {config_json_path} is not a full NemotronVoiceChat combined-checkpoint config")

    full_cfg = DictConfig(raw_config)

    # Ensure required top-level keys
    if "data" not in full_cfg or full_cfg.data is None:
        full_cfg = OmegaConf.merge(
            full_cfg,
            OmegaConf.create(
                {"data": {"target_sample_rate": 22050, "source_sample_rate": 22050, "frame_length": 0.02}}
            ),
        )
    if "exp_manager" not in full_cfg or not hasattr(full_cfg.exp_manager, "explicit_log_dir"):
        with open_dict(full_cfg):
            if "exp_manager" not in full_cfg:
                full_cfg.exp_manager = OmegaConf.create({})
            full_cfg.exp_manager.explicit_log_dir = os.path.join(os.path.dirname(output_path) or ".", "logs")
    if not hasattr(full_cfg.data, "frame_length") or full_cfg.data.frame_length is None:
        with open_dict(full_cfg.data):
            full_cfg.data.frame_length = 0.02

    _apply_tts_inference_config(full_cfg)
    with open_dict(full_cfg):
        full_cfg.model.stt.model.pretrained_s2s_model = None
        full_cfg.model.speech_generation.model.pretrained_model = None
        full_cfg.model.stt.model.pretrained_asr = None
        full_cfg.model.stt.model.pretrained_weights = False
        # zh-phase1 air-gapped support: NemotronVoiceChat -> DuplexSTTModel /
        # DuplexEARTTS load the LLM through cfg.model.stt.model.pretrained_llm
        # and cfg.model.speech_generation.model.pretrained_lm_name; point both
        # at the local files dir. NOTE: downstream code does substring checks
        # ('Nemotron' in pretrained_llm) — the NEMO_LLM_LOCAL_DIR path must
        # keep "Nemotron" in it (see deploy-asr35-zh.md G3).
        local_llm_dir = os.environ.get("NEMO_LLM_LOCAL_DIR", "")
        if local_llm_dir and os.path.isfile(os.path.join(local_llm_dir, "config.json")):
            logger.info(
                f"NEMO_LLM_LOCAL_DIR set - building NemotronVoiceChat with LLM files from {local_llm_dir}"
            )
            full_cfg.model.stt.model.pretrained_llm = local_llm_dir
            full_cfg.model.speech_generation.model.pretrained_lm_name = local_llm_dir

    cfg_dict = OmegaConf.to_container(full_cfg, resolve=True)
    logger.info("Loading NemotronVoiceChat from combined checkpoint...")
    model = NemotronVoiceChat(cfg_dict).eval()

    logger.info(f"Loading weights from {full_checkpoint_path}")
    if full_checkpoint_path.endswith(".safetensors"):
        full_checkpoint = load_file(full_checkpoint_path)
    else:
        full_checkpoint = torch.load(full_checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(full_checkpoint, dict) and "state_dict" in full_checkpoint:
            full_checkpoint = full_checkpoint["state_dict"]
    # Pre-register audio_prompt_latents entries in the ParameterDict so
    # load_state_dict can populate them (they are not present at __init__ time).
    if hasattr(model.tts_model, "maybe_recreate_cached_audio_prompt_latents_structure"):
        model.tts_model.maybe_recreate_cached_audio_prompt_latents_structure(full_checkpoint)

    missing, unexpected = model.load_state_dict(full_checkpoint, strict=False)
    if missing:
        logger.warning(f"Missing keys: {len(missing)}")
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)}")

    tts_model = model.tts_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tts_model = tts_model.to(device)

    logger.info(f"Using pre-baked speaker latent: {speaker_name}")
    logger.info("Setting init inputs...")
    tts_model.set_init_inputs(
        speaker_audio=None, speaker_audio_lens=None, speaker_name=speaker_name, system_prompt=system_prompt,
    )
    init_inputs = tts_model.get_init_inputs(B=1)

    torch.save(init_inputs, output_path)
    logger.info(f"Saved tts_model_init_inputs.pt to {output_path}")
    logger.debug("Init inputs shapes:")
    for key, value in init_inputs.items():
        if value is not None:
            logger.debug(f"  {key}: {value.shape}")
        else:
            logger.debug(f"  {key}: None")


def create_triton_model_config(
    model_name: str = "nemotron-voicechat",
    backend: str = "nemotron-voicechat",
    max_batch_size: int = 0,
    max_sequence_idle_microseconds: int = 300000000,
    system_prompt: str = "You are a helpful AI assistant.",
) -> str:
    """
    Create a Triton model configuration in pbtxt format.

    Args:
        model_name: Name of the model.
        backend: Backend to use (e.g., "python", "pytorch").
        max_batch_size: Maximum batch size (0 for dynamic batching).
        max_sequence_idle_microseconds: Timeout for idle sequences in microseconds.
        system_prompt: Default system prompt for the model.

    Returns:
        String containing the model configuration in pbtxt format.
    """
    import tritonclient.grpc.model_config_pb2 as model_config
    from google.protobuf import text_format

    # Create model config
    config = model_config.ModelConfig()
    config.name = model_name
    config.backend = backend
    config.max_batch_size = max_batch_size

    # Add input
    audio_input = config.input.add()
    audio_input.name = "audio_signal"
    audio_input.data_type = model_config.TYPE_FP32
    audio_input.dims.append(-1)

    # Add input for system prompt
    system_prompt_input = config.input.add()
    system_prompt_input.name = "system_prompt"
    system_prompt_input.data_type = model_config.TYPE_STRING
    system_prompt_input.dims.append(-1)

    # Add input for function response
    function_response_input = config.input.add()
    function_response_input.name = "function_response"
    function_response_input.data_type = model_config.TYPE_STRING
    function_response_input.dims.append(-1)

    # Add output for agent text
    text_output = config.output.add()
    text_output.name = "output_text"
    text_output.data_type = model_config.TYPE_STRING
    text_output.dims.append(-1)

    # Add output for ASR text
    asr_text_output = config.output.add()
    asr_text_output.name = "output_asr_text"
    asr_text_output.data_type = model_config.TYPE_STRING
    asr_text_output.dims.append(-1)

    # Add output for function calling text
    asr_function_output = config.output.add()
    asr_function_output.name = "output_function_text"
    asr_function_output.data_type = model_config.TYPE_STRING
    asr_function_output.dims.append(-1)

    # Add output for audio
    audio_output = config.output.add()
    audio_output.name = "output_audio"
    audio_output.data_type = model_config.TYPE_FP32
    audio_output.dims.append(-1)
    audio_output.dims.append(-1)

    # Add instance group
    instance_group = config.instance_group.add()
    instance_group.count = 1
    instance_group.kind = model_config.ModelInstanceGroup.KIND_GPU

    # Add sequence batching
    sequence_batching = config.sequence_batching
    sequence_batching.max_sequence_idle_microseconds = max_sequence_idle_microseconds

    # Add oldest section
    oldest = sequence_batching.oldest
    oldest.max_candidate_sequences = max_batch_size
    preferred_batch_size = [4, 6, 8, 16, 32, 64, 128, 256, 512, 1024]
    for batch_size in preferred_batch_size:
        if batch_size <= max_batch_size:
            oldest.preferred_batch_size.append(batch_size)
        else:
            break
    oldest.max_queue_delay_microseconds = 10000  # 10ms

    # Add control inputs for sequence management
    # START control
    start_control = sequence_batching.control_input.add()
    start_control.name = "START"
    start_ctrl = start_control.control.add()
    start_ctrl.kind = model_config.ModelSequenceBatching.Control.CONTROL_SEQUENCE_START
    start_ctrl.int32_false_true.append(0)
    start_ctrl.int32_false_true.append(1)

    # READY control
    ready_control = sequence_batching.control_input.add()
    ready_control.name = "READY"
    ready_ctrl = ready_control.control.add()
    ready_ctrl.kind = model_config.ModelSequenceBatching.Control.CONTROL_SEQUENCE_READY
    ready_ctrl.int32_false_true.append(0)
    ready_ctrl.int32_false_true.append(1)

    # END control
    end_control = sequence_batching.control_input.add()
    end_control.name = "END"
    end_ctrl = end_control.control.add()
    end_ctrl.kind = model_config.ModelSequenceBatching.Control.CONTROL_SEQUENCE_END
    end_ctrl.int32_false_true.append(0)
    end_ctrl.int32_false_true.append(1)

    # CORRID control
    corrid_control = sequence_batching.control_input.add()
    corrid_control.name = "CORRID"
    corrid_ctrl = corrid_control.control.add()
    corrid_ctrl.kind = model_config.ModelSequenceBatching.Control.CONTROL_SEQUENCE_CORRID
    corrid_ctrl.data_type = model_config.TYPE_UINT64

    # Add parameters
    config.parameters["system_prompt"].string_value = system_prompt

    # Convert to pbtxt format
    config_text = text_format.MessageToString(config, use_short_repeated_primitives=True)

    return config_text


def generate_triton_model_repository(
    full_checkpoint_path: str,
    output_dir: str,
    speaker_name: Optional[str] = None,
    config_json_path: Optional[str] = None,
) -> None:
    """
    Generate a complete Triton model repository from a full (combined) checkpoint.

    This function creates the directory structure and generates all necessary files
    for a Triton Inference Server deployment:
    - nemotron-voicechat/1/nano-v2-vllm/ (generated via convert_nemotronllm_checkpoint.py)
    - nemotron-voicechat/1/eartts_vllm/ (generated via convert_eartts_checkpoint.py)
    - nemotron-voicechat/1/perception.safetensors
    - nemotron-voicechat/1/codec.safetensors
    - nemotron-voicechat/1/embeddings.safetensors
    - nemotron-voicechat/1/rnnt-asr.safetensors (if checkpoint contains RNNT keys)
    - nemotron-voicechat/1/rnnt_tokenizer/ (if present in checkpoint directory)
    - nemotron-voicechat/1/config.json
    - nemotron-voicechat/config.pbtxt

    The ASR encoder is loaded from the combined checkpoint; no separate .nemo file is needed.

    Args:
        full_checkpoint_path: Path to model.safetensors containing all model weights.
        output_dir: Directory where the model repository will be created.
        speaker_name: Name of pre-baked speaker latent in checkpoint (e.g. "Aria").
        config_json_path: Path to config.json. Auto-discovered from checkpoint directory if None.
    """
    if speaker_name is None:
        speaker_name = _discover_speaker_name(full_checkpoint_path)

    checkpoint_dir = os.path.dirname(os.path.abspath(full_checkpoint_path))

    if config_json_path is None:
        candidate = os.path.join(checkpoint_dir, "config.json")
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"--config not provided and config.json not found in {checkpoint_dir}")
        config_json_path = candidate
        logger.info(f"Auto-discovered config: {config_json_path}")

    import shutil
    import subprocess

    # Create directory structure
    model_name = "nemotron-voicechat"
    model_version = "1"
    model_repo_path = os.path.join(output_dir, model_name)
    model_version_path = os.path.join(model_repo_path, model_version)

    os.makedirs(model_version_path, exist_ok=True)
    logger.info(f"Created model repository directory: {model_version_path}")

    # 1. Copy config.json (first run only) then always patch ASR encoder fields
    dest_config_path = os.path.join(model_version_path, "config.json")
    if os.path.exists(dest_config_path):
        logger.info(f"config.json already exists at {dest_config_path} — skipping copy, applying patch")
    else:
        shutil.copy(config_json_path, dest_config_path)
        logger.info(f"Copied config.json to {dest_config_path}")
    # ASR encoder is in the combined checkpoint; always mark pretrained_weights=False
    # so riva's load_perception builds AudioPerceptionModule from config directly.
    # Applied on every run so reruns don't leave stale values from a prior deploy.
    with open(dest_config_path, "r") as f:
        cfg = json.load(f)
    if "model" in cfg and "stt" in cfg["model"] and "model" in cfg["model"]["stt"]:
        cfg["model"]["stt"]["model"]["pretrained_weights"] = False
        cfg["model"]["stt"]["model"]["pretrained_asr"] = None
        with open(dest_config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        logger.info("Set pretrained_weights=False in config.json (ASR encoder from checkpoint)")

    # 2. Extract and save perception checkpoint
    perception_output = os.path.join(model_version_path, "perception.safetensors")
    if os.path.exists(perception_output):
        logger.info(f"Skipping perception checkpoint (already exists at {perception_output})")
    else:
        logger.info(f"Extracting perception checkpoint to {perception_output}")
        save_perception_checkpoint(full_checkpoint_path, perception_output)

    # 3. Extract and save codec checkpoint
    codec_output = os.path.join(model_version_path, "codec.safetensors")
    if os.path.exists(codec_output):
        logger.info(f"Skipping codec checkpoint (already exists at {codec_output})")
    else:
        logger.info(f"Extracting codec checkpoint to {codec_output}")
        save_codec_checkpoint(full_checkpoint_path, codec_output)

    # 4. Extract and save embeddings
    embeddings_output = os.path.join(model_version_path, "embeddings.safetensors")
    if os.path.exists(embeddings_output):
        logger.info(f"Skipping embeddings (already exists at {embeddings_output})")
    else:
        logger.info(f"Extracting embeddings to {embeddings_output}")
        save_embeddings_checkpoint(full_checkpoint_path, embeddings_output)

    # 5. Extract and save RNNT ASR checkpoint (decoder + joint), if present
    rnnt_asr_output = os.path.join(model_version_path, "rnnt-asr.safetensors")
    if os.path.exists(rnnt_asr_output):
        logger.info(f"Skipping rnnt-asr checkpoint (already exists at {rnnt_asr_output})")
    else:
        rnnt_state_dict = extract_rnnt_asr_state_dict(full_checkpoint_path)
        if rnnt_state_dict:
            logger.info(f"Extracting RNNT ASR checkpoint to {rnnt_asr_output}")
            save_file(rnnt_state_dict, rnnt_asr_output)
            n_decoder = sum(1 for k in rnnt_state_dict if k.startswith("rnnt_decoder."))
            n_joint = sum(1 for k in rnnt_state_dict if k.startswith("rnnt_joint."))
            logger.info(f"Saved RNNT ASR ({n_decoder} decoder tensors, {n_joint} joint tensors)")
        else:
            logger.info("No RNNT ASR keys found in checkpoint — skipping rnnt-asr.safetensors")

    # 5b. Copy rnnt_tokenizer/ directory from checkpoint dir, if present
    src_tokenizer_dir = os.path.join(checkpoint_dir, "rnnt_tokenizer")
    dest_tokenizer_dir = os.path.join(model_version_path, "rnnt_tokenizer")
    if os.path.isdir(src_tokenizer_dir):
        if os.path.exists(dest_tokenizer_dir):
            logger.info(f"Skipping rnnt_tokenizer/ (already exists at {dest_tokenizer_dir})")
        else:
            shutil.copytree(src_tokenizer_dir, dest_tokenizer_dir)
            logger.info(f"Copied rnnt_tokenizer/ to {dest_tokenizer_dir}")
    else:
        logger.info("No rnnt_tokenizer/ found in checkpoint directory — skipping")

    # 5c. Save HuggingFace tokenizer so startup never needs a network download.
    tokenizer_output = os.path.join(model_version_path, "tokenizer")
    if os.path.exists(tokenizer_output):
        logger.info(f"Skipping tokenizer (already exists at {tokenizer_output})")
    else:
        with open(dest_config_path, "r") as f:
            _cfg = json.load(f)
        pretrained_llm = _cfg.get("model", {}).get("stt", {}).get("model", {}).get("pretrained_llm")
        if not pretrained_llm:
            raise RuntimeError("pretrained_llm not found in config.json — cannot bundle tokenizer")
        from transformers import AutoTokenizer as _AutoTokenizer

        # zh-phase1 air-gapped support: prefer a plain local dir holding the
        # LLM's config.json + tokenizer files (e.g. a modelscope/HF snapshot
        # copy) over the HF hub. Mirrored in /s2s/scripts/convert_nemotronllm_checkpoint.py.
        local_llm_dir = os.environ.get("NEMO_LLM_LOCAL_DIR", "")
        if local_llm_dir and os.path.isfile(os.path.join(local_llm_dir, "config.json")):
            logger.info(f"NEMO_LLM_LOCAL_DIR set - bundling tokenizer from {local_llm_dir}")
            pretrained_llm = local_llm_dir

        logger.info(f"Saving tokenizer from {pretrained_llm} to {tokenizer_output} ...")
        _AutoTokenizer.from_pretrained(pretrained_llm).save_pretrained(tokenizer_output)
        logger.info(f"Tokenizer saved to {tokenizer_output}")

    # 6. Generate nano-v2-vllm using convert_nemotronllm_checkpoint.py
    nano_vllm_output = os.path.join(model_version_path, "nano-v2-vllm")
    if os.path.exists(nano_vllm_output) and os.path.isdir(nano_vllm_output):
        logger.info(f"Skipping nano-v2-vllm generation (directory already exists at {nano_vllm_output})")
    else:
        logger.info(f"Generating nano-v2-vllm checkpoint to {nano_vllm_output}")

        # Find the script path relative to this file
        scripts_dir = "/s2s/scripts"
        convert_nemotronllm_script = os.path.join(scripts_dir, "convert_nemotronllm_checkpoint.py")

        if not os.path.exists(convert_nemotronllm_script):
            raise FileNotFoundError(f"convert_nemotronllm_checkpoint.py not found at {convert_nemotronllm_script}")

        nemotron_cmd = [
            "python",
            convert_nemotronllm_script,
            "--checkpoint",
            full_checkpoint_path,
            "--output-dir",
            nano_vllm_output,
            "--config",
            config_json_path,
        ]

        logger.info(f"Running: {' '.join(nemotron_cmd)}")
        subprocess.run(nemotron_cmd, check=True)
        logger.info(f"Successfully generated nano-v2-vllm checkpoint")

    # 7. Generate eartts_vllm using convert_eartts_checkpoint.py
    eartts_vllm_output = os.path.join(model_version_path, "eartts_vllm")
    if os.path.exists(eartts_vllm_output) and os.path.isdir(eartts_vllm_output):
        logger.info(f"Skipping eartts_vllm generation (directory already exists at {eartts_vllm_output})")
    else:
        logger.info(f"Generating eartts_vllm checkpoint to {eartts_vllm_output}")

        scripts_dir = "/s2s/scripts"
        convert_eartts_script = os.path.join(scripts_dir, "convert_eartts_checkpoint.py")

        if not os.path.exists(convert_eartts_script):
            raise FileNotFoundError(f"convert_eartts_checkpoint.py not found at {convert_eartts_script}")

        eartts_cmd = [
            "python",
            convert_eartts_script,
            "--config",
            config_json_path,
            "--model",
            full_checkpoint_path,
            "--outdir",
            eartts_vllm_output,
        ]

        logger.info(f"Running: {' '.join(eartts_cmd)}")
        subprocess.run(eartts_cmd, check=True)
        logger.info(f"Successfully generated eartts_vllm checkpoint")

    # 7b. Patch pretrained_tokenizer_name in eartts_vllm/config.json to local path
    # so vLLM worker subprocesses don't fetch the tokenizer from HuggingFace.
    eartts_config_path = os.path.join(eartts_vllm_output, "config.json")
    if os.path.exists(eartts_config_path):
        with open(eartts_config_path, "r") as f:
            eartts_cfg = json.load(f)
        if "pretrained_tokenizer_name" in eartts_cfg:
            eartts_cfg["pretrained_tokenizer_name"] = tokenizer_output
            with open(eartts_config_path, "w") as f:
                json.dump(eartts_cfg, f, indent=2)
            logger.info(f"Patched pretrained_tokenizer_name in eartts_vllm/config.json to {tokenizer_output}")

    # 8. Generate TTS model init inputs
    os.makedirs(eartts_vllm_output, exist_ok=True)
    tts_init_inputs_path = os.path.join(eartts_vllm_output, "tts_model_init_inputs.pt")
    if os.path.exists(tts_init_inputs_path):
        logger.info(f"Skipping tts_model_init_inputs.pt (already exists at {tts_init_inputs_path})")
    else:
        logger.info(f"Generating tts_model_init_inputs.pt at {tts_init_inputs_path}")
        generate_tts_model_init_inputs(
            config_json_path=config_json_path,
            full_checkpoint_path=full_checkpoint_path,
            output_path=tts_init_inputs_path,
            speaker_name=speaker_name,
        )

    # 10. Create config.pbtxt for Triton
    config_pbtxt_path = os.path.join(model_repo_path, "config.pbtxt")
    if os.path.exists(config_pbtxt_path):
        logger.info(f"Skipping config.pbtxt (already exists at {config_pbtxt_path})")
    else:
        # Generate config using the dedicated function
        config_content = create_triton_model_config(
            model_name="nemotron-voicechat",
            backend="nemotron-voicechat",
            max_batch_size=8,
            max_sequence_idle_microseconds=300000000,  # 5 minutes
            system_prompt="",
        )

        # Write to file
        with open(config_pbtxt_path, "w") as f:
            f.write(config_content)
        logger.info(f"Created config.pbtxt using tritonclient at {config_pbtxt_path}")

    # 11. Create README.md
    readme_path = os.path.join(model_version_path, "README.md")
    if os.path.exists(readme_path):
        logger.info(f"Skipping README.md (already exists at {readme_path})")
    else:
        readme_content = """# Nemotron VoiceChat Model Repository

This directory contains the complete model repository for Nemotron VoiceChat.

## Directory Structure

```
nemotron-voicechat/
├── config.pbtxt                    # Triton model configuration
└── 1/                              # Model version 1
    ├── config.json                 # Model training configuration
    ├── nano-v2-vllm/              # LLM backbone (vLLM format)
    │   ├── model.safetensors
    │   ├── config.json
    │   ├── tokenizer.json
    │   └── ...
    ├── eartts_vllm/               # TTS model (vLLM format)
    │   ├── model.safetensors
    │   ├── model.safetensors.index.json
    │   ├── config.json
    │   └── tts_model_init_inputs.pt
    ├── perception.safetensors     # Audio perception module
    ├── codec.safetensors          # Audio codec
    ├── embeddings.safetensors     # Text and ASR token embeddings
    ├── rnnt-asr.safetensors      # RNN-T decoder + joint
    └── rnnt_tokenizer/            # RNN-T tokenizer files
```

## Usage

Load this model repository with the Triton Inference Server or use it directly
with the Riva server implementation.
"""

        with open(readme_path, "w") as f:
            f.write(readme_content)
        logger.info(f"Created README.md at {readme_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Triton model repository generation completed successfully!")
    logger.info(f"Model repository location: {model_repo_path}")
    logger.info("=" * 60)
