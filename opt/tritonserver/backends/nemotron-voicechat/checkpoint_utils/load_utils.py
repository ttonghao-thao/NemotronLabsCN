import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
from nemo.collections.asr.models import ASRModel
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.modules import AudioPerceptionModule
from nemo.collections.speechlm2.modules.ear_tts_vae_codec import RVQVAEModel
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_nemo
from omegaconf import DictConfig, OmegaConf, open_dict
from safetensors.torch import load_file
from torch import nn
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm_components import EARTTSvLLM

logger = logging.getLogger(__name__)

# Configurable via environment variables
LLM_GPU_MEM_UTIL = float(os.environ.get("LLM_GPU_MEM_UTIL", "0.45"))
TTS_GPU_MEM_UTIL = float(os.environ.get("TTS_GPU_MEM_UTIL", "0.4"))

# =============================================================================
# Configuration Types
# =============================================================================

PerceptionCheckpointType = Literal["pytorch", "cudagraph"]
LLMCheckpointType = Literal["pytorch", "vllm"]
TTSCheckpointType = Literal["pytorch", "vllm"]
CodecCheckpointType = Literal["pytorch"]


@dataclass
class PerceptionConfig:
    """Configuration for the perception (audio encoder) module."""

    state_dict_path: str
    dtype: torch.dtype
    device: torch.device
    checkpoint_type: PerceptionCheckpointType


@dataclass
class LLMConfig:
    """Configuration for the LLM (language model) module."""

    path: str
    dtype: torch.dtype
    device: torch.device
    checkpoint_type: LLMCheckpointType


@dataclass
class TTSConfig:
    """Configuration for the TTS (text-to-speech) module."""

    path: str
    dtype: torch.dtype
    device: torch.device
    prompt_input_path: str
    checkpoint_type: TTSCheckpointType
    guidance_scale: float


@dataclass
class CodecConfig:
    """Configuration for the audio codec module."""

    path: str
    dtype: torch.dtype
    device: torch.device
    checkpoint_type: CodecCheckpointType = "pytorch"


@dataclass
class RNNTConfig:
    """Configuration for the RNN-T (streaming ASR) module."""

    dtype: torch.dtype
    device: torch.device


@dataclass
class VoiceChatModelLoadingConfig:
    """
    Complete configuration for loading a NemotronVoiceChat model.

    This aggregates all component configurations along with the training
    config.json path needed to instantiate the model architecture.
    """

    config_json_path: str
    embeddings_path: str
    perception: PerceptionConfig
    llm: LLMConfig
    tts: TTSConfig
    codec: CodecConfig
    rnnt: Optional[RNNTConfig] = None

    def load_training_config(self) -> DictConfig:
        """Load the training configuration from config.json."""
        import json

        with open(self.config_json_path, "r") as f:
            cfg = json.load(f)
        return OmegaConf.create(cfg)


class ModelComponentsBase(ABC):
    @abstractmethod
    def perception(self) -> AudioPerceptionModule:
        pass

    @abstractmethod
    def codec(self) -> RVQVAEModel:
        pass

    @abstractmethod
    def llm(self) -> AsyncLLM:
        pass

    @abstractmethod
    def tts(self) -> AsyncLLM:
        pass

    @abstractmethod
    def tokenizer(self) -> Any:
        pass

    @abstractmethod
    def embed_tokens(self) -> nn.Embedding:
        pass

    @abstractmethod
    def prompt_input(self) -> torch.Tensor:
        pass

    @abstractmethod
    def device(self) -> torch.device:
        pass

    @abstractmethod
    def model_cfg(self) -> DictConfig:
        pass


class RivaModelComponents(ModelComponentsBase):
    def __init__(
        self,
        perception: AudioPerceptionModule,
        codec: RVQVAEModel,
        llm_engine: AsyncLLM,
        llm_sampling_params: SamplingParams,
        llm_custom_input_specs: list[dict],
        tts_engine: AsyncLLM,
        prompt_input: torch.Tensor,
        model_cfg: DictConfig,
        tokenizer: Any,
        embed_tokens: nn.Embedding,
        device: torch.device,
        rnnt_decoder: Any = None,
    ):
        self._perception = perception
        self._codec = codec
        self.llm_engine = llm_engine
        self.llm_sampling_params = llm_sampling_params
        self.llm_custom_input_specs = llm_custom_input_specs
        self.tts_engine = tts_engine
        self._prompt_input = prompt_input
        self._model_cfg = model_cfg
        self._tokenizer = tokenizer
        self._embed_tokens = embed_tokens
        self._device = device
        self._rnnt_decoder = rnnt_decoder

    @property
    def perception(self) -> AudioPerceptionModule:
        return self._perception

    @property
    def codec(self) -> RVQVAEModel:
        return self._codec

    @property
    def llm(self) -> AsyncLLM:
        return self.llm_engine

    @property
    def tts(self) -> AsyncLLM:
        return self.tts_engine

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self._embed_tokens

    @property
    def prompt_input(self) -> torch.Tensor:
        return self._prompt_input

    @property
    def model_cfg(self) -> DictConfig:
        return self._model_cfg

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def rnnt_decoder(self) -> Any:
        return self._rnnt_decoder


def dtype_from_str(dtype_str: str) -> torch.dtype:
    """Convert a string dtype to torch.dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}. Valid options: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def create_voicechat_loading_config(
    config_json_path: str,
    embeddings_path: str,
    perception_state_dict_path: str,
    perception_dtype: str,
    perception_checkpoint_type: PerceptionCheckpointType,
    llm_path: str,
    llm_dtype: str,
    llm_checkpoint_type: LLMCheckpointType,
    tts_path: str,
    tts_dtype: str,
    prompt_input_path: str,
    tts_checkpoint_type: TTSCheckpointType,
    tts_guidance_scale: float,
    codec_path: str,
    codec_dtype: str,
    codec_checkpoint_type: CodecCheckpointType = "pytorch",
    rnnt_dtype: str = "bfloat16",
) -> VoiceChatModelLoadingConfig:
    """
    Factory function to create a VoiceChatModelLoadingConfig from string arguments.

    This is useful when parsing command-line arguments or loading from
    environment variables where everything comes as strings.

    Args:
        config_json_path: Path to the training config.json file.
        perception_dtype: Data type for perception model (e.g., "bfloat16").
        perception_checkpoint_type: Either "pytorch" or "cudagraph".
        llm_path: Path to the LLM checkpoint.
        llm_dtype: Data type for LLM (e.g., "bfloat16").
        llm_checkpoint_type: Either "pytorch" or "vllm".
        tts_path: Path to the TTS checkpoint.
        tts_dtype: Data type for TTS (e.g., "float32").
        prompt_input_path: Path to the prompt input file.
        tts_guidance_scale: Guidance scale for TTS (e.g., 1.0).
        tts_checkpoint_type: Either "pytorch" or "vllm".
        codec_path: Path to the audio codec checkpoint.
        codec_dtype: Data type for codec (e.g., "float32").
        codec_checkpoint_type: Currently only "pytorch" is supported.
        embeddings_path: Path to embedding safetensors file.
        rnnt_dtype: Data type for RNN-T (e.g., "bfloat16").

    Returns:
        A fully configured VoiceChatModelLoadingConfig instance.
    """
    device_obj = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return VoiceChatModelLoadingConfig(
        config_json_path=config_json_path,
        embeddings_path=embeddings_path,
        perception=PerceptionConfig(
            state_dict_path=perception_state_dict_path,
            dtype=dtype_from_str(perception_dtype),
            device=device_obj,
            checkpoint_type=perception_checkpoint_type,
        ),
        llm=LLMConfig(
            path=llm_path, dtype=dtype_from_str(llm_dtype), device=device_obj, checkpoint_type=llm_checkpoint_type,
        ),
        tts=TTSConfig(
            path=tts_path,
            dtype=dtype_from_str(tts_dtype),
            device=device_obj,
            prompt_input_path=prompt_input_path,
            checkpoint_type=tts_checkpoint_type,
            guidance_scale=tts_guidance_scale,
        ),
        codec=CodecConfig(
            path=codec_path,
            dtype=dtype_from_str(codec_dtype),
            device=device_obj,
            checkpoint_type=codec_checkpoint_type,
        ),
        rnnt=RNNTConfig(dtype=dtype_from_str(rnnt_dtype), device=device_obj,),
    )


def default_voicechat_loading_config(model_repo_path: str,):

    config_json_path = os.path.join(model_repo_path, "config.json")
    perception_state_dict_path = os.path.join(model_repo_path, "perception.safetensors")
    llm_path = os.path.join(model_repo_path, "nano-v2-vllm")
    tts_path = os.path.join(model_repo_path, "eartts_vllm")

    tts_config_path = os.path.join(tts_path, "config.json")
    with open(tts_config_path, "r") as f:
        tts_config = json.load(f)
    tts_guidance_scale = tts_config["guidance_scale"]

    prompt_input_path = os.path.join(tts_path, "tts_model_init_inputs.pt")
    codec_path = os.path.join(model_repo_path, "codec.safetensors")
    embeddings_path = os.path.join(model_repo_path, "embeddings.safetensors")

    return create_voicechat_loading_config(
        config_json_path=config_json_path,
        embeddings_path=embeddings_path,
        perception_state_dict_path=perception_state_dict_path,
        perception_dtype="bfloat16",
        perception_checkpoint_type="cudagraph",
        llm_path=llm_path,
        llm_dtype="bfloat16",
        llm_checkpoint_type="vllm",
        tts_path=tts_path,
        tts_dtype="float32",
        tts_checkpoint_type="vllm",
        tts_guidance_scale=tts_guidance_scale,
        prompt_input_path=prompt_input_path,
        codec_path=codec_path,
        codec_dtype="float32",
        codec_checkpoint_type="pytorch",
        rnnt_dtype="float32",
    )


def load_perception(
    model_cfg: DictConfig, perception_loading_cfg: PerceptionConfig,
):
    pretrained_asr = model_cfg.model.stt.model.get("pretrained_asr", None)
    use_pretrained_weights = model_cfg.model.stt.model.get("pretrained_weights", True)

    if pretrained_asr and use_pretrained_weights:
        user_encoder_config = {}
        if "encoder" in model_cfg.model.stt.model.perception:
            user_encoder_config = OmegaConf.to_container(model_cfg.model.stt.model.perception.encoder, resolve=True)

        asr = load_pretrained_nemo(ASRModel, pretrained_asr).eval()

        with open_dict(model_cfg):
            model_cfg.model.stt.model.perception.preprocessor = asr.cfg.preprocessor
            model_cfg.model.stt.model.perception.encoder = asr.cfg.encoder
            model_cfg.model.stt.model.perception.output_dim = 4480
            # Override with user-specified encoder parameters
            if user_encoder_config:
                for key, value in user_encoder_config.items():
                    if value is not None:  # Only override if user explicitly set a value
                        model_cfg.model.stt.model.perception.encoder[key] = value
    else:
        logger.info("ASR encoder packaged in checkpoint — using perception config from config.json")
        with open_dict(model_cfg):
            model_cfg.model.stt.model.perception.output_dim = 4480

    perception = AudioPerceptionModule(model_cfg.model.stt.model.perception).train()
    # perception.load_state_dict(asr.state_dict(), strict=False)
    logger.debug(f"Load perception state dict from {perception_loading_cfg.state_dict_path}")
    state_dict = load_file(perception_loading_cfg.state_dict_path)
    perception.load_state_dict(state_dict, strict=False)
    perception = perception.to(perception_loading_cfg.device)

    perception.eval()
    perception = perception.to(dtype=torch.bfloat16)

    if perception_loading_cfg.checkpoint_type == "cudagraph":
        logger.debug("Convert perception to CudaGraph Pool")
        from perception_cudagraph import PerceptionCudaGraphPool

        # Use PerceptionCudaGraphPool with predefined batch sizes (1, 4, 8)
        # Automatically picks the closest higher batch size for inference
        perception = PerceptionCudaGraphPool(
            model=perception,
            batch_sizes=(1, 4, 8),
            dtype=perception_loading_cfg.dtype,
            device=perception_loading_cfg.device,
        )
        perception.eval()

    return perception


def load_codec(
    model_cfg: DictConfig, codec_loading_cfg: CodecConfig,
):
    codec = RVQVAEModel(model_cfg.model.speech_generation.model.codec_config)
    state_dict = load_file(codec_loading_cfg.path)
    codec.load_state_dict(state_dict, strict=False)
    codec.to(codec_loading_cfg.dtype)
    codec.to(device=codec_loading_cfg.device)
    codec.eval()

    return codec


def load_llm_backbone_in_vllm(engine_path: str, tokenizer_path: Optional[str] = None):
    engine_args = AsyncEngineArgs(
        model=engine_path,
        tokenizer=tokenizer_path,
        max_model_len=6144,
        max_num_batched_tokens=768,
        gpu_memory_utilization=LLM_GPU_MEM_UTIL,
        trust_remote_code=True,
        mamba_ssm_cache_dtype="float32",
        dtype="bfloat16",
        # cudagraph_mode=PIECEWISE (instead of the v1 default FULL_AND_PIECEWISE). The FULL decode
        # cudagraph captures the whole model.forward — including `inputs_embeds = kwargs.get(
        # "combined_embeds")` — and freezes the custom audio-embedding input at its capture-time
        # (dummy-zero) value, so generated tokens ignore the real audio and the agent text is
        # gibberish. (vLLM 0.22.0's decode effectively ran PIECEWISE for this model, so it never hit
        # this; vLLM 0.24.0 enabled the FULL decode graph. cudagraph_copy_inputs=True does NOT cover
        # custom kwargs.) PIECEWISE keeps torch.compile + piecewise cudagraphs (most of the perf) and
        # reads the real embeddings each step. Proper long-term fix: register the custom-input buffer
        # as a FULL-cudagraph input in the vLLM fork's gpu_model_runner.
        compilation_config={"cudagraph_mode": "PIECEWISE"},
    )

    vllm_config = engine_args.create_engine_config()
    custom_input_specs = vllm_config.model_config.custom_input_specs
    engine = AsyncLLM.from_vllm_config(vllm_config)
    return (
        engine,
        SamplingParams(max_tokens=100000, temperature=0.0, seed=None, stop=[], stop_token_ids=[], ignore_eos=True,),
        custom_input_specs,
    )


def load_tts_model_in_vllm(engine_path: str, tts_config: TTSConfig, tokenizer_path: Optional[str] = None):
    return EARTTSvLLM(
        model_path=engine_path,
        tokenizer_path=tokenizer_path,
        config={
            "max_model_len": 6144,
            "max_num_batched_tokens": 768,
            "gpu_memory_utilization": TTS_GPU_MEM_UTIL,
            "max_tokens": 100000,
            "guidance_scale": tts_config.guidance_scale,
        },
    )


def get_tokenizer(model_cfg: DictConfig):
    tokenizer = AutoTokenizer(model_cfg.model.stt.model.pretrained_llm, use_fast=True)
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<SPECIAL_12>"
    return tokenizer


def load_embeddings(
    embeddings_path: str, dtype: torch.dtype = torch.bfloat16, device: torch.device = None,
) -> nn.Embedding:
    """
    Load embed_tokens from embeddings.safetensors.

    Args:
        embeddings_path: Path to the embeddings.safetensors file.
        dtype: Data type for the embedding layers.
        device: Device to load the embeddings to.

    Returns:
        embed_tokens as nn.Embedding module.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    state_dict = load_file(embeddings_path)

    embed_tokens_weight = state_dict["embed_tokens.weight"]

    vocab_size, hidden_size = embed_tokens_weight.shape

    # Create embedding layers
    embed_tokens = nn.Embedding(vocab_size, hidden_size)
    embed_tokens.weight.data = embed_tokens_weight.to(dtype=dtype)
    embed_tokens = embed_tokens.to(device=device)
    embed_tokens.eval()

    logger.info(f"Loaded embeddings: vocab_size={vocab_size:,}, hidden_size={hidden_size}, dtype={dtype}")

    return embed_tokens


def load_rnnt_decoder(rnnt_cfg: Optional[RNNTConfig], model_dir: Optional[str] = None):
    """
    Load RNN-T decoder+joint from the combined checkpoint.

    Reads _rnnt_merge_info from config.json in model_dir, instantiates
    decoder/joint from the saved configs, loads weights from
    rnnt-asr.safetensors, and loads the tokenizer from rnnt_tokenizer/.

    Args:
        rnnt_cfg: RNNTConfig with device and dtype. If None, RNN-T is disabled.
        model_dir: Directory containing config.json and rnnt-asr.safetensors.

    Returns:
        RNNTBPEDecoding or RNNTDecoding object.

    Raises:
        RuntimeError: if the decoder cannot be loaded for any reason.
    """
    if rnnt_cfg is None:
        raise RuntimeError("RNN-T config not provided; cannot load RNN-T decoder")

    try:
        import glob as _glob

        from nemo.collections.asr.parts.submodules.rnnt_decoding import (
            RNNTBPEDecoding,
            RNNTBPEDecodingConfig,
            RNNTDecoding,
            RNNTDecodingConfig,
        )
        from omegaconf import OmegaConf

        if model_dir is None:
            raise RuntimeError("RNN-T: model_dir not provided — cannot load RNN-T decoder")
        config_json_path = os.path.join(model_dir, "config.json")
        if os.path.isfile(config_json_path):
            with open(config_json_path) as _f:
                _cfg_dict = json.load(_f)
            rnnt_merge_info = _cfg_dict.get("_rnnt_merge_info")
            if rnnt_merge_info:
                logger.info("Found _rnnt_merge_info in config.json — loading RNNT from combined checkpoint")
                decoder_cfg = OmegaConf.create(rnnt_merge_info.get("decoder_config", {}))
                joint_cfg = OmegaConf.create(rnnt_merge_info.get("joint_config", {}))
                decoder_cls_name = rnnt_merge_info.get(
                    "decoder_class", "nemo.collections.asr.modules.rnnt.RNNTDecoder"
                )
                joint_cls_name = rnnt_merge_info.get("joint_class", "nemo.collections.asr.modules.rnnt.RNNTJoint")

                def _import_cls(fqn):
                    import importlib

                    mod_path, cls_name = fqn.rsplit(".", 1)
                    return getattr(importlib.import_module(mod_path), cls_name)

                decoder = _import_cls(decoder_cls_name).from_config_dict(decoder_cfg)
                joint = _import_cls(joint_cls_name).from_config_dict(joint_cfg)

                safetensors_path = os.path.join(model_dir, "rnnt-asr.safetensors")
                if os.path.isfile(safetensors_path):
                    rnnt_state_dict = load_file(safetensors_path)
                    decoder_sd = {
                        k.removeprefix("rnnt_decoder."): v
                        for k, v in rnnt_state_dict.items()
                        if k.startswith("rnnt_decoder.")
                    }
                    joint_sd = {
                        k.removeprefix("rnnt_joint."): v
                        for k, v in rnnt_state_dict.items()
                        if k.startswith("rnnt_joint.")
                    }
                    decoder.load_state_dict(decoder_sd, strict=False)
                    joint.load_state_dict(joint_sd, strict=False)
                    logger.info(
                        f"Loaded RNNT weights from {safetensors_path}: "
                        f"{len(decoder_sd)} decoder, {len(joint_sd)} joint tensors"
                    )
                else:
                    # Fail closed: metadata says RNN-T exists, but its weights are missing.
                    # Loading with randomly-initialized decoder/joint would silently produce
                    # garbage transcripts, so refuse instead of continuing.
                    raise RuntimeError(
                        "_rnnt_merge_info present in config.json but rnnt-asr.safetensors not found "
                        f"at {safetensors_path}; refusing to load RNN-T with uninitialized weights"
                    )

                decoder = decoder.to(dtype=rnnt_cfg.dtype).eval().to(rnnt_cfg.device)
                joint = joint.to(dtype=rnnt_cfg.dtype).eval().to(rnnt_cfg.device)
                joint_vocabulary = getattr(joint, "vocabulary", None)

                tokenizer_dir = os.path.join(model_dir, "rnnt_tokenizer")
                asr_tokenizer = None
                if os.path.isdir(tokenizer_dir):
                    import glob as _glob

                    spm_files = _glob.glob(os.path.join(tokenizer_dir, "*.model"))
                    if spm_files:
                        from nemo.collections.common.tokenizers import SentencePieceTokenizer

                        asr_tokenizer = SentencePieceTokenizer(model_path=spm_files[0])
                        _inner = getattr(asr_tokenizer, "tokenizer", None)
                        if _inner is not None and callable(getattr(_inner, "vocab_size", None)):
                            _inner.vocab_size = _inner.vocab_size()
                        # RNNTBPEDecoding reads tokenizer.supported_punctuation; patch if absent
                        if not hasattr(asr_tokenizer, "supported_punctuation"):
                            asr_tokenizer.supported_punctuation = None
                        logger.info(f"Loaded RNNT tokenizer from {spm_files[0]}")

                strategy = "greedy_batch"
                if asr_tokenizer is not None:
                    decoding_cfg = OmegaConf.structured(RNNTBPEDecodingConfig())
                    decoding_cfg.strategy = strategy
                    decoding_cfg.greedy.max_symbols_per_step = 10
                    decoding_cfg.greedy.use_cuda_graph_decoder = False
                    rnnt_decoder = RNNTBPEDecoding(
                        decoding_cfg=decoding_cfg, decoder=decoder, joint=joint, tokenizer=asr_tokenizer,
                    )
                    logger.info(f"RNN-T: RNNTBPEDecoding loaded from combined checkpoint (strategy={strategy})")
                elif joint_vocabulary is not None:
                    decoding_cfg = OmegaConf.structured(RNNTDecodingConfig())
                    decoding_cfg.strategy = strategy
                    decoding_cfg.greedy.max_symbols_per_step = 10
                    rnnt_decoder = RNNTDecoding(
                        decoding_cfg=decoding_cfg, decoder=decoder, joint=joint, vocabulary=list(joint_vocabulary),
                    )
                    logger.info(f"RNN-T: RNNTDecoding loaded from combined checkpoint (strategy={strategy})")
                else:
                    raise RuntimeError("No tokenizer or vocabulary for RNNT combined checkpoint")
                return rnnt_decoder

        raise RuntimeError(f"No _rnnt_merge_info found in {os.path.join(model_dir, 'config.json')}")

    except Exception as e:
        logger.error(f"Failed to initialize RNN-T decoding: {e}", exc_info=True)
        raise


def load_nemotron_voicechat(model_repo_path: str,):
    start_time = time.perf_counter()
    loading_config = default_voicechat_loading_config(model_repo_path)
    model_cfg = loading_config.load_training_config()

    cfg = model_cfg

    cfg.model.stt.model.pretrained_s2s_model = None
    cfg.model.speech_generation.model.pretrained_model = None
    if cfg.model.stt.model.get("pretrained_asr", None):
        cfg.model.stt.model.pretrained_asr = os.path.join(
            model_repo_path, os.path.basename(cfg.model.stt.model.pretrained_asr)
        )

    # Require the tokenizer to be bundled in the model repo. HF_HUB_OFFLINE=1 is set
    # in the container so any fallback to HuggingFace would fail anyway — fail fast here
    # with a clear message instead of a cryptic network error deep in model loading.
    local_tokenizer = os.path.join(model_repo_path, "tokenizer")
    if not os.path.isdir(local_tokenizer):
        raise FileNotFoundError(
            f"Bundled tokenizer not found at {local_tokenizer}. "
            "Re-run deploy_s2s_model.sh to regenerate the model repo with the tokenizer included."
        )
    logger.info(f"Using local tokenizer from model repo: {local_tokenizer}")
    cfg.model.stt.model.pretrained_llm = local_tokenizer

    logger.info(f"before setting - torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}")
    logger.info(f"before setting - torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")
    logger.info(f"before setting - torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}")

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")

    logger.info(f"after setting - torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}")
    logger.info(f"after setting - torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")
    logger.info(f"after setting - torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}")

    logger.info("=" * 60)
    perception_start_time = time.perf_counter()

    perception = load_perception(model_cfg, loading_config.perception)

    perception_end_time = time.perf_counter()
    logger.info(f"Time taken to load perception: {perception_end_time - perception_start_time} seconds")
    logger.info("=" * 60)

    codec_start_time = time.perf_counter()

    codec = load_codec(model_cfg, loading_config.codec)

    codec_end_time = time.perf_counter()
    logger.info(f"Time taken to load codec: {codec_end_time - codec_start_time} seconds")
    logger.info("=" * 60)

    llm_start_time = time.perf_counter()

    llm_engine, llm_sampling_params, llm_custom_input_specs = load_llm_backbone_in_vllm(
        loading_config.llm.path, tokenizer_path=local_tokenizer if os.path.isdir(local_tokenizer) else None,
    )

    llm_end_time = time.perf_counter()
    logger.info(f"Time taken to load LLM: {llm_end_time - llm_start_time} seconds")
    logger.info("=" * 60)

    tts_start_time = time.perf_counter()

    # zh-phase1 dual-GPU support: place the TTS vLLM engine on a different
    # physical GPU than the LLM (created above, on GPU 0). vLLM EngineCore
    # subprocesses inherit the environment at creation time, so temporarily
    # point CUDA_VISIBLE_DEVICES at TTS_CUDA_VISIBLE_DEVICES; unset, behavior
    # is identical to stock. Restored immediately after creation so the rest
    # of the Triton stub still sees all GPUs. Note: this module's logger has
    # no INFO handler in the Triton backend path — verify placement with
    # nvidia-smi, not docker logs.
    tts_visible_devices = os.environ.pop("TTS_CUDA_VISIBLE_DEVICES", None)
    _prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if tts_visible_devices:
            logger.info(f"Placing TTS engine on CUDA_VISIBLE_DEVICES={tts_visible_devices}")
            os.environ["CUDA_VISIBLE_DEVICES"] = tts_visible_devices
        tts_engine = load_tts_model_in_vllm(
            loading_config.tts.path,
            loading_config.tts,
            tokenizer_path=local_tokenizer if os.path.isdir(local_tokenizer) else None,
        )
    finally:
        if tts_visible_devices:
            if _prev_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = _prev_cvd

    tts_end_time = time.perf_counter()
    logger.info(f"Time taken to load TTS: {tts_end_time - tts_start_time} seconds")
    logger.info("=" * 60)

    # Load tokenizer
    tokenizer = get_tokenizer(model_cfg)

    # Load embedding layers from standalone embeddings.safetensors
    # These are needed for computing BOS embeddings, etc. when using vLLM
    embeddings_start_time = time.perf_counter()
    embed_tokens = load_embeddings(
        embeddings_path=loading_config.embeddings_path,
        dtype=loading_config.perception.dtype,
        device=loading_config.perception.device,
    )
    embeddings_end_time = time.perf_counter()
    logger.info(f"Time taken to load embeddings: {embeddings_end_time - embeddings_start_time} seconds")
    logger.info("=" * 60)

    # Load RNN-T decoding (decoder + joint). Required for a combined checkpoint:
    # load_rnnt_decoder() raises if _rnnt_merge_info or rnnt-asr.safetensors is missing.
    rnnt_start_time = time.perf_counter()
    rnnt_decoder = load_rnnt_decoder(rnnt_cfg=loading_config.rnnt, model_dir=model_repo_path)
    rnnt_end_time = time.perf_counter()
    logger.info(f"Time taken to load RNN-T: {rnnt_end_time - rnnt_start_time} seconds")
    logger.info("=" * 60)

    prompt_input = torch.load(loading_config.tts.prompt_input_path)

    end_time = time.perf_counter()
    logger.info(f"Time taken to load model: {end_time - start_time} seconds")

    return RivaModelComponents(
        perception=perception,
        codec=codec,
        llm_engine=llm_engine,
        llm_sampling_params=llm_sampling_params,
        llm_custom_input_specs=llm_custom_input_specs,
        tts_engine=tts_engine,
        prompt_input=prompt_input,
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        embed_tokens=embed_tokens,
        device=loading_config.perception.device,
        rnnt_decoder=rnnt_decoder,
    )
