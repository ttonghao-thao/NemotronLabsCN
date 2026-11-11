from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from nemo.collections.speechlm2.modules.ear_tts_vae_codec import CausalConv1dCache
from runtime_setup import MODEL_STEPS_PER_CALL


@dataclass
class InferenceOptions:
    batch_size: int = 1
    num_chunks_per_inference: int = MODEL_STEPS_PER_CALL

    num_samples_per_chunk: int = int(16000 * 0.08)
    max_chunks_for_inference: int = 70

    @property
    def max_samples_for_inference(self):
        return self.max_chunks_for_inference * self.num_samples_per_chunk


@dataclass
class TritonSeqState:
    seq_id: int
    audio_buffer: torch.Tensor
    generated_text_tokens: torch.Tensor
    generated_asr_tokens: torch.Tensor
    generated_audio_tokens: torch.Tensor
    generated_function_tokens: torch.Tensor
    audio_chunk_idx: int
    codec_cache: CausalConv1dCache
    input_embeddings: torch.Tensor = None
    decoder_global_step: int = 0
    generated_text: List[str] = field(default_factory=list)
    generated_asr_text: List[str] = field(default_factory=list)
    generated_function_text: List[str] = field(default_factory=list)
    generated_audio: List[torch.Tensor] = field(default_factory=list)
    start: bool = False
    end: bool = False
    speech_state: torch.Tensor = None
    vllm_llm_generator = None
    vllm_llm_start = False
    system_prompt: str = ""
    tool_ack_messages: Dict[str, List[str]] = field(default_factory=dict)  # tool_name -> ack_messages

    # Function calling state
    fn_call: Dict[str, Any] = field(
        default_factory=lambda: {
            "state": "idle",
            "request": "",
            "response": "",
            "response_tokens": deque(),
            "ack_tokens": deque(),
            "frames_in_state": 0,
            "extract_cancelled": False,
            "inject_cancelled": False,
        }
    )
    # Function calling request and response timeouts in number of 80ms frames
    fn_call_request_timeout: int = (10 / 0.08)  # 10 seconds
    fn_call_response_timeout: int = (10 / 0.08)  # 10 seconds

    # RNNT streaming state for EOU-only path (NeMo-style low-level decoder.predict + joint.joint)
    rnnt_eou_state: Optional[dict] = None

    # RNNT transcript display side-channel. This is intentionally separate from
    # rnnt_eou_state and the RNNT turn-taking counters below.
    rnnt_display_state: Optional[dict] = None
    rnnt_display_emitted_len: int = 0
    rnnt_display_turn_open: bool = False
    rnnt_display_pending_text: str = ""

    # RNNT-based speech activity detection
    rnnt_user_speaking: bool = False  # speech_confirmed: True after BOU threshold, False after EOU/BOS/EOS
    rnnt_silent_frames: int = 0  # blank_count: consecutive blank frames (NeMo: blank_count)
    rnnt_consecutive_speech_frames: int = 0  # nonblank_consec: consecutive non-blank frames (NeMo: nonblank_consec)
    rnnt_nonblank_total: int = 0  # nonblank_total: cumulative non-blank frames (NeMo: nonblank_total)
    rnnt_rolling_density: float = 0.0  # EMA of is_speech; adapts BOU min-frames threshold
    rnnt_agent_speaking: bool = False  # True from agent BOS until agent EOS (NeMo: agent_speaking)
    rnnt_first_turn: bool = True  # True until first agent BOS; enables relaxed BOU threshold
    rnnt_agent_talking_frames: int = 0  # frames since agent BOS (NeMo: _agent_talking_frames)
    rnnt_turn_text_tokens: int = 0  # non-special tokens in current agent turn (NeMo: _turn_text_tokens)

    # Set when decoder_global_step reaches the pre-allocated token buffer limit.
    # Once set, infer() returns an error response instead of processing the sequence.
    steps_exhausted: bool = False

    # TTS PAD-silence gating state (per TTS PAD-silence fix)
    # agent_idle gates PAD→silence substitution in vllm_tts_step:
    #   True  → no agent turn open; all PAD frames produce silence
    #   False → turn open; PAD frames are allowed until in-turn tail budget is exhausted
    # Reset to True on EOS, BOS→False, and whenever FC returns to idle.
    tts_agent_idle: bool = True
    tts_in_turn_content: int = 0  # content tokens emitted since last BOS
    tts_in_turn_pads: int = 0  # consecutive PAD frames since last content token


@dataclass
class SeqInfo:
    start_flags: torch.Tensor
    end_flags: torch.Tensor
    sequence_ids: torch.Tensor
    seq_to_batch_idx: Dict[int, int]


@dataclass
class InferenceResult:
    generated_text: str
    generated_asr_text: str
    generated_function_text: str
    generated_audio: torch.Tensor
    skipped: bool = False
    error: Optional[str] = None
