from typing import Any, Dict, List

import torch
import triton_python_backend_utils as pb_utils

from data_types import TritonSeqState
from nemo.collections.speechlm2.modules.ear_tts_vae_codec import CausalConv1dCache
from runtime_setup import DTYPE


class SequenceManager:
    """Owns the per-stream TritonSeqState lifecycle."""

    def __init__(
        self, device: torch.device, num_audio_quantizers: int,
    ):
        self._states: Dict[Any, TritonSeqState] = {}
        self._device = device
        self._num_audio_quantizers = num_audio_quantizers

    # -- dict-like access (drop-in for the old self._seq_state dict) --

    def __getitem__(self, seq_id) -> TritonSeqState:
        return self._states[seq_id]

    def __contains__(self, seq_id) -> bool:
        return seq_id in self._states

    def remove(self, seq_id) -> None:
        if seq_id in self._states:
            del self._states[seq_id]

    # -- lifecycle --

    def init_state(self, seq_id) -> None:
        """Initialize sequence state for a new conversation."""
        logger = pb_utils.Logger
        logger.log_info(f"Initializing sequence state for {seq_id}")

        with torch.no_grad():
            audio_buffer = torch.tensor([[]], dtype=DTYPE, device=self._device)

            generated_text_tokens = torch.empty(1, 5000, device=self._device, dtype=torch.long)
            generated_asr_tokens = torch.empty(1, 5000, device=self._device, dtype=torch.long)
            generated_function_tokens = torch.empty(1, 5000, device=self._device, dtype=torch.long)
            generated_audio_tokens = torch.empty(
                1, 5000, self._num_audio_quantizers, device=self._device, dtype=torch.long,
            )

            speech_state = torch.zeros(1, device=self._device, dtype=torch.long)

        self._states[seq_id] = TritonSeqState(
            seq_id=seq_id,
            start=True,
            end=False,
            audio_buffer=audio_buffer,
            generated_text_tokens=generated_text_tokens,
            generated_asr_tokens=generated_asr_tokens,
            generated_function_tokens=generated_function_tokens,
            generated_audio_tokens=generated_audio_tokens,
            audio_chunk_idx=0,
            codec_cache=CausalConv1dCache(),
            decoder_global_step=0,
            speech_state=speech_state,
        )

    def mark_start(self, seq_id) -> None:
        """Mark the sequence as started."""
        if seq_id not in self._states:
            self.init_state(seq_id)
        self._states[seq_id].start = True
        self._states[seq_id].decoder_global_step = 0

    def mark_end(self, seq_id) -> None:
        """Mark the sequence as ended."""
        if seq_id in self._states:
            self._states[seq_id].end = True

    def is_started(self, seq_id) -> bool:
        """Check if a sequence has been started."""
        return seq_id in self._states

    def is_ended(self, seq_id) -> bool:
        """Check if a sequence has been ended."""
        return seq_id in self._states and self._states[seq_id].end

    def get_active(self) -> List[int]:
        """Get a list of all active sequence IDs."""
        return [seq_id for seq_id, state in self._states.items() if state.start and not state.end]
