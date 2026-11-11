import json

import torch
from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2.modules import AudioPerceptionModule
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_nemo
from omegaconf import DictConfig, OmegaConf, open_dict


def load_perception_model(config_file: str, model_path: str):
    with open(config_file, "r") as f:
        cfg = json.load(f)

    cfg = DictConfig(cfg)

    user_encoder_config = None
    if "encoder" in cfg.model.stt.model.perception:
        user_encoder_config = OmegaConf.to_container(cfg.model.stt.model.perception.encoder, resolve=True)

    asr = load_pretrained_nemo(ASRModel, model_path).eval()

    with open_dict(cfg):
        cfg.model.stt.model.perception.preprocessor = asr.cfg.preprocessor
        cfg.model.stt.model.perception.encoder = asr.cfg.encoder
        cfg.model.stt.model.perception.output_dim = 4480
        # Override with user-specified encoder parameters
        if user_encoder_config:
            for key, value in user_encoder_config.items():
                if value is not None:  # Only override if user explicitly set a value
                    cfg.model.stt.model.perception.encoder[key] = value
    perception = AudioPerceptionModule(cfg.model.stt.model.perception)
    perception.load_state_dict(asr.state_dict(), strict=False)
    perception.eval()  # Use eval mode for deterministic batch norm and disabled dropout

    return perception


class PerceptionCudaGraph(torch.nn.Module):
    """
    CUDA Graph optimized perception module.

    Strategy: Decouple preprocessing (spectrogram) from encoder.
    - Preprocessing runs outside CUDA graph on actual-sized input (no padding artifacts)
    - Spectrogram is padded to fixed size
    - Encoder runs through CUDA graph with padded spectrogram

    Batch Size Handling:
    - Graph is captured with max_batch_size (default=32)
    - Smaller batches are zero-padded to max_batch_size before graph replay
    - Output is sliced back to the actual batch size after replay
    - This allows a single graph to handle all batch sizes <= max_batch_size
    """

    def __init__(
        self, model, max_spec_frames=561, max_batch_size=32, dtype=torch.float32, device=torch.device("cuda"),
    ):
        super(PerceptionCudaGraph, self).__init__()
        self.model = model
        self.preprocessor = model.preprocessor
        self.dtype = dtype
        self.device = device
        self.max_batch_size = max_batch_size
        self.model.eval()

        # Determine spectrogram feature dimension by running a sample through preprocessor
        with torch.no_grad():
            sample_audio = torch.randn(max_batch_size, 16000, device=self.device)
            sample_len = torch.tensor([16000] * max_batch_size, dtype=torch.long, device=self.device)
            sample_spec, _ = self.preprocessor(input_signal=sample_audio, length=sample_len)
            # sample_spec shape: (B, D, T) where D is feature dim
            self.spec_feature_dim = sample_spec.shape[1]

        # Static buffers for CUDA graph - spectrogram input (B, D, T)
        self.max_spec_frames = max_spec_frames
        self.static_processed_signal = torch.zeros(
            (max_batch_size, self.spec_feature_dim, self.max_spec_frames), dtype=self.dtype, device=self.device,
        )
        self.static_processed_signal_length = torch.tensor(
            [self.max_spec_frames] * max_batch_size, dtype=torch.long, device=self.device
        )

        # Warmup
        with torch.no_grad():
            with torch.cuda.device(self.device):
                torch.cuda.synchronize()
                for _ in range(3):
                    _ = self.model(
                        processed_signal=self.static_processed_signal,
                        processed_signal_length=self.static_processed_signal_length,
                    )
                torch.cuda.synchronize()

            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g):
                self.encoded_audio, self.encoded_audio_length, self.encoder_emb = self.model(
                    processed_signal=self.static_processed_signal,
                    processed_signal_length=self.static_processed_signal_length,
                    return_encoder_emb=True,
                )

        self.static_processed_signal.zero_()
        self.static_processed_signal_length.zero_()
        self.encoded_audio.zero_()
        self.encoded_audio_length.zero_()
        self.encoder_emb.zero_()

    def forward(self, input_signal, input_signal_length, return_encoder_emb=True):
        """
        Forward pass with CUDA graph optimization.

        Args:
            input_signal: (B, T) - raw audio signal (B <= max_batch_size)
            input_signal_length: (B,) - actual length of audio
            return_encoder_emb: bool - whether to return encoder embeddings

        Returns:
            encoded_audio, encoded_audio_length, encoder_emb (if return_encoder_emb=True)

        Note:
            Batch sizes smaller than max_batch_size are zero-padded before graph
            replay and sliced back to actual batch size after.
        """
        actual_batch_size = input_signal.shape[0]

        if actual_batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size ({actual_batch_size}) exceeds max_batch_size ({self.max_batch_size}). "
                f"Increase max_batch_size when creating PerceptionCudaGraph."
            )

        with torch.no_grad():
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )
        # processed_signal shape: (B, D, T_actual)

        actual_spec_frames = processed_signal.shape[2]

        if actual_spec_frames > self.max_spec_frames:
            raise ValueError(
                f"Spectrogram frames ({actual_spec_frames}) exceeds max_spec_frames ({self.max_spec_frames}). "
                f"Increase max_spec_frames or reduce input audio length."
            )

        # Zero out static buffer and copy actual data (pad batch if needed)
        self.static_processed_signal.zero_()
        self.static_processed_signal[:actual_batch_size, :, :actual_spec_frames] = processed_signal
        self.static_processed_signal_length.zero_()
        self.static_processed_signal_length[:actual_batch_size].copy_(processed_signal_length)

        # Replay the captured graph
        self.g.replay()

        # Use the encoder's exact valid length. A proportional estimate is not
        # reliable across the first/subsequent subsampling boundaries.
        actual_output_frames = max(1, int(self.encoded_audio_length[:actual_batch_size].max().item()))

        # Slice output to actual batch size and actual output frames
        encoded_audio = self.encoded_audio[:actual_batch_size, :actual_output_frames, :].clone()
        encoded_audio_length = self.encoded_audio_length[:actual_batch_size].clone()
        encoder_emb = self.encoder_emb[:actual_batch_size, :actual_output_frames, :].clone()

        if return_encoder_emb:
            return encoded_audio, encoded_audio_length, encoder_emb
        else:
            return encoded_audio, encoded_audio_length


class PerceptionCudaGraphPool(torch.nn.Module):
    """
    Pool of CUDA Graphs for different batch sizes.

    Maintains separate CUDA graphs for predefined batch sizes (1, 4, 8, 16, 32)
    and automatically selects the closest higher batch size for inference.
    This minimizes wasted compute from zero-padding while maintaining the
    performance benefits of CUDA graphs.

    Example:
        - Input batch_size=2 → uses graph for batch_size=4
        - Input batch_size=5 → uses graph for batch_size=8
        - Input batch_size=17 → uses graph for batch_size=32
    """

    # Predefined batch sizes for CUDA graphs
    BATCH_SIZES = (1, 4, 8, 16, 32)

    def __init__(
        self,
        model,
        max_spec_frames=561,
        batch_sizes: tuple[int, ...] = None,
        dtype=torch.float32,
        device=torch.device("cuda"),
        lazy_init: bool = True,
    ):
        """
        Initialize the CUDA Graph pool.

        Args:
            model: The AudioPerceptionModule to wrap.
            max_spec_frames: Maximum spectrogram frames (default 561 for ~5s audio).
            batch_sizes: Tuple of batch sizes to create graphs for.
                         Defaults to (1, 4, 8, 16, 32).
            dtype: Data type for the model.
            device: Device to run on.
            lazy_init: If True, graphs are created on first use. If False,
                       all graphs are created during __init__.
        """
        super(PerceptionCudaGraphPool, self).__init__()
        self.model = model
        self.preprocessor = model.preprocessor
        self.dtype = dtype
        self.device = device
        self.max_spec_frames = max_spec_frames
        self.batch_sizes = tuple(sorted(batch_sizes or self.BATCH_SIZES))
        self.max_batch_size = max(self.batch_sizes)

        # Dictionary to store CUDA graph instances
        self._graphs: dict[int, PerceptionCudaGraph] = {}

        if not lazy_init:
            # Create all graphs upfront
            print(f"Creating CUDA graphs for batch sizes: {self.batch_sizes}")
            for bs in self.batch_sizes:
                self._get_or_create_graph(bs)
            print(f"✅ All {len(self.batch_sizes)} CUDA graphs created!")

    def _get_closest_batch_size(self, batch_size: int) -> int:
        """Find the smallest predefined batch size >= input batch_size."""
        for bs in self.batch_sizes:
            if bs >= batch_size:
                return bs
        raise ValueError(
            f"Batch size {batch_size} exceeds maximum supported batch size {self.max_batch_size}. "
            f"Supported batch sizes: {self.batch_sizes}"
        )

    def _get_or_create_graph(self, graph_batch_size: int) -> PerceptionCudaGraph:
        """Get existing graph or create a new one for the given batch size."""
        if graph_batch_size not in self._graphs:
            print(f"  Creating CUDA graph for batch_size={graph_batch_size}...")
            self._graphs[graph_batch_size] = PerceptionCudaGraph(
                model=self.model,
                max_spec_frames=self.max_spec_frames,
                max_batch_size=graph_batch_size,
                dtype=self.dtype,
                device=self.device,
            )
        return self._graphs[graph_batch_size]

    def forward(self, input_signal, input_signal_length, return_encoder_emb=True):
        """
        Forward pass using the optimal CUDA graph for the input batch size.

        Automatically selects the closest higher batch size graph to minimize
        wasted compute from zero-padding.

        Args:
            input_signal: (B, T) - raw audio signal
            input_signal_length: (B,) - actual length of audio
            return_encoder_emb: bool - whether to return encoder embeddings

        Returns:
            encoded_audio, encoded_audio_length, encoder_emb (if return_encoder_emb=True)
        """
        actual_batch_size = input_signal.shape[0]
        graph_batch_size = self._get_closest_batch_size(actual_batch_size)
        graph = self._get_or_create_graph(graph_batch_size)

        return graph(
            input_signal=input_signal, input_signal_length=input_signal_length, return_encoder_emb=return_encoder_emb,
        )

    @property
    def created_graphs(self) -> list[int]:
        """Return list of batch sizes for which graphs have been created."""
        return sorted(self._graphs.keys())
