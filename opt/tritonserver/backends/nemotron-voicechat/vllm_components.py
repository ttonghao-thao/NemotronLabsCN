import asyncio
import json
import os

import torch
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.attention.selector import _cached_get_attn_backend
from vllm.v1.engine.async_llm import AsyncLLM


class EARTTSvLLM:
    def __init__(self, model_path: str, config: dict, dtype: str = "float32", tokenizer_path: str = None):
        self.config = config
        self.prompt_input_file = os.path.join(model_path, "tts_model_init_inputs.pt")

        os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
        # TF32 matmul precision to match TTS training ("medium").
        # torch.set_float32_matmul_precision is process-local and does NOT
        # propagate to vLLM's spawned worker processes; this CUDA-level env
        # var is inherited by child processes.
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
        _cached_get_attn_backend.cache_clear()

        # load vllm engine
        engine_args = AsyncEngineArgs(
            model=model_path,
            tokenizer=tokenizer_path,
            dtype=dtype,
            max_model_len=config["max_model_len"],
            max_num_batched_tokens=config["max_num_batched_tokens"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
            skip_tokenizer_init=True,  # Skip tokenizer since we're using embeddings directly
            enable_prefix_caching=False,
            enable_log_requests=True,
        )

        engine = AsyncLLM.from_engine_args(engine_args)
        # guidance_scale: Typical values are 1.0-15.0. Set to None to disable CFG.
        sampling_params = SamplingParams(
            max_tokens=config["max_tokens"],
            skip_sampling=True,
            ignore_eos=True,
            guidance_scale=config["guidance_scale"],
        )
        self.engine = engine
        self.sampling_params = sampling_params
        self.iterator_dict = {}
        self.event_loop = None

        # Default zero latent for decode steps — shape [1, H]. Overwritten by
        # get_prompt_input() when tts_model_init_inputs.pt contains a real latent.
        with open(os.path.join(model_path, "config.json")) as _f:
            _hidden_size = json.load(_f)["hidden_size"]
        self._audio_prompt_latent = torch.zeros(1, _hidden_size)

        os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
        _cached_get_attn_backend.cache_clear()

    # Destructor (Finalizer)
    def __del__(self):
        self.shutdown()

    def shutdown(self):
        if hasattr(self, 'event_loop') and self.event_loop is not None:
            self.event_loop.close()
            self.event_loop = None
        if hasattr(self, 'engine') and self.engine:
            self.engine.shutdown()
            self.engine = None

    def get_prompt_input(self):
        torch_type = getattr(torch, "float32")
        inputs = torch.load(self.prompt_input_file)

        # Shift acoustic tokens left: remove last frame, prepend a zero frame.
        # This matches teacher-forcing alignment used during training.
        if inputs["code"][0].shape[0] > 1:
            inputs["code"][0] = torch.nn.functional.pad(inputs["code"][0][:-1], [0, 0, 1, 0])

        prompt_len = inputs["code"].shape[1]

        # Apply edge detection to match native model's BOS placement logic:
        # BOS should only be applied at the FIRST position where non_prompt_mask is True
        non_prompt_mask = inputs["non_prompt_mask"].squeeze(0)  # T
        # Compute edge: positions where mask is True AND previous position is False
        padded_prev = torch.nn.functional.pad(non_prompt_mask[:-1], [1, 0], value=False)
        bos_mask = (non_prompt_mask & (~padded_prev)).to(dtype=torch_type).detach().cpu()

        custom_inputs = {
            "acoustic_tokens": inputs["code"].detach().cpu()[0],
            "text_tokens": inputs["subword_ids"].squeeze(0).detach().cpu(),
            "text_mask": inputs["subword_mask"].squeeze(0).detach().cpu(),
            "bos_mask": bos_mask,
        }

        audio_prompt_latent = inputs.get("audio_prompt_latent", None)
        if audio_prompt_latent is not None:
            # squeeze batch dim → [1, H], cache for decode steps, expand to [prompt_len, H] for prefill
            latent = audio_prompt_latent.squeeze(0).detach().cpu()
            self._audio_prompt_latent = latent
            custom_inputs["audio_prompt_latent"] = latent.expand(prompt_len, -1).contiguous()

        prompt_inputs = {
            # dummy tokens
            "prompt_token_ids": [0] * prompt_len,
            # actual inputs to the model in prefill stage
            "custom_inputs": custom_inputs,
        }
        return prompt_inputs

    def create_stream(self, request_id):
        if request_id in self.iterator_dict:
            return

        if self.event_loop is None:
            self.event_loop = asyncio.new_event_loop()
        prompt_input = self.get_prompt_input()
        iterator = self.engine.generate(
            {'type': 'tokens', **prompt_input}, sampling_params=self.sampling_params, request_id=str(request_id)
        ).__aiter__()
        self.iterator_dict[request_id] = iterator

    def stream_get_output(self, request_id):
        iterator = self.iterator_dict[request_id]
        try:
            output = self.event_loop.run_until_complete(iterator.__anext__())
            return False, output
        except StopAsyncIteration:
            del self.iterator_dict[request_id]
            return True, None

    def stream_submit_request(self, request_id, new_custom_inputs):
        future = asyncio.ensure_future(
            self.engine.append_request(request_id=str(request_id), custom_inputs=new_custom_inputs),
            loop=self.event_loop,
        )
        return future

    def stream_abort_request(self, request_id):
        self.event_loop.run_until_complete(self.engine.abort(request_id=str(request_id)))
        self.iterator_dict[request_id].aclose()
        del self.iterator_dict[request_id]
