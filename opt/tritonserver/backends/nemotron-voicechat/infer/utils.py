import copy
import functools
import json
import re
import time
from contextlib import contextmanager
from typing import Set

import torch
import triton_python_backend_utils as pb_utils
from data_types import TritonSeqState
from runtime_setup import DEBUG_MODE, FORCE_TURN_TAKING_THRESHOLD


def _log_time(label: str, debug_only: bool = True, cuda_sync: bool = False):
    """Decorator: logs execution time of a method and emits an NVTX range.

    When debug_only=True (default), timing is skipped unless DEBUG_MODE is set.
    NVTX ranges are always emitted (no-op overhead when not profiling).
    When cuda_sync=True, torch.cuda.synchronize() is called before reading the
    end time so GPU-bound operations are fully measured.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            torch.cuda.nvtx.range_push(label)
            try:
                if debug_only and not DEBUG_MODE:
                    return fn(*args, **kwargs)
                t0 = time.perf_counter()
                result = fn(*args, **kwargs)
                if cuda_sync:
                    torch.cuda.synchronize()
                pb_utils.Logger.log_info(f"{label}: {(time.perf_counter() - t0) * 1000:.1f}ms")
                return result
            finally:
                torch.cuda.nvtx.range_pop()

        return wrapper

    return decorator


@contextmanager
def _time_block(label: str, debug_only: bool = True):
    """Context manager: logs execution time of a code block and emits an NVTX range.

    When debug_only=True (default), timing is skipped unless DEBUG_MODE is set.
    NVTX ranges are always emitted (no-op overhead when not profiling).
    """
    torch.cuda.nvtx.range_push(label)
    try:
        if debug_only and not DEBUG_MODE:
            yield
            return
        t0 = time.perf_counter()
        yield
        pb_utils.Logger.log_info(f"{label}: {(time.perf_counter() - t0) * 1000:.1f}ms")
    finally:
        torch.cuda.nvtx.range_pop()


_inflect_engine = []  # lazy singleton


_TEMP_RE = re.compile(r'^(-?\d+(?:\.\d+)?)\s*°\s*([CF])$', re.IGNORECASE)


def _nums_to_words(obj):
    """Recursively convert int/float leaf values and temperature strings to spoken words."""
    if not _inflect_engine:
        import inflect

        _inflect_engine.append(inflect.engine())
    eng = _inflect_engine[0]

    if isinstance(obj, dict):
        return {k: _nums_to_words(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nums_to_words(v) for v in obj]
    if isinstance(obj, float):
        try:
            int_part = int(obj)
            frac_str = str(obj).split(".")[1] if "." in str(obj) else ""
            result = eng.number_to_words(int_part, andword="")
            if frac_str:
                result += " point " + " ".join(eng.number_to_words(int(d)) for d in frac_str)
            return result
        except Exception:
            return str(obj)
    if isinstance(obj, int):
        try:
            return eng.number_to_words(obj, andword="")
        except Exception:
            return str(obj)
    if isinstance(obj, str):
        m = _TEMP_RE.match(obj)
        if m:
            try:
                num_str, scale = m.group(1), m.group(2).upper()
                int_part = int(float(num_str))
                frac_str = num_str.split(".")[1] if "." in num_str else ""
                word = eng.number_to_words(int_part, andword="")
                if frac_str:
                    word += " point " + " ".join(eng.number_to_words(int(d)) for d in frac_str)
                unit = "Celsius" if scale == "C" else "Fahrenheit"
                return f"{word} degrees {unit}"
            except Exception:
                pass
        obj = obj.replace('—', ' ').replace('–', ' ')  # em dash, en dash
    return obj


_TOOL_RESPONSE_RE = re.compile(r'(<TOOL_RESPONSE>)(.*?)(</TOOL_RESPONSE>)', re.DOTALL)


def normalize_tool_response(response: str) -> str:
    """Convert numeric JSON values inside <TOOL_RESPONSE> to spoken words.

    String values (dates, times, etc.) are passed through unchanged.
    Matches NeMo's _convert_tool_response_nums_to_text behaviour.
    """

    def _replace(match):
        open_tag, json_str, close_tag = match.group(1), match.group(2), match.group(3)
        try:
            converted = _nums_to_words(json.loads(json_str))
            return open_tag + json.dumps(converted) + close_tag
        except (json.JSONDecodeError, TypeError):
            return match.group(0)

    result = _TOOL_RESPONSE_RE.sub(_replace, response)
    if result != response:
        pb_utils.Logger.log_info(f"[normalize_tool_response] before: {response!r}")
        pb_utils.Logger.log_info(f"[normalize_tool_response]  after: {result!r}")
    return result


def format_tool_response_for_nemo(response: str) -> str:
    """Return a normalized NeMo-format tool response string for function-channel injection."""
    response = response.strip()
    if response.startswith("<TOOL_RESPONSE>") and response.endswith("</TOOL_RESPONSE>"):
        return normalize_tool_response(response)

    try:
        json.loads(response)
        payload = response
    except (json.JSONDecodeError, TypeError):
        payload = json.dumps(response)
    return normalize_tool_response(f"<TOOL_RESPONSE>[{payload}]</TOOL_RESPONSE>")


def sample_text_token(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    current_step: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    special_token_ids: Set[int],
) -> torch.Tensor:
    """
    Sample text tokens with optional top-p sampling and repetition penalty.
    Special tokens (pad, eos, bos) bypass sampling - if they have highest probability, return them directly.

    Refer _sample_text_token in nemo/collections/speechlm2/inference/model_wrappers/model_factory.py.

    Args:
        logits: Logits tensor of shape (B, V).
        generated_tokens: Previously generated tokens of shape (B, T).
        current_step: Current decoding step (used to slice generated_tokens).
        top_p: Nucleus sampling threshold.
        temperature: Temperature for scaling logits.
        repetition_penalty: Penalty factor for repeated tokens.
        special_token_ids: Token IDs that bypass sampling.

    Returns:
        Sampled token ids of shape (B,).
    """
    B, V = logits.shape

    greedy_tokens = logits.argmax(dim=-1)  # (B,)

    # No sampling needed
    if top_p >= 1.0 and repetition_penalty == 1.0 and temperature in (0.0, 1.0):
        return greedy_tokens

    sampled_tokens = greedy_tokens.clone()

    for b in range(B):
        # Special tokens are never sampled
        if greedy_tokens[b].item() in special_token_ids:
            continue

        batch_logits = logits[b].clone()  # (V,)

        # Repetition penalty
        if repetition_penalty != 1.0 and current_step > 0:
            prev_tokens = generated_tokens[b, :current_step]
            unique_prev = prev_tokens.unique()
            # Exclude special tokens from penalty
            if special_token_ids:
                # Use unique_prev.device to ensure tensors are on the same device
                # (generated_tokens may be on a different device than logits, e.g., vLLM returns CPU logits)
                special_tensor = torch.tensor(list(special_token_ids), device=unique_prev.device)
                mask = ~torch.isin(unique_prev, special_tensor)
                unique_prev = unique_prev[mask]

            for token_id in unique_prev:
                token_id = token_id.item()
                if batch_logits[token_id] > 0:
                    batch_logits[token_id] = batch_logits[token_id] / repetition_penalty
                else:
                    batch_logits[token_id] = batch_logits[token_id] * repetition_penalty

        # temperature=0 remains deterministic, but uses the repetition-adjusted
        # logits instead of bypassing the configured penalty.
        if temperature == 0.0:
            sampled_tokens[b] = batch_logits.argmax(dim=-1)
            continue

        # Temperature scaling
        if temperature != 1.0:
            batch_logits = batch_logits / temperature

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(batch_logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative prob > top_p, keeping at least one
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift to keep the first token that exceeds threshold
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            # Set to -inf
            batch_logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')

        probs = torch.softmax(batch_logits, dim=-1)
        sampled_tokens[b] = torch.multinomial(probs, num_samples=1).item()

    return sampled_tokens


def resolve_single_token_id(tokenizer, token_str: str) -> int:
    token_ids = tokenizer.text_to_ids(token_str)
    if not token_ids:
        raise ValueError(f"Tokenizer did not return an id for token: {token_str}")
    if len(token_ids) != 1:
        raise ValueError(f"Expected a single id for token {token_str}, got {token_ids}")
    return int(token_ids[0])


def clone_cache(cache):
    """Deep clone cache structures to ensure complete isolation between streams."""
    if cache is None:
        return None
    if isinstance(cache, torch.Tensor):
        return cache.detach().clone()
    if isinstance(cache, (list, tuple)):
        return type(cache)(clone_cache(x) for x in cache)
    if isinstance(cache, dict):
        return {k: clone_cache(v) for k, v in cache.items()}
    if hasattr(cache, "__dict__"):
        return copy.deepcopy(cache)
    return cache
