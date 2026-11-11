"""Tests for text normalization utilities in utils.py."""
import json
import sys
import types
from pathlib import Path

# Allow running directly without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub out Triton dependencies not available outside the container
for _mod in ("triton_python_backend_utils", "data_types", "runtime_setup"):
    if _mod not in sys.modules:
        _stub = types.ModuleType(_mod)
        if _mod == "runtime_setup":
            _stub.DEBUG_MODE = False
            _stub.FORCE_TURN_TAKING_THRESHOLD = 40
        if _mod == "data_types":
            _stub.TritonSeqState = object
        if _mod == "triton_python_backend_utils":
            _stub.Logger = types.SimpleNamespace(log_info=lambda *a, **k: None, log_error=lambda *a, **k: None)
        sys.modules[_mod] = _stub

from utils import format_tool_response_for_nemo, normalize_tool_response


# ---------------------------------------------------------------------------
# Tool response normalization — converts int/float values only inside
# <TOOL_RESPONSE> tags; string values pass through unchanged.
# ---------------------------------------------------------------------------
def _wrap(payload: str) -> str:
    return f"<TOOL_RESPONSE>{payload}</TOOL_RESPONSE>"


def _unwrap(result: str) -> str:
    assert result.startswith("<TOOL_RESPONSE>") and result.endswith("</TOOL_RESPONSE>")
    return result[len("<TOOL_RESPONSE>") : -len("</TOOL_RESPONSE>")]


def test_tool_response_int_converted():
    # int values should be spoken
    payload = json.dumps([{"count": 42}])
    result = normalize_tool_response(_wrap(payload))
    parsed = json.loads(_unwrap(result))
    assert parsed[0]["count"] == "forty-two"


def test_tool_response_float_converted():
    payload = json.dumps([{"bmi": 22.5}])
    result = normalize_tool_response(_wrap(payload))
    parsed = json.loads(_unwrap(result))
    assert "twenty-two" in parsed[0]["bmi"]


def test_tool_response_string_unchanged():
    # String values (dates, times, non-temperature units) must NOT be converted
    payload = json.dumps([{"date": "2026-07-23", "time": "18:11:42", "speed": "15 mph"}])
    result = normalize_tool_response(_wrap(payload))
    parsed = json.loads(_unwrap(result))
    assert parsed[0]["date"] == "2026-07-23"
    assert parsed[0]["time"] == "18:11:42"
    assert parsed[0]["speed"] == "15 mph"


def test_tool_response_em_dash_removed():
    payload = json.dumps([{"desc": "Sunny—great day"}])
    result = normalize_tool_response(_wrap(payload))
    parsed = json.loads(_unwrap(result))
    assert parsed[0]["desc"] == "Sunny great day"


def test_tool_response_en_dash_removed():
    payload = json.dumps([{"range": "Mon–Fri"}])
    result = normalize_tool_response(_wrap(payload))
    parsed = json.loads(_unwrap(result))
    assert parsed[0]["range"] == "Mon Fri"


def test_tool_response_temperature_fahrenheit():
    payload = json.dumps([{"temperature": "72°F"}])
    result = normalize_tool_response(_wrap(payload))
    parsed = json.loads(_unwrap(result))
    assert parsed[0]["temperature"] == "seventy-two degrees Fahrenheit"


def test_tool_response_temperature_celsius():
    payload = json.dumps([{"temp": "22°C"}])
    result = normalize_tool_response(_wrap(payload))
    parsed = json.loads(_unwrap(result))
    assert parsed[0]["temp"] == "twenty-two degrees Celsius"


def test_tool_response_temperature_decimal():
    payload = json.dumps([{"feels_like": "98.6°F"}])
    result = normalize_tool_response(_wrap(payload))
    parsed = json.loads(_unwrap(result))
    assert "ninety-eight" in parsed[0]["feels_like"]
    assert "six" in parsed[0]["feels_like"]
    assert "Fahrenheit" in parsed[0]["feels_like"]


def test_tool_response_temperature_negative():
    payload = json.dumps([{"low": "-5°C"}])
    result = normalize_tool_response(_wrap(payload))
    parsed = json.loads(_unwrap(result))
    assert "five" in parsed[0]["low"]
    assert "Celsius" in parsed[0]["low"]


def test_tool_response_temperature_with_spaces():
    # Spaces around the degree sign should still match
    payload = json.dumps([{"temp": "100 °F"}])
    result = normalize_tool_response(_wrap(payload))
    parsed = json.loads(_unwrap(result))
    assert parsed[0]["temp"] == "one hundred degrees Fahrenheit"


def test_tool_response_nested():
    payload = json.dumps([{"data": {"value": 100, "unit": "kg"}}])
    result = normalize_tool_response(_wrap(payload))
    parsed = json.loads(_unwrap(result))
    assert "hundred" in parsed[0]["data"]["value"]
    assert parsed[0]["data"]["unit"] == "kg"


def test_tool_response_no_tags_unchanged():
    # Without TOOL_RESPONSE tags the string is returned as-is
    plain = "The answer is 42"
    assert normalize_tool_response(plain) == plain


def test_tool_response_invalid_json_unchanged():
    # Invalid JSON inside tags → returned unchanged
    wrapped = _wrap("not valid json {")
    assert normalize_tool_response(wrapped) == wrapped


# ---------------------------------------------------------------------------
# format_tool_response_for_nemo — wraps in TOOL_RESPONSE; no normalization
# ---------------------------------------------------------------------------
def _parse_nemo_tool_response(result: str):
    assert result.startswith("<TOOL_RESPONSE>")
    assert result.endswith("</TOOL_RESPONSE>")
    inner = result[len("<TOOL_RESPONSE>") : -len("</TOOL_RESPONSE>")]
    return json.loads(inner)


def test_format_tool_response_wraps_json():
    response = json.dumps({"temperature": "72°F", "wind_speed": "15 mph"})
    result = format_tool_response_for_nemo(response)
    parsed = _parse_nemo_tool_response(result)
    assert isinstance(parsed, list)
    # String values are NOT normalized by format_tool_response_for_nemo
    assert parsed[0]["temperature"] == "72°F"


def test_format_tool_response_wraps_plain_string():
    result = format_tool_response_for_nemo("The answer is 42")
    parsed = _parse_nemo_tool_response(result)
    # Plain string is JSON-escaped and wrapped; no normalization
    assert parsed == ["The answer is 42"]


def test_format_tool_response_already_wrapped():
    response = "<TOOL_RESPONSE>[{\"answer\": 42}]</TOOL_RESPONSE>"
    assert format_tool_response_for_nemo(response) == response


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(failed)
