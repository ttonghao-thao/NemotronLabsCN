import argparse
import json
import logging
import random
import sys
import time
import uuid

import librosa
import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient
from tqdm import tqdm
from tritonclient.utils import np_to_triton_dtype

logger = logging.getLogger(__name__)

model_name = "nemotron-voicechat"

INPUT_CHUNK_SIZE_SAMPLES = int(16000 * 0.08)  # number of samples per input chunk

times_spend_on_inference = []

# Default template for augmenting function-calling system prompts that only contain <AVAILABLE_TOOLS>...</AVAILABLE_TOOLS>.
# Use {tools_content} as placeholder; literal braces in the template are escaped ({{ }}) for .format(tools_content=...).
# Matches NeMo s2s_dataset.DEFAULT_FC_SYSTEM_PROMPT_TEMPLATE and _augment_fc_system_prompt.
DEFAULT_FC_SYSTEM_PROMPT_TEMPLATE = """You can use the following tools to assist the user if required:
<AVAILABLE_TOOLS>{tools_content}</AVAILABLE_TOOLS>

If you decide to call any tool(s), use the following format:
<TOOLCALL>[{{"name": "tool_name1", "arguments": "tool_args1"}}, {{"name": "tool_name2", "arguments": "tool_args2"}}]</TOOLCALL>

The user will execute tool-calls and return responses from tool(s) in this format:
<TOOL_RESPONSE>[{{"tool_response1"}}, {{"tool_response2"}}]</TOOL_RESPONSE>

Based on the tool responses, you can call additional tools if needed, correct tool calls if any errors are found, or just respond to the user."""

# SYSTEM_PROMPT = "You are an AI assistant developed by NVIDIA. Your name is NVIDIA VoiceChat. Your job is to be helpful and have engaging conversations in English. Maintain a warm tone. Keep the dialogue open and ongoing. You must provide diverse responses, rephrase answers if user asks the same questions. Wait for the user to finish speaking before responding."

SYSTEM_PROMPT = ""
# TOOL_PARAMETER = ""
TOOL_PARAMETER = """[{"name": "convert_currency", "description": "Convert a certain amount of currency from one currency to another", "parameters": {"type": "dict", "properties": {"amount": {"ty    pe": "number", "description": "The amount of currency to convert"}, "from_currency": {"type": "string", "description": "The currency to convert from"}, "to_currency": {"type": "string", "description": "The     currency to convert to"}}, "required": ["amount", "from_currency", "to_currency"]}}, {"name": "get_news_headlines", "description": "Get the latest news headlines", "parameters": {"type": "dict", "properties    ": {"country": {"type": "string", "description": "The country for which news headlines should be retrieved"}, "category": {"type": "string", "description": "The category of news, e.g. business, sports, ente    rtainment"}}, "required": []}}, {"name": "calculate_bmi", "description": "Calculate the Body Mass Index (BMI)", "parameters": {"type": "dict", "properties": {"weight": {"type": "number", "description": "The weight in kilograms"}, "height": {"type": "number", "description": "The height in meters"}}, "required": ["weight", "height"]}}, {"name": "generate_random_number", "description": "Generate a random number within specified range", "parameters": {"type": "dict", "properties": {"min": {"type": "number", "description": "The minimum value of the range"}, "max": {"type": "number", "description": "The maximum value of the range"}}, "required": ["min", "max"]}}]"""
TOOL_RESPONSE = '[{"converted_amount": 595.5, "from_currency": "Euros", "to_currency": "US Dollars"}]'
TOOL_RESPONSE_ERROR = '<TOOL_RESPONSE>[{"error": "Failed to convert currency"}]</TOOL_RESPONSE>'

parser = argparse.ArgumentParser()
parser.add_argument("--audio_file", type=str, required=True)
parser.add_argument("--no-output", action="store_true", required=False)
args = parser.parse_args()
audio_file = args.audio_file
if args.no_output:
    logging.basicConfig(level=logging.ERROR)


def get_audio_as_chunks(audio_file):
    audio_signal, sr = librosa.load(audio_file, sr=16000)
    audio_signal = np.expand_dims(audio_signal, axis=0)

    padded_len_samples = int(INPUT_CHUNK_SIZE_SAMPLES - audio_signal.shape[1] % INPUT_CHUNK_SIZE_SAMPLES)
    audio_signal = np.pad(audio_signal, ((0, 0), (0, padded_len_samples)), "constant", constant_values=0)

    audio_signal_chunks = [
        audio_signal[:, i : i + INPUT_CHUNK_SIZE_SAMPLES]
        for i in range(0, audio_signal.shape[1], INPUT_CHUNK_SIZE_SAMPLES)
    ]
    logger.info(f"Zero padded {padded_len_samples} samples. Final number of samples: {audio_signal.shape[1]}")

    return audio_signal_chunks


def send_sequence_end(client, sequence_id):
    """Send a final request with sequence_end=True to properly clean up the sequence"""
    try:
        logger.info(f"Sending sequence_end=True for sequence_id={sequence_id}")

        # Send empty audio chunk with sequence_end=True
        empty_audio = np.zeros((1, INPUT_CHUNK_SIZE_SAMPLES), dtype=np.float32)

        inputs = [
            grpcclient.InferInput("audio_signal", empty_audio.shape, np_to_triton_dtype(empty_audio.dtype)),
            grpcclient.InferInput("system_prompt", (1, 1), "BYTES"),
            grpcclient.InferInput("function_response", (1, 1), "BYTES"),
        ]
        inputs[0].set_data_from_numpy(empty_audio)
        inputs[1].set_data_from_numpy(np.array([[b""]], dtype=object))
        inputs[2].set_data_from_numpy(np.array([[b""]], dtype=object))

        outputs = [
            grpcclient.InferRequestedOutput("output_text"),
            grpcclient.InferRequestedOutput("output_asr_text"),
            grpcclient.InferRequestedOutput("output_function_text"),
            grpcclient.InferRequestedOutput("output_audio"),
        ]

        response = client.infer(
            model_name,
            inputs,
            request_id=str(uuid.uuid4()),
            outputs=outputs,
            sequence_id=sequence_id,
            sequence_start=False,
            sequence_end=True,  # This is the key - properly end the sequence
        )
        logger.info("Sequence ended successfully")

    except Exception as e:
        logger.error(f"Error ending sequence: {e}")


def merge_system_and_tool_prompt(system_prompt: str, tools_parameter: str):
    tools_with_template = ""
    if tools_parameter.strip():
        tools_with_template = DEFAULT_FC_SYSTEM_PROMPT_TEMPLATE.format(tools_content=tools_parameter.strip())
    return system_prompt + "\n" + tools_with_template


with grpcclient.InferenceServerClient("localhost:8001") as client:
    audio_signal_chunks = get_audio_as_chunks(audio_file)

    function_response = ""

    generated_text = []
    generated_asr_text = []
    generated_function_text = []
    generated_audio = []

    # Generate a numeric sequence ID instead of string UUID to match UINT64 type
    sequence_id = random.randint(1, 2 ** 63 - 1)  # Generate random uint64 value

    config = client.get_model_config(model_name, as_json=True)
    logger.info(f"Model config: {config}")
    system_prompt = config.get("config", {}).get("parameters", {}).get("system_prompt", "").get("string_value", "")
    logger.info(f"Defaults system prompt: {system_prompt}")

    try:
        # Pre-send system prompt with silence to start the sequence
        silence = np.zeros((1, INPUT_CHUNK_SIZE_SAMPLES), dtype=np.float32)
        prompt_inputs = [
            grpcclient.InferInput("audio_signal", silence.shape, np_to_triton_dtype(silence.dtype)),
            grpcclient.InferInput("system_prompt", (1, 1), "BYTES"),
            grpcclient.InferInput("function_response", (1, 1), "BYTES"),
        ]
        prompt_inputs[0].set_data_from_numpy(silence)
        input_data = merge_system_and_tool_prompt(SYSTEM_PROMPT, TOOL_PARAMETER)
        prompt_inputs[1].set_data_from_numpy(np.array([[input_data.encode("utf-8")]], dtype=object))
        prompt_inputs[2].set_data_from_numpy(np.array([[b""]], dtype=object))
        prompt_outputs = [
            grpcclient.InferRequestedOutput("output_text"),
            grpcclient.InferRequestedOutput("output_asr_text"),
            grpcclient.InferRequestedOutput("output_audio"),
        ]
        client.infer(
            model_name,
            prompt_inputs,
            request_id=str(uuid.uuid4()),
            outputs=prompt_outputs,
            sequence_id=sequence_id,
            sequence_start=True,
            sequence_end=False,
        )

        for idx, audio_chunk in tqdm(enumerate(audio_signal_chunks)):
            inputs = [
                grpcclient.InferInput("audio_signal", audio_chunk.shape, np_to_triton_dtype(audio_chunk.dtype)),
                grpcclient.InferInput("system_prompt", (1, 1), "BYTES"),
                grpcclient.InferInput("function_response", (1, 1), "BYTES"),
            ]
            inputs[0].set_data_from_numpy(audio_chunk)
            inputs[1].set_data_from_numpy(np.array([[b""]], dtype=object))
            inputs[2].set_data_from_numpy(np.array([[function_response.encode("utf-8")]], dtype=object))
            if function_response:
                logger.info(f"function_response: {function_response} sent. Resetting to empty string.")
                function_response = ""

            outputs = [
                grpcclient.InferRequestedOutput("output_text"),
                grpcclient.InferRequestedOutput("output_asr_text"),
                grpcclient.InferRequestedOutput("output_function_text"),
                grpcclient.InferRequestedOutput("output_audio"),
            ]

            if idx % 2 != 0:
                start_time = time.perf_counter()
            response = client.infer(
                model_name,
                inputs,
                request_id=str(uuid.uuid4()),
                outputs=outputs,
                sequence_id=sequence_id,
                sequence_start=False,
                sequence_end=idx == len(audio_signal_chunks) - 1,
            )
            if idx % 2 != 0:
                end_time = time.perf_counter()
                # latency = end_time - start_time
                # logger.info(f"time to infer: {latency}")

            result = response.get_response()
            output_text = response.as_numpy("output_text")
            output_asr_text = response.as_numpy("output_asr_text")
            output_function_text = response.as_numpy("output_function_text")
            output_audio = response.as_numpy("output_audio")

            logger.debug(f"input chunk shape: {audio_chunk.shape}")
            logger.debug(f"output chunk shape: {output_audio.shape}")
            # logger.info(f"time to infer chunk: {end_time - start_time}")

            generated_text.extend([i.decode("utf-8") for i in output_text])
            generated_asr_text.extend([i.decode("utf-8") for i in output_asr_text])
            generated_function_text.extend([i.decode("utf-8") for i in output_function_text])

            if generated_text:
                logger.info(f"generated_text: {''.join(generated_text)}")
            if generated_asr_text:
                logger.info(f"generated_asr_text: {''.join(generated_asr_text)}")

            function_text = [i.decode("utf-8") for i in output_function_text]
            function_text = ''.join(function_text)
            if function_text:
                logger.info(f"function_text: {function_text}")
                function_request = json.loads(function_text)
                if 'name' in function_request[0]:
                    function_name = function_request[0]['name']
                    function_arguments = function_request[0]['arguments']
                    logger.info(f"function_name: {function_name}")
                    logger.info(f"function_arguments: {function_arguments}")
                    function_response = TOOL_RESPONSE  # if random.random() < 0.5 else TOOL_RESPONSE_ERROR

            if output_audio.shape[1] > 0:
                times_spend_on_inference.append(end_time - start_time)
                generated_audio.append(output_audio.squeeze(0).squeeze(0))
                logger.info(f"time to infer: {end_time - start_time}")

    except KeyboardInterrupt:
        logger.info("\nKeyboardInterrupt received! Calling send_sequence_end...")
        send_sequence_end(client, sequence_id)
        logger.info("Sequence cleanup completed. Exiting...")
        sys.exit(0)

    logger.info("Agent text: " + "".join([str(i) for i in generated_text]))
    logger.info("ASR text: " + "".join([str(i) for i in generated_asr_text]))
    logger.info("Function text: " + "".join([str(i) for i in generated_function_text]))
    generated_audio = np.concatenate(generated_audio)
    audio_int16 = (generated_audio * 32768.0).astype(np.int16)

    if not args.no_output:
        # sf.write("output_audio.wav", generated_audio.squeeze(0), 22050)
        sf.write("output_audio.wav", audio_int16, 22050, subtype='PCM_16')
        logger.info("Agent audio written to output_audio.wav")

        # save audio file with both input and output in each channel
        # resample input to 22050 Hz, and pad shorter file to the same length as the longer one
        input_audio, sr = librosa.load(audio_file, sr=22050)
        generated_audio_1d = generated_audio
        maxlen = max(input_audio.shape[0], generated_audio_1d.shape[0])
        input_audio = np.pad(input_audio, (0, maxlen - input_audio.shape[0]), mode="constant")
        generated_audio_1d = np.pad(generated_audio_1d, (0, maxlen - generated_audio_1d.shape[0]), mode="constant",)
        both_audio = np.column_stack([input_audio, generated_audio_1d])  # Create stereo: [T, 2]
        sf.write("input_and_output_combined.wav", both_audio, 22050)
        logger.info("Input and output combined audio written to input_and_output_combined.wav")
    else:
        print("Warmup inference complete")

    logger.info(f"Average time spend on inference: {np.mean(times_spend_on_inference)}")
    logger.info(f"std of time spend on inference: {np.std(times_spend_on_inference)}")
    logger.info(f"Median time spend on inference: {np.median(times_spend_on_inference)}")
    logger.info(f"Min time spend on inference: {np.min(times_spend_on_inference)}")
    logger.info(f"Max time spend on inference: {np.max(times_spend_on_inference)}")
    logger.info(f"All times spend on inference: {[round(i, 4) for i in times_spend_on_inference]}")
    logger.info(f"Number of chunks: {len(times_spend_on_inference)}")
