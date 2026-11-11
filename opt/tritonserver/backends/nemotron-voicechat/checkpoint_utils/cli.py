"""
CLI argument parsing and dispatch for checkpoint_utils.
"""

import argparse
import logging
import os

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser and subparsers."""
    parser = argparse.ArgumentParser(
        description="Checkpoint utilities for NemotronVoiceChat models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging output",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ##########################
    # extract-codec subcommand
    extract_codec_parser = subparsers.add_parser(
        "extract-codec", help="Extract codec weights from a full model checkpoint",
    )
    extract_codec_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the full model.safetensors checkpoint",
    )
    extract_codec_parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory to save the extracted codec checkpoint",
    )
    extract_codec_parser.add_argument(
        "--filename", type=str, default="codec.safetensors", help="Output filename (default: codec.safetensors)",
    )

    ##########################
    # extract-perception subcommand
    extract_perception_parser = subparsers.add_parser(
        "extract-perception", help="Extract perception weights from a full model checkpoint",
    )
    extract_perception_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the full model.safetensors checkpoint",
    )
    extract_perception_parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory to save the extracted perception checkpoint",
    )
    extract_perception_parser.add_argument(
        "--filename",
        type=str,
        default="perception.safetensors",
        help="Output filename (default: perception.safetensors)",
    )

    ##########################
    # extract-embeddings subcommand
    extract_embeddings_parser = subparsers.add_parser(
        "extract-embeddings", help="Extract embedding weights from a full model checkpoint",
    )
    extract_embeddings_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the full model.safetensors checkpoint",
    )
    extract_embeddings_parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory to save the extracted embeddings",
    )
    extract_embeddings_parser.add_argument(
        "--filename",
        type=str,
        default="embeddings.safetensors",
        help="Output filename (default: embeddings.safetensors)",
    )

    ##########################
    # list-keys subcommand
    list_keys_parser = subparsers.add_parser(
        "list-keys", help="List all keys from a safetensors checkpoint to a text file",
    )
    list_keys_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the .safetensors checkpoint file",
    )
    list_keys_parser.add_argument(
        "--output", type=str, required=True, help="Path to the output text file (one key per line)",
    )

    ##########################
    # load-model subcommand (Riva mode - vLLM engines)
    load_model_parser = subparsers.add_parser(
        "load-model", help="Load the full NemotronVoiceChat model and print a summary",
    )
    load_model_parser.add_argument(
        "--model-repo", type=str, required=True, help="Path to the model repository containing all checkpoints",
    )

    ##########################
    # generate-triton-repo subcommand
    generate_repo_parser = subparsers.add_parser(
        "generate-triton-repo", help="Generate a complete Triton model repository from a full checkpoint",
    )
    generate_repo_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the full model.safetensors checkpoint",
    )
    generate_repo_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the config.json file (auto-discovered from --checkpoint directory if omitted)",
    )
    generate_repo_parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for the model repository",
    )
    ##########################
    # generate-tts-input subcommand
    generate_tts_input_parser = subparsers.add_parser(
        "generate-tts-input", help="Generate TTS model initialization inputs (tts_model_init_inputs.pt)",
    )
    generate_tts_input_parser.add_argument(
        "--config", type=str, required=True, help="Path to the config.json file",
    )
    generate_tts_input_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the full model.safetensors checkpoint",
    )
    generate_tts_input_parser.add_argument(
        "--output", type=str, required=True, help="Output path for tts_model_init_inputs.pt",
    )

    return parser


def main() -> None:
    """Parse arguments, configure logging, and dispatch to the appropriate command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(level=logging.INFO)

    from .import_utils import (
        generate_triton_model_repository,
        generate_tts_model_init_inputs,
        list_checkpoint_keys,
        save_codec_checkpoint,
        save_embeddings_checkpoint,
        save_perception_checkpoint,
    )
    from .load_utils import load_nemotron_voicechat

    if args.command == "extract-codec":
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.filename)
        save_codec_checkpoint(args.checkpoint, output_path)
    elif args.command == "extract-perception":
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.filename)
        save_perception_checkpoint(args.checkpoint, output_path)
    elif args.command == "extract-embeddings":
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.filename)
        save_embeddings_checkpoint(args.checkpoint, output_path)
    elif args.command == "list-keys":
        list_checkpoint_keys(args.checkpoint, args.output)
    elif args.command == "load-model":
        logger.info(f"Loading NemotronVoiceChat from: {args.model_repo}")
        logger.info("=" * 60)

        artifacts = load_nemotron_voicechat(args.model_repo)

        logger.info("\n" + "=" * 60)
        logger.info("MODEL SUMMARY")
        logger.info("=" * 60)

        # Perception summary
        logger.info("\n[Perception]")
        num_perception_params = sum(p.numel() for p in artifacts.perception.parameters())
        logger.info(f"  Parameters: {num_perception_params:,}")
        logger.info(f"  Dtype: {next(artifacts.perception.parameters()).dtype}")

        # Codec summary
        logger.info("\n[Codec]")
        num_codec_params = sum(p.numel() for p in artifacts.codec.parameters())
        logger.info(f"  Parameters: {num_codec_params:,}")
        logger.info(f"  Dtype: {next(artifacts.codec.parameters()).dtype}")

        # LLM summary
        logger.info("\n[LLM (vLLM)]")
        logger.info(f"  Engine: {type(artifacts.llm_engine).__name__}")

        # TTS summary
        logger.info("\n[TTS (vLLM)]")
        logger.info(f"  Engine: {type(artifacts.tts_engine).__name__}")

        logger.info("\n" + "=" * 60)
        logger.info("Model loaded successfully!")
        logger.info("=" * 60)
    elif args.command == "generate-triton-repo":
        logger.info(f"Generating Triton model repository from checkpoint: {args.checkpoint}")
        logger.info("=" * 60)

        generate_triton_model_repository(
            full_checkpoint_path=args.checkpoint, config_json_path=args.config, output_dir=args.output_dir,
        )
    elif args.command == "generate-tts-input":
        logger.info("Generating TTS model initialization inputs")
        logger.info("=" * 60)
        logger.info(f"Config: {args.config}")
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"Output: {args.output}")
        logger.info("=" * 60)

        generate_tts_model_init_inputs(
            config_json_path=args.config, full_checkpoint_path=args.checkpoint, output_path=args.output,
        )

        logger.info("\n" + "=" * 60)
        logger.info("✅ TTS initialization inputs generated successfully!")
        logger.info(f"Output saved to: {args.output}")
        logger.info("=" * 60)
    else:
        parser.print_help()
