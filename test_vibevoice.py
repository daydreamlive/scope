#!/usr/bin/env python3
"""Test script for VibeVoice pipeline integration.

This script tests the VibeVoice text-to-speech pipeline by:
1. Loading the pipeline with default settings
2. Generating audio from sample text
3. Streaming the audio in chunks
4. Saving the output to a WAV file
"""

import argparse
import logging
import sys
import wave
from pathlib import Path

import numpy as np
import torch

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from scope.core.pipelines.vibevoice.pipeline import VibeVoicePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def save_audio_chunks(chunks: list[np.ndarray], output_path: Path, sample_rate: int):
    """Save audio chunks to a WAV file.
    
    Args:
        chunks: List of audio chunks (float32, [-1, 1] range)
        output_path: Path to save the WAV file
        sample_rate: Sample rate of the audio
    """
    # Concatenate all chunks
    audio = np.concatenate(chunks)
    
    # Convert to int16
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    
    # Save to WAV file
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    logger.info(f"Saved audio to {output_path}")
    logger.info(f"Duration: {len(audio) / sample_rate:.2f}s")


def test_basic_generation(
    pipeline: VibeVoicePipeline,
    text: str,
    output_path: Path,
):
    """Test basic audio generation from text.
    
    Args:
        pipeline: VibeVoice pipeline instance
        text: Text to synthesize
        output_path: Path to save the output audio
    """
    logger.info("=" * 80)
    logger.info("Testing basic generation")
    logger.info("=" * 80)
    logger.info(f"Text: {text}")
    
    # Prepare the pipeline (generates audio internally)
    logger.info("Preparing pipeline (generating audio)...")
    reqs = pipeline.prepare(text=text)
    logger.info(f"Requirements: {reqs}")
    
    # Stream the audio in chunks
    logger.info("Streaming audio chunks...")
    chunks = []
    chunk_count = 0
    
    while True:
        chunk = pipeline()
        if chunk is None:
            break
        
        chunks.append(chunk.numpy())
        chunk_count += 1
        
        if chunk_count % 50 == 0:
            logger.info(f"Streamed {chunk_count} chunks ({len(chunks) * pipeline.chunk_size / pipeline.target_sample_rate:.2f}s)")
    
    logger.info(f"Streamed {chunk_count} chunks total")
    
    # Save the audio
    if chunks:
        save_audio_chunks(chunks, output_path, pipeline.target_sample_rate)
    else:
        logger.warning("No audio chunks generated!")


def test_multiple_generations(
    pipeline: VibeVoicePipeline,
    texts: list[str],
    output_dir: Path,
):
    """Test multiple consecutive generations.
    
    Args:
        pipeline: VibeVoice pipeline instance
        texts: List of texts to synthesize
        output_dir: Directory to save output files
    """
    logger.info("=" * 80)
    logger.info("Testing multiple consecutive generations")
    logger.info("=" * 80)
    
    for i, text in enumerate(texts):
        logger.info(f"\nGeneration {i+1}/{len(texts)}")
        logger.info(f"Text: {text[:100]}...")
        
        output_path = output_dir / f"output_{i+1}.wav"
        test_basic_generation(pipeline, text, output_path)


def main():
    parser = argparse.ArgumentParser(description="Test VibeVoice pipeline")
    parser.add_argument(
        "--model-path",
        type=str,
        default="microsoft/VibeVoice-Realtime-0.5B",
        help="Path to the HuggingFace model",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Emma",
        help="Speaker name (e.g., Emma, Mike, Grace)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (cuda/mps/cpu), auto-detected if not specified",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./test_outputs"),
        help="Directory to save test outputs",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Custom text to synthesize (if not specified, uses sample texts)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.5,
        help="CFG scale for generation",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample texts for testing
    sample_texts = [
        "Hello, this is a test of the VibeVoice text-to-speech system. "
        "The quick brown fox jumps over the lazy dog.",
        
        "VibeVoice is a novel framework designed for generating expressive, "
        "long-form, multi-speaker conversational audio.",
        
        "Testing numbers and punctuation: one, two, three! "
        "Can you hear this? Yes, I can. That's amazing!",
    ]
    
    # Use custom text if provided, otherwise use samples
    if args.text:
        texts = [args.text]
    else:
        texts = sample_texts
    
    logger.info("Initializing VibeVoice pipeline...")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Speaker: {args.speaker}")
    logger.info(f"Device: {args.device or 'auto-detect'}")
    logger.info(f"CFG Scale: {args.cfg_scale}")
    
    try:
        # Initialize pipeline
        pipeline = VibeVoicePipeline(
            model_path=args.model_path,
            speaker_name=args.speaker,
            device=args.device,
            cfg_scale=args.cfg_scale,
        )
        
        # Run tests
        if len(texts) == 1:
            output_path = args.output_dir / "output.wav"
            test_basic_generation(pipeline, texts[0], output_path)
        else:
            test_multiple_generations(pipeline, texts, args.output_dir)
        
        logger.info("=" * 80)
        logger.info("All tests completed successfully!")
        logger.info(f"Output files saved to: {args.output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

