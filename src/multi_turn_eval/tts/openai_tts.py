"""OpenAI TTS integration for generating audio from text.

This module provides Text-to-Speech functionality using the OpenAI TTS API.
It's a simpler alternative to ElevenLabs and uses the same OPENAI_API_KEY
that's likely already configured for running evaluations.

Usage:
    # Generate audio for a single text
    tts = OpenAITTS()
    audio_path = await tts.generate("Hello, how can I help you?", "output.wav")
    
    # Generate audio for all turns in a benchmark
    await generate_audio_for_benchmark("tau_bench_airline", provider="openai")
"""

import asyncio
import os
import wave
from pathlib import Path
from typing import Literal, Optional

from loguru import logger

# OpenAI TTS voices
OPENAI_VOICES = {
    "alloy": "Neutral, balanced voice",
    "echo": "Warm, engaging male voice",
    "fable": "Expressive, British-accented voice",
    "onyx": "Deep, authoritative male voice",
    "nova": "Friendly, warm female voice",
    "shimmer": "Clear, expressive female voice",
}

# TTS models
OPENAI_TTS_MODELS = {
    "tts-1": "Standard quality, fastest",
    "tts-1-hd": "High quality, slower",
}


class OpenAITTS:
    """OpenAI Text-to-Speech client.
    
    Uses the OpenAI TTS API to generate high-quality speech from text.
    Supports multiple voices and output formats.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "nova",
        model: Literal["tts-1", "tts-1-hd"] = "tts-1",
        speed: float = 1.0,
    ):
        """Initialize the TTS client.
        
        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
            voice: Voice to use. Options:
                - alloy: Neutral, balanced
                - echo: Warm male voice
                - fable: British-accented
                - onyx: Deep male voice
                - nova: Friendly female voice (default, good for customer service)
                - shimmer: Clear female voice
            model: TTS model to use:
                - tts-1: Standard quality, faster
                - tts-1-hd: High quality, slower
            speed: Speech speed (0.25 to 4.0). Default 1.0.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning(
                "OPENAI_API_KEY not set. TTS functionality will not work. "
                "Set the environment variable or pass api_key to constructor."
            )
        
        self.voice = voice
        self.model = model
        self.speed = max(0.25, min(4.0, speed))
        self.sample_rate = 24000  # OpenAI TTS outputs at 24kHz
    
    async def generate(
        self,
        text: str,
        output_path: str | Path,
        voice: Optional[str] = None,
        response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "pcm",
    ) -> Path:
        """Generate audio from text.
        
        Args:
            text: The text to convert to speech.
            output_path: Path to save the audio file.
            voice: Optional voice override.
            response_format: Audio format. Use 'pcm' for raw audio or 'wav' for wrapped.
            
        Returns:
            Path to the generated audio file.
            
        Raises:
            RuntimeError: If API key is not set or API call fails.
        """
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        
        # Import here to avoid dependency issues if openai is not installed
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Install with: pip install openai"
            )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        client = AsyncOpenAI(api_key=self.api_key)
        
        try:
            response = await client.audio.speech.create(
                model=self.model,
                voice=voice or self.voice,
                input=text,
                response_format=response_format,
                speed=self.speed,
            )
            
            # Get the audio content
            audio_data = response.content
            
            # For PCM format, wrap in WAV header
            if response_format == "pcm":
                with wave.open(str(output_path), 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_data)
            else:
                # Write raw audio data for other formats
                output_path.write_bytes(audio_data)
            
            logger.info(f"Generated audio: {output_path} ({len(text)} chars)")
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"OpenAI TTS API error: {e}")
    
    async def generate_batch(
        self,
        texts: list[str],
        output_dir: str | Path,
        prefix: str = "audio",
        concurrency: int = 3,
    ) -> list[Path]:
        """Generate audio for multiple texts with controlled concurrency.
        
        Args:
            texts: List of texts to convert.
            output_dir: Directory to save audio files.
            prefix: Prefix for audio filenames.
            concurrency: Max concurrent API calls.
            
        Returns:
            List of paths to generated audio files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def generate_one(i: int, text: str) -> Path:
            async with semaphore:
                output_path = output_dir / f"{prefix}_{i:03d}.wav"
                return await self.generate(text, output_path)
        
        tasks = [generate_one(i, text) for i, text in enumerate(texts)]
        return await asyncio.gather(*tasks)


async def generate_audio_for_benchmark(
    benchmark_name: str,
    voice: Optional[str] = None,
    force_regenerate: bool = False,
    model: str = "tts-1",
) -> list[Path]:
    """Generate audio files for all turns in a benchmark using OpenAI TTS.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., "tau_bench_airline").
        voice: Optional voice to use (alloy, echo, fable, onyx, nova, shimmer).
        force_regenerate: If True, regenerate even if audio files exist.
        model: TTS model (tts-1 or tts-1-hd).
        
    Returns:
        List of paths to generated audio files.
    """
    import importlib
    
    # Load benchmark
    try:
        module = importlib.import_module(f"benchmarks.{benchmark_name}.config")
        config = module.BenchmarkConfig
    except ModuleNotFoundError as e:
        raise ValueError(f"Benchmark '{benchmark_name}' not found: {e}")
    
    turns = config.turns
    audio_dir = config.audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    tts = OpenAITTS(voice=voice or "nova", model=model)
    
    generated_paths = []
    for i, turn in enumerate(turns):
        audio_path = config.get_audio_path(i)
        
        # Skip if already exists and not forcing regeneration
        if audio_path.exists() and not force_regenerate:
            logger.info(f"Audio already exists: {audio_path}")
            generated_paths.append(audio_path)
            continue
        
        # Get the input text
        input_text = turn.get("input", "")
        if not input_text:
            logger.warning(f"Turn {i} has no input text, skipping")
            continue
        
        # Generate audio
        path = await tts.generate(input_text, audio_path)
        generated_paths.append(path)
    
    logger.info(f"Generated {len(generated_paths)} audio files for {benchmark_name}")
    return generated_paths


# CLI for generating audio
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate TTS audio using OpenAI")
    parser.add_argument("benchmark", help="Benchmark name (e.g., tau_bench_airline)")
    parser.add_argument("--voice", help="Voice to use", default="nova",
                       choices=list(OPENAI_VOICES.keys()))
    parser.add_argument("--model", help="TTS model", default="tts-1",
                       choices=list(OPENAI_TTS_MODELS.keys()))
    parser.add_argument("--force", action="store_true", help="Regenerate existing files")
    args = parser.parse_args()
    
    asyncio.run(generate_audio_for_benchmark(
        args.benchmark, args.voice, args.force, args.model
    ))

