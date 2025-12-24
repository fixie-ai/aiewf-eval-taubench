"""ElevenLabs TTS integration for generating audio from text.

This module provides Text-to-Speech functionality using the ElevenLabs API.
Generated audio can be used to test speech-to-speech models by converting
text-based benchmark inputs to audio format.

Usage:
    # Generate audio for a single text
    tts = ElevenLabsTTS()
    audio_path = await tts.generate("Hello, how can I help you?", "output.wav")
    
    # Generate audio for all turns in a benchmark
    await generate_audio_for_benchmark("tau_bench_airline")
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

import aiohttp
from loguru import logger


# ElevenLabs API configuration
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

# Default voice IDs from ElevenLabs
VOICE_IDS = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",  # Rachel - warm, conversational
    "domi": "AZnzlk1XvdvUeBnXmlld",    # Domi - young, bright
    "bella": "EXAVITQu4vr4xnSDxMaL",   # Bella - soft, pleasant
    "antoni": "ErXwobaYiN019PkySvjV",  # Antoni - calm, professional
    "elli": "MF3mGyEYCl7XYWbV9V6O",    # Elli - bright, upbeat
    "josh": "TxGEqnHWrfWFTfGW9XjX",    # Josh - deep, mature
    "arnold": "VR6AewLTigWG4xSOukaG", # Arnold - crisp, narrator
    "adam": "pNInz6obpgDQGcFmaJgB",    # Adam - deep, confident
    "sam": "yoZ06aMxZJJ28mfd3POQ",     # Sam - raspy, young
}


class ElevenLabsTTS:
    """ElevenLabs Text-to-Speech client.
    
    Generates high-quality audio from text using the ElevenLabs API.
    Supports multiple voices and audio formats.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_turbo_v2_5",
        output_format: str = "mp3_44100_128",  # MP3 for better compatibility
    ):
        """Initialize the TTS client.
        
        Args:
            api_key: ElevenLabs API key. If not provided, reads from ELEVENLABS_API_KEY env var.
            voice_id: Voice to use. Defaults to "rachel" for warm conversational tone.
            model_id: ElevenLabs model to use. Options:
                - eleven_turbo_v2_5: Fastest, good quality
                - eleven_multilingual_v2: Best for non-English
                - eleven_monolingual_v1: Original English model
            output_format: Audio format. Options:
                - mp3_44100_128: MP3 at 44.1kHz, 128kbps (recommended)
                - mp3_44100_192: MP3 at 44.1kHz, 192kbps (higher quality)
                - pcm_16000: 16-bit PCM at 16kHz
                - pcm_22050: 16-bit PCM at 22.05kHz
                - pcm_24000: 16-bit PCM at 24kHz
                - pcm_44100: 16-bit PCM at 44.1kHz
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            logger.warning(
                "ELEVENLABS_API_KEY not set. TTS functionality will not work. "
                "Set the environment variable or pass api_key to constructor."
            )
        
        self.voice_id = voice_id or VOICE_IDS["rachel"]
        self.model_id = model_id
        self.output_format = output_format
        
        # Determine sample rate from format
        if "16000" in output_format:
            self.sample_rate = 16000
        elif "22050" in output_format:
            self.sample_rate = 22050
        elif "24000" in output_format:
            self.sample_rate = 24000
        elif "44100" in output_format:
            self.sample_rate = 44100
        else:
            self.sample_rate = 24000
    
    async def generate(
        self,
        text: str,
        output_path: str | Path,
        voice_id: Optional[str] = None,
    ) -> Path:
        """Generate audio from text.
        
        Args:
            text: The text to convert to speech.
            output_path: Path to save the audio file.
            voice_id: Optional voice override.
            
        Returns:
            Path to the generated audio file.
            
        Raises:
            RuntimeError: If API key is not set or API call fails.
        """
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set")
        
        voice = voice_id or self.voice_id
        url = f"{ELEVENLABS_API_URL}/{voice}"
        
        headers = {
            "Accept": "audio/mpeg" if "mp3" in self.output_format else "audio/wav",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key,
        }
        
        data = {
            "text": text,
            "model_id": self.model_id,
            "output_format": self.output_format,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"ElevenLabs API error {response.status}: {error_text}"
                    )
                
                audio_data = await response.read()
                
                # For MP3 format, save directly
                if "mp3" in self.output_format:
                    output_path.write_bytes(audio_data)
                # For PCM format, convert to WAV using soundfile (more reliable than wave module)
                elif "pcm" in self.output_format:
                    import numpy as np
                    import soundfile as sf
                    
                    # Convert bytes to numpy array (16-bit PCM)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Write as WAV file
                    sf.write(
                        str(output_path),
                        audio_array,
                        self.sample_rate,
                        subtype='PCM_16'
                    )
                else:
                    # Unknown format, save as-is
                    output_path.write_bytes(audio_data)
        
        logger.info(f"Generated audio: {output_path} ({len(text)} chars)")
        return output_path
    
    async def generate_batch(
        self,
        texts: list[str],
        output_dir: str | Path,
        prefix: str = "audio",
    ) -> list[Path]:
        """Generate audio for multiple texts.
        
        Args:
            texts: List of texts to convert.
            output_dir: Directory to save audio files.
            prefix: Prefix for audio filenames.
            
        Returns:
            List of paths to generated audio files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = []
        for i, text in enumerate(texts):
            output_path = output_dir / f"{prefix}_{i:03d}.wav"
            path = await self.generate(text, output_path)
            paths.append(path)
        
        return paths


async def generate_audio_for_benchmark(
    benchmark_name: str,
    voice_id: Optional[str] = None,
    force_regenerate: bool = False,
) -> list[Path]:
    """Generate audio files for all turns in a benchmark.
    
    This function loads a benchmark, extracts the input text from each turn,
    and generates corresponding audio files using ElevenLabs TTS.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., "tau_bench_airline").
        voice_id: Optional voice ID to use.
        force_regenerate: If True, regenerate even if audio files exist.
        
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
    
    tts = ElevenLabsTTS(voice_id=voice_id)
    
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
    
    parser = argparse.ArgumentParser(description="Generate TTS audio for benchmarks")
    parser.add_argument("benchmark", help="Benchmark name (e.g., tau_bench_airline)")
    parser.add_argument("--voice", help="Voice ID to use", default=None)
    parser.add_argument("--force", action="store_true", help="Regenerate existing files")
    args = parser.parse_args()
    
    asyncio.run(generate_audio_for_benchmark(args.benchmark, args.voice, args.force))

