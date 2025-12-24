"""Text-to-Speech integration for multi-turn evaluation.

This module provides TTS functionality to convert text inputs to audio
for testing speech-to-speech models.

Supported providers:
- OpenAI TTS (recommended, uses existing OPENAI_API_KEY)
- ElevenLabs (higher quality, requires ELEVENLABS_API_KEY)
"""
from .elevenlabs import ElevenLabsTTS
from .elevenlabs import generate_audio_for_benchmark as generate_audio_elevenlabs
from .openai_tts import OpenAITTS
from .openai_tts import generate_audio_for_benchmark as generate_audio_openai

__all__ = [
    "ElevenLabsTTS",
    "OpenAITTS",
    "generate_audio_elevenlabs",
    "generate_audio_openai",
]

