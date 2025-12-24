"""Audio buffer processor for recording conversation audio.

This module provides a processor that captures both user and bot audio
during a conversation, maintaining wall-clock alignment for stereo output.

The processor:
1. Captures InputAudioRawFrame (user audio from microphone/TTS)
2. Captures OutputAudioRawFrame (bot audio from LLM)
3. Aligns both tracks to wall-clock time
4. Emits on_track_audio_data event when recording stops
"""

import asyncio
import time
from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class WallClockAlignedAudioBufferProcessor(FrameProcessor):
    """Records user and bot audio with wall-clock alignment.
    
    This processor captures audio frames and aligns them to wall-clock time,
    allowing stereo output where user and bot audio are on separate channels.
    
    Events:
        on_track_audio_data(processor, user_audio, bot_audio, sample_rate, num_channels):
            Emitted when stop_recording() is called with the accumulated audio data.
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        num_channels: int = 2,
        **kwargs
    ):
        """Initialize the audio buffer.
        
        Args:
            sample_rate: Expected sample rate for audio frames.
            num_channels: Number of output channels (2 for stereo).
        """
        super().__init__(**kwargs)
        self._init_sample_rate = sample_rate
        self._num_channels = num_channels
        
        # Recording state
        self._recording = False
        self._recording_start_time: Optional[float] = None
        
        # Audio buffers (raw bytes)
        self._user_audio_chunks: list[tuple[float, bytes]] = []  # (timestamp, audio_bytes)
        self._bot_audio_chunks: list[tuple[float, bytes]] = []
        
        # Track last audio timestamps for gap detection
        self._last_user_time: Optional[float] = None
        self._last_bot_time: Optional[float] = None
    
    async def start_recording(self) -> None:
        """Start recording audio.
        
        Call this before sending audio through the pipeline.
        """
        self._recording = True
        self._recording_start_time = time.monotonic()
        self._user_audio_chunks = []
        self._bot_audio_chunks = []
        self._last_user_time = None
        self._last_bot_time = None
        logger.info(f"[AudioBuffer] Started recording at {self._recording_start_time}")
    
    async def stop_recording(self) -> None:
        """Stop recording and emit the on_track_audio_data event.
        
        This will emit the accumulated audio data for both tracks.
        """
        if not self._recording:
            logger.warning("[AudioBuffer] stop_recording called but not recording")
            return
        
        self._recording = False
        recording_end_time = time.monotonic()
        duration = recording_end_time - (self._recording_start_time or recording_end_time)
        
        logger.info(f"[AudioBuffer] Stopped recording after {duration:.2f}s")
        
        # Compile audio data
        user_audio = self._compile_audio_track(
            self._user_audio_chunks,
            duration,
            self._init_sample_rate
        )
        bot_audio = self._compile_audio_track(
            self._bot_audio_chunks,
            duration,
            self._init_sample_rate
        )
        
        logger.info(
            f"[AudioBuffer] Compiled audio: user={len(user_audio)} bytes, "
            f"bot={len(bot_audio)} bytes"
        )
        
        # Emit event
        await self._call_event_handler(
            "on_track_audio_data",
            user_audio,
            bot_audio,
            self._init_sample_rate,
            self._num_channels,
        )
    
    def _compile_audio_track(
        self,
        chunks: list[tuple[float, bytes]],
        total_duration: float,
        sample_rate: int,
    ) -> bytes:
        """Compile audio chunks into a continuous buffer with silence for gaps.
        
        Args:
            chunks: List of (timestamp, audio_bytes) tuples.
            total_duration: Total recording duration in seconds.
            sample_rate: Sample rate for calculating silence.
            
        Returns:
            Continuous audio buffer with gaps filled with silence.
        """
        if not chunks:
            # Return silence for the entire duration
            silence_samples = int(total_duration * sample_rate)
            return bytes(silence_samples * 2)  # 16-bit = 2 bytes per sample
        
        # Sort chunks by timestamp
        sorted_chunks = sorted(chunks, key=lambda x: x[0])
        
        # Build output buffer
        output = bytearray()
        bytes_per_second = sample_rate * 2  # 16-bit mono
        
        current_time = 0.0
        
        for timestamp, audio_bytes in sorted_chunks:
            # Calculate time relative to recording start
            relative_time = timestamp - (self._recording_start_time or timestamp)
            
            # Add silence if there's a gap
            if relative_time > current_time:
                gap_duration = relative_time - current_time
                silence_bytes = int(gap_duration * bytes_per_second)
                output.extend(bytes(silence_bytes))
                current_time = relative_time
            
            # Add the audio chunk
            output.extend(audio_bytes)
            chunk_duration = len(audio_bytes) / bytes_per_second
            current_time += chunk_duration
        
        # Pad to total duration if needed
        if current_time < total_duration:
            remaining_silence = int((total_duration - current_time) * bytes_per_second)
            output.extend(bytes(remaining_silence))
        
        return bytes(output)
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process audio frames and record them."""
        await super().process_frame(frame, direction)
        
        if self._recording:
            current_time = time.monotonic()
            
            if isinstance(frame, InputAudioRawFrame):
                # User audio (from mic/TTS)
                self._user_audio_chunks.append((current_time, frame.audio))
                self._last_user_time = current_time
                
            elif isinstance(frame, OutputAudioRawFrame):
                # Bot audio (from LLM)
                self._bot_audio_chunks.append((current_time, frame.audio))
                self._last_bot_time = current_time
        
        # Always forward the frame
        await self.push_frame(frame, direction)

