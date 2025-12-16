#
# Null Audio Output Transport for Pipecat
#
# A minimal output transport that inherits BaseOutputTransport's speaking detection
# logic (BotStartedSpeakingFrame/BotStoppedSpeakingFrame) but discards audio output.
#
# This is useful for pipelines that need speaking detection without actual audio playback,
# such as evaluation/test pipelines or speech-to-speech model testing.
#
# IMPORTANT: This transport simulates real-time audio playback timing. Without this,
# BotStoppedSpeakingFrame would be generated based on when audio frames stop arriving
# (LLM generation time) rather than when audio would finish playing (content duration).
# LLMs generate audio faster than real-time, so we must pace frame consumption to match
# actual playback duration.
#

import asyncio
import time

from loguru import logger
from pipecat.frames.frames import Frame, InterruptionFrame, OutputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams


class NullAudioOutputTransport(BaseOutputTransport):
    """Output transport that tracks audio for speaking detection but discards output.

    This transport extends BaseOutputTransport to inherit its MediaSender logic,
    which automatically generates BotStartedSpeakingFrame and BotStoppedSpeakingFrame
    based on audio output timing. However, it doesn't actually play or send the audio
    anywhere - it just discards it.

    CRITICAL: This transport simulates real-time audio playback by sleeping
    proportionally to each audio frame's duration. This is necessary because:
    - LLMs generate audio faster than real-time (e.g., 33s of audio in 10s)
    - BotStoppedSpeakingFrame is triggered when audio queue is empty for BOT_VAD_STOP_SECS
    - Without timing simulation, BotStoppedSpeakingFrame fires too early
    - This would cause turn advancement before audio "finishes playing"

    This is useful for:
    - Test/evaluation pipelines where you don't need audio playback
    - Speech-to-speech model testing where you only need transcripts
    - Pipelines that need speaking state tracking without audio output hardware

    The key mechanism inherited from BaseOutputTransport:
    - MediaSender tracks TTSAudioRawFrame timing
    - After BOT_VAD_STOP_SECS of no audio, generates BotStoppedSpeakingFrame
    - BotStoppedSpeakingFrame flows upstream to trigger response finalization
    """

    def __init__(self, params: TransportParams, **kwargs):
        """Initialize the null audio output transport.

        Args:
            params: Transport configuration parameters. Should have audio_out_enabled=True
                    and appropriate sample rate settings.
            **kwargs: Additional arguments passed to BaseOutputTransport.
        """
        super().__init__(params, **kwargs)
        # Timing state for simulating real-time audio playback
        # Based on WebsocketServerOutputTransport's timing implementation
        self._next_send_time = 0.0
        self._total_audio_duration = 0.0
        self._total_sleep_time = 0.0
        self._frame_count = 0
        self._playback_start_time = 0.0

    async def start(self, frame: StartFrame):
        """Start the transport and initialize the MediaSender.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        # Call set_transport_ready to initialize MediaSender which handles
        # BotStartedSpeakingFrame/BotStoppedSpeakingFrame generation
        await self.set_transport_ready(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle interruption timing reset.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # Reset timing on interruption (matches WebsocketServerOutputTransport behavior)
        if isinstance(frame, InterruptionFrame):
            self._next_send_time = 0.0

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Simulate audio playback timing, then discard the frame.

        This method sleeps to simulate real-time audio playback. Without this,
        the audio queue would drain at LLM generation speed (faster than real-time),
        causing BotStoppedSpeakingFrame to fire too early.

        The timing algorithm (from WebsocketServerOutputTransport):
        1. Calculate when this frame should finish "playing"
        2. Sleep until that time (if in the future)
        3. Advance the next send time by this frame's duration

        Args:
            frame: The audio frame to "write" (actually discarded after timing).

        Returns:
            True always, indicating the frame was "successfully written".
        """
        # Calculate this frame's duration in seconds
        # Formula: num_bytes / (sample_rate * num_channels * bytes_per_sample)
        # For 16-bit audio, bytes_per_sample = 2
        num_samples = len(frame.audio) // (frame.num_channels * 2)
        frame_duration = num_samples / frame.sample_rate

        self._frame_count += 1
        self._total_audio_duration += frame_duration

        # Log first frame and periodic updates
        if self._frame_count == 1:
            self._playback_start_time = time.monotonic()
            logger.info(
                f"[NullAudioOutput] First audio frame: {frame_duration*1000:.1f}ms, "
                f"samples={num_samples}, sr={frame.sample_rate}"
            )

        # Simulate real-time playback with pacing
        await self._simulate_playback_timing(frame_duration)

        # Log every 100 frames
        if self._frame_count % 100 == 0:
            elapsed = time.monotonic() - self._playback_start_time
            logger.info(
                f"[NullAudioOutput] Frame {self._frame_count}: "
                f"total_audio={self._total_audio_duration:.2f}s, "
                f"total_sleep={self._total_sleep_time:.2f}s, "
                f"wall_elapsed={elapsed:.2f}s"
            )

        # Don't actually play/send the audio - just discard it
        # Return True so MediaSender continues to track speaking state
        return True

    async def _simulate_playback_timing(self, duration: float):
        """Sleep to simulate real-time audio playback.

        This implements a pacing algorithm that ensures audio is "consumed"
        at real-time speed, not at LLM generation speed.

        Args:
            duration: The duration of the audio frame in seconds.
        """
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)

        if sleep_duration > 0:
            await asyncio.sleep(sleep_duration)
            self._total_sleep_time += sleep_duration
            # We caught up to schedule, advance by frame duration
            self._next_send_time += duration
        else:
            # We're behind or just starting, reset baseline and advance
            self._next_send_time = time.monotonic() + duration

    def _log_playback_summary(self):
        """Log a summary of playback timing (called when speaking stops)."""
        if self._frame_count > 0:
            elapsed = time.monotonic() - self._playback_start_time
            logger.info(
                f"[NullAudioOutput] Playback summary: "
                f"frames={self._frame_count}, "
                f"audio_duration={self._total_audio_duration:.2f}s, "
                f"sleep_time={self._total_sleep_time:.2f}s, "
                f"wall_elapsed={elapsed:.2f}s"
            )
            # Reset for next speaking segment
            self._frame_count = 0
            self._total_audio_duration = 0.0
            self._total_sleep_time = 0.0
