#!/usr/bin/env python3
"""
Conversation test script specifically for AWS Nova Sonic.

Nova Sonic has unique behavior that requires special handling:
1. Speech-to-speech model: audio in, audio out
2. Requires 16kHz audio input
3. Text transcripts arrive AFTER audio (8+ seconds delay)
4. Requires special "trigger" mechanism to start first turn assistant response (Nova Sonic general requirement)
5. Uses AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION in system instruction
6. Connection timeout after 8 minutes - handled via automatic reconnection

This script does NOT use NullAudioOutputTransport because BotStoppedSpeakingFrame
triggers premature response finalization before text arrives from the server.

Reconnection Handling:
    Nova Sonic has an 8-minute connection limit. When this timeout occurs:
    1. Pipecat automatically reconnects and reloads conversation context
    2. NovaSonicTurnEndDetector detects the ErrorFrame with "timed out"
    3. After a 3-second delay for reconnection, the assistant response is re-triggered
    4. The conversation continues seamlessly from where it left off

Usage:
    uv run python convo-test-nova-sonic.py [--model MODEL_NAME]
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional

from loguru import logger
from dotenv import load_dotenv
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.llm_service import FunctionCallParams
from pipecat.frames.frames import (
    Frame,
    MetricsFrame,
    CancelFrame,
    ErrorFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TranscriptionMessage,
    InputAudioRawFrame,
    LLMRunFrame,
    LLMMessagesAppendFrame,
    TTSStoppedFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    LLMTokenUsage,
    TTFBMetricsData,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.transports.base_transport import TransportParams
from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService

from turns import turns
from tools_schema import ToolsSchemaForTest
from system_instruction_short import system_instruction
import soundfile as sf
from scripts.paced_input_transport import PacedInputTransport
from scripts.tool_call_recorder import ToolCallRecorder

load_dotenv()

# Enable DEBUG logging for Nova Sonic LLM service
import logging

logging.getLogger("pipecat.services.aws.nova_sonic.llm").setLevel(logging.DEBUG)

logger.info("Starting Nova Sonic conversation test...")


# -------------------------
# Custom Frame for Nova Sonic Completion End
# -------------------------

from dataclasses import dataclass
from pipecat.frames.frames import DataFrame


@dataclass
class NovaSonicCompletionEndFrame(DataFrame):
    """Signal that Nova Sonic has finished generating the complete response.

    This frame is emitted when Nova Sonic's `completionEnd` event is received,
    indicating that all text chunks should arrive soon. Use this to know when
    to start the final text collection timeout.
    """
    pass


@dataclass
class NovaSonicTextTurnEndFrame(DataFrame):
    """Signal that Nova Sonic has finished generating text for this turn.

    This frame is emitted when we receive a FINAL TEXT content with stopReason=END_TURN,
    indicating that the transcript for this assistant response is complete.
    """
    pass


# -------------------------
# Custom Nova Sonic LLM Service with Completion End Signal
# -------------------------


class NovaSonicLLMServiceWithCompletionSignal(AWSNovaSonicLLMService):
    """Extended Nova Sonic service that emits frames for turn completion detection.

    The base AWSNovaSonicLLMService handles events but doesn't expose key signals.
    This subclass:
    1. Tracks the current content being received (type, role, generationStage)
    2. Emits NovaSonicTextTurnEndFrame when FINAL TEXT ends with END_TURN
    3. Emits NovaSonicCompletionEndFrame when the session's completionEnd arrives
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_content_type = None
        self._current_content_role = None
        self._current_generation_stage = None

    async def _handle_completion_start_event(self, event_json):
        """Log when a new completion starts."""
        logger.debug("NovaSonicLLM: === completionStart ===")
        await super()._handle_completion_start_event(event_json)

    async def _handle_content_start_event(self, event_json):
        """Track content block info for detecting turn end."""
        content_start = event_json.get("contentStart", {})
        self._current_content_type = content_start.get("type")
        self._current_content_role = content_start.get("role")

        # Parse generationStage from additionalModelFields
        additional = content_start.get("additionalModelFields")
        if additional:
            import json
            try:
                fields = json.loads(additional) if isinstance(additional, str) else additional
                self._current_generation_stage = fields.get("generationStage")
            except:
                self._current_generation_stage = None
        else:
            self._current_generation_stage = None

        logger.debug(
            f"NovaSonicLLM: contentStart type={self._current_content_type} "
            f"role={self._current_content_role} stage={self._current_generation_stage}"
        )
        await super()._handle_content_start_event(event_json)

    async def _handle_content_end_event(self, event_json):
        """Detect when FINAL TEXT ends with END_TURN - this is the turn's transcript end."""
        content_end = event_json.get("contentEnd", {})
        stop_reason = content_end.get("stopReason", "?")

        # Check for FINAL ASSISTANT TEXT with END_TURN - transcript complete for this turn
        if (self._current_content_type == "TEXT" and
            self._current_content_role == "ASSISTANT" and
            self._current_generation_stage == "FINAL" and
            stop_reason == "END_TURN"):
            logger.info(
                f"NovaSonicLLM: *** TEXT TURN END *** FINAL text with END_TURN - pushing signal"
            )
            await self.push_frame(NovaSonicTextTurnEndFrame())
        else:
            logger.debug(f"NovaSonicLLM: contentEnd stopReason={stop_reason}")

        # Clear tracking
        self._current_content_type = None
        self._current_content_role = None
        self._current_generation_stage = None

        await super()._handle_content_end_event(event_json)

    async def _handle_completion_end_event(self, event_json):
        """Handle Nova Sonic's completionEnd event by pushing a signal frame."""
        logger.info("NovaSonicLLM: === completionEnd === pushing signal frame")
        await self.push_frame(NovaSonicCompletionEndFrame())


# -------------------------
# Nova Sonic Turn End Detector
# -------------------------


class NovaSonicTurnEndDetector(FrameProcessor):
    """Detects end of Nova Sonic turn based on text arrival (not audio silence).

    Nova Sonic has unique behavior where assistant text arrives significantly
    AFTER audio output finishes (8+ seconds delay). This detector:

    1. Watches for TTSTextFrame with non-empty content
    2. Buffers all text for the current response
    3. After no more text arrives for `text_timeout_sec`, triggers end-of-turn

    This approach works because:
    - Without BotStoppedSpeakingFrame, Nova Sonic's `_assistant_is_responding` stays True
    - This allows text to be pushed when it arrives from the server
    - We detect response end by watching for text, not audio silence
    """

    def __init__(
        self,
        end_of_turn_callback: Callable[[str], Any],
        text_timeout_sec: float = 5.0,
        post_completion_timeout_sec: float = 3.0,  # Shorter timeout after completionEnd
        response_timeout_sec: float = 30.0,  # Max time to wait for any response
        metrics_callback: Optional[Callable[[MetricsFrame], None]] = None,
    ):
        super().__init__()
        self._end_of_turn_callback = end_of_turn_callback
        self._metrics_callback = metrics_callback
        self._text_timeout = text_timeout_sec
        self._post_completion_timeout = post_completion_timeout_sec
        self._response_timeout = response_timeout_sec

        # State tracking
        self._response_active = False
        self._response_text = ""
        self._last_text_time: Optional[float] = None
        self._last_audio_time: Optional[float] = None  # Track when last audio arrived
        self._timeout_task: Optional[asyncio.Task] = None
        self._response_timeout_task: Optional[asyncio.Task] = None
        self._audio_check_task: Optional[asyncio.Task] = None  # Check for audio completion
        self._audio_frame_count = 0
        self._waiting_for_response = False
        self._trigger_time: Optional[float] = None
        self._processing_turn_end = False  # Guard against concurrent turn completions
        self._completion_ended = False  # True when completionEnd OR audio stops
        self._text_turn_ended = False  # True when NovaSonicTextTurnEndFrame received

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Track metrics
        if isinstance(frame, MetricsFrame) and self._metrics_callback:
            self._metrics_callback(frame)

        # Track response lifecycle
        if isinstance(frame, LLMFullResponseStartFrame):
            self._response_active = True
            self._waiting_for_response = False  # Response received!
            self._response_text = ""
            self._audio_frame_count = 0
            # Cancel response timeout since we got a response
            if self._response_timeout_task:
                self._response_timeout_task.cancel()
                self._response_timeout_task = None
            logger.debug("NovaSonicTurnEndDetector: Response started")

        elif isinstance(frame, TTSAudioRawFrame):
            self._audio_frame_count += 1
            self._last_audio_time = time.monotonic()
            # Start/restart audio completion check
            self._start_audio_check()

        elif isinstance(frame, TTSStoppedFrame):
            # TTSStoppedFrame indicates the response audio is done
            # For Nova Sonic without output transport, this is our turn end signal
            logger.info("NovaSonicTurnEndDetector: TTSStoppedFrame - triggering turn end")
            # Give a moment for any text to arrive, then trigger
            asyncio.create_task(self._delayed_turn_end())
        elif isinstance(frame, LLMFullResponseEndFrame):
            logger.debug("NovaSonicTurnEndDetector: Received LLMFullResponseEndFrame")

        # Handle Nova Sonic completion end signal
        elif isinstance(frame, NovaSonicCompletionEndFrame):
            logger.info(
                f"NovaSonicTurnEndDetector: CompletionEnd received! "
                f"Text so far: {len(self._response_text)} chars. "
                f"Switching to {self._post_completion_timeout}s post-completion timeout."
            )
            self._completion_ended = True
            # Restart timeout with shorter post-completion timeout
            if self._response_text:
                self._start_timeout_check()

        # Handle Nova Sonic text turn end signal - transcript is now complete!
        # This is the most reliable signal that all text for this turn has arrived.
        elif isinstance(frame, NovaSonicTextTurnEndFrame):
            logger.info(
                f"NovaSonicTurnEndDetector: *** TEXT TURN END *** received! "
                f"Text collected: {len(self._response_text)} chars. Triggering turn end."
            )
            self._text_turn_ended = True
            # Cancel any pending timeout - we're ending the turn now
            if self._timeout_task:
                self._timeout_task.cancel()
                self._timeout_task = None
            # Trigger turn end immediately since we know the transcript is complete
            asyncio.create_task(self._handle_text_turn_end())

        # Watch for text frames - this is the key signal for Nova Sonic
        # Only accept text when we're actively waiting for/receiving a response
        # This prevents late-arriving text from previous responses from being counted
        if isinstance(frame, TTSTextFrame):
            text = getattr(frame, "text", None)
            if text:
                # Only accept text if we're waiting for response OR already receiving one
                # AND not currently processing a turn end
                if (self._waiting_for_response or self._response_active) and not self._processing_turn_end:
                    logger.info(
                        f"NovaSonicTurnEndDetector: Processing text ({len(text)} chars): {text[:100]}..."
                    )
                    self._response_text += text
                    self._last_text_time = time.monotonic()
                    self._start_timeout_check()
                else:
                    logger.warning(
                        f"NovaSonicTurnEndDetector: IGNORING late text ({len(text)} chars) - "
                        f"waiting={self._waiting_for_response}, active={self._response_active}, "
                        f"processing={self._processing_turn_end}"
                    )

        await self.push_frame(frame, direction)

    async def _delayed_turn_end(self):
        """Wait briefly then trigger turn end (gives text time to arrive)."""
        await asyncio.sleep(1.0)  # Wait 1 second for any late text

        # Guard against concurrent turn completions
        if self._processing_turn_end:
            logger.debug("NovaSonicTurnEndDetector: Delayed turn end but already processing, ignoring")
            return

        if self._response_active:
            self._processing_turn_end = True
            try:
                # Get accumulated text (might be empty for Nova Sonic)
                text = self._response_text or "[audio response]"
                logger.info(
                    f"NovaSonicTurnEndDetector: Turn ended. Text: {len(self._response_text)} chars, "
                    f"Audio frames: {self._audio_frame_count}"
                )
                self._reset()  # Reset BEFORE callback
                await self._end_of_turn_callback(text)
            finally:
                self._processing_turn_end = False

    async def _handle_text_turn_end(self):
        """Handle turn end when NovaSonicTextTurnEndFrame is received.

        This is the most reliable way to know the transcript is complete.
        We give a very short delay (0.5s) to collect any remaining text that
        might be in flight, then end the turn.
        """
        # Short delay to collect any text still in transit
        await asyncio.sleep(0.5)

        # Guard against concurrent turn completions
        if self._processing_turn_end:
            logger.debug("NovaSonicTurnEndDetector: Text turn end but already processing, ignoring")
            return

        self._processing_turn_end = True
        try:
            # Get accumulated text
            final_text = self._response_text or "[no text captured]"
            audio_frames = self._audio_frame_count
            logger.info(
                f"NovaSonicTurnEndDetector: Turn complete via TEXT_TURN_END signal. "
                f"Text: {len(final_text)} chars, Audio frames: {audio_frames}"
            )
            self._reset()  # Reset BEFORE callback
            await self._end_of_turn_callback(final_text)
        finally:
            self._processing_turn_end = False

    def _start_timeout_check(self):
        """Start or restart the timeout check for more text."""
        if self._timeout_task:
            self._timeout_task.cancel()
        self._timeout_task = asyncio.create_task(self._check_timeout())

    async def _check_timeout(self):
        """Wait for timeout and trigger end-of-turn if no more text."""
        try:
            # Use shorter timeout after completionEnd signal
            timeout = self._post_completion_timeout if self._completion_ended else self._text_timeout
            await asyncio.sleep(timeout)

            # Guard against concurrent turn completions
            if self._processing_turn_end:
                logger.debug("NovaSonicTurnEndDetector: Timeout fired but already processing turn end, ignoring")
                return

            # If we get here, no more text arrived for timeout seconds
            if self._response_text:
                self._processing_turn_end = True
                try:
                    # Capture text and reset state BEFORE the async callback
                    # This prevents late-arriving text from accumulating during the callback
                    final_text = self._response_text
                    audio_frames = self._audio_frame_count
                    completion_status = "after completionEnd" if self._completion_ended else "before completionEnd"
                    logger.info(
                        f"NovaSonicTurnEndDetector: Turn complete after {timeout}s silence ({completion_status}). "
                        f"Text: {len(final_text)} chars, Audio frames: {audio_frames}"
                    )
                    self._reset()  # Reset BEFORE callback to prevent accumulation
                    await self._end_of_turn_callback(final_text)
                finally:
                    self._processing_turn_end = False
        except asyncio.CancelledError:
            pass  # New text arrived, timer was reset

    def _start_audio_check(self):
        """Start or restart the audio completion check."""
        if self._audio_check_task:
            self._audio_check_task.cancel()
        self._audio_check_task = asyncio.create_task(self._check_audio_completion())

    async def _check_audio_completion(self):
        """Check if audio output has stopped, indicating response is complete.

        When audio stops, we switch to the shorter post-completion timeout for text.
        This is more reliable than waiting for completionEnd which may not arrive.
        """
        AUDIO_DONE_THRESHOLD = 2.0  # Consider audio done after 2s of silence
        try:
            while True:
                await asyncio.sleep(AUDIO_DONE_THRESHOLD)

                if self._last_audio_time is None:
                    continue

                time_since_audio = time.monotonic() - self._last_audio_time
                if time_since_audio >= AUDIO_DONE_THRESHOLD and not self._completion_ended:
                    logger.info(
                        f"NovaSonicTurnEndDetector: Audio stopped ({time_since_audio:.1f}s ago). "
                        f"Switching to {self._post_completion_timeout}s post-audio timeout for text."
                    )
                    self._completion_ended = True  # Reuse this flag for audio completion
                    # Restart text timeout with shorter duration
                    if self._response_text:
                        self._start_timeout_check()
                    break  # Done checking audio
        except asyncio.CancelledError:
            pass

    def _reset(self):
        """Reset state for next turn."""
        self._response_active = False
        self._response_text = ""
        self._last_text_time = None
        self._last_audio_time = None
        self._audio_frame_count = 0
        self._waiting_for_response = False
        self._trigger_time = None
        self._completion_ended = False
        self._text_turn_ended = False
        if self._response_timeout_task:
            self._response_timeout_task.cancel()
            self._response_timeout_task = None
        if self._audio_check_task:
            self._audio_check_task.cancel()
            self._audio_check_task = None

    def signal_trigger_sent(self):
        """Called when assistant response is triggered - start response timeout."""
        self._waiting_for_response = True
        self._trigger_time = time.monotonic()
        logger.info(
            f"NovaSonicTurnEndDetector: Trigger sent, waiting for response (timeout={self._response_timeout}s)"
        )
        if self._response_timeout_task:
            self._response_timeout_task.cancel()
        self._response_timeout_task = asyncio.create_task(self._check_response_timeout())

    async def _check_response_timeout(self):
        """Check if response started within timeout period."""
        try:
            await asyncio.sleep(self._response_timeout)

            # Guard against concurrent turn completions
            if self._processing_turn_end:
                logger.debug("NovaSonicTurnEndDetector: Response timeout but already processing, ignoring")
                return

            # If we get here, no response started within timeout
            if self._waiting_for_response and not self._response_active:
                self._processing_turn_end = True
                try:
                    logger.warning(
                        f"NovaSonicTurnEndDetector: No response received within {self._response_timeout}s - "
                        f"ending turn with timeout"
                    )
                    self._reset()  # Reset BEFORE callback
                    await self._end_of_turn_callback("[NO RESPONSE - TIMEOUT]")
                finally:
                    self._processing_turn_end = False
        except asyncio.CancelledError:
            pass  # Response started or turn reset

    def reset_for_reconnection(self):
        """Reset state after a connection timeout/reconnection.

        Called by the PipelineTask error handler when Nova Sonic times out.
        Pipecat automatically reconnects and reloads context, but we need to
        reset our internal state so we're ready for the re-triggered response.
        """
        logger.info("NovaSonicTurnEndDetector: Resetting state for reconnection")

        # Cancel any pending timeout tasks
        if self._timeout_task:
            self._timeout_task.cancel()
            self._timeout_task = None
        if self._response_timeout_task:
            self._response_timeout_task.cancel()
            self._response_timeout_task = None
        if self._audio_check_task:
            self._audio_check_task.cancel()
            self._audio_check_task = None

        # Reset state
        self._response_active = False
        self._response_text = ""
        self._last_text_time = None
        self._last_audio_time = None
        self._audio_frame_count = 0
        self._waiting_for_response = False
        self._trigger_time = None
        self._processing_turn_end = False
        self._completion_ended = False
        self._text_turn_ended = False


# -------------------------
# Utilities for persistence
# -------------------------


def now_iso() -> str:
    try:
        from datetime import UTC

        return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except Exception:
        return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


class RunRecorder:
    """Accumulates per-turn data and writes JSONL + summary."""

    def __init__(self, model_name: str):
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        self.run_dir = Path("runs") / f"nova-sonic-{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = self.run_dir / "transcript.jsonl"
        self.fp = self.out_path.open("a", encoding="utf-8")
        self.model_name = model_name

        # per-turn working state
        self.turn_start_monotonic: Optional[float] = None
        self.turn_usage: Dict[str, Any] = {}
        self.turn_calls: List[Dict[str, Any]] = []
        self.turn_results: List[Dict[str, Any]] = []
        self.turn_index: int = 0
        self.turn_ttfb_ms: Optional[int] = None

        self.total_turns_scored = 0

    def start_turn(self, turn_index: int):
        self.turn_index = turn_index
        self.turn_start_monotonic = time.monotonic()
        self.turn_usage = {}
        self.turn_calls = []
        self.turn_results = []
        self.turn_ttfb_ms = None

    def record_ttfb(self, ttfb_seconds: float):
        if self.turn_ttfb_ms is None:
            self.turn_ttfb_ms = int(ttfb_seconds * 1000)

    def record_usage_metrics(self, m: LLMTokenUsage, model: Optional[str] = None):
        self.turn_usage = {
            "prompt_tokens": m.prompt_tokens,
            "completion_tokens": m.completion_tokens,
            "total_tokens": m.total_tokens,
            "cache_read_input_tokens": m.cache_read_input_tokens,
            "cache_creation_input_tokens": m.cache_creation_input_tokens,
        }
        if model:
            self.model_name = model

    def record_tool_call(self, name: str, args: Dict[str, Any]):
        self.turn_calls.append({"name": name, "args": args})

    def record_tool_result(self, name: str, response: Dict[str, Any]):
        self.turn_results.append({"name": name, "response": response})

    def write_turn(self, *, user_text: str, assistant_text: str):
        latency_ms = None
        if self.turn_start_monotonic is not None:
            latency_ms = int((time.monotonic() - self.turn_start_monotonic) * 1000)

        rec = {
            "ts": now_iso(),
            "turn": self.turn_index,
            "model_name": self.model_name,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "tool_calls": self.turn_calls,
            "tool_results": self.turn_results,
            "tokens": self.turn_usage or None,
            "ttfb_ms": self.turn_ttfb_ms,
            "latency_ms": latency_ms,
        }
        self.fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.fp.flush()
        self.total_turns_scored += 1
        logger.info(f"Recorded turn {self.turn_index}: {assistant_text[:100]}...")

    def write_summary(self):
        runtime = {
            "model_name": self.model_name,
            "turns": self.total_turns_scored,
            "note": "Nova Sonic specific test run",
        }
        (self.run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")


# -------------------------
# Tool call stub
# -------------------------

recorder: Optional[RunRecorder] = None


async def function_catchall(params: FunctionCallParams):
    logger.info(f"Function call: {params}")
    result = {"status": "success"}
    await params.result_callback(result)


# -------------------------
# Frame logger for debugging
# -------------------------


class FrameLogger(FrameProcessor):
    """Logs frames passing through the pipeline."""

    def __init__(self, name: str = "FrameLogger"):
        super().__init__()
        self._name = name
        self._input_audio_count = 0
        self._output_audio_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Track audio frames with periodic logging
        if isinstance(frame, InputAudioRawFrame):
            self._input_audio_count += 1
            if self._input_audio_count == 1 or self._input_audio_count % 100 == 0:
                logger.info(
                    f"[{self._name}] InputAudioRawFrame #{self._input_audio_count} ({len(frame.audio)} bytes)"
                )
        elif isinstance(frame, TTSAudioRawFrame):
            self._output_audio_count += 1
            if self._output_audio_count == 1 or self._output_audio_count % 100 == 0:
                logger.info(
                    f"[{self._name}] TTSAudioRawFrame #{self._output_audio_count} ({len(frame.audio)} bytes)"
                )
        elif isinstance(frame, TranscriptionMessage):
            logger.info(
                f"[{self._name}] TranscriptionMessage: '{frame.message}' (role={frame.role})"
            )
        else:
            logger.debug(f"[{self._name}] {frame.__class__.__name__} ({direction})")

        await self.push_frame(frame, direction)


# -------------------------
# Main
# -------------------------


async def main(model_name: str, max_turns: Optional[int] = None):
    turn_idx = 0

    # Validate model name
    n = model_name.lower()
    if "nova-sonic" not in n and "nova_sonic" not in n:
        logger.warning(f"Model '{model_name}' may not be a Nova Sonic model. Proceeding anyway.")

    # AWS credentials
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session_token = os.getenv("AWS_SESSION_TOKEN")
    aws_region = os.getenv("AWS_REGION", "us-east-1")

    if not (aws_access_key_id and aws_secret_access_key):
        raise EnvironmentError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required for Nova Sonic"
        )

    # Nova Sonic requires the trigger instruction appended to system instruction
    nova_sonic_system_instruction = (
        f"{system_instruction} "
        f"{AWSNovaSonicLLMService.AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION}"
    )
    logger.info(f"Using full system instruction ({len(nova_sonic_system_instruction)} chars)")

    # Create Nova Sonic LLM service with completion end signal
    # Using our custom subclass that emits NovaSonicCompletionEndFrame
    llm = NovaSonicLLMServiceWithCompletionSignal(
        secret_access_key=aws_secret_access_key,
        access_key_id=aws_access_key_id,
        session_token=aws_session_token,
        region=aws_region,
        model=model_name if ":" in model_name else "amazon.nova-sonic-v1:0",
        voice_id="tiffany",
        system_instruction=nova_sonic_system_instruction,
        tools=ToolsSchemaForTest,
    )
    llm.register_function(None, function_catchall)

    # Set up recorder
    global recorder
    recorder = RunRecorder(model_name=model_name)
    recorder.start_turn(turn_idx)

    # Context - Nova Sonic ONLY accepts SPEECH input (not text!)
    # We provide a system message but NO user message
    # The user's question comes as AUDIO via PacedInputTransport
    messages = [
        {"role": "system", "content": system_instruction},
        # NO user message - Nova Sonic only accepts audio input!
    ]
    context = LLMContext(messages, tools=ToolsSchemaForTest)
    logger.info("Context initialized (user input will be audio, not text)")
    logger.info(f"Context messages count: {len(context.get_messages())}")
    for i, msg in enumerate(context.get_messages()):
        logger.info(
            f"  Message {i}: role={msg.get('role')}, content_len={len(str(msg.get('content', '')))}"
        )
    context_aggregator = LLMContextAggregatorPair(context)

    # Pipeline task reference (will be set after task creation)
    task: Optional[PipelineTask] = None
    done = False

    def handle_metrics(frame: MetricsFrame):
        for md in frame.data:
            if isinstance(md, LLMUsageMetricsData):
                recorder.record_usage_metrics(md.value, getattr(md, "model", None))
            elif isinstance(md, TTFBMetricsData):
                recorder.record_ttfb(md.value)

    async def end_of_turn(assistant_text: str):
        """Called when turn detector determines response is complete."""
        nonlocal turn_idx, done

        if done:
            logger.info("end_of_turn called but already done")
            return

        # Record this turn
        recorder.write_turn(
            user_text=turns[turn_idx].get("input", ""),
            assistant_text=assistant_text,
        )

        turn_idx += 1

        # Check if we should continue - respect max_turns limit
        turn_limit = max_turns if max_turns else len(turns)
        if turn_idx < turn_limit:
            recorder.start_turn(turn_idx)
            logger.info(f"Starting turn {turn_idx}: {turns[turn_idx]['input'][:50]}...")

            # Queue audio for next turn
            audio_path = turns[turn_idx].get("audio_file")
            if audio_path and paced_input:
                try:
                    # Calculate audio duration to know when it will finish streaming
                    data, sr = sf.read(audio_path, dtype="int16")
                    audio_duration_sec = len(data) / sr
                    logger.info(f"Audio duration for turn {turn_idx}: {audio_duration_sec:.2f}s")

                    paced_input.enqueue_wav_file(audio_path)
                    logger.info(f"Queued audio for turn {turn_idx}")

                    # Wait for audio to finish streaming
                    wait_time = audio_duration_sec + 0.5
                    logger.info(f"Waiting {wait_time:.2f}s for audio to finish streaming...")
                    await asyncio.sleep(wait_time)

                    await llm.trigger_assistant_response()
                    turn_detector.signal_trigger_sent()
                    logger.info(f"Triggered assistant response for turn {turn_idx}")
                except Exception as e:
                    logger.exception(f"Failed to queue audio for turn {turn_idx}: {e}")
                    # Fall back to text
                    await task.queue_frames(
                        [
                            LLMMessagesAppendFrame(
                                messages=[{"role": "user", "content": turns[turn_idx]["input"]}],
                                run_llm=False,
                            )
                        ]
                    )
                    await asyncio.sleep(0.5)
                    await llm.trigger_assistant_response()
                    turn_detector.signal_trigger_sent()
            else:
                # No audio file, use text
                await task.queue_frames(
                    [
                        LLMMessagesAppendFrame(
                            messages=[{"role": "user", "content": turns[turn_idx]["input"]}],
                            run_llm=False,
                        )
                    ]
                )
                await asyncio.sleep(0.5)
                await llm.trigger_assistant_response()
                turn_detector.signal_trigger_sent()
        else:
            logger.info("Conversation complete!")
            recorder.write_summary()
            done = True
            await llm.push_frame(CancelFrame())

    # Create turn detector
    # Strategy:
    # - Before completionEnd: use long 8-second timeout (text chunks have gaps)
    # - After completionEnd: use short 3-second timeout (all text should be arriving)
    # - Response timeout: 60 seconds (Nova Sonic can take a while to start)
    turn_detector = NovaSonicTurnEndDetector(
        end_of_turn_callback=end_of_turn,
        text_timeout_sec=8.0,  # Before completionEnd: wait 8s after last text
        post_completion_timeout_sec=3.0,  # After completionEnd: wait 3s for stragglers
        response_timeout_sec=60.0,  # If no response within 60s after trigger, skip
        metrics_callback=handle_metrics,
    )

    # Create paced input transport for audio
    # Nova Sonic requires 16kHz input
    input_params = TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=16000,
        audio_in_channels=1,
        audio_in_passthrough=True,
    )
    paced_input = PacedInputTransport(
        input_params,
        pre_roll_ms=100,
        continuous_silence=True,
        wait_for_ready=True,  # Wait for LLM to be ready before sending audio
    )

    # Recorder accessor for ToolCallRecorder
    def current_recorder():
        global recorder
        return recorder

    # Build pipeline
    # NOTE: No NullAudioOutputTransport! It causes BotStoppedSpeakingFrame too quickly,
    # which sets _assistant_is_responding = False and text gets ignored.
    # We rely on TTSStoppedFrame for turn end detection instead.
    pipeline = Pipeline(
        [
            paced_input,
            context_aggregator.user(),
            FrameLogger("PreLLM"),
            llm,
            FrameLogger("PostLLM"),
            ToolCallRecorder(current_recorder),
            turn_detector,  # Detects turn end based on TTSStoppedFrame
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        idle_timeout_secs=60,  # Longer timeout for Nova Sonic's delayed responses
        # These frames reset the idle timer when received
        idle_timeout_frames=(TTSAudioRawFrame, TTSTextFrame, InputAudioRawFrame, MetricsFrame),
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Track reconnection state to avoid duplicate handling
    reconnecting = False
    RECONNECTION_DELAY = 3.0  # Seconds to wait for Pipecat to reconnect

    @task.event_handler("on_error")
    async def handle_task_error(error: ErrorFrame):
        """Handle errors at the PipelineTask level.

        This catches ErrorFrames that flow upstream to the task source,
        including Nova Sonic timeout errors. When we detect a timeout,
        Pipecat automatically reconnects - we just need to reset state
        and re-trigger the assistant response.
        """
        nonlocal turn_idx, done, reconnecting

        error_msg = getattr(error, "error", "") or ""

        # Check for Nova Sonic timeout
        if "timed out" in error_msg.lower() and not done:
            if reconnecting:
                logger.debug("handle_task_error: Already reconnecting, skipping")
                return

            reconnecting = True
            try:
                logger.warning(
                    f"handle_task_error: Nova Sonic timeout detected on turn {turn_idx}. "
                    f"Pipecat will auto-reconnect, waiting {RECONNECTION_DELAY}s then re-triggering..."
                )

                # Reset turn detector state
                turn_detector.reset_for_reconnection()

                # Wait for Pipecat to reconnect and reload context
                await asyncio.sleep(RECONNECTION_DELAY)

                # Re-trigger assistant response for current turn
                if not done:
                    logger.info(f"handle_task_error: Re-triggering assistant response for turn {turn_idx}")
                    await llm.trigger_assistant_response()
                    turn_detector.signal_trigger_sent()
                    logger.info(f"handle_task_error: Successfully re-triggered turn {turn_idx}")
            except Exception as e:
                logger.exception(f"handle_task_error: Failed to handle reconnection: {e}")
            finally:
                reconnecting = False
        else:
            logger.warning(f"handle_task_error: Non-timeout error: {error_msg[:200]}")

    async def queue_first_turn(delay: float = 1.0):
        """Queue the first turn - send user question as AUDIO, then trigger."""
        await asyncio.sleep(delay)

        # Queue LLMRunFrame to establish connection
        logger.info("Queuing LLMRunFrame to establish connection...")
        await task.queue_frames([LLMRunFrame()])

        # Wait for connection to establish
        await asyncio.sleep(1.0)

        # Signal LLM ready to receive audio
        logger.info("Signaling LLM ready for audio...")
        paced_input.signal_ready()

        # Queue user's question as AUDIO (Nova Sonic only accepts speech input!)
        audio_path = turns[0].get("audio_file")
        if audio_path:
            # Calculate audio duration
            data, sr = sf.read(audio_path, dtype="int16")
            audio_duration_sec = len(data) / sr
            logger.info(f"Audio duration: {audio_duration_sec:.2f}s")

            paced_input.enqueue_wav_file(audio_path)
            logger.info(f"Queued user question audio: {audio_path}")

            # Wait for audio to finish streaming (plus small buffer)
            wait_time = audio_duration_sec + 0.5
            logger.info(f"Waiting {wait_time:.2f}s for audio to finish streaming...")
            await asyncio.sleep(wait_time)

            # NOW trigger assistant response (after user audio is sent)
            logger.info("Triggering assistant response after user audio...")
            await llm.trigger_assistant_response()
            turn_detector.signal_trigger_sent()
            logger.info("Triggered assistant response")
        else:
            logger.error("No audio file for first turn - Nova Sonic requires audio input!")
            await task.cancel()

    # Start first turn
    asyncio.create_task(queue_first_turn())

    # Run pipeline
    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conversation test for AWS Nova Sonic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python convo-test-nova-sonic.py
    uv run python convo-test-nova-sonic.py --model amazon.nova-sonic-v1:0

Environment variables:
    AWS_ACCESS_KEY_ID     - AWS access key (required)
    AWS_SECRET_ACCESS_KEY - AWS secret key (required)
    AWS_SESSION_TOKEN     - AWS session token (optional)
    AWS_REGION            - AWS region (default: us-east-1)
""",
    )
    parser.add_argument(
        "--model",
        default="amazon.nova-sonic-v1:0",
        help="Nova Sonic model name (default: amazon.nova-sonic-v1:0)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum number of turns to run (default: all turns)",
    )

    args = parser.parse_args()

    logger.info(f"Running Nova Sonic test with model: {args.model}")
    if args.max_turns:
        logger.info(f"Limiting to {args.max_turns} turns")
    asyncio.run(main(args.model, max_turns=args.max_turns))
