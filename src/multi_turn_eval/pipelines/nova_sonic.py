"""Nova Sonic pipeline components for AWS Bedrock Nova Sonic models."""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from loguru import logger

from pipecat.frames.frames import (
    DataFrame,
    Frame,
    InputAudioRawFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
    MetricsFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService


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


class NovaSonicLLMServiceWithCompletionSignal(AWSNovaSonicLLMService):
    """Extended Nova Sonic service that emits frames for turn completion detection.

    The base AWSNovaSonicLLMService handles events but doesn't expose key signals.
    This subclass:
    1. Tracks the current content being received (type, role, generationStage)
    2. Emits NovaSonicTextTurnEndFrame when FINAL TEXT ends with END_TURN
    3. Emits NovaSonicCompletionEndFrame when the session's completionEnd arrives
    4. Emits TTFB metrics (time from trigger to first audio)
    5. Supports Nova 2 Sonic VAD configuration (endpointingSensitivity)
    6. Overrides reset_conversation() with retry limits to prevent infinite error cascade
    """

    def __init__(
        self,
        endpointing_sensitivity: str = None,
        max_reconnect_attempts: int = 3,
        max_context_turns: int = 15,
        on_reconnecting: Optional[Callable[[], None]] = None,
        on_reconnected: Optional[Callable[[], None]] = None,
        on_retriggered: Optional[Callable[[], None]] = None,
        on_max_reconnects_exceeded: Optional[Callable[[], Any]] = None,
        **kwargs,
    ):
        """Initialize the Nova Sonic service.

        Args:
            endpointing_sensitivity: VAD sensitivity for Nova 2 Sonic only.
                Options: "HIGH" (quick cutoff), "MEDIUM" (default), "LOW" (longer wait).
                Only applicable to amazon.nova-2-sonic-v1:0 model.
                Nova Sonic v1 does not support this parameter.
            max_reconnect_attempts: Maximum reconnection attempts before giving up.
            max_context_turns: Maximum number of user/assistant turn pairs to keep during
                reconnection. Older turns are truncated to avoid exceeding Nova Sonic's
                context limits. System messages are always preserved.
            on_reconnecting: Callback when reconnection starts (pause audio input).
            on_reconnected: Callback when reconnection completes (resume audio input).
            on_retriggered: Callback after assistant response is re-triggered (signal turn detector).
            on_max_reconnects_exceeded: Async callback when max reconnects exceeded (cancel task).
        """
        super().__init__(**kwargs)
        self._current_content_type = None
        self._current_content_role = None
        self._current_generation_stage = None
        self._ttfb_started = False  # Track if we've started TTFB timing for this turn
        self._endpointing_sensitivity = endpointing_sensitivity

        # Reconnection handling
        self._max_reconnect_attempts = max_reconnect_attempts
        self._max_context_turns = max_context_turns
        self._reconnect_attempts = 0
        self._is_reconnecting = False
        self._need_retrigger_after_reconnect = False
        self._on_reconnecting = on_reconnecting
        self._on_reconnected = on_reconnected
        self._on_retriggered = on_retriggered
        self._on_max_reconnects_exceeded = on_max_reconnects_exceeded

    def can_generate_metrics(self) -> bool:
        """Enable metrics generation for TTFB tracking.

        The base FrameProcessor returns False by default, which prevents
        start_ttfb_metrics() and stop_ttfb_metrics() from working.
        """
        return True

    def is_reconnecting(self) -> bool:
        """Check if currently reconnecting (for external coordination)."""
        return self._is_reconnecting

    def reset_reconnect_counter(self):
        """Reset the reconnection attempt counter (call on successful turn completion).

        Does NOT reset if currently reconnecting, to prevent race conditions where
        a turn completes during reconnection and resets the counter mid-cycle.
        """
        if self._is_reconnecting:
            logger.debug(
                f"Not resetting reconnect counter during reconnection (current: {self._reconnect_attempts})"
            )
            return
        if self._reconnect_attempts > 0:
            logger.info(f"Resetting reconnect counter (was {self._reconnect_attempts})")
        self._reconnect_attempts = 0

    def _truncate_context_for_reconnection(self):
        """Truncate context for reconnection to fit within Nova Sonic's limits.

        Nova Sonic has strict context limits during session reconnection (~5-10K chars total).
        Strategy: Use a minimal system prompt (just core instructions) + most recent 1 turn (2 messages).

        Based on testing:
        - 21.6K chars: "Chat history is over max limit" error
        - 10.8K chars: Still too large
        - Need to stay under ~5K chars total for reliable reconnection

        Returns the number of messages removed, or 0 if no truncation was needed.
        """
        if not self._context:
            return 0

        messages = self._context.get_messages()
        if not messages:
            return 0

        # Separate system messages from conversation messages
        system_messages = []
        conversation_messages = []
        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                system_messages.append(msg)
            else:
                conversation_messages.append(msg)

        # Use a minimal system prompt for reconnection (~300 chars)
        # This preserves core behavior while fitting within strict limits
        minimal_system_prompt = """You are a helpful voice assistant for the AI Engineer World's Fair 2025 (June 3-5, San Francisco).
Answer questions about the conference schedule, sessions, and speakers.
Be conversational and concise. If you don't have specific information, say so politely."""

        if system_messages:
            original_content = str(system_messages[0].get("content", ""))
            original_len = len(original_content)
            system_messages = [{"role": "system", "content": minimal_system_prompt}]
            logger.warning(
                f"Using minimal system prompt for reconnection: {original_len} chars -> {len(minimal_system_prompt)} chars"
            )

        # Keep ZERO conversation messages - Nova Sonic's internal state may be corrupted
        # after 8-minute timeout, and any conversation history causes "over max limit" errors
        max_messages = 0
        if len(conversation_messages) > max_messages:
            messages_removed = len(conversation_messages)
            truncated_conversation = []  # Keep nothing
            logger.warning(
                f"Truncating conversation for reconnection: keeping last {max_messages} messages. "
                f"Removing {messages_removed} older messages."
            )
        else:
            messages_removed = 0
            truncated_conversation = conversation_messages
            logger.debug(
                f"Conversation truncation not needed: {len(conversation_messages)} messages "
                f"<= {max_messages} max"
            )

        # Rebuild: minimal system + recent conversation
        new_messages = system_messages + truncated_conversation

        # Update the context with truncated messages
        self._context.set_messages(new_messages)
        return messages_removed

    async def reset_conversation(self):
        """Override to add retry limits, context truncation, and preserve trigger state.

        The base class calls this automatically when errors occur in the receive task.
        Without retry limits, connection errors can cascade infinitely.

        Key improvements:
        1. Retry limits - gives up after max_reconnect_attempts
        2. Context truncation - removes old messages to fit within Nova Sonic limits
        3. Preserves trigger state - re-triggers assistant response after reconnection
        4. Callbacks - notifies external components to pause/resume audio input
        """
        # Check retry limit
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(
                f"Max reconnect attempts ({self._max_reconnect_attempts}) reached. "
                f"Giving up on reconnection."
            )
            await self.push_error(
                error_msg=f"Nova Sonic: Max reconnect attempts ({self._max_reconnect_attempts}) exceeded"
            )
            self._wants_connection = False

            # Call the max reconnects exceeded callback to terminate gracefully
            if self._on_max_reconnects_exceeded:
                try:
                    result = self._on_max_reconnects_exceeded()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.exception(f"Error in on_max_reconnects_exceeded callback: {e}")
            return

        self._reconnect_attempts += 1
        self._is_reconnecting = True

        logger.warning(
            f"Nova Sonic reset_conversation() attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}"
        )

        # Remember if we need to re-trigger after reconnection
        # This is lost in _disconnect() so we must capture it here
        self._need_retrigger_after_reconnect = (
            self._triggering_assistant_response or self._assistant_is_responding
        )
        logger.info(
            f"Nova Sonic: Will re-trigger after reconnect: {self._need_retrigger_after_reconnect} "
            f"(triggering={self._triggering_assistant_response}, responding={self._assistant_is_responding})"
        )

        # Truncate context to avoid exceeding Nova Sonic's limits during reconnection
        messages_removed = self._truncate_context_for_reconnection()
        if messages_removed > 0:
            logger.info(f"Nova Sonic: Removed {messages_removed} old messages before reconnection")

        # Notify external components to pause audio input
        if self._on_reconnecting:
            try:
                self._on_reconnecting()
            except Exception as e:
                logger.warning(f"Error in on_reconnecting callback: {e}")

        # Call parent implementation (handles disconnect/reconnect/context reload)
        try:
            await super().reset_conversation()
        except Exception as e:
            logger.exception(f"Error in parent reset_conversation: {e}")
            self._is_reconnecting = False
            raise

        self._is_reconnecting = False

        # Notify external components reconnection is complete
        if self._on_reconnected:
            try:
                self._on_reconnected()
            except Exception as e:
                logger.warning(f"Error in on_reconnected callback: {e}")

        # Re-trigger assistant response if we were in the middle of one
        # NOTE: Disabled re-trigger after reconnection as it causes "Chat history over max limit"
        # errors. The user will need to re-send their audio to continue the conversation.
        if self._need_retrigger_after_reconnect:
            logger.warning(
                "Nova Sonic: Skipping re-trigger after reconnection (causes errors). "
                "User must re-send audio to continue."
            )
            self._need_retrigger_after_reconnect = False
            # Don't trigger - let the next user audio input restart the conversation

    async def _send_session_start_event(self):
        """Override to add endpointingSensitivity for Nova 2 Sonic VAD control.

        Nova 2 Sonic supports VAD configuration via endpointingSensitivity:
        - HIGH: Very sensitive to pauses (quick cutoff)
        - MEDIUM: Balanced sensitivity (default)
        - LOW: Less sensitive to pauses (longer wait before cutoff)

        Nova Sonic v1 does not support this parameter.
        """
        # Build inference configuration
        inference_config = {
            "maxTokens": self._params.max_tokens,
            "topP": self._params.top_p,
            "temperature": self._params.temperature,
        }

        # Add endpointingSensitivity for Nova 2 Sonic
        if self._endpointing_sensitivity:
            inference_config["endpointingSensitivity"] = self._endpointing_sensitivity
            logger.info(f"NovaSonicLLM: Using endpointingSensitivity={self._endpointing_sensitivity}")

        session_start = json.dumps({
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": inference_config
                }
            }
        })
        await self._send_client_event(session_start)

    async def start_ttfb_for_user_audio_complete(self):
        """Start TTFB timing when user audio delivery is complete.

        This should be called when the last byte of user audio has been
        delivered to the model. TTFB = time from this point to first model audio.
        """
        logger.info("NovaSonicLLM: Starting TTFB metrics (user audio complete)")
        await self.start_ttfb_metrics()
        self._ttfb_started = True
        self._audio_output_count = 0  # Reset for new turn

    async def trigger_assistant_response(self):
        """Override to trigger assistant response."""
        logger.info("NovaSonicLLM: Triggering assistant response")
        await super().trigger_assistant_response()

    async def _receive_task_handler(self):
        """Override to add custom event handling for turn end detection.

        This extends the parent's receive handler to:
        1. Track content metadata (type, role, generationStage)
        2. Emit NovaSonicTextTurnEndFrame when AUDIO ends with END_TURN
        3. Emit NovaSonicCompletionEndFrame when completionEnd arrives
        4. Capture TTFB metrics on first audio output
        """
        try:
            while self._stream and not self._disconnecting:
                output = await self._stream.await_output()
                result = await output[1].receive()

                if result.value and result.value.bytes_:
                    response_data = result.value.bytes_.decode("utf-8")
                    json_data = json.loads(response_data)

                    if "event" in json_data:
                        event_json = json_data["event"]

                        # Route events to handlers
                        if "completionStart" in event_json:
                            await self._handle_completion_start_event(event_json)
                        elif "contentStart" in event_json:
                            await self._handle_content_start_event(event_json)
                        elif "textOutput" in event_json:
                            await self._handle_text_output_event(event_json)
                        elif "audioOutput" in event_json:
                            await self._handle_audio_output_event(event_json)
                        elif "toolUse" in event_json:
                            await self._handle_tool_use_event(event_json)
                        elif "contentEnd" in event_json:
                            await self._handle_content_end_event(event_json)
                        elif "completionEnd" in event_json:
                            await self._handle_completion_end_event(event_json)
        except Exception as e:
            if self._disconnecting:
                logger.debug(f"NovaSonicLLM: _receive_task_handler exception during disconnect: {e}")
                return
            logger.error(f"NovaSonicLLM: Error in receive task: {e}")
            await self.push_error(error_msg=f"Error processing responses: {e}", exception=e)
            if self._wants_connection:
                await self.reset_conversation()

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
            try:
                fields = json.loads(additional) if isinstance(additional, str) else additional
                self._current_generation_stage = fields.get("generationStage")
            except:
                self._current_generation_stage = None
        else:
            self._current_generation_stage = None

        # Track content block depth
        if not hasattr(self, '_content_depth'):
            self._content_depth = 0
        self._content_depth += 1

        logger.info(
            f"NovaSonicLLM: >>> contentStart [{self._content_depth}] "
            f"type={self._current_content_type} role={self._current_content_role} "
            f"stage={self._current_generation_stage}"
        )
        await super()._handle_content_start_event(event_json)

    async def _handle_text_output_event(self, event_json):
        """Log text output events and emit SPECULATIVE text for transcription."""
        text_output = event_json.get("textOutput", {})
        content = text_output.get("content", "")

        # Log the text
        logger.debug(
            f"NovaSonicLLM:     textOutput type={self._current_content_type} "
            f"role={self._current_content_role} stage={self._current_generation_stage} "
            f"content={content[:80]!r}..."
        )

        # Emit SPECULATIVE ASSISTANT text as TTSTextFrame for transcription
        # This arrives in real-time with audio, unlike FINAL which is delayed 30+ seconds
        if (self._current_content_role == "ASSISTANT" and
            self._current_generation_stage == "SPECULATIVE" and
            content):
            from pipecat.frames.frames import AggregationType
            logger.info(f"NovaSonicLLM: Emitting SPECULATIVE text ({len(content)} chars): {content[:60]}...")
            frame = TTSTextFrame(content, aggregated_by=AggregationType.SENTENCE)
            await self.push_frame(frame)

        await super()._handle_text_output_event(event_json)

    async def _handle_audio_output_event(self, event_json):
        """Log audio output events and capture TTFB on first audio."""
        if not hasattr(self, '_audio_output_count'):
            self._audio_output_count = 0
        self._audio_output_count += 1

        # Stop TTFB metrics on first audio output (this is the "first byte" for speech-to-speech)
        if self._audio_output_count == 1 and self._ttfb_started:
            logger.info("NovaSonicLLM: Stopping TTFB metrics on first audio output")
            await self.stop_ttfb_metrics()
            self._ttfb_started = False

        if self._audio_output_count == 1 or self._audio_output_count % 50 == 0:
            logger.info(
                f"NovaSonicLLM:     audioOutput #{self._audio_output_count} "
                f"role={self._current_content_role}"
            )
        await super()._handle_audio_output_event(event_json)

    async def _handle_content_end_event(self, event_json):
        """Detect when AUDIO ends with END_TURN - this signals the turn is complete.

        Since we're using SPECULATIVE text (which arrives with audio), we use AUDIO END_TURN
        as the turn completion signal instead of waiting for FINAL text.
        """
        content_end = event_json.get("contentEnd", {})
        stop_reason = content_end.get("stopReason", "?")

        # Track content block depth
        if not hasattr(self, '_content_depth'):
            self._content_depth = 0
        depth_before = self._content_depth
        self._content_depth = max(0, self._content_depth - 1)

        logger.debug(
            f"NovaSonicLLM: <<< contentEnd [{depth_before}->{self._content_depth}] "
            f"type={self._current_content_type} role={self._current_content_role} "
            f"stage={self._current_generation_stage} stopReason={stop_reason}"
        )

        # Check for AUDIO with END_TURN - this means the assistant is done speaking
        # Since we capture SPECULATIVE text (which arrives with audio), this is our turn end signal
        if (self._current_content_type == "AUDIO" and
            self._current_content_role == "ASSISTANT" and
            stop_reason == "END_TURN"):
            logger.info(
                f"NovaSonicLLM: *** AUDIO TURN END *** Assistant audio complete - pushing signal"
            )
            await self.push_frame(NovaSonicTextTurnEndFrame())

        # Clear tracking
        self._current_content_type = None
        self._current_content_role = None
        self._current_generation_stage = None

        await super()._handle_content_end_event(event_json)

    async def _handle_completion_end_event(self, event_json):
        """Handle Nova Sonic's completionEnd event by pushing a signal frame."""
        logger.info("NovaSonicLLM: === completionEnd === pushing signal frame")
        await self.push_frame(NovaSonicCompletionEndFrame())


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
            # NOTE: We intentionally do NOT call _start_audio_check() here.
            # Nova Sonic generates responses in multiple audio segments with
            # 2+ second pauses between them. The audio silence detection would
            # misinterpret these inter-segment gaps as "response complete" and
            # trigger premature turn completion before AUDIO END_TURN arrives.
            # Instead, we rely solely on NovaSonicTextTurnEndFrame (AUDIO END_TURN)
            # as the authoritative turn completion signal.

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

        # Watch for text frames FIRST - process text before checking turn end signals
        # This ensures we capture text that arrives in the same batch as the turn end signal
        if isinstance(frame, TTSTextFrame):
            text = getattr(frame, "text", None)
            if text:
                # Accept text if:
                # - We're waiting for response OR already receiving one
                # - OR we received the text turn end signal (collecting late text)
                # AND not currently in the final turn end processing
                can_accept = (
                    (self._waiting_for_response or self._response_active or self._text_turn_ended)
                    and not self._processing_turn_end
                )
                if can_accept:
                    logger.info(
                        f"NovaSonicTurnEndDetector: Processing text ({len(text)} chars): {text[:100]}..."
                    )
                    self._response_text += text
                    self._last_text_time = time.monotonic()
                    self._start_timeout_check()
                # else: Ignore FINAL text chunks that arrive after turn completion.
                # Nova Sonic sends text twice: SPECULATIVE (with audio) and FINAL (4-6s later).
                # We capture SPECULATIVE text during the turn, so FINAL chunks are duplicates.

        # Handle Nova Sonic text turn end signal - transcript is now complete!
        # This is the most reliable signal that all text for this turn has arrived.
        # Process AFTER text frames so we capture text in the same batch.
        if isinstance(frame, NovaSonicTextTurnEndFrame):
            logger.info(
                f"NovaSonicTurnEndDetector: *** TEXT TURN END *** received! "
                f"Text collected: {len(self._response_text)} chars."
            )
            self._text_turn_ended = True
            # Cancel any pending timeout - we're ending the turn now
            if self._timeout_task:
                self._timeout_task.cancel()
                self._timeout_task = None
            # Use a short delay to let any remaining text frames be processed
            asyncio.create_task(self._handle_text_turn_end())

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

        Since we now use SPECULATIVE text (which arrives with audio), we only need
        a short delay after AUDIO END_TURN to collect any remaining text chunks.
        """
        # Short delay - SPECULATIVE text arrives with audio, so by the time
        # AUDIO END_TURN fires, most text should already be captured
        logger.info("NovaSonicTurnEndDetector: AUDIO END_TURN received, waiting 1s for final text...")
        await asyncio.sleep(1.0)

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


# ============================================================================
# Nova Sonic Pipeline
# ============================================================================


class NovaSonicPipeline:
    """Pipeline for AWS Nova Sonic speech-to-speech models.

    Nova Sonic has unique behavior that requires special handling:
    1. Speech-to-speech model: audio in, audio out
    2. Requires 16kHz audio input
    3. Text transcripts arrive AFTER audio (8+ seconds delay)
    4. Requires special "trigger" mechanism to start assistant response
    5. Uses AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION in system instruction
    6. Connection timeout after 8 minutes - handled via automatic reconnection

    This pipeline creates its own LLM service (requires_service=False).
    """

    requires_service = False  # We create our own LLM

    def __init__(self, benchmark):
        """Initialize the pipeline.

        Args:
            benchmark: A BenchmarkConfig instance with turns, tools, and system instruction.
        """
        import os

        self.benchmark = benchmark
        self.turns = benchmark.turns
        self.turn_idx = 0
        self.done = False
        self.recorder = None
        self.task = None
        self.context = None
        self.llm = None
        self.model_name = None
        self._turn_indices = None

        # Track tool calls to detect duplicates within a turn
        self._seen_tool_calls: set = set()
        # Track tool_call_ids that are duplicates (for filtering in ToolCallRecorder)
        self._duplicate_tool_call_ids: set = set()

        # Nova Sonic specific
        self.paced_input = None
        self.turn_detector = None
        self.context_aggregator = None

        # AWS credentials (needed for LLM creation)
        self._aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self._aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self._aws_session_token = os.getenv("AWS_SESSION_TOKEN")
        self._aws_region = os.getenv("AWS_REGION", "us-east-1")

    @property
    def effective_turns(self):
        """Get the turns to run (filtered by turn_indices if set)."""
        if self._turn_indices is not None:
            return [self.turns[i] for i in self._turn_indices if i < len(self.turns)]
        return self.turns

    def _get_actual_turn_index(self, effective_index: int) -> int:
        """Convert effective turn index to actual turn index."""
        if self._turn_indices is not None:
            return self._turn_indices[effective_index]
        return effective_index

    def _get_current_turn(self) -> dict:
        """Get the current turn data."""
        return self.effective_turns[self.turn_idx]

    def _get_audio_path_for_turn(self, turn_index: int):
        """Get the audio file path for a turn.

        Prefers benchmark.get_audio_path() if available, falls back to
        the turn's audio_file field.

        Args:
            turn_index: The effective turn index (index into effective_turns).

        Returns:
            Path to audio file as string, or None if not available.
        """
        from pathlib import Path

        # Try benchmark's get_audio_path method first (uses audio_dir)
        if hasattr(self.benchmark, "get_audio_path"):
            actual_index = self._get_actual_turn_index(turn_index)
            path = self.benchmark.get_audio_path(actual_index)
            if path and path.exists():
                return str(path)

        # Fall back to turn's audio_file field
        turn = self.effective_turns[turn_index]
        return turn.get("audio_file")

    async def run(
        self,
        recorder,
        model: str,
        service_class=None,
        turn_indices=None,
    ) -> None:
        """Run the complete benchmark.

        Args:
            recorder: TranscriptRecorder for saving results.
            model: Model name/identifier.
            service_class: Ignored for Nova Sonic (we create our own LLM).
            turn_indices: Optional list of turn indices to run (for debugging).
        """
        import os
        import soundfile as sf
        from pathlib import Path

        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
        from pipecat.transports.base_transport import TransportParams

        from multi_turn_eval.processors.tool_call_recorder import ToolCallRecorder
        from multi_turn_eval.transports.paced_input import PacedInputTransport

        self.recorder = recorder
        self.model_name = model
        self._turn_indices = turn_indices

        # Validate AWS credentials
        if not (self._aws_access_key_id and self._aws_secret_access_key):
            raise EnvironmentError(
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required for Nova Sonic"
            )

        # Get system instruction and tools from benchmark
        system_instruction = getattr(self.benchmark, "system_instruction", "")
        tools = getattr(self.benchmark, "tools_schema", None)

        # Nova Sonic requires the trigger instruction appended to system instruction
        from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService

        nova_sonic_system_instruction = (
            f"{system_instruction} "
            f"{AWSNovaSonicLLMService.AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION}"
        )
        logger.info(f"Using full system instruction ({len(nova_sonic_system_instruction)} chars)")

        # Create Nova Sonic LLM service
        self.llm = NovaSonicLLMServiceWithCompletionSignal(
            secret_access_key=self._aws_secret_access_key,
            access_key_id=self._aws_access_key_id,
            session_token=self._aws_session_token,
            region=self._aws_region,
            model=model if ":" in model else "amazon.nova-sonic-v1:0",
            voice_id="tiffany",
            system_instruction=nova_sonic_system_instruction,
            tools=tools,
            endpointing_sensitivity="HIGH",  # Quick cutoff for faster responses
        )

        # Register function handler
        from pipecat.services.llm_service import FunctionCallParams

        async def function_catchall(params: FunctionCallParams):
            # Create a key for duplicate detection (function_name + args)
            call_key = (params.function_name, str(params.arguments or {}))

            # Check for duplicate tool call
            if call_key in self._seen_tool_calls:
                logger.warning(
                    f"Skipping duplicate tool call: {params.function_name} "
                    f"(tool_call_id={getattr(params, 'tool_call_id', 'unknown')})"
                )
                await params.result_callback({"status": "duplicate_skipped"})
                return

            # Track this call
            self._seen_tool_calls.add(call_key)

            logger.info(f"Function call: {params}")
            result = {"status": "success"}
            await params.result_callback(result)

        self.llm.register_function(None, function_catchall)

        # Create context - Nova Sonic only accepts SPEECH input, so no user message
        messages = [
            {"role": "system", "content": system_instruction},
        ]
        self.context = LLMContext(messages, tools=tools)
        self.context_aggregator = LLMContextAggregatorPair(self.context)

        # Metrics handler
        def handle_metrics(frame: MetricsFrame):
            from pipecat.metrics.metrics import LLMUsageMetricsData, TTFBMetricsData

            for md in frame.data:
                if isinstance(md, LLMUsageMetricsData):
                    self.recorder.record_usage_metrics(md.value, getattr(md, "model", None))
                elif isinstance(md, TTFBMetricsData):
                    self.recorder.record_ttfb(md.value)

        # End of turn callback
        async def end_of_turn(assistant_text: str):
            if self.done:
                logger.info("end_of_turn called but already done")
                return

            # Record this turn
            self.recorder.write_turn(
                user_text=self._get_current_turn().get("input", ""),
                assistant_text=assistant_text,
            )

            # Reset reconnect counter on successful turn completion
            self.llm.reset_reconnect_counter()

            self.turn_idx += 1

            # Reset tool call tracking for the new turn
            self._seen_tool_calls.clear()

            if self.turn_idx < len(self.effective_turns):
                actual_idx = self._get_actual_turn_index(self.turn_idx)
                self.recorder.start_turn(actual_idx)
                logger.info(f"Starting turn {self.turn_idx}: {self._get_current_turn()['input'][:50]}...")
                await self._queue_next_turn()
            else:
                logger.info("Conversation complete!")
                self.recorder.write_summary()
                self.done = True
                await self.task.cancel()

        # Create turn detector
        self.turn_detector = NovaSonicTurnEndDetector(
            end_of_turn_callback=end_of_turn,
            text_timeout_sec=5.0,
            post_completion_timeout_sec=2.0,
            response_timeout_sec=60.0,
            metrics_callback=handle_metrics,
        )

        # Create paced input transport (Nova Sonic requires 16kHz)
        input_params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_in_channels=1,
            audio_in_passthrough=True,
        )
        self.paced_input = PacedInputTransport(
            input_params,
            pre_roll_ms=100,
            continuous_silence=True,
            wait_for_ready=True,  # Wait for LLM to be ready before sending audio
        )

        # Track interrupted turn state for reconnection handling
        self._interrupted_turn_text = ""
        self._was_responding_at_disconnect = False

        # Set up reconnection callbacks
        def on_reconnecting():
            logger.info("Reconnection starting: pausing audio input and resetting turn detector")
            self.paced_input.pause()

            # Capture accumulated text and response state BEFORE reset
            # We need this to handle interrupted turns after reconnection
            self._interrupted_turn_text = self.turn_detector._response_text or ""
            self._was_responding_at_disconnect = self.turn_detector._response_active

            if self._was_responding_at_disconnect:
                logger.warning(
                    f"Turn {self.turn_idx} interrupted mid-response. "
                    f"Captured {len(self._interrupted_turn_text)} chars before reset."
                )

            self.turn_detector.reset_for_reconnection()

        def on_reconnected():
            logger.info("Reconnection complete: waiting 2s before resuming audio")
            import threading
            import asyncio

            def delayed_resume():
                import time

                time.sleep(2.0)
                logger.info("Delayed audio resume: signaling ready now")
                self.paced_input.signal_ready()

                # If we were mid-response when disconnected, we need to:
                # 1. Complete the interrupted turn (with whatever text was collected)
                # 2. This will trigger _queue_next_turn() to queue the next turn's audio
                if self._was_responding_at_disconnect:
                    logger.warning(
                        f"Handling interrupted turn {self.turn_idx} after reconnection. "
                        f"Text captured: {len(self._interrupted_turn_text)} chars"
                    )
                    time.sleep(0.5)  # Small delay to let signal_ready settle

                    # Bridge to async: run end_of_turn in a new event loop
                    text = self._interrupted_turn_text or "[Turn interrupted by 8-minute reconnection]"

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(end_of_turn(text))
                        logger.info(f"Successfully advanced to turn {self.turn_idx} after interrupted turn")
                    except Exception as e:
                        logger.error(f"Error handling interrupted turn: {e}")
                    finally:
                        loop.close()

                    # Reset state
                    self._was_responding_at_disconnect = False
                    self._interrupted_turn_text = ""

            threading.Thread(target=delayed_resume, daemon=True).start()

        def on_retriggered():
            logger.info("Assistant response re-triggered after reconnection")
            self.turn_detector.signal_trigger_sent()

        async def on_max_reconnects_exceeded():
            logger.error("Max reconnect attempts exceeded - terminating pipeline")
            self.done = True
            self.recorder.write_summary()
            await self.task.cancel()

        self.llm._on_reconnecting = on_reconnecting
        self.llm._on_reconnected = on_reconnected
        self.llm._on_retriggered = on_retriggered
        self.llm._on_max_reconnects_exceeded = on_max_reconnects_exceeded

        # Recorder accessor for ToolCallRecorder
        def recorder_accessor():
            return self.recorder

        def duplicate_ids_accessor():
            return self._duplicate_tool_call_ids

        # Build pipeline
        pipeline = Pipeline(
            [
                self.paced_input,
                self.context_aggregator.user(),
                self.llm,
                ToolCallRecorder(recorder_accessor, duplicate_ids_accessor),
                self.turn_detector,
                self.context_aggregator.assistant(),
            ]
        )

        self.task = PipelineTask(
            pipeline,
            idle_timeout_secs=60,  # Longer timeout for Nova Sonic's delayed responses
            idle_timeout_frames=(TTSAudioRawFrame, TTSTextFrame, InputAudioRawFrame, MetricsFrame),
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        # Initialize first turn
        actual_first_idx = self._get_actual_turn_index(0)
        self.recorder.start_turn(actual_first_idx)

        # Queue first turn
        asyncio.create_task(self._queue_first_turn())

        # Run pipeline
        runner = PipelineRunner(handle_sigint=True)
        await runner.run(self.task)

    async def _queue_first_turn(self, delay: float = 1.0):
        """Queue the first turn - send user question as AUDIO, then trigger."""
        import soundfile as sf
        from pathlib import Path

        from pipecat.frames.frames import LLMRunFrame

        await asyncio.sleep(delay)

        # Queue LLMRunFrame to establish connection
        logger.info("Queuing LLMRunFrame to establish connection...")
        await self.task.queue_frames([LLMRunFrame()])

        # Wait for connection to establish
        await asyncio.sleep(1.0)

        # Signal LLM ready to receive audio
        logger.info("Signaling LLM ready for audio...")
        self.paced_input.signal_ready()

        # Queue user's question as AUDIO
        turn = self._get_current_turn()
        audio_path = self._get_audio_path_for_turn(self.turn_idx)
        if audio_path:
            # Calculate audio duration
            data, sr = sf.read(audio_path, dtype="int16")
            audio_duration_sec = len(data) / sr
            logger.info(f"Audio duration: {audio_duration_sec:.2f}s")

            self.paced_input.enqueue_wav_file(audio_path)
            logger.info(f"Queued user question audio: {audio_path}")

            # Signal trigger as soon as we start sending audio.
            # This tells the turn detector to start accepting text.
            # We don't send audio until previous turn ended, so we can safely
            # clear buffers and accept any incoming text from this point.
            self.turn_detector.signal_trigger_sent()

            # Wait for audio to finish streaming (plus small buffer)
            wait_time = audio_duration_sec + 0.5
            logger.info(f"Waiting {wait_time:.2f}s for audio to finish streaming...")
            await asyncio.sleep(wait_time)

            # Start TTFB timing now that user audio is complete
            self.recorder.reset_ttfb()
            await self.llm.start_ttfb_for_user_audio_complete()

            # Trigger assistant response
            # Nova Sonic v1 needs explicit audio trigger, Nova 2 Sonic auto-triggers via VAD
            logger.info("Triggering assistant response after user audio...")
            if self.llm._is_assistant_response_trigger_needed():
                await self.llm.trigger_assistant_response()
            else:
                # Nova 2 Sonic - send LLMRunFrame to trigger context push
                logger.info("Using LLMRunFrame for Nova 2 Sonic")
                await self.task.queue_frames([LLMRunFrame()])
            logger.info("Triggered assistant response")
        else:
            logger.error("No audio file for first turn - Nova Sonic requires audio input!")
            await self.task.cancel()

    async def _queue_next_turn(self):
        """Queue audio for the next turn."""
        import soundfile as sf
        from pathlib import Path

        from pipecat.frames.frames import LLMMessagesAppendFrame

        turn = self._get_current_turn()
        audio_path = self._get_audio_path_for_turn(self.turn_idx)

        if audio_path:
            try:
                # Wait before starting next turn to let Nova Sonic settle
                logger.info("Waiting 5s before starting next turn...")
                await asyncio.sleep(5.0)

                # Calculate audio duration
                data, sr = sf.read(audio_path, dtype="int16")
                audio_duration_sec = len(data) / sr
                logger.info(f"Audio duration for turn {self.turn_idx}: {audio_duration_sec:.2f}s")

                self.paced_input.enqueue_wav_file(audio_path)
                logger.info(f"Queued audio for turn {self.turn_idx}")

                # Signal trigger as soon as we start sending audio.
                # This tells the turn detector to start accepting text.
                self.turn_detector.signal_trigger_sent()

                # Wait for audio to finish streaming
                wait_time = audio_duration_sec + 0.5
                logger.info(f"Waiting {wait_time:.2f}s for audio to finish streaming...")
                await asyncio.sleep(wait_time)

                # Start TTFB timing
                self.recorder.reset_ttfb()
                await self.llm.start_ttfb_for_user_audio_complete()

                # Trigger assistant response
                # Nova Sonic v1 needs explicit audio trigger, Nova 2 Sonic auto-triggers via VAD
                if self.llm._is_assistant_response_trigger_needed():
                    await self.llm.trigger_assistant_response()
                else:
                    # Nova 2 Sonic - send LLMRunFrame to trigger context push
                    logger.info("Using LLMRunFrame for Nova 2 Sonic")
                    await self.task.queue_frames([LLMRunFrame()])
                logger.info(f"Triggered assistant response for turn {self.turn_idx}")
            except Exception as e:
                logger.exception(f"Failed to queue audio for turn {self.turn_idx}: {e}")
                # Fall back to text
                await self.task.queue_frames(
                    [
                        LLMMessagesAppendFrame(
                            messages=[{"role": "user", "content": turn["input"]}],
                            run_llm=False,
                        )
                    ]
                )
                await asyncio.sleep(0.5)
                # Trigger assistant response (model-specific)
                if self.llm._is_assistant_response_trigger_needed():
                    await self.llm.trigger_assistant_response()
                else:
                    await self.task.queue_frames([LLMRunFrame()])
                self.turn_detector.signal_trigger_sent()
        else:
            # No audio file, use text
            await self.task.queue_frames(
                [
                    LLMMessagesAppendFrame(
                        messages=[{"role": "user", "content": turn["input"]}],
                        run_llm=False,
                    )
                ]
            )
            await asyncio.sleep(0.5)
            # Trigger assistant response (model-specific)
            if self.llm._is_assistant_response_trigger_needed():
                await self.llm.trigger_assistant_response()
            else:
                await self.task.queue_frames([LLMRunFrame()])
            self.turn_detector.signal_trigger_sent()
