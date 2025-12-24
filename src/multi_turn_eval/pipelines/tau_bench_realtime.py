"""TAU-bench realtime pipeline with TTS-to-speech loop.

This pipeline implements TAU-bench evaluation for speech-to-speech models:
1. User simulator generates text response
2. OpenAI TTS converts text to audio
3. Audio is sent to the realtime model (OpenAI Realtime, Ultravox, Gemini Live)
4. Model's audio response is transcribed
5. Transcript is passed back to user simulator
6. Loop continues until task is complete

This enables evaluating speech models on TAU-bench without pre-recorded audio.
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    TranscriptionMessage,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.services.openai.realtime import events as rt_events
from pipecat.services.ultravox.llm import OneShotInputParams
from pipecat.transports.base_transport import TransportParams

from multi_turn_eval.pipelines.base import BasePipeline
from multi_turn_eval.processors.audio_buffer import WallClockAlignedAudioBufferProcessor
from multi_turn_eval.processors.tool_call_recorder import ToolCallRecorder
from multi_turn_eval.processors.tts_transcript import (
    TTSStoppedAssistantTranscriptProcessor,
)
from multi_turn_eval.transports.null_audio_output import NullAudioOutputTransport
from multi_turn_eval.transports.paced_input import PacedInputTransport
from multi_turn_eval.tts.openai_tts import OpenAITTS
from multi_turn_eval.user_simulator import LLMUserSimulator, UserSimulatorConfig
from multi_turn_eval.user_simulator.llm_user import STOP_SIGNAL


class TurnGate(FrameProcessor):
    """Gates turn advancement until bot finishes speaking."""

    def __init__(self, on_turn_ready, audio_drain_delay: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self._on_turn_ready = on_turn_ready
        self._audio_drain_delay = audio_drain_delay
        self._pending_transcript: Optional[str] = None
        self._turn_end_task: Optional[asyncio.Task] = None

    def set_pending_transcript(self, text: str):
        logger.info(f"[TurnGate] Storing pending transcript ({len(text)} chars)")
        self._pending_transcript = text

    def clear_pending(self):
        self._pending_transcript = None
        if self._turn_end_task and not self._turn_end_task.done():
            self._turn_end_task.cancel()
            self._turn_end_task = None

    async def _delayed_turn_end(self, text: str):
        try:
            logger.info(f"[TurnGate] Waiting {self._audio_drain_delay}s for audio to drain...")
            await asyncio.sleep(self._audio_drain_delay)
            logger.info(f"[TurnGate] Triggering turn end with transcript ({len(text)} chars)")
            await self._on_turn_ready(text)
        except asyncio.CancelledError:
            logger.info("[TurnGate] Turn end cancelled")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStoppedSpeakingFrame):
            logger.info("[TurnGate] BotStoppedSpeakingFrame received")
            if self._pending_transcript is not None:
                text = self._pending_transcript
                self._pending_transcript = None
                if self._turn_end_task and not self._turn_end_task.done():
                    self._turn_end_task.cancel()
                self._turn_end_task = asyncio.create_task(self._delayed_turn_end(text))

        await self.push_frame(frame, direction)


class LLMFrameLogger(FrameProcessor):
    """Logs frames and captures TTFB metrics."""

    def __init__(self, recorder_accessor):
        super().__init__()
        self._recorder_accessor = recorder_accessor

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, InputAudioRawFrame):
            logger.debug(f"[LLM→] {frame.__class__.__name__} ({direction})")
        if isinstance(frame, MetricsFrame):
            for md in frame.data:
                if isinstance(md, TTFBMetricsData):
                    recorder = self._recorder_accessor()
                    if recorder:
                        recorder.record_ttfb(md.value)
        await self.push_frame(frame, direction)


class TauBenchRealtimePipeline(BasePipeline):
    """TAU-bench pipeline for speech-to-speech models.

    Implements the full TTS-to-speech evaluation loop:
    1. User simulator generates text response
    2. OpenAI TTS converts text to audio
    3. Audio is sent to the realtime model
    4. Model's audio response is transcribed
    5. Transcript is passed back to user simulator
    6. Loop continues until task is complete or max turns reached
    """

    requires_service = True

    def __init__(
        self,
        benchmark,
        user_config: Optional[UserSimulatorConfig] = None,
        tts_provider: str = "openai",
        tts_voice: str = "nova",
        tts_model: str = "tts-1",
    ):
        """Initialize the TAU-bench realtime pipeline.

        Args:
            benchmark: Benchmark configuration with tasks and tools.
            user_config: Configuration for the user simulator.
            tts_provider: TTS provider to use ("openai" or "elevenlabs")
            tts_voice: Voice to use:
                - OpenAI: "nova", "alloy", "echo", "fable", "onyx", "shimmer"
                - ElevenLabs: "rachel", "domi", "bella", "antoni", "josh", etc.
            tts_model: Model to use:
                - OpenAI: "tts-1" or "tts-1-hd"
                - ElevenLabs: "eleven_turbo_v2_5" (fast) or "eleven_multilingual_v2"
        """
        super().__init__(benchmark)
        self.user_config = user_config or UserSimulatorConfig()
        # Enable TTS mode for speech-friendly responses (e.g., "brian jones 3256" instead of "brian_jones_3256")
        self.user_config.tts_mode = True
        self.user_simulator: Optional[LLMUserSimulator] = None
        
        # Initialize TTS based on provider
        self.tts_provider = tts_provider.lower()
        if self.tts_provider == "openai":
            self.tts = OpenAITTS(voice=tts_voice, model=tts_model)
        elif self.tts_provider == "elevenlabs":
            from multi_turn_eval.tts.elevenlabs import ElevenLabsTTS, VOICE_IDS
            # Map friendly name to voice ID if needed
            voice_id = VOICE_IDS.get(tts_voice.lower(), tts_voice)
            self.tts = ElevenLabsTTS(voice_id=voice_id, model_id=tts_model)
        else:
            raise ValueError(f"Unsupported TTS provider: {tts_provider}. Use 'openai' or 'elevenlabs'.")

        # Pipeline components
        self.context_aggregator = None
        self.paced_input: Optional[PacedInputTransport] = None
        self.transcript = None
        self.assistant_shim = None
        self.audio_buffer: Optional[WallClockAlignedAudioBufferProcessor] = None
        self.turn_gate: Optional[TurnGate] = None
        self.output_transport: Optional[NullAudioOutputTransport] = None

        # Conversation state
        self.current_task_idx: int = 0
        self.conversation_turns: List[Dict[str, str]] = []
        self.max_conversation_turns: int = 20
        self.conversation_complete: bool = False
        self.all_tool_calls: List[Dict[str, Any]] = []

        # Temp directory for TTS audio files
        self._temp_dir = tempfile.mkdtemp(prefix="tau_bench_tts_")
        self._sample_rate = 24000  # OpenAI TTS outputs at 24kHz

    def _is_gemini_live(self) -> bool:
        if not self.model_name:
            return False
        m = self.model_name.lower()
        return (m.startswith("gemini") or m.startswith("models/gemini")) and (
            "live" in m or "native-audio" in m
        )

    def _is_openai_realtime(self) -> bool:
        if not self.model_name:
            return False
        m = self.model_name.lower()
        return "realtime" in m and m.startswith("gpt")

    def _is_ultravox_realtime(self) -> bool:
        if not self.model_name:
            return False
        m = self.model_name.lower()
        return "ultravox" in m

    def _create_llm(self, service_class: Optional[type], model: str) -> FrameProcessor:
        """Create LLM service with proper configuration for realtime models."""
        if service_class is None:
            raise ValueError("--service is required for this pipeline")

        class_name = service_class.__name__
        system_instruction = getattr(self.benchmark, "system_instruction", "")
        tools = getattr(self.benchmark, "tools_schema", None)

        if "OpenAIRealtime" in class_name:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY environment variable is required")

            session_props = rt_events.SessionProperties(
                instructions=system_instruction,
                tools=tools,
            )
            return service_class(
                api_key=api_key,
                model=model,
                system_instruction=system_instruction,
                session_properties=session_props,
            )
        elif "UltravoxRealtime" in class_name:
            api_key = os.getenv("ULTRAVOX_API_KEY")
            if not api_key:
                raise EnvironmentError("ULTRAVOX_API_KEY environment variable is required")

            params = OneShotInputParams(
                api_key=api_key,
                system_prompt=system_instruction,
                temperature=1.0,
                model=model,
            )
            return service_class(
                params=params,
                one_shot_selected_tools=tools,
            )
        else:
            # Gemini Live and others
            return super()._create_llm(service_class, model)

    def _setup_context(self) -> None:
        """Create LLMContext with system prompt and tools."""
        system_instruction = getattr(self.benchmark, "system_instruction", "")
        tools = getattr(self.benchmark, "tools_schema", None)
        messages = [{"role": "system", "content": system_instruction}]
        self.context = LLMContext(messages, tools=tools)
        self.context_aggregator = LLMContextAggregatorPair(self.context)

    def _setup_llm(self) -> None:
        """Configure LLM with tool handlers."""
        self.llm.register_function(None, self._function_catchall_with_tracking)

    async def _function_catchall_with_tracking(self, params) -> None:
        """Handle tool calls and track them for TAU-bench evaluation.
        
        Args:
            params: FunctionCallParams with function_name, arguments, tool_call_id, result_callback
        """
        function_name = params.function_name
        args = params.arguments or {}
        tool_call_id = getattr(params, 'tool_call_id', None)
        
        # Track the tool call
        self.all_tool_calls.append({
            "name": function_name,
            "args": args,
            "tool_call_id": tool_call_id,
        })
        logger.info(f"[TAU-bench Realtime] Tool call: {function_name}({args})")

        # Execute via environment if available (TAU-bench style)
        # Check for get_environment() method (TAU-bench pattern)
        if hasattr(self.benchmark, 'get_environment'):
            try:
                env = self.benchmark.get_environment()
                result_str = env.invoke_tool(function_name, **args)
                
                # Parse result - TAU-bench tools return either:
                # 1. JSON string (success): json.dumps({...})
                # 2. Plain string (like "Transfer successful")
                # 3. Empty string (for "think" tool)
                if not result_str or result_str.strip() == "":
                    # Empty is expected for "think" tool (logs reasoning without action)
                    if function_name != "think":
                        logger.warning(f"[TAU-bench Realtime] Tool {function_name} returned empty result")
                    result = {"status": "success", "message": "Tool executed successfully"}
                else:
                    try:
                        # Try parsing as JSON first
                        result = json.loads(result_str)
                    except json.JSONDecodeError:
                        # If not JSON, treat as plain string response (e.g., "Transfer successful")
                        # Wrap it in a dict so the API accepts it
                        result = {"result": result_str}
                
                # Log result info (handle both dict and list responses)
                if isinstance(result, dict):
                    logger.info(f"[TAU-bench Realtime] Tool result keys: {list(result.keys())}")
                elif isinstance(result, list):
                    logger.info(f"[TAU-bench Realtime] Tool result: list with {len(result)} items")
                else:
                    logger.info(f"[TAU-bench Realtime] Tool result: {type(result).__name__}")
            except Exception as e:
                logger.error(f"[TAU-bench Realtime] Tool execution failed: {e}")
                result = {"status": "error", "message": str(e)}
        # Fallback to get_tool_handler pattern
        elif hasattr(self.benchmark, 'get_tool_handler'):
            try:
                handler = self.benchmark.get_tool_handler()
                result_str = handler(function_name, **args)
                
                # Parse result - same logic as above
                if not result_str or result_str.strip() == "":
                    if function_name != "think":
                        logger.warning(f"[TAU-bench Realtime] Tool {function_name} returned empty result")
                    result = {"status": "success", "message": "Tool executed successfully"}
                else:
                    try:
                        result = json.loads(result_str)
                    except json.JSONDecodeError:
                        result = {"result": result_str}
                
                if isinstance(result, dict):
                    logger.info(f"[TAU-bench Realtime] Tool result keys: {list(result.keys())}")
                elif isinstance(result, list):
                    logger.info(f"[TAU-bench Realtime] Tool result: list with {len(result)} items")
                else:
                    logger.info(f"[TAU-bench Realtime] Tool result: {type(result).__name__}")
            except Exception as e:
                logger.error(f"[TAU-bench Realtime] Tool execution failed: {e}")
                result = {"status": "error", "message": str(e)}
        else:
            logger.warning(f"[TAU-bench Realtime] No environment available for tool {function_name}")
            # Fallback to generic success
            result = {"status": "success"}

        await params.result_callback(result)

        # Check if conversation should end
        if function_name in ("end_session", "transfer_to_human_agents"):
            logger.info(f"[TAU-bench Realtime] Conversation ending tool called: {function_name}")
            self.conversation_complete = True

    def _build_task(self) -> None:
        """Build the pipeline with paced input and transcript processors."""

        def recorder_accessor():
            return self.recorder

        def duplicate_ids_accessor():
            return self._duplicate_tool_call_ids

        # Create paced input transport at 24kHz (OpenAI TTS sample rate)
        input_params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=self._sample_rate,
            audio_in_channels=1,
            audio_in_passthrough=True,
        )
        self.paced_input = PacedInputTransport(
            input_params,
            pre_roll_ms=100,
            continuous_silence=True,
        )

        # Create transcript processors
        self.transcript = TranscriptProcessor()
        self.assistant_shim = TTSStoppedAssistantTranscriptProcessor()

        # Create audio buffer for recording
        self.audio_buffer = WallClockAlignedAudioBufferProcessor(
            sample_rate=self._sample_rate,
            num_channels=2,
        )

        # Register event handler to save audio when recording stops
        @self.audio_buffer.event_handler("on_track_audio_data")
        async def on_track_audio_data(
            processor, user_audio: bytes, bot_audio: bytes, sample_rate: int, num_channels: int
        ):
            """Save conversation audio with user and bot on separate channels."""
            logger.info(
                f"[AudioRecording] on_track_audio_data: "
                f"user={len(user_audio)} bytes, bot={len(bot_audio)} bytes"
            )

            if not self.recorder or not hasattr(self.recorder, "run_dir"):
                logger.error("[AudioRecording] Cannot save audio: no recorder or run_dir")
                return

            # Convert to numpy for processing
            user_np = np.frombuffer(user_audio, dtype=np.int16)
            bot_np = np.frombuffer(bot_audio, dtype=np.int16)

            # Pad shorter track to match longer
            max_len = max(len(user_np), len(bot_np))
            if len(user_np) < max_len:
                user_np = np.concatenate([user_np, np.zeros(max_len - len(user_np), dtype=np.int16)])
            if len(bot_np) < max_len:
                bot_np = np.concatenate([bot_np, np.zeros(max_len - len(bot_np), dtype=np.int16)])

            # Interleave for stereo: user=left, bot=right
            stereo = np.zeros(max_len * 2, dtype=np.int16)
            stereo[0::2] = user_np
            stereo[1::2] = bot_np

            output_path = self.recorder.run_dir / f"conversation_task_{self.current_task_idx}.wav"
            logger.info(f"[AudioRecording] Saving audio to {output_path}")

            try:
                with wave.open(str(output_path), "wb") as wf:
                    wf.setnchannels(2)  # Stereo
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(sample_rate)
                    wf.writeframes(stereo.tobytes())

                duration_secs = max_len / sample_rate
                logger.info(f"[AudioRecording] Saved {output_path} ({duration_secs:.1f}s)")
            except Exception as e:
                logger.exception(f"[AudioRecording] Failed to save audio: {e}")

        # Register transcript handler
        @self.assistant_shim.event_handler("on_transcript_update")
        async def on_transcript_update(processor, frame):
            for msg in frame.messages:
                if isinstance(msg, TranscriptionMessage) and getattr(msg, "role", None) == "assistant":
                    logger.info(f"[TAU-bench Realtime] Agent transcript: {msg.content[:100]}...")
                    # Store the transcript for the turn gate
                    self.turn_gate.set_pending_transcript(msg.content)

        # Create TurnGate to coordinate turn transitions
        self.turn_gate = TurnGate(on_turn_ready=self._on_agent_response)

        # Create null output transport
        import pipecat.transports.base_output as base_output_module
        base_output_module.BOT_VAD_STOP_SECS = 2.0

        self.output_transport = NullAudioOutputTransport(
            TransportParams(
                audio_out_enabled=True,
                audio_out_sample_rate=self._sample_rate,
            )
        )

        llm_logger = LLMFrameLogger(recorder_accessor)

        pipeline = Pipeline(
            [
                self.paced_input,
                self.context_aggregator.user(),
                self.transcript.user(),
                self.llm,
                llm_logger,
                ToolCallRecorder(recorder_accessor, duplicate_ids_accessor),
                self.assistant_shim,
                self.turn_gate,
                self.context_aggregator.assistant(),
                self.output_transport,
                self.audio_buffer,
            ]
        )

        self.task = PipelineTask(
            pipeline,
            idle_timeout_secs=60,  # Longer timeout for multi-turn
            idle_timeout_frames=(InputAudioRawFrame, OutputAudioRawFrame, MetricsFrame),
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

    async def _text_to_audio_bytes(self, text: str) -> bytes:
        """Convert text to audio using OpenAI TTS.

        Args:
            text: The text to convert to speech.

        Returns:
            Raw PCM audio bytes at 24kHz mono 16-bit.
        """
        # Generate unique filename
        audio_path = Path(self._temp_dir) / f"user_turn_{time.time():.0f}.wav"

        try:
            await self.tts.generate(text, audio_path)

            # Read the audio file and return raw bytes
            data, sr = sf.read(str(audio_path), dtype="int16")

            # Ensure it's at the expected sample rate
            if sr != self._sample_rate:
                logger.warning(f"TTS returned {sr}Hz audio, expected {self._sample_rate}Hz")

            # Convert to bytes
            audio_bytes = data.tobytes()
            logger.info(f"[TTS] Generated {len(audio_bytes)} bytes of audio for: {text[:50]}...")
            return audio_bytes

        except Exception as e:
            logger.error(f"[TTS] Failed to generate audio: {e}")
            # Return silence as fallback
            silence_duration = 1.0  # 1 second of silence
            silence_samples = int(self._sample_rate * silence_duration)
            return bytes(silence_samples * 2)  # 16-bit = 2 bytes per sample

    async def _send_user_audio(self, text: str) -> None:
        """Convert user text to audio and send to the realtime model.

        Args:
            text: The user's message text.
        """
        audio_bytes = await self._text_to_audio_bytes(text)

        # Send audio to the paced input transport
        self.paced_input.enqueue_bytes(
            audio_bytes,
            num_channels=1,
            sample_rate=self._sample_rate,
        )

        logger.info(f"[TAU-bench Realtime] Sent user audio ({len(audio_bytes)} bytes)")

    async def _on_agent_response(self, transcript: str) -> None:
        """Called when the agent finishes responding.

        Args:
            transcript: The agent's response text (from audio transcription).
        """
        logger.info(f"[TAU-bench Realtime] Agent response: {transcript[:100]}...")

        # Store in conversation history
        self.conversation_turns.append({"role": "assistant", "content": transcript})

        # Check if conversation should end (from tool calls)
        if self.conversation_complete:
            logger.info("[TAU-bench Realtime] Conversation marked complete")
            await self._finish_current_task()
            return

        # Check turn limit
        if len(self.conversation_turns) >= self.max_conversation_turns * 2:
            logger.warning("[TAU-bench Realtime] Max turns reached")
            self.conversation_complete = True
            await self._finish_current_task()
            return

        # Get user simulator response
        user_response = self.user_simulator.step(transcript)

        # Check if user is done
        if self.user_simulator.is_done(user_response):
            clean_response = self.user_simulator.get_clean_response(user_response)
            if clean_response:  # Only add and send if there's actual content
                self.conversation_turns.append({"role": "user", "content": clean_response})
                logger.info(f"[TAU-bench Realtime] User (final): {clean_response[:100]}...")
                
                # Send final user message as audio and wait for agent response
                await self._send_user_audio(clean_response)
                await asyncio.sleep(3.0)
            else:
                logger.info("[TAU-bench Realtime] User sent STOP signal with no content, ending task")
            
            self.conversation_complete = True
            await self._finish_current_task()
        else:
            self.conversation_turns.append({"role": "user", "content": user_response})
            logger.info(f"[TAU-bench Realtime] User: {user_response[:100]}...")

            # Send user response as audio (this will trigger next agent response)
            await self._send_user_audio(user_response)

    async def _finish_current_task(self) -> None:
        """Record task results and signal completion."""
        # Record the task result
        instruction = self.effective_turns[self.current_task_idx].get("instruction", "")
        final_response = ""
        if self.conversation_turns:
            # Find last assistant message
            for turn in reversed(self.conversation_turns):
                if turn["role"] == "assistant":
                    final_response = turn["content"]
                    break

        self.recorder.write_turn(
            user_text=instruction,
            assistant_text=final_response,
            extra_data={
                "conversation_turns": len(self.conversation_turns),
                "tool_calls": self.all_tool_calls,
                "full_conversation": self.conversation_turns,
            }
        )

        logger.info(f"[TAU-bench Realtime] Task {self.current_task_idx} complete "
                    f"({len(self.conversation_turns)} turns, {len(self.all_tool_calls)} tool calls)")

        # Mark as done to exit the run loop
        self.done = True

    async def _queue_first_turn(self) -> None:
        """Queue the initial user message for the first task."""
        # Start audio recording
        logger.info("[TAU-bench Realtime] Starting audio recording")
        await self.audio_buffer.start_recording()

        if self.output_transport is not None:
            self.output_transport.reset_recording_baseline(
                recording_sample_rate=self.audio_buffer._init_sample_rate
            )

        # For Gemini Live, push context frame
        if self._is_gemini_live():
            await self.task.queue_frames([LLMContextFrame(self.context)])

        # Give the pipeline a moment to start
        await asyncio.sleep(1.0)

        # Get task and initialize user simulator
        task = self.effective_turns[self.current_task_idx]
        instruction = task.get("instruction", "")
        
        # Use instruction as-is (it already contains user_id in TAU-bench tasks)
        # Reset user simulator and get initial message
        initial_message = self.user_simulator.reset(instruction)
        self.conversation_turns.append({"role": "user", "content": initial_message})

        logger.info(f"[TAU-bench Realtime] Initial message: {initial_message[:100]}...")

        # Send initial user message as audio
        await self._send_user_audio(initial_message)

    async def _queue_next_turn(self) -> None:
        """Not used for TAU-bench - turns are driven by user simulator."""
        pass

    def _reset_context_for_task(self) -> None:
        """Reset the LLM context for a new task."""
        system_instruction = getattr(self.benchmark, "system_instruction", "")
        tools = getattr(self.benchmark, "tools_schema", None)
        messages = [{"role": "system", "content": system_instruction}]
        self.context = LLMContext(messages, tools=tools)
        self.context_aggregator = LLMContextAggregatorPair(self.context)

    async def run(
        self,
        recorder,
        model: str,
        service_class: Optional[type] = None,
        service_name: Optional[str] = None,
        turn_indices: Optional[List[int]] = None,
    ) -> None:
        """Run TAU-bench evaluation with user simulator and TTS.

        Each "turn" in the benchmark is a complete task that requires
        multiple conversation turns to complete.
        """
        self.recorder = recorder
        self.model_name = model
        self.service_name = service_name
        self._turn_indices = turn_indices

        # Initialize user simulator
        self.user_simulator = LLMUserSimulator(self.user_config)

        try:
            # Run each task
            for task_idx, task in enumerate(self.effective_turns):
                if turn_indices and task_idx not in turn_indices:
                    continue

                logger.info(f"[TAU-bench Realtime] Starting task {task_idx}")
                self.current_task_idx = task_idx
                self.conversation_turns = []
                self.all_tool_calls = []
                self.conversation_complete = False
                self.done = False

                # Reset environment for this task
                if hasattr(self.benchmark, 'reset_environment'):
                    self.benchmark.reset_environment()

                # Create LLM service (fresh for each task)
                self.llm = self._create_llm(service_class, model)

                # Setup
                self._setup_context()
                self._setup_llm()
                self._build_task()

                # Start recorder turn
                self.recorder.start_turn(task_idx)

                # Run the pipeline
                await self._queue_first_turn()
                runner = PipelineRunner(handle_sigint=False)  # Don't intercept sigint
                await runner.run(self.task)

                # Stop audio recording (this triggers saving)
                await self.audio_buffer.stop_recording()

            # Write summary only if all tasks completed
            self.recorder.write_summary()
            logger.info("[TAU-bench Realtime] Evaluation complete")

        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("[TAU-bench Realtime] Interrupted by user")
            raise
        finally:
            # Cleanup temp files
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir: {e}")
