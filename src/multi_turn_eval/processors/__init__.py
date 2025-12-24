"""Frame processors for multi-turn evaluation pipelines."""

from multi_turn_eval.processors.audio_buffer import WallClockAlignedAudioBufferProcessor
from multi_turn_eval.processors.tool_call_recorder import ToolCallRecorder
from multi_turn_eval.processors.tts_transcript import TTSStoppedAssistantTranscriptProcessor

__all__ = [
    "WallClockAlignedAudioBufferProcessor",
    "ToolCallRecorder",
    "TTSStoppedAssistantTranscriptProcessor",
]
