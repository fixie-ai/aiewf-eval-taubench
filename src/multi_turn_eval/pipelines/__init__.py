"""Pipeline implementations for different LLM service types.

Pipelines handle the full execution of multi-turn benchmarks including:
- Creating and configuring LLM services
- Managing turn flow (queuing turns, detecting end-of-turn)
- Recording transcripts and metrics
- Handling reconnection for long-running sessions

Available pipelines:
- TextPipeline: For text-based LLM services (OpenAI, Anthropic, Google, etc.)
- RealtimePipeline: For speech-to-speech services (OpenAI Realtime, Gemini Live)
- GrokRealtimePipeline: For xAI Grok Voice Agent API
- NovaSonicPipeline: For AWS Nova Sonic speech-to-speech service

Note: Realtime pipelines are imported lazily to avoid dependency issues
when only using text pipelines.
"""

from multi_turn_eval.pipelines.base import BasePipeline
from multi_turn_eval.pipelines.text import TextPipeline

# Lazy imports for realtime pipelines to avoid import errors when
# audio_buffer or other realtime-specific dependencies are missing.
# These are imported on-demand by the CLI's get_pipeline_class function.

__all__ = [
    "BasePipeline",
    "TextPipeline",
]


def __getattr__(name: str):
    """Lazy import for realtime pipeline classes."""
    if name in ("RealtimePipeline", "GeminiLiveLLMServiceWithReconnection"):
        from multi_turn_eval.pipelines.realtime import (
            RealtimePipeline,
            GeminiLiveLLMServiceWithReconnection,
        )
        if name == "RealtimePipeline":
            return RealtimePipeline
        return GeminiLiveLLMServiceWithReconnection
    
    if name in ("GrokRealtimePipeline", "XAIRealtimeLLMService"):
        from multi_turn_eval.pipelines.grok_realtime import (
            GrokRealtimePipeline,
            XAIRealtimeLLMService,
        )
        if name == "GrokRealtimePipeline":
            return GrokRealtimePipeline
        return XAIRealtimeLLMService
    
    if name in ("NovaSonicPipeline", "NovaSonicLLMServiceWithCompletionSignal", "NovaSonicTurnGate"):
        from multi_turn_eval.pipelines.nova_sonic import (
            NovaSonicPipeline,
            NovaSonicLLMServiceWithCompletionSignal,
            NovaSonicTurnGate,
        )
        if name == "NovaSonicPipeline":
            return NovaSonicPipeline
        if name == "NovaSonicLLMServiceWithCompletionSignal":
            return NovaSonicLLMServiceWithCompletionSignal
        return NovaSonicTurnGate
    
    if name == "TauBenchRealtimePipeline":
        from multi_turn_eval.pipelines.tau_bench_realtime import TauBenchRealtimePipeline
        return TauBenchRealtimePipeline
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
