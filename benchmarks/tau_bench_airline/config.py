"""Configuration for the TAU-bench airline scenario benchmark.

This configuration aligns with the original TAU-bench format:
https://github.com/sierra-research/tau-bench
"""
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .turns import tasks  # Original TAU-bench uses 'tasks'
from .tools import AirlineToolsSchema
from .data import AirlineEnvironment

# Load wiki.md as system instruction (original TAU-bench format)
WIKI_PATH = Path(__file__).parent / "wiki.md"
system_instruction = WIKI_PATH.read_text()


class BenchmarkConfig:
    """Configuration for the TAU-bench airline benchmark."""

    # Benchmark metadata
    name = "tau_bench_airline"
    description = "TAU-bench airline scenario - tool-agent-user interaction benchmark"
    
    # Data - use 'tasks' from original TAU-bench format
    turns = tasks
    tools_schema = AirlineToolsSchema
    
    # Audio directory for TTS-generated audio (created on demand)
    audio_dir = Path(__file__).parent / "audio"
    
    # System prompt
    system_instruction = system_instruction
    
    # TAU-bench specific: evaluation mode
    evaluation_mode = "action_match"  # Compare actions instead of semantic judgment
    
    # TAU-bench specific: disable duplicate detection
    # TAU-bench tasks may require calling the same tool multiple times
    skip_duplicate_detection = True
    
    # TAU-bench specific: use stateful environment for tool execution
    _environment: Optional[AirlineEnvironment] = None
    
    @classmethod
    def get_environment(cls) -> AirlineEnvironment:
        """Get or create the airline environment for tool execution."""
        if cls._environment is None:
            cls._environment = AirlineEnvironment()
        return cls._environment
    
    @classmethod
    def reset_environment(cls) -> None:
        """Reset the environment to initial state."""
        if cls._environment is not None:
            cls._environment.reset()
        else:
            cls._environment = AirlineEnvironment()
    
    @classmethod
    def get_tool_handler(cls) -> Callable[[str, Dict[str, Any]], str]:
        """Get a function that handles tool calls using the environment.
        
        Returns:
            A callable that takes (tool_name, arguments) and returns the result.
        """
        env = cls.get_environment()
        return env.invoke_tool
    
    @classmethod
    def get_audio_path(cls, turn_index: int) -> Path:
        """Get the audio file path for a specific turn.
        
        For TAU-bench, audio files are generated from the task input using TTS.
        """
        return cls.audio_dir / f"task_{turn_index:03d}.wav"
    
    @classmethod
    def get_task(cls, task_index: int) -> dict:
        """Get a specific task by index."""
        if 0 <= task_index < len(cls.turns):
            return cls.turns[task_index]
        raise IndexError(f"Task index {task_index} out of range (0-{len(cls.turns)-1})")

