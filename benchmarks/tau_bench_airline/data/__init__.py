"""TAU-bench airline environment using original tool logic.

This module provides a stateful environment that uses the original TAU-bench
data loading and tool invocation logic.
"""
import json
import os
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

# Mapping of tool names to their original classes for invocation
# We import here to avoid circular dependency
from ..tools import TOOL_CLASSES

FOLDER_PATH = os.path.dirname(__file__)

def load_data() -> dict[str, Any]:
    """Original TAU-bench data loading logic."""
    with open(os.path.join(FOLDER_PATH, "flights.json")) as f:
        flight_data = json.load(f)
    with open(os.path.join(FOLDER_PATH, "reservations.json")) as f:
        reservation_data = json.load(f)
    with open(os.path.join(FOLDER_PATH, "users.json")) as f:
        user_data = json.load(f)
    return {
        "flights": flight_data,
        "reservations": reservation_data,
        "users": user_data,
    }

def to_hashable(obj: Any) -> Any:
    """Convert an object to a hashable format (stable sort for dicts/lists)."""
    if isinstance(obj, dict):
        return sorted((k, to_hashable(v)) for k, v in obj.items())
    elif isinstance(obj, list):
        return tuple(to_hashable(v) for v in obj)
    return obj

def consistent_hash(obj: Any) -> str:
    """Generate a stable hash for a database state.
    
    Uses SHA256 to match the original TAU-bench implementation.
    """
    content = str(to_hashable(obj))  # Convert to string representation
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

class AirlineEnvironment:
    """Stateful environment using original TAU-bench tool logic."""
    
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """Initialize the environment."""
        # Use our local load_data which now has the original logic
        self.data = data if data else load_data()
        self.initial_data = deepcopy(self.data)
        self.tool_call_history = []
    
    def reset(self) -> None:
        """Reset the environment to initial state."""
        self.data = deepcopy(self.initial_data)
        self.tool_call_history = []
    
    def get_data_snapshot(self) -> Dict[str, Any]:
        """Get a deep copy of the current data state."""
        return deepcopy(self.data)

    def get_state_hash(self) -> str:
        """Get consistent hash of the current database state."""
        return consistent_hash(self.data)
    
    def invoke_tool(self, tool_name: str, **kwargs) -> str:
        """Invoke a tool using the original Tool class logic."""
        if tool_name not in TOOL_CLASSES:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        
        tool_class = TOOL_CLASSES[tool_name]
        
        try:
            # Original TAU-bench tools have a static invoke method:
            # invoke(data: Dict[str, Any], **kwargs) -> str
            result = tool_class.invoke(self.data, **kwargs)
            
            # Track history for evaluation
            self.tool_call_history.append((tool_name, kwargs))
            
            return result
        except Exception as e:
            return json.dumps({"error": str(e)})

__all__ = ["load_data", "AirlineEnvironment", "consistent_hash"]
