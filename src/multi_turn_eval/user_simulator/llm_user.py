"""LLM-based user simulator for TAU-bench style evaluation.

This module implements a user simulator that uses an LLM to roleplay as a
customer interacting with an agent. The simulator:
1. Follows the task instruction (user's goal)
2. Responds to agent questions naturally
3. Provides confirmations when agent asks for them
4. Signals completion when the task is done

Based on the original TAU-bench user simulator design.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


# Completion signal that indicates the user is done
STOP_SIGNAL = "###STOP###"


@dataclass
class UserSimulatorConfig:
    """Configuration for the user simulator."""
    
    # LLM settings
    model: str = "gpt-4o"
    provider: str = "openai"  # openai, anthropic, google
    temperature: float = 0.7
    max_tokens: int = 256
    
    # Behavior settings
    max_turns: int = 20  # Maximum conversation turns before giving up
    include_reasoning: bool = False  # Include chain-of-thought in responses
    tts_mode: bool = False  # Generate TTS-friendly responses (for speech-to-speech models)


# System prompt for the user simulator (aligned with original TAU-bench)
USER_SIMULATOR_SYSTEM_PROMPT = """You are a user interacting with an agent.

Instruction: {instruction}

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '{stop_signal}' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""

# TTS-friendly system prompt (for speech-to-speech models)
USER_SIMULATOR_TTS_PROMPT = """You are a user interacting with an agent via voice conversation. Your responses will be converted to speech using text-to-speech.

Instruction: {instruction}

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '{stop_signal}' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction.

**IMPORTANT - TTS-Friendly Speech Rules:**
- When providing IDs with underscores or special characters (like "brian_jones_3256"), make sure to pronounce each part individually (e.g. "Brian underscore Jones underscore 3256")
- If your name is misspelled by the agent, you should correct it by saying the name with the corrected letters spelled out. Spelling one letter at a time, like "B-R-I-A-N" will allow the TTS engine to clarify the different spelling.
- Speak conversationally as if you're actually on the phone with a customer service agent.
- Numbers should be said naturally: "3256" becomes "three two five six" or "thirty-two fifty-six" depending on context."""


class LLMUserSimulator:
    """LLM-based user simulator for multi-turn evaluation.
    
    Uses an LLM to roleplay as a customer following a specific task instruction.
    The simulator maintains conversation history and generates contextual responses.
    """
    
    def __init__(self, config: Optional[UserSimulatorConfig] = None):
        """Initialize the user simulator.
        
        Args:
            config: Configuration for the simulator. Uses defaults if not provided.
        """
        self.config = config or UserSimulatorConfig()
        self.instruction: str = ""
        self.conversation_history: List[Dict[str, str]] = []
        self.turn_count: int = 0
        self._client = None
    
    def _get_client(self):
        """Get or create the LLM client."""
        if self._client is not None:
            return self._client
        
        if self.config.provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self._client = OpenAI(api_key=api_key)
        elif self.config.provider == "anthropic":
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
        
        return self._client
    
    def reset(self, instruction: str) -> str:
        """Reset the simulator with a new task instruction.
        
        Args:
            instruction: The user's goal/instruction for this task.
            
        Returns:
            The initial user message to start the conversation.
        """
        self.instruction = instruction
        self.turn_count = 0
        
        # Choose prompt based on TTS mode
        prompt_template = USER_SIMULATOR_TTS_PROMPT if self.config.tts_mode else USER_SIMULATOR_SYSTEM_PROMPT
        
        # Original TAU-bench: Agent says "Hi! How can I help you today?" first
        # User responds to this greeting
        agent_greeting = "Hi! How can I help you today?"
        self.conversation_history = [
            {"role": "system", "content": prompt_template.format(
                instruction=instruction,
                stop_signal=STOP_SIGNAL,
            )},
            {"role": "user", "content": agent_greeting}
        ]
        
        # Generate user's response to the greeting
        initial_message = self._generate_response()
        
        logger.info(f"[UserSimulator] Reset with instruction: {instruction[:100]}...")
        logger.info(f"[UserSimulator] Initial message: {initial_message[:100]}...")
        
        return initial_message
    
    
    def step(self, agent_message: str) -> str:
        """Generate user response to an agent message.
        
        Args:
            agent_message: The agent's latest response.
            
        Returns:
            The user's response. Contains STOP_SIGNAL if conversation should end.
        """
        self.turn_count += 1
        
        # Check turn limit
        if self.turn_count >= self.config.max_turns:
            logger.warning(f"[UserSimulator] Max turns ({self.config.max_turns}) reached")
            return f"I think we've been going in circles. Let me call back later. {STOP_SIGNAL}"
        
        # Add agent message to history (as "user" from simulator's perspective)
        self.conversation_history.append({"role": "user", "content": agent_message})
        
        # Generate user response (returns "assistant" from simulator's perspective)
        # _generate_response now adds the response to conversation_history
        response = self._generate_response()
        
        logger.info(f"[UserSimulator] Turn {self.turn_count}: {response[:100]}...")
        
        return response
    
    def _generate_response(self) -> str:
        """Generate a response using the LLM."""
        try:
            client = self._get_client()
            
            if self.config.provider == "openai":
                # conversation_history already includes system prompt
                response = client.chat.completions.create(
                    model=self.config.model,
                    messages=self.conversation_history,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                user_response = response.choices[0].message.content.strip()
                
            elif self.config.provider == "anthropic":
                # Anthropic expects system separate from messages
                system_prompt = self.conversation_history[0]["content"]
                messages = self.conversation_history[1:]  # Skip system message
                
                response = client.messages.create(
                    model=self.config.model,
                    system=system_prompt,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                user_response = response.content[0].text.strip()
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
            
            # Add the response to history
            self.conversation_history.append({"role": "assistant", "content": user_response})
            return user_response
                
        except Exception as e:
            logger.error(f"[UserSimulator] Error generating response: {e}")
            return f"I'm having trouble understanding. Can we try again? {STOP_SIGNAL}"
    
    def is_done(self, response: str) -> bool:
        """Check if the conversation should end.
        
        Args:
            response: The latest user response.
            
        Returns:
            True if STOP_SIGNAL is in the response.
        """
        return STOP_SIGNAL in response
    
    def get_clean_response(self, response: str) -> str:
        """Remove the stop signal from a response.
        
        Args:
            response: Response that may contain STOP_SIGNAL.
            
        Returns:
            Response with STOP_SIGNAL removed.
        """
        return response.replace(STOP_SIGNAL, "").strip()

