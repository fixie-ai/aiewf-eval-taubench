"""TAU-bench multi-turn pipeline with user simulator.

This pipeline implements the full TAU-bench evaluation loop:
1. User simulator generates initial message from task instruction
2. Agent responds (may ask clarifying questions)
3. User simulator responds to agent
4. Repeat until task complete or max turns reached
5. Evaluate based on final database state

For speech-to-speech models, user responses are converted to audio via TTS
before being sent to the model.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from multi_turn_eval.pipelines.text import TextPipeline
from multi_turn_eval.user_simulator import LLMUserSimulator, UserSimulatorConfig
from multi_turn_eval.user_simulator.llm_user import STOP_SIGNAL


class TauBenchTextPipeline(TextPipeline):
    """TAU-bench pipeline for text-based models with user simulator.
    
    Extends TextPipeline to support multi-turn conversations where a
    user simulator (LLM) plays the role of the customer.
    """
    
    def __init__(self, benchmark, user_config: Optional[UserSimulatorConfig] = None):
        """Initialize the TAU-bench pipeline.
        
        Args:
            benchmark: Benchmark configuration with tasks and tools.
            user_config: Configuration for the user simulator.
        """
        super().__init__(benchmark)
        self.user_config = user_config or UserSimulatorConfig()
        self.user_simulator: Optional[LLMUserSimulator] = None
        self.current_task: Optional[Dict[str, Any]] = None
        self.conversation_turns: List[Dict[str, str]] = []
        self.max_conversation_turns = 20
    
    def _setup_context(self) -> None:
        """Override to do nothing; context is managed per task in _run_task_conversation."""
        # TAU-bench uses multi-turn conversations where context is reset for each task
        pass
    
    def _build_task(self) -> None:
        """Override to do nothing; we don't use Pipecat task for text-only multi-turn."""
        # TAU-bench text pipeline uses direct OpenAI API calls instead of Pipecat pipeline
        pass
    
    async def run(
        self,
        recorder,
        model: str,
        service_class: Optional[type] = None,
        service_name: Optional[str] = None,
        turn_indices: Optional[List[int]] = None,
    ) -> None:
        """Run TAU-bench evaluation with user simulator.
        
        Each "turn" in the benchmark is actually a complete task that may
        require multiple conversation turns to complete.
        """
        self.recorder = recorder
        self.model_name = model
        self.service_name = service_name
        self._turn_indices = turn_indices
        
        # Initialize user simulator
        self.user_simulator = LLMUserSimulator(self.user_config)
        
        # Run each task
        # Note: self.effective_turns is already filtered by turn_indices (see BasePipeline)
        # But we need the original task indices for recording, so we track them
        logger.info(f"[TAU-bench] Running {len(self.effective_turns)} tasks")
        for enum_idx, task in enumerate(self.effective_turns):
            # Get the actual task index from the benchmark
            if turn_indices:
                task_idx = turn_indices[enum_idx]
            else:
                task_idx = enum_idx
            
            logger.info(f"[TAU-bench] Starting task {task_idx}")
            self.current_task = task
            
            # Reset environment for this task
            if hasattr(self.benchmark, 'reset_environment'):
                self.benchmark.reset_environment()
            
            # Get user instruction from task
            instruction = task.get("instruction", "")
            user_id = task.get("user_id", "")
            
            # Use instruction as-is (it already contains user_id in TAU-bench tasks)
            full_instruction = instruction
            
            # Run multi-turn conversation for this task
            await self._run_task_conversation(task_idx, full_instruction)
        
        # Write summary
        self.recorder.write_summary()
        logger.info("[TAU-bench] Evaluation complete")
    
    async def _run_task_conversation(self, task_idx: int, instruction: str) -> None:
        """Run a multi-turn conversation for a single task.
        
        Args:
            task_idx: Index of the current task.
            instruction: The user's goal/instruction.
        """
        self.conversation_turns = []
        self.recorder.start_turn(task_idx)
        
        # Reset context for new conversation
        self._reset_context_for_task()
        
        # Get initial user message
        initial_message = self.user_simulator.reset(instruction)
        self.conversation_turns.append({"role": "user", "content": initial_message})
        
        # Add to LLM context
        self.context.add_messages([{"role": "user", "content": initial_message}])
        
        # Conversation loop
        conversation_complete = False
        turn_count = 0
        all_tool_calls = []
        final_agent_response = ""
        
        while not conversation_complete and turn_count < self.max_conversation_turns:
            turn_count += 1
            logger.info(f"[TAU-bench] Task {task_idx}, conversation turn {turn_count}")
            
            # Get agent response
            agent_response, tool_calls = await self._get_agent_response()
            all_tool_calls.extend(tool_calls)
            final_agent_response = agent_response
            
            self.conversation_turns.append({"role": "assistant", "content": agent_response})
            logger.info(f"[TAU-bench] Agent: {agent_response[:100]}...")
            
            # Check if agent transferred or ended
            if self._agent_ended_conversation(tool_calls):
                logger.info("[TAU-bench] Agent ended conversation (transfer/end_session)")
                conversation_complete = True
                break
            
            # Get user simulator response
            user_response = self.user_simulator.step(agent_response)
            
            # Check if user is done
            if self.user_simulator.is_done(user_response):
                clean_response = self.user_simulator.get_clean_response(user_response)
                if clean_response:  # Only add if there's actual content
                    self.conversation_turns.append({"role": "user", "content": clean_response})
                    logger.info(f"[TAU-bench] User (final): {clean_response[:100]}...")
                else:
                    logger.info("[TAU-bench] User sent STOP signal with no content")
                conversation_complete = True
            else:
                self.conversation_turns.append({"role": "user", "content": user_response})
                self.context.add_messages([{"role": "user", "content": user_response}])
                logger.info(f"[TAU-bench] User: {user_response[:100]}...")
        
        # Record the task result
        self.recorder.write_turn(
            user_text=instruction,  # Original instruction
            assistant_text=final_agent_response,
            extra_data={
                "conversation_turns": len(self.conversation_turns),
                "tool_calls": all_tool_calls,
                "full_conversation": self.conversation_turns,
            }
        )
    
    def _reset_context_for_task(self) -> None:
        """Reset the LLM context for a new task while keeping system prompt."""
        system_instruction = getattr(self.benchmark, "system_instruction", "")
        tools = getattr(self.benchmark, "tools_schema", None)
        
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
        
        messages = [{"role": "system", "content": system_instruction}]
        self.context = LLMContext(messages, tools=tools)
        self.context_aggregator = LLMContextAggregatorPair(self.context)
    
    async def _get_agent_response(self) -> tuple[str, List[Dict[str, Any]]]:
        """Get a response from the agent.
        
        Returns:
            Tuple of (response_text, tool_calls)
        """
        from pipecat.frames.frames import LLMRunFrame, TextFrame, FunctionCallResultFrame
        
        # Collect response
        response_text = ""
        tool_calls = []
        
        # Simple synchronous call for text models
        # This is a simplified version - in production you'd use the full pipeline
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Get tools in OpenAI format from the benchmark tools schema
            tools_schema = getattr(self.benchmark, "tools_schema", None)
            tools_list = None
            if tools_schema:
                # Use standard_tools which now contains schemas derived from original tools
                tools_list = [
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": {
                                "type": "object",
                                "properties": t.properties,
                                "required": t.required or [],
                            }
                        }
                    }
                    for t in tools_schema.standard_tools
                ]
            
            # Build messages
            messages = self.context.get_messages()
            
            # Make API call
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools_list,
                temperature=0.7,
            )
            
            choice = response.choices[0]
            
            # Handle tool calls
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)
                    
                    tool_calls.append({
                        "name": tool_name,
                        "args": tool_args,
                        "tool_call_id": tc.id,
                    })
                    
                    # Execute tool
                    tool_handler = getattr(self.benchmark, 'get_tool_handler', None)
                    if tool_handler:
                        handler = tool_handler()
                        result_str = handler(tool_name, **tool_args)
                        logger.info(f"[TAU-bench] Tool {tool_name} returned: {result_str[:200] if result_str else '(empty)'}")
                        
                        # Parse result - TAU-bench tools return either:
                        # 1. JSON string (success): json.dumps({...})
                        # 2. Plain string (error): "Error: ..."
                        # 3. Empty string (expected for "think" tool)
                        if not result_str or result_str.strip() == "":
                            # Empty is expected for "think" tool (logs reasoning without action)
                            if tool_name != "think":
                                logger.warning(f"[TAU-bench] Tool {tool_name} returned empty result")
                            result = {"status": "success", "message": "Tool executed successfully"}
                        else:
                            try:
                                # Try parsing as JSON first
                                result = json.loads(result_str)
                            except json.JSONDecodeError:
                                # If not JSON, treat as plain string response (likely error message)
                                # Wrap it in a dict so OpenAI API accepts it
                                result = {"result": result_str}
                    else:
                        result = {"status": "success"}
                    
                    # Add tool call and result to context
                    self.context.add_messages([
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tc.function.arguments,
                                }
                            }]
                        },
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result),
                        }
                    ])
                
                # Get follow-up response after tool calls
                messages = self.context.get_messages()
                follow_up = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools_list,
                    temperature=0.7,
                )
                response_text = follow_up.choices[0].message.content or ""
                
                # Handle any additional tool calls
                if follow_up.choices[0].message.tool_calls:
                    # Recursive handling could go here
                    pass
            else:
                response_text = choice.message.content or ""
            
            # Add assistant response to context
            if response_text:
                self.context.add_messages([{"role": "assistant", "content": response_text}])
            
        except Exception as e:
            logger.error(f"[TAU-bench] Error getting agent response: {e}")
            response_text = f"I apologize, I'm having technical difficulties. Error: {e}"
        
        return response_text, tool_calls
    
    def _agent_ended_conversation(self, tool_calls: List[Dict[str, Any]]) -> bool:
        """Check if the agent ended the conversation.
        
        Args:
            tool_calls: List of tool calls made by the agent.
            
        Returns:
            True if agent called end_session or transfer_to_human_agents.
        """
        ending_tools = {"end_session", "transfer_to_human_agents"}
        for tc in tool_calls:
            if tc.get("name") in ending_tools:
                return True
        return False

