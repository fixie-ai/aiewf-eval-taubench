"""TAU-bench specific judge for action-based and state-based evaluation.

This judge matches the original TAU-bench evaluation logic:
1. State-based matching: The final database state must match ground truth.
2. Output matching: For tasks with expected outputs, they must appear in the final response.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from benchmarks.tau_bench_airline.data import AirlineEnvironment, consistent_hash

@dataclass
class TauBenchJudgment:
    """Judgment result for a TAU-bench task."""
    task_id: int
    passed: bool
    reward: float
    state_matched: bool
    outputs_matched: Optional[bool]
    reasoning: str
    gt_hash: str
    actual_hash: str

def calculate_ground_truth_state(task: Dict[str, Any]) -> str:
    """Run expected actions against a clean database to get ground truth hash."""
    env = AirlineEnvironment()
    expected_actions = task.get("actions", task.get("expected_actions", []))
    
    # Original TAU-bench tools: book_reservation, etc.
    for action in expected_actions:
        name = action["name"]
        args = action.get("arguments", {})
        env.invoke_tool(name, **args)
        
    return env.get_state_hash()

def judge_task(
    task: Dict[str, Any],
    actual_calls: List[Dict[str, Any]],
    final_response: str = "",
) -> TauBenchJudgment:
    """Judge a single TAU-bench task using state-based comparison."""
    # 1. Get ground truth hash
    gt_hash = calculate_ground_truth_state(task)
    
    # 2. Get actual final hash from the agent's environment
    # Note: The agent's final state is captured in the transcript or needs to be reconstructed
    # For now, we reconstruct it from the tool calls in the transcript
    env = AirlineEnvironment()
    for call in actual_calls:
        name = call["name"]
        args = call.get("args", call.get("arguments", {}))
        env.invoke_tool(name, **args)
    
    actual_hash = env.get_state_hash()
    state_matched = (actual_hash == gt_hash)
    
    # 3. Check outputs (if required by task)
    expected_outputs = task.get("outputs", [])
    outputs_matched = True
    output_details = []
    
    if expected_outputs:
        for output in expected_outputs:
            found = output.lower() in final_response.lower().replace(",", "")
            if not found:
                outputs_matched = False
                output_details.append(f"Missing output: '{output}'")
    else:
        outputs_matched = None # Not applicable
        
    # 4. Final reward (must match BOTH state and outputs)
    passed = state_matched and (outputs_matched is not False)
    reward = 1.0 if passed else 0.0
    
    # Reasoning
    reasoning_parts = []
    if state_matched:
        reasoning_parts.append("✅ Final database state matched ground truth.")
    else:
        reasoning_parts.append("❌ Database state mismatch.")
        reasoning_parts.append(f"  GT Hash: {gt_hash}")
        reasoning_parts.append(f"  Actual:  {actual_hash}")
        
    if outputs_matched is False:
        reasoning_parts.append(f"❌ Output mismatch: {'; '.join(output_details)}")
    elif outputs_matched is True:
        reasoning_parts.append("✅ All expected outputs found in final response.")
        
    return TauBenchJudgment(
        task_id=task.get("task_id", task.get("annotator", 0)),
        passed=passed,
        reward=reward,
        state_matched=state_matched,
        outputs_matched=outputs_matched,
        reasoning="\n".join(reasoning_parts),
        gt_hash=gt_hash,
        actual_hash=actual_hash
    )

def load_transcript(run_dir: Path) -> List[Dict[str, Any]]:
    """Load transcript.jsonl from run directory."""
    path = run_dir / "transcript.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No transcript.jsonl in {run_dir}")
    
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def judge_run(
    run_dir: Path,
    expected_turns: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Judge a complete TAU-bench run."""
    records = load_transcript(run_dir)
    
    judgments = {}
    total_passed = 0
    total_reward = 0.0
    
    for record in records:
        turn_idx = record.get("turn", 0)
        if turn_idx >= len(expected_turns):
            continue
            
        task = expected_turns[turn_idx]
        tool_calls = record.get("tool_calls", [])
        final_response = record.get("assistant_text", "")
        
        judgment = judge_task(task, tool_calls, final_response)
        
        judgments[turn_idx] = {
            "passed": judgment.passed,
            "reward": judgment.reward,
            "state_matched": judgment.state_matched,
            "outputs_matched": judgment.outputs_matched,
            "reasoning": judgment.reasoning,
            "gt_hash": judgment.gt_hash,
            "actual_hash": judgment.actual_hash,
        }
        
        if judgment.passed:
            total_passed += 1
        total_reward += judgment.reward
    
    return {
        "judgments": judgments,
        "summary": {
            "total_tasks": len(records),
            "passed": total_passed,
            "pass_rate": total_passed / len(records) if records else 0,
            "total_reward": total_reward,
            "avg_reward": total_reward / len(records) if records else 0,
        },
        "judge_version": "tau_bench_state_v1",
        "judged_at": datetime.utcnow().isoformat() + "Z",
    }

def write_outputs(
    run_dir: Path,
    result: Dict[str, Any],
    model_name: str,
) -> None:
    """Write judgment outputs to the run directory."""
    judgments = result["judgments"]
    summary = result["summary"]
    
    summary_data = {
        "model_name": model_name,
        "tau_pass_rate": summary["pass_rate"],
        "tau_avg_reward": summary["avg_reward"],
        "tasks_total": summary["total_tasks"],
        "tasks_passed": summary["passed"],
        "judge_version": result["judge_version"],
        "judged_at": result["judged_at"],
    }
    
    (run_dir / "tau_summary.json").write_text(
        json.dumps(summary_data, indent=2) + "\n",
        encoding="utf-8",
    )
    
    lines = [
        "# TAU-bench State-Based Evaluation",
        "",
        f"**Model**: {model_name}",
        f"**Judge Version**: {result['judge_version']}",
        f"**Judged**: {result['judged_at']}",
        "",
        "## Summary Metrics",
        "",
        f"- **Pass Rate**: {summary['passed']}/{summary['total_tasks']} ({summary['pass_rate']*100:.1f}%)",
        f"- **Average Reward**: {summary['avg_reward']:.3f}",
        "",
        "## Per-Task Results",
        "",
        "| Task | Passed | Reward | State Match | Output Match |",
        "|------|--------|--------|-------------|--------------|",
    ]
    
    for turn_idx, judgment in sorted(judgments.items()):
        status = "✅" if judgment["passed"] else "❌"
        state = "✓" if judgment["state_matched"] else "✗"
        output = "-" if judgment["outputs_matched"] is None else ("✓" if judgment["outputs_matched"] else "✗")
        lines.append(f"| {turn_idx} | {status} | {judgment['reward']:.1f} | {state} | {output} |")
    
    lines.extend(["", "## Detailed Reasoning", ""])
    for turn_idx, judgment in sorted(judgments.items()):
        if not judgment["passed"]:
            lines.append(f"### Task {turn_idx}")
            lines.append(judgment["reasoning"])
            lines.append("")
            
    (run_dir / "tau_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote TAU-bench state-based outputs to {run_dir}")

__all__ = ["judge_task", "judge_run", "write_outputs", "TauBenchJudgment"]
