from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import time

class IterationSnapshot(BaseModel):
    """Telemetry for a single optimization iteration."""
    iteration: int
    best_score: float
    best_prompt_length: int
    cumulative_tokens_in: int
    cumulative_tokens_out: int
    cumulative_total_cost: float
    cumulative_invoke_calls: int
    cumulative_batch_calls: int
    cumulative_api_wait_sec: float = Field(description="Time spent waiting for LLM responses")

class OptimizationTelemetry(BaseModel):
    """Final telemetry report for the entire optimization run."""
    method_name: str
    task_type: str
    start_time: datetime
    end_time: datetime
    total_wall_clock_sec: float
    total_api_wait_sec: float
    
    total_tokens_in: int
    total_tokens_out: int
    total_tokens: int
    total_cost_usd: float
    
    total_invoke_calls: int
    total_batch_calls: int
    total_batch_items: int
    total_api_requests: int = Field(description="Sum of invoke_calls and batch_calls")
    
    initial_score: float
    final_score: float
    score_improvement: float
    trajectory: List[IterationSnapshot]


class TelemetryCollector:
    """Helper to aggregate telemetry during a PromptTuner run."""
    def __init__(self, method_name: str, task_type: str, target_model):
        self.method_name = method_name
        self.task_type = task_type
        self.target_model = target_model
        self.start_time = datetime.now()
        self.start_wall_clock = time.time()
        self.trajectory = []

    def on_iteration_end(self, iteration: int, best_score: float, best_prompt: str) -> None:
        stats = self.target_model.get_stats() if hasattr(self.target_model, "get_stats") else {}
        stats = stats or {}
        self.trajectory.append(IterationSnapshot(
            iteration=iteration,
            best_score=best_score,
            best_prompt_length=len(best_prompt),
            cumulative_tokens_in=stats.get("prompt_tokens", 0),
            cumulative_tokens_out=stats.get("completion_tokens", 0),
            cumulative_total_cost=stats.get("total_cost", 0.0),
            cumulative_invoke_calls=stats.get("invoke_calls", 0),
            cumulative_batch_calls=stats.get("batch_calls", 0),
            cumulative_api_wait_sec=stats.get("api_wait_sec", 0.0),
        ))

    def finalize(self, initial_score: float, final_score: float) -> OptimizationTelemetry:
        stats = self.target_model.get_stats() if hasattr(self.target_model, "get_stats") else {}
        stats = stats or {}
        end_wall_clock = time.time()
        return OptimizationTelemetry(
            method_name=self.method_name,
            task_type=self.task_type,
            start_time=self.start_time,
            end_time=datetime.now(),
            total_wall_clock_sec=end_wall_clock - self.start_wall_clock,
            total_api_wait_sec=stats.get("api_wait_sec", 0.0),
            total_tokens_in=stats.get("prompt_tokens", 0),
            total_tokens_out=stats.get("completion_tokens", 0),
            total_tokens=stats.get("total_tokens", 0),
            total_cost_usd=stats.get("total_cost", 0.0),
            total_invoke_calls=stats.get("invoke_calls", 0),
            total_batch_calls=stats.get("batch_calls", 0),
            total_batch_items=stats.get("batch_items", 0),
            total_api_requests=stats.get("invoke_calls", 0) + stats.get("batch_calls", 0),
            initial_score=initial_score if initial_score is not None else 0.0,
            final_score=final_score if final_score is not None else 0.0,
            score_improvement=(final_score - initial_score) if (initial_score is not None and final_score is not None) else 0.0,
            trajectory=self.trajectory,
        )