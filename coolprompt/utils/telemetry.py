from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

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