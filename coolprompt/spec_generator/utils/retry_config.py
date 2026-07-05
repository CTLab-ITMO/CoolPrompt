from dataclasses import dataclass


@dataclass
class ValidationConfig:
    max_topup_attempts: int = 3

    judge_enabled: bool = True
    judge_quality_threshold: float = 0.7
    judge_batch_size: int = 10