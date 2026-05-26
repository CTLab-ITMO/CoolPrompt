"""
Pydantic schemas для валидации конфигурации RIDER.

Этот модуль определяет схемы для валидации YAML конфигураций:
- ExperimentConfig - конфигурация эксперимента
- RIDERConfig - параметры алгоритма RIDER
- DatasetConfig - настройки датасетов
- LLMConfig - настройки LLM провайдеров
- EvaluationConfig - параметры оценки

Все конфигурации валидируются при загрузке из YAML.
"""

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LLMConfig(BaseModel):
    """Конфигурация LLM провайдера."""

    model_config = ConfigDict(extra='forbid')

    provider: Literal['openai', 'openrouter', 'deepseek'] = Field(
        default='openai',
        description="LLM provider (use 'openrouter' + google/gemini-* for Gemini models)"
    )
    model: str = Field(
        default='gpt-3.5-turbo',
        description="Model name"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Base temperature for generation"
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=32000,
        description="Maximum tokens for generation"
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts"
    )

class DatasetConfig(BaseModel):
    """Конфигурация датасета."""

    model_config = ConfigDict(extra='forbid')

    name: Literal['GSM8K', 'AG_News', 'SQuAD_2', 'CommonGen', 'XSum', 'CodeSearchNet', 'HotpotQA'] = Field(
        description="Dataset name"
    )
    train_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Train split size"
    )
    val_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Validation split size"
    )
    dev_size: int = Field(
        default=50,
        ge=1,
        le=5000,
        description="Dev split size (for final selection)"
    )
    test_size: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Test split size"
    )
    data_offset: int = Field(
        default=0,
        ge=0,
        le=100000,
        description="Number of samples to skip (for sequential data loading)"
    )

    @model_validator(mode='after')
    def validate_test_size(self):
        """Проверяет что test >= val."""
        if self.test_size < self.val_size:
            raise ValueError(
                f"test_size ({self.test_size}) should be >= val_size ({self.val_size})"
            )
        return self


class RIDERConfig(BaseModel):
    """Конфигурация алгоритма RIDER."""

    model_config = ConfigDict(extra='forbid')

    population_size: int = Field(
        default=15,
        ge=5,
        le=100,
        description="Population size"
    )
    num_generations: int = Field(
        default=12,
        ge=1,
        le=100,
        description="Number of evolution generations"
    )
    elite_size: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Elite size (preserved each generation)"
    )
    tournament_size: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Tournament selection size"
    )

    # Diversity
    diversity_threshold: float = Field(
        default=0.72,  # ИСПРАВЛЕНО: 0.85→0.72 (DPP best practice for semantic diversity)
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for diversity filtering"
    )

    # UCB/Thompson Sampling parameters
    ucb_c: float = Field(
        default=1.414,  # √2
        ge=0.0,
        le=10.0,
        description="UCB exploration constant"
    )
    use_thompson_sampling: bool = Field(
        default=True,
        description="Use Thompson Sampling instead of UCB1 (recommended for better exploration-exploitation)"
    )

    # Adaptive diversity threshold (EvoPrompt++, 2024)
    adaptive_diversity: bool = Field(
        default=True,
        description="Use adaptive diversity threshold based on generation progress"
    )
    diversity_threshold_min: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum diversity threshold (early generations, strict)"
    )
    diversity_threshold_max: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Maximum diversity threshold (late generations, relaxed)"
    )

    # Adaptive hyperparameters
    temperature_min: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Minimum adaptive temperature"
    )
    temperature_max: float = Field(
        default=1.2,
        ge=0.0,
        le=2.0,
        description="Maximum adaptive temperature"
    )

    # Memory
    max_memory_patterns: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Maximum patterns in long-term memory"
    )
    memory_update_interval: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Generations between memory updates"
    )

    # Selection
    use_pareto_selection: bool = Field(
        default=False,
        description="Use Pareto selection instead of greedy"
    )
    ensemble_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Final ensemble size (k-DPP)"
    )

    # Warm start
    warm_start_from: Optional[str] = Field(
        default=None,
        description="Path to previous experiment results dir to warm-start population from"
    )

    # Cross-Experiment Memory (RIDER-unique). Set to False for fair baseline comparison.
    cross_experiment_memory: bool = Field(
        default=True,
        description="Enable loading best prompts from previous experiments on same dataset"
    )

    @model_validator(mode='after')
    def validate_ranges(self):
        """Проверяет связанные ограничения конфигурации."""
        if self.elite_size >= self.population_size:
            raise ValueError(
                f"elite_size ({self.elite_size}) must be < population_size ({self.population_size})"
            )
        if self.temperature_max <= self.temperature_min:
            raise ValueError(
                f"temperature_max ({self.temperature_max}) must be > "
                f"temperature_min ({self.temperature_min})"
            )
        return self


class EvaluationConfig(BaseModel):
    """Конфигурация evaluation."""

    model_config = ConfigDict(extra='forbid')

    use_cache: bool = Field(
        default=True,
        description="Use evaluation cache"
    )
    max_cache_size: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum cache size"
    )
    show_progress: bool = Field(
        default=True,
        description="Show progress bars"
    )
    save_predictions: bool = Field(
        default=True,
        description="Save predictions to disk"
    )

class ExperimentConfig(BaseModel):
    """Полная конфигурация эксперимента."""

    model_config = ConfigDict(extra='forbid')

    # Metadata
    experiment_name: str = Field(
        description="Experiment name"
    )
    description: Optional[str] = Field(
        default=None,
        description="Experiment description"
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility"
    )

    # Datasets
    datasets: List[DatasetConfig] = Field(
        description="List of datasets to run"
    )

    # Algorithms
    methods: List[Literal['RIDER', 'EvoPrompt-GA', 'EvoPrompt-DE', 'APE', 'ReEvo', 'PromptBreeder', 'ZeroShot']] = Field(
        default=['RIDER'],
        description="Methods to compare"
    )

    # LLM
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration"
    )

    # RIDER
    rider: RIDERConfig = Field(
        default_factory=RIDERConfig,
        description="RIDER configuration"
    )

    # Evaluation
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Evaluation configuration"
    )

    # Output
    output_dir: str = Field(
        default='./results',
        description="Output directory for results"
    )
    save_history: bool = Field(
        default=True,
        description="Save evolution history"
    )
    save_checkpoints: bool = Field(
        default=False,
        description="Save checkpoints during evolution"
    )
    log_detailed_evaluations: bool = Field(
        default=True,
        description="Enable detailed evaluation logging for ALL methods (predictions vs ground truth)"
    )

    # Parallelization settings
    num_parallel_workers: int = Field(
        default=1,
        ge=1,
        le=64,
        description="Number of parallel workers for dataset-level parallelization"
    )
    parallel_strategy: Literal['sequential', 'datasets', 'methods', 'hybrid'] = Field(
        default='sequential',
        description=(
            "Parallelization strategy:\n"
            "- sequential: No parallelization (default)\n"
            "- datasets: Parallelize across datasets (recommended)\n"
            "- methods: Parallelize across methods (requires thread-safe cache)\n"
            "- hybrid: Both datasets and methods (experimental)"
        )
    )

    @field_validator('datasets')
    @classmethod
    def validate_datasets_not_empty(cls, v):
        """Проверяет что список датасетов не пуст."""
        if not v:
            raise ValueError("At least one dataset must be specified")
        return v

    @field_validator('methods')
    @classmethod
    def validate_methods_not_empty(cls, v):
        """Проверяет что список методов не пуст."""
        if not v:
            raise ValueError("At least one method must be specified")
        return v

    @model_validator(mode='after')
    def validate_rider_in_methods(self):
        """Проверяет что если RIDER в methods, то rider config задан."""
        if 'RIDER' in self.methods and self.rider is None:
            raise ValueError("RIDER config must be specified when RIDER is in methods")
        return self

# Helper functions

def load_config_from_dict(config_dict: Dict) -> ExperimentConfig:
    """
    Загружает и валидирует конфигурацию из словаря.

    Args:
        config_dict: Словарь с конфигурацией

    Returns:
        Валидированный ExperimentConfig

    Raises:
        ValidationError: Если конфигурация невалидна
    """
    return ExperimentConfig(**config_dict)


def load_config_from_yaml(yaml_path: str) -> ExperimentConfig:
    """
    Загружает конфигурацию из YAML файла.

    Args:
        yaml_path: Путь к YAML файлу

    Returns:
        Валидированный ExperimentConfig

    Raises:
        ValidationError: Если конфигурация невалидна
        FileNotFoundError: Если файл не найден
    """
    import yaml

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    return load_config_from_dict(config_dict)


def create_default_config(
    experiment_name: str,
    datasets: List[str],
    output_dir: str = './results'
) -> ExperimentConfig:
    """
    Создает конфигурацию с дефолтными параметрами.

    Args:
        experiment_name: Название эксперимента
        datasets: Список названий датасетов
        output_dir: Директория для результатов

    Returns:
        ExperimentConfig с дефолтными параметрами
    """
    dataset_configs = [
        DatasetConfig(name=ds) for ds in datasets
    ]

    return ExperimentConfig(
        experiment_name=experiment_name,
        datasets=dataset_configs,
        output_dir=output_dir
    )

