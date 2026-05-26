"""
RIDER CLI - Command Line Interface.

Команды:
- rider init [template] - Инициализация конфигурации из шаблона
- rider run <config> - Запуск эксперимента
- rider analyze <results_dir> - Анализ результатов
- rider visualize <results_dir> - Визуализация результатов
- rider diagnose <history_dir> - Глубокий диагностический анализ истории эволюции

Example:
    $ rider init quick_test
    $ rider run config.yaml
    $ rider analyze ./results/experiment_name
    $ rider visualize ./results/experiment_name
    $ rider diagnose ./results/experiment_name
"""

import argparse
import sys
import os
import logging
import json
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress httpx HTTP 200 OK spam — only show warnings/errors
logging.getLogger("httpx").setLevel(logging.WARNING)

# Thread-safe file writing
results_lock = threading.Lock()


def save_results_atomic(results: Dict, summary_path: Path):
    """
    Атомарная запись результатов в файл.

    Использует временный файл + rename (atomic operation) для безопасной записи.
    Thread-safe с помощью Lock.
    ВАЖНО: Читает текущий файл и мёрджит, чтобы не перетереть результаты,
    сохранённые другими потоками через save_single_result.

    Args:
        results: Dictionary with experiment results
        summary_path: Path to summary.json file
    """
    with results_lock:
        # Read current file on disk and merge (prevents overwriting save_single_result data)
        if summary_path.exists():
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    current_on_disk = json.load(f)
                # Merge: disk data takes precedence (it's more up-to-date from save_single_result)
                merged = dict(current_on_disk)
                merged.update(results)
            except Exception:
                merged = results
        else:
            merged = results

        # Write to temporary file
        temp_path = summary_path.with_suffix('.json.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, default=str, ensure_ascii=False)

        # Atomic rename with retry (Windows PermissionError)
        for attempt in range(5):
            try:
                temp_path.replace(summary_path)
                break
            except PermissionError as e:
                if attempt < 4:
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    logger.warning(f"Failed to replace summary.json in save_results_atomic: {e}")


def load_progress(progress_path: Path) -> Dict:
    """
    Загружает progress file или создает новый.
    С retry логикой для Windows PermissionError (race condition с параллельными потоками).

    Args:
        progress_path: Path to progress.json file

    Returns:
        Dict with progress data
    """
    default = {
        'experiment_name': '',
        'last_updated': datetime.now().isoformat(),
        'total_experiments': 0,
        'completed_experiments': 0,
        'experiments': {}
    }
    if not progress_path.exists():
        return default

    max_retries = 5
    for attempt in range(max_retries):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (PermissionError, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (2 ** attempt))
            else:
                logger.warning(f"Failed to load progress after {max_retries} attempts: {e}")
                return default


def update_progress(
    progress_path: Path,
    result_key: str,
    status: str,
    **metadata
):
    """
    Атомарное обновление progress file.

    Args:
        progress_path: Path to progress.json
        result_key: Experiment key like "GSM8K_ZeroShot"
        status: "pending" | "in_progress" | "completed" | "failed"
        **metadata: Additional fields (fitness, error, started_at, etc.)
    """
    with results_lock:
        # Load existing
        progress = load_progress(progress_path)

        # Update entry
        if result_key not in progress['experiments']:
            progress['experiments'][result_key] = {}

        progress['experiments'][result_key]['status'] = status
        progress['experiments'][result_key]['last_updated'] = datetime.now().isoformat()
        progress['experiments'][result_key].update(metadata)

        # Update counters
        progress['last_updated'] = datetime.now().isoformat()
        progress['completed_experiments'] = sum(
            1 for exp in progress['experiments'].values()
            if exp.get('status') == 'completed'
        )

        # Write atomically with retry logic for Windows PermissionError
        temp_path = progress_path.with_suffix('.json.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, default=str, ensure_ascii=False)

        # Retry logic for file replacement (handles Windows PermissionError)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                temp_path.replace(progress_path)
                break
            except PermissionError as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: 0.1s, 0.2s, 0.4s, 0.8s, 1.6s
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    # Last attempt failed - log error but don't crash
                    logger.warning(f"Failed to update progress file after {max_retries} attempts: {e}")
                    # Clean up temp file
                    if temp_path.exists():
                        try:
                            temp_path.unlink()
                        except Exception as cleanup_error:
                            logger.warning(
                                f"Failed to clean up temporary progress file {temp_path}: {cleanup_error}"
                            )


def get_progress_summary(progress_path: Path) -> str:
    """
    Человеко-читаемая сводка прогресса.

    Args:
        progress_path: Path to progress.json

    Returns:
        String summary like "Progress: 5/10 completed, 1 in progress"
    """
    progress = load_progress(progress_path)

    completed = progress['completed_experiments']
    total = progress['total_experiments']

    in_progress = [
        key for key, exp in progress['experiments'].items()
        if exp.get('status') == 'in_progress'
    ]

    return f"Progress: {completed}/{total} completed, {len(in_progress)} in progress"


def save_single_result(summary_path: Path, result_key: str, result: Dict):
    """
    Атомарное добавление ОДНОГО результата в summary.json.
    Thread-safe с помощью Lock + retry для Windows PermissionError.

    Args:
        summary_path: Path to summary.json
        result_key: Experiment key like "GSM8K_ZeroShot"
        result: Result dictionary
    """
    with results_lock:
        # Read existing results with retry
        all_results = {}
        if summary_path.exists():
            for attempt in range(5):
                try:
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        all_results = json.load(f)
                    break
                except (PermissionError, json.JSONDecodeError) as e:
                    if attempt < 4:
                        time.sleep(0.1 * (2 ** attempt))
                    else:
                        logger.warning(f"Failed to read summary.json: {e}")

        # Add new result
        all_results[result_key] = result

        # Write atomically (temp file + rename) with retry
        temp_path = summary_path.with_suffix('.json.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)

        for attempt in range(5):
            try:
                temp_path.replace(summary_path)
                break
            except PermissionError as e:
                if attempt < 4:
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    logger.warning(f"Failed to replace summary.json: {e}")


def run_dataset_worker(
    dataset_name: str,
    data: Dict[str, List],
    config,
    llm_client,
    evaluator,
    sentence_encoder,
    output_dir: Path,
    existing_results: Dict,
    progress_path: Path,  # NEW: Progress tracking file
    baseline_evaluator=None  # Evaluator без prompt repetition для бейзлайнов
) -> Dict[str, Dict]:
    """Worker function для параллельного выполнения экспериментов на одном датасете."""
    from rider.algorithms.rider import RIDER
    local_results = {}
    summary_path = output_dir / 'summary.json'

    logger.info(f"\n{'='*70}")
    logger.info(f"[Worker] Dataset: {dataset_name}")
    logger.info(f"{'='*70}")

    for method in config.methods:
        result_key = f"{dataset_name}_{method}"

        # Load current progress to check status
        progress = load_progress(progress_path)

        # CHECK IF ALREADY COMPLETED (resume capability via progress.json)
        if progress['experiments'].get(result_key, {}).get('status') == 'completed':
            # Only skip if result exists in summary.json
            if result_key in existing_results:
                logger.info(f"⏭️  Skipping {result_key} (already completed)")
                local_results[result_key] = existing_results[result_key]
                continue
            else:
                # Completed in progress.json but missing in summary.json - reset and rerun
                logger.warning(f"⚠️  {result_key} marked completed but missing result - will rerun")
                update_progress(progress_path, result_key, 'pending')

        # Mark as in_progress
        update_progress(progress_path, result_key, 'in_progress', started_at=datetime.now().isoformat())

        logger.info(f"\n[Worker] Running {method} on {dataset_name}...")

        # Выбираем evaluator: RIDER использует prompt repetition, бейзлайны — нет
        method_evaluator = evaluator if method == 'RIDER' else (baseline_evaluator or evaluator)

        # Сброс счётчиков API перед каждым экспериментом
        llm_client.reset_usage()

        try:
            # Factory pattern - copy from existing code below
            if method == 'RIDER':
                algorithm = RIDER(
                    llm_client=llm_client,
                    evaluator=method_evaluator,
                    sentence_encoder=sentence_encoder,
                    dataset_name=dataset_name,
                    config={
                        'population_size': config.rider.population_size,
                        'num_generations': config.rider.num_generations,
                        'elite_size': config.rider.elite_size,
                        'tournament_size': config.rider.tournament_size,
                        'diversity_threshold': config.rider.diversity_threshold,
                        'ucb_c': config.rider.ucb_c,
                        'use_thompson_sampling': config.rider.use_thompson_sampling,
                        'adaptive_diversity': config.rider.adaptive_diversity,
                        'diversity_threshold_min': config.rider.diversity_threshold_min,
                        'diversity_threshold_max': config.rider.diversity_threshold_max,
                        'max_memory_patterns': config.rider.max_memory_patterns,
                        'memory_update_interval': config.rider.memory_update_interval,
                        'use_pareto_selection': config.rider.use_pareto_selection,
                        'ensemble_size': config.rider.ensemble_size,
                        'log_detailed_evaluations': config.log_detailed_evaluations,
                        'cross_experiment_memory': config.rider.cross_experiment_memory,
                    },
                    model=config.llm.model,
                    temperature=config.llm.temperature,
                    experiment_name=config.experiment_name
                )
                # Crash recovery — resume from latest checkpoint
                initial_population = None
                start_generation = 0
                result_dir = Path(config.output_dir) / config.experiment_name
                crash_pop, crash_gen = RIDER.load_crash_recovery_checkpoint(
                    result_dir, dataset_name
                )
                if crash_pop and crash_gen > 0:
                    logger.info(
                        f"🔄 Crash recovery: resuming {result_key} from Gen {crash_gen} "
                        f"(best={max(p.fitness for p in crash_pop):.4f})"
                    )
                    initial_population = crash_pop
                    start_generation = crash_gen
                elif config.rider.warm_start_from:
                    # Warm start: загружаем финальную популяцию из предыдущего эксперимента
                    warm_start_path = Path(config.rider.warm_start_from)
                    if not warm_start_path.is_absolute():
                        warm_start_path = Path(config.output_dir) / config.rider.warm_start_from
                    logger.info(f"Warm start: loading population from {warm_start_path}")
                    initial_population = RIDER.load_population_from_checkpoint(
                        warm_start_path, dataset_name
                    )
                    if initial_population:
                        logger.info(f"Warm start: loaded {len(initial_population)} prompts")
                    else:
                        logger.warning("Warm start: no prompts loaded, starting from scratch")
                best_prompt = algorithm.run(
                    train_data=data['train'],
                    val_data=data['val'],
                    dev_data=data['dev'],
                    show_progress=config.evaluation.show_progress,
                    initial_population=initial_population,
                    start_generation=start_generation
                )
                if config.save_history:
                    algorithm.history.save()

            elif method == 'ZeroShot':
                from rider.algorithms.zeroshot import ZeroShot
                algorithm = ZeroShot(
                    llm_client=llm_client,
                    evaluator=method_evaluator,
                    dataset_name=dataset_name,
                    num_prompts=10,
                    model=config.llm.model,
                    temperature=config.llm.temperature,
                    save_history=config.save_history,
                    log_detailed_evaluations=config.log_detailed_evaluations,
                    experiment_name=config.experiment_name
                )
                best_prompt = algorithm.run(
                    train_data=data['train'],
                    val_data=data['val'],
                    dev_data=data['dev'],
                    test_data=data['test']
                )

            elif method == 'APE':
                from rider.algorithms.ape import APE
                algorithm = APE(
                    llm_client=llm_client,
                    evaluator=method_evaluator,
                    dataset_name=dataset_name,
                    num_prompts=50,
                    num_demos=5,
                    model=config.llm.model,
                    temperature=1.0,
                    top_p=0.99,
                    save_history=config.save_history,
                    log_detailed_evaluations=config.log_detailed_evaluations,
                    experiment_name=config.experiment_name
                )
                best_prompt = algorithm.run(
                    train_data=data['train'],
                    val_data=data['val'],
                    dev_data=data['dev'],
                    test_data=data['test']
                )

            elif method == 'EvoPrompt-GA':
                from rider.algorithms.evoprompt_ga import EvoPromptGA
                algorithm = EvoPromptGA(
                    llm_client=llm_client,
                    evaluator=method_evaluator,
                    dataset_name=dataset_name,
                    population_size=10,
                    num_generations=10,
                    model=config.llm.model,
                    temperature=0.5,
                    top_p=0.95,
                    save_history=config.save_history,
                    log_detailed_evaluations=config.log_detailed_evaluations,
                    experiment_name=config.experiment_name
                )
                best_prompt = algorithm.run(
                    train_data=data['train'],
                    val_data=data['val'],
                    dev_data=data['dev'],
                    test_data=data['test']
                )

            elif method == 'EvoPrompt-DE':
                from rider.algorithms.evoprompt_de import EvoPromptDE
                algorithm = EvoPromptDE(
                    llm_client=llm_client,
                    evaluator=method_evaluator,
                    dataset_name=dataset_name,
                    population_size=10,
                    num_generations=10,
                    model=config.llm.model,
                    temperature=0.5,
                    top_p=0.95,
                    save_history=config.save_history,
                    log_detailed_evaluations=config.log_detailed_evaluations,
                    experiment_name=config.experiment_name
                )
                best_prompt = algorithm.run(
                    train_data=data['train'],
                    val_data=data['val'],
                    dev_data=data['dev'],
                    test_data=data['test']
                )

            elif method == 'PromptBreeder':
                from rider.algorithms.promptbreeder import PromptBreeder
                algorithm = PromptBreeder(
                    llm_client=llm_client,
                    evaluator=method_evaluator,
                    dataset_name=dataset_name,
                    population_size=10,
                    num_generations=10,
                    num_mutation_prompts=5,
                    num_thinking_styles=3,
                    model=config.llm.model,
                    temperature=0.7,
                    top_p=0.95,
                    save_history=config.save_history,
                    log_detailed_evaluations=config.log_detailed_evaluations,
                    experiment_name=config.experiment_name
                )
                best_prompt = algorithm.run(
                    train_data=data['train'],
                    val_data=data['val'],
                    dev_data=data['dev'],
                    test_data=data['test']
                )
            else:
                logger.warning(f"Method {method} not implemented, skipping")
                continue

            # Save results
            result = {
                'dataset': dataset_name,
                'method': method,
                'best_prompt': best_prompt.text,
                'best_fitness': best_prompt.fitness,
                'completed_at': datetime.now().isoformat()
            }

            if method == 'RIDER':
                result['statistics'] = algorithm.get_statistics()

            # API usage для всех методов (для таблицы затрат)
            result['api_usage'] = llm_client.get_usage_stats()

            # Update in-memory results
            local_results[result_key] = result

            # ✅ SAVE TO DISK IMMEDIATELY (per-method saving!)
            save_single_result(summary_path, result_key, result)

            # ✅ UPDATE PROGRESS: completed
            update_progress(
                progress_path,
                result_key,
                'completed',
                fitness=best_prompt.fitness,
                result_saved=True,
                completed_at=datetime.now().isoformat()
            )

            logger.info(f"✅ {method} on {dataset_name}: SAVED (fitness={best_prompt.fitness:.4f})")

        except Exception as e:
            logger.error(f"❌ Error with {method} on {dataset_name}: {e}")
            traceback.print_exc()

            # ✅ UPDATE PROGRESS: failed
            update_progress(progress_path, result_key, 'failed', error=str(e))

    return local_results

def init_command(args):
    """
    Инициализация конфигурации из шаблона.

    Args:
        args: Parsed arguments
    """
    import shutil

    template_name = args.template or 'quick_test'
    output_path = args.output or f'{template_name}.yaml'

    # Путь к шаблону
    package_dir = Path(__file__).parent.parent
    template_path = package_dir / 'config' / 'templates' / f'{template_name}.yaml'

    if not template_path.exists():
        logger.error(f"Template not found: {template_name}")
        logger.info("Available templates: experiments, quick_test")
        return 1

    # Копируем шаблон
    shutil.copy(template_path, output_path)
    logger.info(f"Created configuration: {output_path}")
    logger.info(f"Edit this file and then run: rider run {output_path}")

    return 0


def run_command(args):
    """
    Запуск эксперимента.

    Args:
        args: Parsed arguments
    """
    from rider.config.schema import load_config_from_yaml
    from rider.llm.client import LLMClient
    from rider.evaluation.metrics import MetricsEvaluator
    from rider.evaluation.evaluator import PromptEvaluator
    from rider.data.loader import UnifiedDatasetLoader
    from sentence_transformers import SentenceTransformer

    config_path = args.config

    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return 1

    # Load and validate config
    logger.info(f"Loading configuration from {config_path}")
    try:
        config = load_config_from_yaml(config_path)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Datasets: {[d.name for d in config.datasets]}")
    logger.info(f"Methods: {config.methods}")

    # Create output directory with timestamp to prevent overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.experiment_name}_{timestamp}"
    output_dir = Path(config.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    # Override experiment_name with timestamped run_name so ALL paths use it
    config.experiment_name = run_name
    logger.info(f"Run ID: {run_name}")
    logger.info(f"Output directory: {output_dir}")

    # Initialize components
    logger.info("Initializing components...")

    # LLM Client (ИСПРАВЛЕНО: правильные параметры)
    try:
        llm_client = LLMClient(
            provider=config.llm.provider,
            max_retries=config.llm.max_retries
        )
    except NotImplementedError as e:
        logger.error(str(e))
        return 1

    # Metrics
    metrics_evaluator = MetricsEvaluator()

    # Evaluator для RIDER (с prompt repetition)
    evaluator = PromptEvaluator(
        llm_client=llm_client,
        metrics_evaluator=metrics_evaluator,
        model=config.llm.model,
        temperature=0.0,  # Greedy for evaluation
        max_cache_size=config.evaluation.max_cache_size,
        max_workers=getattr(config, 'num_parallel_workers', 32),
        use_prompt_repetition=True
    )

    # Evaluator для бейзлайнов (без prompt repetition)
    baseline_evaluator = PromptEvaluator(
        llm_client=llm_client,
        metrics_evaluator=metrics_evaluator,
        model=config.llm.model,
        temperature=0.0,  # Greedy for evaluation
        max_cache_size=config.evaluation.max_cache_size,
        max_workers=getattr(config, 'num_parallel_workers', 32),
        use_prompt_repetition=False
    )

    # Dataset Loader
    loader = UnifiedDatasetLoader(
        cache_dir='.cache/datasets',
        seed=config.seed
    )

    # SentenceTransformer for diversity
    logger.info("Loading SentenceTransformer...")
    sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    # Load datasets
    logger.info("Loading datasets...")
    datasets = {}
    for dataset_config in config.datasets:
        try:
            data = loader.load_dataset(
                dataset_name=dataset_config.name,
                train_size=dataset_config.train_size,
                val_size=dataset_config.val_size,
                dev_size=dataset_config.dev_size,
                test_size=dataset_config.test_size,
                data_offset=dataset_config.data_offset
            )
            datasets[dataset_config.name] = data
            logger.info(
                f"Loaded {dataset_config.name}: "
                f"train={len(data['train'])}, val={len(data['val'])}, "
                f"dev={len(data['dev'])}, test={len(data['test'])}"
            )
        except Exception as e:
            logger.error(f"Failed to load {dataset_config.name}: {e}")
            continue

    if not datasets:
        logger.error("No datasets loaded, aborting")
        return 1

    # Setup paths for results
    summary_path = output_dir / 'summary.json'
    progress_path = output_dir / 'progress.json'

    # Load existing results if file exists (resume capability)
    if summary_path.exists():
        logger.info(f"📂 Found existing results: {summary_path}")
        with open(summary_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        logger.info(f"   Loaded {len(results)} existing experiments")
    else:
        results = {}

    # ✅ LOAD PROGRESS FILE
    progress = load_progress(progress_path)

    # Initialize total experiments count if not set
    if progress['total_experiments'] == 0:
        progress['total_experiments'] = len(datasets) * len(config.methods)
        progress['experiment_name'] = config.experiment_name
        update_progress(progress_path, '_init', 'pending')  # Trigger save
        progress = load_progress(progress_path)  # Reload

    # ✅ ANALYZE INTERRUPTED EXPERIMENTS
    interrupted = [
        key for key, exp in progress['experiments'].items()
        if exp.get('status') == 'in_progress' and key != '_init'
    ]

    if interrupted:
        logger.warning(f"⚠️  Found {len(interrupted)} interrupted experiments:")
        for key in interrupted:
            exp = progress['experiments'][key]
            logger.warning(f"   - {key}: last update at {exp.get('last_updated', 'unknown')}")

        # Reset interrupted to pending (will be restarted)
        for key in interrupted:
            update_progress(progress_path, key, 'pending')

    # ✅ DISPLAY PROGRESS SUMMARY
    logger.info(get_progress_summary(progress_path))

    # Run experiments
    if config.parallel_strategy == 'sequential' or config.num_parallel_workers == 1:
        # Sequential execution
        logger.info("\n🔄 Running experiments sequentially")

        for dataset_name, data in datasets.items():
            local_results = run_dataset_worker(
                dataset_name, data, config, llm_client, evaluator,
                sentence_encoder, output_dir, results, progress_path,
                baseline_evaluator=baseline_evaluator
            )
            results.update(local_results)
            save_results_atomic(results, summary_path)
            logger.info(f"✅ Completed and SAVED {dataset_name} (total: {len(results)} experiments)")

    elif config.parallel_strategy == 'datasets':
        # Dataset-level parallelization
        logger.info(f"\n🚀 Starting parallel execution with {config.num_parallel_workers} workers")
        logger.info(f"   Strategy: {config.parallel_strategy}")
        logger.info(f"   Total experiments: {len(datasets)} datasets × {len(config.methods)} methods")

        with ThreadPoolExecutor(max_workers=config.num_parallel_workers) as executor:
            future_to_dataset = {
                executor.submit(
                    run_dataset_worker,
                    dataset_name, data, config, llm_client, evaluator,
                    sentence_encoder, output_dir, results, progress_path,
                    baseline_evaluator
                ): dataset_name
                for dataset_name, data in datasets.items()
            }

            for future in as_completed(future_to_dataset):
                dataset_name = future_to_dataset[future]
                try:
                    local_results = future.result()
                    results.update(local_results)
                    save_results_atomic(results, summary_path)
                    logger.info(f"✅ Completed and SAVED {dataset_name} (total: {len(results)} experiments)")
                except Exception as e:
                    logger.error(f"❌ Failed to process {dataset_name}: {e}")
                    traceback.print_exc()
    else:
        raise ValueError(f"Unsupported parallel_strategy: {config.parallel_strategy}")

    # Final save
    save_results_atomic(results, summary_path)
    logger.info(f"\nFinal save to {summary_path}")

    logger.info(f"\n{'='*70}")
    logger.info("Experiment completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Total experiments: {len(results)}")
    logger.info(f"{'='*70}")

    return 0
def analyze_command(args):
    """
    Анализ результатов эксперимента.

    Args:
        args: Parsed arguments
    """
    results_dir = args.results_dir

    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        return 1

    logger.info(f"Analyzing results in {results_dir}")

    # Load summary
    summary_path = Path(results_dir) / 'summary.json'
    if not summary_path.exists():
        logger.error(f"Summary file not found: {summary_path}")
        return 1

    with open(summary_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("Results Summary")
    logger.info(f"{'='*70}")

    for key, result in results.items():
        logger.info(f"\n{key}:")
        logger.info(f"  Dataset: {result['dataset']}")
        logger.info(f"  Method: {result['method']}")
        logger.info(f"  Best Fitness: {result['best_fitness']:.4f}")
        logger.info(f"  Best Prompt: {result['best_prompt'][:80]}...")

    logger.info(f"\n{'='*70}")

    # Check for history files
    history_files = list(Path(results_dir).glob('*_history.json'))
    if history_files:
        logger.info(f"\nFound {len(history_files)} history files")
        logger.info("Use 'rider visualize' to create visualizations")

    return 0


def visualize_command(args):
    """
    Визуализация результатов.

    Args:
        args: Parsed arguments
    """
    results_dir = args.results_dir

    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        return 1

    logger.info(f"Creating visualizations for {results_dir}")
    logger.warning("Visualization not yet fully implemented")

    # FUTURE WORK: re-enable the visualization pipeline here once
    # convergence/operator/diversity plot generation is restored.

    return 0


def diagnose_command(args):
    """
    Глубокий анализ истории для поиска слабых мест RIDER.

    Args:
        args: Parsed arguments
    """
    from rider.execution.history import EvolutionHistory

    input_dir = Path(args.history_dir)

    if not input_dir.exists():
        logger.error(f"History directory not found: {input_dir}")
        return 1

    def _looks_like_history_dir(path: Path) -> bool:
        return (
            (path / "full_history.json").exists()
            or (path / "generation_summaries.json").exists()
            or (path / "evolution_steps.json").exists()
        )

    history_path = None
    fallback_mode = False
    if input_dir.is_dir() and _looks_like_history_dir(input_dir):
        history_path = input_dir
    else:
        candidates = [
            p for p in input_dir.rglob("*")
            if p.is_dir() and _looks_like_history_dir(p)
        ]
        if candidates:
            candidates.sort(key=lambda p: (
                0 if (p / "full_history.json").exists() else 1,
                len(p.parts),
                str(p)
            ))
            history_path = candidates[0]

    if history_path is None:
        legacy_files = list(input_dir.rglob('*_history.json'))
        if legacy_files:
            logger.error(
                "Legacy single-file histories are no longer loaded directly; "
                "use a history/<experiment>/ directory with generation_summaries.json."
            )
        else:
            logger.error(f"No history directory found under {input_dir}")
            logger.info("Expected history/<experiment>/generation_summaries.json")
        return 1

    fallback_mode = not (history_path / "full_history.json").exists()
    logger.info(f"Loading history from {history_path}")

    if history_path.parent.name == "history":
        save_dir = history_path.parent.parent
        experiment_id = history_path.name
    else:
        save_dir = input_dir
        experiment_id = history_path.name

    history = EvolutionHistory(save_dir=save_dir, experiment_id=experiment_id)
    history.load(history_path)

    experiment = history.metadata.get("experiment_id") or experiment_id

    print("="*70)
    print("RIDER DIAGNOSTIC REPORT")
    print("="*70)
    print(f"Experiment: {experiment}")
    if fallback_mode:
        print("Loaded via generation_summaries.json fallback")

    # 1. Operator analysis
    print("\n[1] OPERATOR PERFORMANCE")
    operator_stats = history.get_operator_analysis()
    if operator_stats:
        for op, stats in sorted(operator_stats.items(),
                                key=lambda x: x[1]['avg_fitness_improvement'],
                                reverse=True):
            print(f"  {op:30s} | Uses: {stats['total_uses']:3d} | "
                  f"Success: {stats['success_rate']:.1%} | "
                  f"Avg Δfitness: {stats['avg_fitness_improvement']:+.4f}")
    else:
        print("  No operator data available")

    # 2. Elite trajectory
    print("\n[2] ELITE TRAJECTORY")
    summaries = history.generation_summaries
    if summaries:
        for summary in summaries[:10]:
            print(f"  Gen {summary.generation:2d} | Best: {summary.best_fitness:.4f} | "
                  f"Avg: {summary.avg_fitness:.4f} | "
                  f"Δ: {summary.improvement_over_previous:+.4f}")
        if len(summaries) > 10:
            print(f"  ... ({len(summaries) - 10} more generations)")
    else:
        print("  No trajectory data available")

    # 3. Trends
    print("\n[3] TRENDS")
    if summaries:
        first = summaries[0].best_fitness
        last = summaries[-1].best_fitness
        print(f"  Best fitness: {first:.4f} -> {last:.4f}")
        print(f"  Generations: {len(summaries)}")
    else:
        print("  No trend data available")

    # 4. Top errors
    print("\n[4] TOP ERRORS")
    try:
        failures = history.analyze_failures(top_k=5)
        if failures['error_type_distribution']:
            for error_type, count in failures['error_type_distribution'].items():
                print(f"  {error_type:30s}: {count:4d}")
        else:
            print("  No error data available (run with log_detailed_evaluations=True)")
    except Exception as e:
        logger.warning(f"Failure analysis skipped: {e}")
        print("  No error data available")

    # Сохранить отчёт
    report_file = input_dir / "diagnostic_report.txt"
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RIDER DIAGNOSTIC REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Experiment: {experiment}\n")
            if fallback_mode:
                f.write("Loaded via generation_summaries.json fallback\n")
            f.write("\n")
            f.write("[1] OPERATOR PERFORMANCE\n\n")
            if operator_stats:
                for op, stats in sorted(operator_stats.items(),
                                        key=lambda x: x[1]['avg_fitness_improvement'],
                                        reverse=True):
                    f.write(f"  {op:30s} | Uses: {stats['total_uses']:3d} | "
                            f"Success: {stats['success_rate']:.1%} | "
                            f"Avg Δfitness: {stats['avg_fitness_improvement']:+.4f}\n")
            else:
                f.write("  No operator data available\n")
            f.write("\n[2] ELITE TRAJECTORY\n\n")
            if summaries:
                for summary in summaries:
                    f.write(f"  Gen {summary.generation:2d} | Best: {summary.best_fitness:.4f} | "
                            f"Avg: {summary.avg_fitness:.4f} | "
                            f"Δ: {summary.improvement_over_previous:+.4f}\n")
            else:
                f.write("  No trajectory data available\n")
            f.write("\n[3] TRENDS\n\n")
            if summaries:
                f.write(f"  Best fitness: {summaries[0].best_fitness:.4f} -> "
                        f"{summaries[-1].best_fitness:.4f}\n")
                f.write(f"  Generations: {len(summaries)}\n")
            else:
                f.write("  No trend data available\n")
            f.write("\n[4] TOP ERRORS\n\n")
            try:
                failures = history.analyze_failures(top_k=5)
                if failures['error_type_distribution']:
                    for error_type, count in failures['error_type_distribution'].items():
                        f.write(f"  {error_type:30s}: {count:4d}\n")
                else:
                    f.write("  No error data available\n")
            except Exception:
                f.write("  No error data available\n")
        print(f"\n{'='*70}")
        print(f"Report saved to: {report_file}")
        print(f"{'='*70}")
    except Exception as e:
        logger.warning(f"Failed to save report: {e}")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='rider',
        description='RIDER - Reflective Iterative Diversity-Enhanced Reasoning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rider init quick_test              Create quick test configuration
  rider init experiments             Create full experiment configuration
  rider run config.yaml              Run experiment with config
  rider analyze ./results/exp_name   Analyze experiment results
  rider visualize ./results/exp_name Create visualizations
  rider diagnose ./results/exp_name  Deep diagnostic analysis
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize configuration from template')
    init_parser.add_argument(
        'template',
        nargs='?',
        default='quick_test',
        choices=['quick_test', 'experiments'],
        help='Configuration template (default: quick_test)'
    )
    init_parser.add_argument(
        '-o', '--output',
        help='Output file path (default: <template>.yaml)'
    )

    # Run command
    run_parser = subparsers.add_parser('run', help='Run experiment')
    run_parser.add_argument(
        'config',
        help='Path to configuration file (YAML)'
    )

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze experiment results')
    analyze_parser.add_argument(
        'results_dir',
        help='Path to results directory'
    )

    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Visualize experiment results')
    visualize_parser.add_argument(
        'results_dir',
        help='Path to results directory'
    )

    # Diagnose command
    diagnose_parser = subparsers.add_parser('diagnose', help='Deep diagnostic analysis of evolution history')
    diagnose_parser.add_argument(
        'history_dir',
        help='Path to history directory (e.g., ./results/experiment_name/)'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    if args.command == 'init':
        return init_command(args)
    elif args.command == 'run':
        return run_command(args)
    elif args.command == 'analyze':
        return analyze_command(args)
    elif args.command == 'visualize':
        return visualize_command(args)
    elif args.command == 'diagnose':
        return diagnose_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
