## Инструмент для примения и оценки базовых методов промптинга

#### Реализованные в данный момент методы:
- zero-shot
- few-shot
- zero-shot-chain-of-thoughts
- few-shot-chain-of-thoughts
- role-based
- self-discover

## Применение методов
Работа методов осуществляется с помощью скрипта `base_prompting_optimizer.py`

#### Пример запуска:
```
python base_prompting_optimizer.py --method few-shot
--input-file-path basic_prompts.json --output-file-path ./logs/few_shot_result.json --num-shots 5 --labels-file-path labels.json 
```

#### Аргументы:
- `method`: название метода
- `input-file-path`: путь до JSON файла с базовыми промптами (по умолчанию `base_prompts.json`). Ожидается структура {task: prompt}
- `output-file-path`: путь до файла, в который запишется результат
- `num-shots`: необязательный аргумент для few-shot методов
- `labels-file-path`: путь до файла с метками для задач классификации (по умолчанию `labels.json`)

#### Пример результата
```
{
    "bbh/boolean_expressions": "Your role is Computer Scientist. Evaluate the result of a random Boolean expression.",
    "bbh/object_counting": "Your role is Quantitative Analyst. Questions that involve enumerating objects and asking the model to count them.",
    "mnli": "Your role is Textual Analysis Specialist. In this task, you're given a pair of sentences, premise and hypothesis. Your job is to choose whether the two sentences clearly agree/disagree with each other, or if this cannot be determined.",
    "openbookqa": "Your role is Climate Scientist. Answer the following question: ",
    "ethos": "Your role is Content Moderation Specialist. Please decide if the following statement is a hate speech or not"
}
```

## Оценка методов

Оценка осуещствляется с помощью скрипта `prompts_scoring.py`. Внутри заводится `DefaultLLM` модель в обертке `ModelLoader` из файла `model_loader.py`.

#### Пример запуска
```
python prompts_scoring.py --input-file-path ./logs/fs_1.json --output-file-path ./logs/fs_1_results.json --generation-metric bleu --classification-metric accuracy --full --gen-only
```

#### Аргументы:
- `input-file-path`: путь до JSON файла с промптами для оценки. Ожидается структура {task: prompt}
- `output-file-path`: путь до файла, в который запишется результат. Структура результата: {task: {metric: {name: metric_name, score: score}, prompt: prompt}}
- `generation-metric`: метрика для генерации (по умолчанию meteor)
- `classification-metric`: метрика для классификации (по умолчанию f1)
- `full`: необязательный флаг для оценки на всем тестовом датасете 
- `gen-only`: необязательный флаг для оценки только на задачах генерации

#### Пример результата
```
{
    "bbh/object_counting": {
        "metric": {
            "name": "bleu",
            "score": 0.0
        },
        "prompt": "Questions that involve enumerating objects and asking the model to count them."
    }
}
```