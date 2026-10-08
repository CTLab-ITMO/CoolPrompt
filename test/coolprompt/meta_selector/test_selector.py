import csv

from coolprompt.meta_selector import APOMetaSelector


class _JsonModel:
    def __init__(self, *responses):
        self._responses = iter(responses)

    def invoke(self, prompt):
        return next(self._responses)


def _write_metadata(path, rows):
    fields = [
        "method",
        "model",
        "dataset",
        "description",
        "domain",
        "task_type",
        "split",
        "metric",
        "metric_value",
        "cost_usd",
        "time_sec",
        "llm_calls",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _row(method, quality, cost, runtime):
    return {
        "method": method,
        "model": "openai/gpt-4o-mini",
        "dataset": "benchmark",
        "description": "Summarize a news document.",
        "domain": "news",
        "task_type": "generation",
        "split": "150/100/300",
        "metric": "BERTScore F1",
        "metric_value": quality,
        "cost_usd": cost,
        "time_sec": runtime,
        "llm_calls": "10",
    }


def test_selector_ranks_quality_cost_time_and_maps_sapo(tmp_path):
    table = tmp_path / "metadata.csv"
    _write_metadata(
        table,
        [
            _row("SAPO", "0.90", "0.10", "10"),
            _row("HyPER", "0.80", "0.20", "20"),
            _row("RIDER", "0.70", "0.30", "30"),
        ],
    )
    selector = APOMetaSelector(table)
    model = _JsonModel(
        '{"description":"Summarize news", "domain":"news", "task_type":"generation"}',
        '{"top_datasets":["benchmark"]}',
    )

    result = selector.select(model, "Summarize", "generation")

    assert result.recommended_method == "SAPO"
    assert result.selected_method == "sapo"
    assert result.fallback_reason is None
    assert result.candidates[0].final_score > result.candidates[-1].final_score


def test_selector_penalizes_missing_cost_and_time():
    assert APOMetaSelector._rank_scores([0.9, None], True) == [1.0, 0.0]
    assert APOMetaSelector._rank_scores([0.1, None], False) == [1.0, 0.0]


def test_selector_uses_custom_table_and_json_fallback(tmp_path):
    table = tmp_path / "metadata.csv"
    _write_metadata(table, [_row("RIDER", "0.90", "0.20", "10")])
    selector = APOMetaSelector(table)

    result = selector.select(object(), "Summarize the text", "generation")

    assert result.recommended_method == "RIDER"
    assert result.selected_method == "rider"
    assert result.similar_datasets == ["benchmark"]


def test_no_dataset_result_uses_hyper_light():
    result = APOMetaSelector.no_dataset_result()

    assert result.selected_method == "hyper_light"
    assert "No dataset" in result.fallback_reason
