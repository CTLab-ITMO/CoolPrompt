"""Integration tests for full telemetry pipeline across all optimizers.

These tests verify that telemetry tracking works correctly end-to-end
with OpenRouter API calls. Requires OPENROUTER_API_KEY environment variable.

Example:
    export OPENROUTER_API_KEY=sk-or-v1-...
    pytest test/coolprompt/test_telemetry_full_pipeline.py -v
"""
from __future__ import annotations

import os
import json
import pytest
from typing import Any, Dict
from pathlib import Path
from langchain_openai import ChatOpenAI
from coolprompt.assistant import PromptTuner
from coolprompt.utils.telemetry import OptimizationTelemetry


def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from environment."""
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY or OPENAI_API_KEY not set")
    return api_key


def create_chat_model() -> ChatOpenAI:
    """Create a ChatOpenAI model configured for OpenRouter."""
    api_key = get_openrouter_api_key()
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.environ.get("INTEGRATION_LLM_MODEL", "openai/gpt-4o-mini")
    
    return ChatOpenAI(
        model=model,
        temperature=0.3,
        max_completion_tokens=2048,
        max_retries=3,
        base_url=base_url,
        api_key=api_key,
    )


@pytest.fixture
def chat_model():
    """Fixture providing a configured ChatOpenAI model."""
    return create_chat_model()


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture providing a temporary directory for telemetry outputs."""
    output_dir = tmp_path / "telemetry_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


class TestTelemetryHyperLight:
    """Test telemetry tracking for HyPER Light optimizer."""
    
    def test_hyper_light_with_telemetry(self, chat_model, temp_output_dir):
        """Test that HyPER Light correctly tracks and exports telemetry."""
        tuner = PromptTuner(
            target_model=chat_model,
            system_model=chat_model,
            logs_dir=temp_output_dir / "logs"
        )
        
        telemetry_path = str(temp_output_dir / "hyper_light_telemetry")
        
        final_prompt = tuner.run(
            start_prompt="Classify the sentiment of this text as POSITIVE or NEGATIVE.",
            task="classification",
            dataset=[
                "I love this product, it's amazing!",
                "This is terrible, worst purchase ever.",
                "Great quality and fast shipping.",
                "Broken on arrival, very disappointed.",
            ],
            target=["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE"],
            method="hyper_light",
            metric="f1",
            problem_description="Binary sentiment classification",
            validation_size=0.5,
            enable_telemetry=True,
            export_telemetry=True,
            telemetry_format="both",
            telemetry_path=telemetry_path,
            verbose=1,
        )
        
        assert final_prompt is not None
        assert len(final_prompt) > 0
        
        assert hasattr(tuner, "telemetry_report")
        assert tuner.telemetry_report is not None
        
        report = tuner.telemetry_report
        assert isinstance(report, OptimizationTelemetry)
        assert report.method_name == "hyper_light"
        assert report.task_type == "classification"
        
        assert report.total_api_wait_sec >= 0
        assert report.total_tokens >= 0
        assert report.total_cost_usd >= 0
        
        assert len(report.trajectory) >= 1
        assert report.trajectory[0].iteration == 1
        
        json_path = Path(f"{telemetry_path}.json")
        csv_path = Path(f"{telemetry_path}_trajectory.csv")
        
        assert json_path.exists(), f"JSON telemetry not found at {json_path}"
        assert csv_path.exists(), f"CSV telemetry not found at {csv_path}"
        
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        assert json_data["method_name"] == "hyper_light"
        assert "total_cost_usd" in json_data
        assert "trajectory" in json_data
        assert len(json_data["trajectory"]) >= 1


class TestTelemetryHyper:
    """Test telemetry tracking for HyPER optimizer (multi-iteration)."""
    
    def test_hyper_with_telemetry(self, chat_model, temp_output_dir):
        """Test that HyPER correctly tracks telemetry across iterations."""
        tuner = PromptTuner(
            target_model=chat_model,
            system_model=chat_model,
            logs_dir=temp_output_dir / "logs"
        )
        
        telemetry_path = str(temp_output_dir / "hyper_telemetry")
        
        final_prompt = tuner.run(
            start_prompt="Solve this math problem step by step.",
            task="generation",
            dataset=[
                "If a train travels 120 miles in 2 hours, what is its average speed?",
                "A rectangle has length 8 and width 5. What is its area?",
                "What is 15% of 200?",
            ],
            target=["60 mph", "40", "30"],
            method="hyper",
            metric="em",
            problem_description="Math word problem solving",
            validation_size=0.5,
            enable_telemetry=True,
            export_telemetry=True,
            telemetry_format="json",
            telemetry_path=telemetry_path,
            verbose=1,
            n_iterations=2,
            n_candidates=2,
            mini_batch_size=2,
            patience=None,
        )
        
        assert final_prompt is not None
        assert len(final_prompt) > 0
        
        assert hasattr(tuner, "telemetry_report")
        report = tuner.telemetry_report
        
        assert report.method_name == "hyper"
        assert report.task_type == "generation"
        
        assert len(report.trajectory) >= 2, \
            f"Expected at least 2 iterations, got {len(report.trajectory)}"
        
        for i, snapshot in enumerate(report.trajectory):
            assert snapshot.iteration == i + 1
            assert snapshot.best_score is not None
            assert snapshot.best_prompt_length > 0
        
        if len(report.trajectory) > 1:
            for i in range(1, len(report.trajectory)):
                assert report.trajectory[i].cumulative_tokens_in >= \
                       report.trajectory[i-1].cumulative_tokens_in
                assert report.trajectory[i].cumulative_total_cost >= \
                       report.trajectory[i-1].cumulative_total_cost
        
        json_path = Path(f"{telemetry_path}.json")
        assert json_path.exists()
        
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        assert len(json_data["trajectory"]) >= 2


class TestTelemetryCompress:
    """Test telemetry tracking for Compressor optimizer."""
    
    def test_compress_with_telemetry(self, chat_model, temp_output_dir):
        """Test that Compressor correctly tracks telemetry."""
        tuner = PromptTuner(
            target_model=chat_model,
            system_model=chat_model,
            logs_dir=temp_output_dir / "logs"
        )
        
        telemetry_path = str(temp_output_dir / "compress_telemetry")
        
        long_prompt = """You are an expert assistant. Your task is to analyze the provided text 
        carefully and comprehensively. You should consider all aspects of the text including 
        its structure, content, tone, and purpose. After your analysis, provide a detailed 
        summary that captures the main points, key arguments, and important details. 
        Your response should be well-organized and easy to understand."""
        
        final_prompt = tuner.run(
            start_prompt=long_prompt,
            task="generation",
            method="compress",
            enable_telemetry=True,
            export_telemetry=True,
            telemetry_format="both",
            telemetry_path=telemetry_path,
            verbose=1,
        )
        
        assert final_prompt is not None
        assert len(final_prompt) < len(long_prompt)
        
        assert hasattr(tuner, "telemetry_report")
        report = tuner.telemetry_report
        
        assert report.method_name == "compress"
        assert len(report.trajectory) == 1
        
        assert report.trajectory[0].best_score == 0.0
        
        assert Path(f"{telemetry_path}.json").exists()
        assert Path(f"{telemetry_path}_trajectory.csv").exists()


class TestTelemetryReflective:
    """Test telemetry tracking for ReflectivePrompt optimizer."""
    
    def test_reflective_with_telemetry(self, chat_model, temp_output_dir):
        """Test that ReflectivePrompt correctly tracks telemetry."""
        tuner = PromptTuner(
            target_model=chat_model,
            system_model=chat_model,
            logs_dir=temp_output_dir / "logs"
        )
        
        telemetry_path = str(temp_output_dir / "reflective_telemetry")
        
        final_prompt = tuner.run(
            start_prompt="Answer the question based on the context.",
            task="generation",
            dataset=[
                "Context: The sky is blue. Question: What color is the sky?",
                "Context: Water boils at 100°C. Question: At what temperature does water boil?",
                "Context: Paris is the capital of France. Question: What is the capital of France?",
            ],
            target=["Blue", "100°C", "Paris"],
            method="reflective",
            metric="em",
            problem_description="Question answering from context",
            validation_size=0.5,
            enable_telemetry=True,
            export_telemetry=True,
            telemetry_format="both",
            telemetry_path=telemetry_path,
            verbose=1,
            population_size=3,
            num_epochs=2,
        )
        
        assert final_prompt is not None
        assert len(final_prompt) > 0
        
        assert hasattr(tuner, "telemetry_report")
        report = tuner.telemetry_report
        
        assert report.method_name == "reflective"
        assert len(report.trajectory) >= 2
        
        csv_path = Path(f"{telemetry_path}_trajectory.csv")
        assert csv_path.exists()
        
        import csv
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) >= 2
        assert "iteration" in rows[0]
        assert "best_score" in rows[0]
        assert "cumulative_tokens_in" in rows[0]


class TestTelemetryDistill:
    """Test telemetry tracking for DistillPrompt optimizer."""
    
    def test_distill_with_telemetry(self, chat_model, temp_output_dir):
        """Test that DistillPrompt correctly tracks telemetry."""
        tuner = PromptTuner(
            target_model=chat_model,
            system_model=chat_model,
            logs_dir=temp_output_dir / "logs"
        )
        
        telemetry_path = str(temp_output_dir / "distill_telemetry")
        
        final_prompt = tuner.run(
            start_prompt="Summarize the text.",
            task="generation",
            dataset=[
                "The quick brown fox jumps over the lazy dog.",
                "Artificial intelligence is transforming many industries.",
                "Climate change poses significant challenges to our planet.",
            ],
            target=["Fox jumps over dog", "AI transforms industries", "Climate change challenges"],
            method="distill",
            metric="bertscore",
            problem_description="Text summarization",
            validation_size=0.5,
            enable_telemetry=True,
            export_telemetry=True,
            telemetry_path=telemetry_path,
            verbose=1,
            num_epochs=2,
        )
        
        assert final_prompt is not None
        
        assert hasattr(tuner, "telemetry_report")
        report = tuner.telemetry_report
        
        assert report.method_name == "distill"
        assert len(report.trajectory) >= 2


class TestTelemetryReGPS:
    """Test telemetry tracking for ReGPS optimizer."""
    
    def test_regps_with_telemetry(self, chat_model, temp_output_dir):
        """Test that ReGPS correctly tracks telemetry."""
        tuner = PromptTuner(
            target_model=chat_model,
            system_model=chat_model,
            logs_dir=temp_output_dir / "logs"
        )
        
        telemetry_path = str(temp_output_dir / "regps_telemetry")
        
        final_prompt = tuner.run(
            start_prompt="Classify the text.",
            task="classification",
            dataset=[
                "This movie was fantastic!",
                "I hated this film.",
                "The acting was superb.",
            ],
            target=["POSITIVE", "NEGATIVE", "POSITIVE"],
            method="regps",
            metric="f1",
            problem_description="Sentiment classification",
            validation_size=0.5,
            enable_telemetry=True,
            export_telemetry=True,
            telemetry_path=telemetry_path,
            verbose=1,
            population_size=3,
            num_epochs=2,
            bad_examples_number=2,
        )
        
        assert final_prompt is not None
        
        assert hasattr(tuner, "telemetry_report")
        report = tuner.telemetry_report
        
        assert report.method_name == "regps"
        assert len(report.trajectory) >= 2
