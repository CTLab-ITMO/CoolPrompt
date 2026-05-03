"""FeedbackModule for generating prompt improvement recommendations."""

import random
from typing import Any, List, Optional

from coolprompt.evaluator.evaluator import FailedExampleDetailed
from coolprompt.utils.parsing import extract_json, get_model_answer_extracted
from coolprompt.utils.prompt_templates.hyper_templates import FEEDBACK_PROMPT_TEMPLATE, FILTER_RECOMMENDATIONS_PROMPT
from coolprompt.optimizer.structured_schemas.hype import (
    RecommendationResponse,
    FilteredRecommendationsResponse,
)


class FeedbackModule:
    """Generates recommendations for improving prompts based on failed examples."""

    def __init__(self, model: Any, use_structured_output: bool = False) -> None:
        self.model = model
        self.use_structured_output = use_structured_output

    def generate_recommendation(
        self,
        prompt: str,
        instance: str,
        model_answer: str,
        model_answer_parsed: Optional[str] = None,
        metric_value: float | int = 0.0,
        ground_truth: str | int = "",
    ) -> str:
        """Generate a single recommendation for a failed example.

        Args:
            prompt: The original prompt that was used.
            instance: The task instance (input/question).
            model_answer: The model's answer (incorrect, raw).
            model_answer_parsed: The model's parsed answer (for metric calculation).
            metric_value: The metric value for this answer.
            ground_truth: The correct answer.

        Returns:
            A recommendation string for improving the prompt.
        """
        formatted_prompt = FEEDBACK_PROMPT_TEMPLATE.format(
            prompt=prompt,
            instance=instance,
            model_answer=model_answer,
            model_answer_parsed=model_answer_parsed or "",
            metric_value=metric_value,
            ground_truth=ground_truth,
        )
        if self.use_structured_output:
            structured = self.model.with_structured_output(
                RecommendationResponse, method="json_schema"
            )
            return structured.invoke(formatted_prompt).recommendation
        result = get_model_answer_extracted(self.model, formatted_prompt)
        return self._process_output(result)

    def generate_recommendations(
        self,
        prompt: str,
        failed_examples: List[FailedExampleDetailed],
    ) -> List[str]:
        """Generate recommendations for all failed examples.

        Args:
            prompt: The original prompt that was used.
            failed_examples: List of failed examples.

        Returns:
            List of recommendation strings.
        """
        return [
            self.generate_recommendation(
                prompt=prompt,
                instance=fe.instance,
                model_answer=fe.assistant_answer,
                model_answer_parsed=fe.model_answer_parsed,
                metric_value=fe.metric_value,
                ground_truth=fe.ground_truth,
            )
            for fe in failed_examples
        ]

    def filter_recommendations(self, recommendations: List[str]) -> List[str]:
        """Filter and deduplicate recommendations using LLM.

        Args:
            recommendations: List of recommendation strings.

        Returns:
            Deduplicated and filtered list of recommendations.
        """
        if not recommendations:
            return []

        formatted_recs = "\n".join(
            f"{i + 1}. {rec}" for i, rec in enumerate(recommendations)
        )
        prompt = FILTER_RECOMMENDATIONS_PROMPT.format(
            recommendations=formatted_recs
        )
        if self.use_structured_output:
            try:
                structured = self.model.with_structured_output(
                    FilteredRecommendationsResponse, method="json_schema"
                )
                return list(structured.invoke(prompt).recommendations)
            except Exception:
                return random.sample(
                    recommendations, min(3, len(recommendations))
                )

        result = get_model_answer_extracted(self.model, prompt)
        try:
            data = extract_json(result)
            if data and isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass

        return random.sample(recommendations, min(3, len(recommendations)))

    def _process_output(self, output: Any) -> str:
        """Process model output to extract recommendation."""
        return output if isinstance(output, str) else str(output)
