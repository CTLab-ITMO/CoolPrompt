"""FeedbackModule for generating prompt improvement recommendations."""

import random
from typing import Any, List, Optional

from coolprompt.evaluator.evaluator import FailedExampleDetailed
from coolprompt.utils.parsing import extract_json, get_model_answer_extracted


FEEDBACK_PROMPT_TEMPLATE = """You are an expert prompt engineer.

The prompt was evaluated on benchmark task and failed on some examples. You will be given with a prompt and an example.

Prompt:
<prompt>
{prompt}
</prompt>

Failed task: 
<failed_task>
{instance}
</failed_task>

Model answer (raw): 
<model_answer>
{model_answer}
</model_answer>

Model answer (parsed):
<model_answer_parsed>
{model_answer_parsed}
</model_answer_parsed>

Metric value: {metric_value}

Сorrect answer:
<ground_truth>
{ground_truth}
</ground_truth>

Identify the core reasoning error pattern.

Give ONE general, universal recommendation to improve the prompt (no task-special details).

Format: Consice, max 20-25 words, starts with action verb. Output nothing but the actual recommendation. Avoid meta‑comments (e.g., "similar to…", "as before…") – the recommendation must stand alone.

Example: "Require step-by-step reasoning before classifying."

Recommendation:
"""

FILTER_RECOMMENDATIONS_PROMPT = """You have a list of recommendations for prompt improvement:

{recommendations}

TASK:
1. Group them into conceptual clusters (similar ideas).
2. For each cluster, **synthesize a single, new recommendation** that captures the essence of all items in that cluster. Do not just copy an existing one.
3. Rank clusters by size (largest first). If some clusters conflict - drop the less ones.
4. Output ONLY a JSON array of the synthesized recommendations, in rank order.

GOOD EXAMPLES:
Input: ["step-by-step", "break down calc", "don't show work", "format clearly"]
Correct output: ["Require detailed step-by-step reasoning with calculations", "Specify the desired output format explicitly"]
Why good:
- Captured main ideas of reasoning cluster into 1 strong rec
- Didn't loose cluster from "format clearly"
- Resolved conflict: "don't show work" is less frequent recommendation, so its cluster was dropped

BAD EXAMPLES:
Input: ["Focus on clarifying the output format requirements",
        "Add examples of expected responses to the prompt",
        "Make sure to specify exact sentiment labels",
        "Include examples to avoid confusion with similar labels",
        "Focus on tone analysis in the text",
        "Clarify what constitutes positive vs negative",
        "Add examples of positive responses",
        "Similar to previous - add more examples"]
Wrong output: ["Similar to previous - add more examples", "Add examples of positive responses", "Make sure to specify exact sentiment labels", "Focus on tone analysis in the text"]
Why bad:
- "Similar to previous" = meta-trash
- No synthesis of 6+ example recs into 1 strong rec, uses only existing recommendations
- Two different recommendations with a similiar intent: adding examples (duplicates)
"""


class FeedbackModule:
    """Generates recommendations for improving prompts based on failed examples."""

    def __init__(self, model: Any) -> None:
        self.model = model

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
