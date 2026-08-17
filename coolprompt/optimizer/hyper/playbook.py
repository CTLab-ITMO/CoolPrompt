"""HyPER Light variant that enriches the meta-prompt with a generated playbook."""

from __future__ import annotations

from typing import Any, override

from coolprompt.optimizer.hyper.meta_prompt import (
    HyPERLightMethod,
    MetaPromptOptimizer,
)
from coolprompt.utils.parsing import extract_json, get_model_answer_extracted


PLAYBOOK_GENERATION_PROMPT = """You are an expert prompt strategist.

Given the starting prompt below, create a reusable playbook for solving the task.
The playbook is a structured collection of strategies, decision rules, execution
steps, quality checks, and common failure modes that an assistant can use when
answering requests of this type.

Base the playbook on the task described by the starting prompt. Do not solve a
particular user instance, do not invent missing task details, and preserve the
language and constraints expressed by the starting prompt.

Return ONLY one valid JSON object with this structure:
{{
  "task_summary": "short summary of the task",
  "strategies": [
    {{
      "name": "strategy name",
      "when_to_use": "when this strategy applies",
      "steps": ["concrete step 1", "concrete step 2"],
      "checks": ["quality check 1"]
    }}
  ],
  "decision_rules": ["rule for choosing or applying a strategy"],
  "common_failure_modes": ["failure mode and how to avoid it"],
  "output_contract": ["requirements for the final answer"]
}}

<start_prompt>
{initial_prompt}
</start_prompt>
"""


class HyPERLightPlaybookMethod(HyPERLightMethod):
    """HyPER Light with a preliminary playbook-generation LLM call.

    The method first derives a reusable task playbook from ``initial_prompt``.
    It then passes that playbook as ``meta_info`` to the regular HyPER Light
    meta-prompt optimizer, which generates the final prompt.
    """

    def _generate_playbook(self, model: Any, initial_prompt: str) -> dict[str, Any]:
        """Generate and parse a playbook, retaining raw output on parse failure."""
        request = PLAYBOOK_GENERATION_PROMPT.format(initial_prompt=initial_prompt)
        raw_result = get_model_answer_extracted(model, request)
        parsed = extract_json(raw_result)
        if isinstance(parsed, dict):
            return parsed
        return {"raw_playbook": raw_result}

    @override
    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split=None,
        evaluator=None,
        problem_description=None,
        **kwargs,
    ):
        """Generate a playbook, then run the standard HyPER Light step."""
        telemetry_callback = kwargs.pop("telemetry_callback", None)
        hyper_meta_info = kwargs.pop("hyper_meta_info", None)
        hyper_meta_prompt = kwargs.pop("hyper_meta_prompt", None)
        use_structured_output = kwargs.pop("use_structured_output", False)
        playbook_prompt = kwargs.pop("playbook_prompt", None)

        if playbook_prompt is None:
            playbook = self._generate_playbook(model, initial_prompt)
        else:
            playbook_request = playbook_prompt.format(initial_prompt=initial_prompt)
            raw_playbook = get_model_answer_extracted(model, playbook_request)
            parsed_playbook = extract_json(raw_playbook)
            playbook = (
                parsed_playbook
                if isinstance(parsed_playbook, dict)
                else {"raw_playbook": raw_playbook}
            )

        optimizer_kwargs = {
            "model": model,
            "use_structured_output": use_structured_output,
            **kwargs,
        }
        if hyper_meta_prompt is not None:
            optimizer_kwargs["meta_prompt"] = hyper_meta_prompt
        optimizer = MetaPromptOptimizer(**optimizer_kwargs)

        meta_info = hyper_meta_info.copy() if hyper_meta_info else {}
        meta_info["playbook"] = playbook
        if "problem_description" not in meta_info:
            meta_info["problem_description"] = problem_description

        final_prompt = optimizer.optimize(
            prompt=initial_prompt,
            meta_info=meta_info,
            n_prompts=1,
        )

        if telemetry_callback is not None:
            telemetry_callback(
                iteration=1,
                best_score=0.0,
                best_prompt=final_prompt,
            )

        return final_prompt

    @property
    @override
    def name(self) -> str:
        return "hyper_light_playbook"
