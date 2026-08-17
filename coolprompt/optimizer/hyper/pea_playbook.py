"""HyPER Light variant using the MR.PEA abstraction knowledge format."""

from __future__ import annotations

from typing import Any, override

from coolprompt.optimizer.hyper.playbook import HyPERLightPlaybookMethod
from coolprompt.utils.parsing import extract_json, get_model_answer_extracted


PEA_PLAYBOOK_SYSTEM_PROMPT = """You are a Meta-Reasoning Specialist (Abstraction).
Your role is to create or refine reusable, task-agnostic knowledge for the given task.

Objectives:
1. If no existing knowledge is provided, generate abstract strategies, principles,
   evaluation criteria, and knowledge-gap hypotheses.
2. If existing knowledge is provided, improve it by clarifying language, removing
   redundancy, adding missing insights, improving organization, and making items
   more actionable.
3. Set need_change to false only when the existing knowledge is truly optimal.

OUTPUT CONTRACT (STRICT JSON ONLY):
If knowledge should change, return:
{
  "need_change": true,
  "strategies": ["..."],
  "principles": ["..."],
  "evaluation_criteria": ["..."],
  "gap_hypotheses": ["..."],
  "change_rationale": "..."
}

If no change is needed, return:
{
  "need_change": false,
  "change_rationale": "Existing knowledge is already optimal"
}

Rules:
- Return only valid minified JSON. Do not use Markdown or extra text.
- Prefer need_change=true whenever any improvement is possible.
- Return 2-8 concise strategies, principles, and evaluation criteria.
- Return 1-3 concise gap hypotheses.
- Keep each item at most 20 words.
- Do not include task-specific examples.
"""


PEA_PLAYBOOK_USER_PROMPT = """Analyze the task and produce abstract strategies.

Task Description: {task_description}
Sample Question: {sample_question}
Existing Knowledge: {latest_knowledge}
"""


class HyPERLightPEAPlaybookMethod(HyPERLightPlaybookMethod):
    """HyPER Light with an MR.PEA-style abstraction knowledge step.

    The generated object is intentionally the flat MR.PEA knowledge structure:
    ``strategies``, ``principles``, ``evaluation_criteria``,
    ``gap_hypotheses``, and ``change_rationale``. It is then passed as the
    ``playbook`` meta-information to the regular HyPER Light optimizer.
    """

    def _generate_playbook(self, model: Any, initial_prompt: str) -> dict[str, Any]:
        """Generate MR.PEA knowledge from the initial prompt."""
        request = (
            PEA_PLAYBOOK_SYSTEM_PROMPT
            + "\n\n"
            + PEA_PLAYBOOK_USER_PROMPT.format(
                task_description=initial_prompt,
                sample_question="",
                latest_knowledge="",
            )
        )
        raw_result = get_model_answer_extracted(model, request)
        parsed = extract_json(raw_result)
        if isinstance(parsed, dict):
            return parsed
        return {"raw_playbook": raw_result}

    @property
    @override
    def name(self) -> str:
        return "hyper_light_pea_playbook"
