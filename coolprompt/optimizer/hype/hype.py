from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from coolprompt.utils.parsing import extract_answer, get_model_answer_extracted
from coolprompt.utils.prompt_templates.hyper_templates import (
    HypeMetaPromptBuilder,
    HypeMetaPromptConfig,
    META_INFO_SECTION,
    META_PROMPT_SECTIONS,
    SECTION_CONSTRAINTS,
    SECTION_OUTPUT_FORMAT,
    SECTION_PROMPT_STRUCTURE,
    SECTION_RECOMMENDATIONS,
    SECTION_ROLE,
)


def _build_full_meta_prompt_template(builder: HypeMetaPromptBuilder) -> str:
    body = builder.build_meta_prompt()
    return (
        body
        + "\n\nUser query:\n<user_query>\n{QUERY}\n</user_query>\n"
        + "{META_INFO_BLOCK}"
    )


class Optimizer(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def optimize(self):
        pass


class HyPEOptimizer(Optimizer):
    def __init__(
        self,
        model,
        config: Optional[HypeMetaPromptConfig] = None,
        meta_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(model)
        self.builder = HypeMetaPromptBuilder(config)
        if meta_prompt is not None:
            self.meta_prompt = meta_prompt
        else:
            self.meta_prompt = _build_full_meta_prompt_template(self.builder)

    def get_section(self, name: str) -> Any:
        """
        Return the current value of a meta-prompt section.

        Args:
            name: Section name (one of META_PROMPT_SECTIONS).

        Returns:
            List[str] for 'recommendations'/'constraints', str for others.
        """
        return self.builder.get_section(name)

    def update_section(
        self,
        name: str,
        value: Union[str, List[str]],
    ) -> None:
        """
        Update a section value and rebuild the meta-prompt.

        Args:
            name: Section name (one of META_PROMPT_SECTIONS).
            value: New value (List[str] for recommendations/constraints, str for others).
        """
        self.builder.set_section(name, value)
        self._rebuild_meta_prompt()

    def _rebuild_meta_prompt(self) -> None:
        """Rebuild the full meta-prompt from the current builder state."""
        self.meta_prompt = _build_full_meta_prompt_template(self.builder)

    def set_meta_prompt(self, meta_prompt: str) -> None:
        """Override the meta-prompt with a custom string."""
        self.meta_prompt = meta_prompt

    def optimize(
        self,
        prompt: str,
        meta_info: Optional[dict[str, Any]] = None,
        n_prompts: int = 1,
    ) -> Union[str, List[str]]:
        """
        Generate an optimized prompt using the HyPE method.

        Args:
            prompt: The user query/prompt to optimize.
            meta_info: Optional dict of task metadata (e.g., problem_description).
            n_prompts: Number of prompt variants to generate.

        Returns:
            Single optimized prompt string if n_prompts=1, else list of prompts.
        """
        query = self._format_meta_prompt(prompt, **(meta_info or {}))
        raw_result = get_model_answer_extracted(self.model, query, n=n_prompts)
        if n_prompts == 1:
            return self._process_model_output(raw_result)
        return [self._process_model_output(r) for r in raw_result]

    def _format_meta_prompt(self, prompt: str, **kwargs) -> str:
        if kwargs:
            meta_info_content = "\n".join([f"{k}: {v}" for k, v in kwargs.items()])
            meta_info_block = META_INFO_SECTION.format(
                meta_info_content=meta_info_content
            )
        else:
            meta_info_block = ""

        return self.meta_prompt.format(QUERY=prompt, META_INFO_BLOCK=meta_info_block)

    RESULT_PROMPT_TAGS = ("<result_prompt>", "</result_prompt>")

    def _process_model_output(self, output: Any) -> str:
        """Extract the result prompt from model output."""
        result = extract_answer(
            output,
            self.RESULT_PROMPT_TAGS,
            format_mismatch_label=output,
        )
        return result if isinstance(result, str) else str(result)


def hype_optimizer(
    model,
    prompt: str,
    problem_description: Optional[str] = None,
) -> str:
    """
    Optimize a prompt using the HyPE method.

    Args:
        model: Language model to use for optimization.
        prompt: The initial prompt to optimize.
        problem_description: Optional description of the problem/task.

    Returns:
        The optimized prompt string.
    """
    optimizer = HyPEOptimizer(model)
    meta_info = {}
    if problem_description:
        meta_info["problem_description"] = problem_description
    return optimizer.optimize(prompt, meta_info=meta_info if meta_info else None)
