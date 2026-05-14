from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union, override

from coolprompt.optimizer.autoprompting_method import (
    AutoPromptingMethod,
    BenchmarkContext,
)
from coolprompt.utils.parsing import extract_answer, get_model_answer_extracted
from coolprompt.utils.prompt_templates.hyper_templates import (
    HypeMetaPromptBuilder,
    HypeMetaPromptConfig,
    META_INFO_SECTION,
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


class HyPEMethod(AutoPromptingMethod):
    """HyPE (Hypothetical Prompt Enhancer) for ``PromptTuner`` / benchmarks."""

    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split=None,
        evaluator=None,
        problem_description=None,
        **kwargs,
    ):
        hype_meta_info = kwargs.pop("hype_meta_info", None)
        optimizer = HyPEOptimizer(model=model, **kwargs)
        meta_info = hype_meta_info.copy() if hype_meta_info else {}
        if "problem_description" not in meta_info:
            meta_info["problem_description"] = problem_description
        return optimizer.optimize(
            prompt=initial_prompt,
            meta_info=meta_info if meta_info else None,
            n_prompts=1,
        )

    def run_configured_benchmark(
        self,
        ctx: BenchmarkContext,
        start_prompt: str,
    ) -> str:
        meta = dict(ctx.config.get("meta_info", {}))
        return self.optimize(
            ctx.model,
            start_prompt,
            problem_description=ctx.config.get("problem_description"),
            hype_meta_info=meta if meta else None,
        )

    def is_data_driven(self):
        return False

    @property
    @override
    def name(self):
        return "hype"
