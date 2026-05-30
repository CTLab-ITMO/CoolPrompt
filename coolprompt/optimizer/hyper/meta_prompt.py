"""Meta-prompt single-step optimizer built on the shared template builder."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union, override

from coolprompt.optimizer.autoprompting_method import (
    AutoPromptingMethod,
    BenchmarkContext,
)
from coolprompt.utils.parsing import extract_answer, get_model_answer_extracted
from coolprompt.utils.prompt_templates.hyper_templates import (
    META_INFO_SECTION,
    MetaPromptBuilder,
    MetaPromptConfig,
    Recommendation,
)
from coolprompt.utils.structured_schemas.optimizer.hyper import (
    ResultPromptResponse,
)


def _build_full_meta_prompt_template(builder: MetaPromptBuilder) -> str:
    """Append user-query and meta-info placeholders to the builder meta-prompt body."""
    body = builder.build_meta_prompt()
    return (
        body
        + "\n\nUser query:\n<user_query>\n{QUERY}\n</user_query>\n"
        + "{META_INFO_BLOCK}"
    )


class Optimizer(ABC):
    """Abstract base for optimizers that consume a LangChain-compatible ``model``."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @abstractmethod
    def optimize(self, *args: Any, **kwargs: Any) -> Any:
        """Run one optimization step; signature is defined by subclasses."""
        ...


class MetaPromptOptimizer(Optimizer):
    """Single-shot meta-prompt optimizer: one structured LLM call per ``optimize``."""

    def __init__(
        self,
        model: Any,
        config: Optional[MetaPromptConfig] = None,
        meta_prompt: Optional[str] = None,
        use_structured_output: bool = False,
    ) -> None:
        super().__init__(model)
        self.builder = MetaPromptBuilder(config)
        if meta_prompt is not None:
            self.meta_prompt = meta_prompt
        else:
            self.meta_prompt = _build_full_meta_prompt_template(self.builder)
        self.use_structured_output = use_structured_output

    def get_section(self, name: str) -> Any:
        """Return the current value stored for a named meta-prompt section."""
        return self.builder.get_section(name)

    def update_section(
        self,
        name: str,
        value: Union[str, List[str], List[Recommendation]],
    ) -> None:
        """Update a section and rebuild the cached full meta-prompt string."""
        self.builder.set_section(name, value)
        self._rebuild_meta_prompt()

    def _rebuild_meta_prompt(self) -> None:
        self.meta_prompt = _build_full_meta_prompt_template(self.builder)

    def set_meta_prompt(self, meta_prompt: str) -> None:
        """Replace the entire meta-prompt template string."""
        self.meta_prompt = meta_prompt

    def optimize(
        self,
        prompt: str,
        meta_info: Optional[dict[str, Any]] = None,
        n_prompts: int = 1,
    ) -> Union[str, List[str]]:
        """Generate improved prompt(s) via the meta-prompt + LLM path."""
        query = self._format_meta_prompt(prompt, **(meta_info or {}))
        if self.use_structured_output:
            structured = self.model.with_structured_output(
                ResultPromptResponse, method="json_schema"
            )
            if n_prompts == 1:
                return structured.invoke(query).result_prompt
            return [
                r.result_prompt
                for r in structured.batch([query] * n_prompts)
            ]
        raw_result = get_model_answer_extracted(self.model, query, n=n_prompts)
        if n_prompts == 1:
            return self._process_model_output(raw_result)
        return [self._process_model_output(r) for r in raw_result]

    def _format_meta_prompt(self, prompt: str, **kwargs: Any) -> str:
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
        result = extract_answer(
            output,
            self.RESULT_PROMPT_TAGS,
            format_mismatch_label=output,
        )
        return result if isinstance(result, str) else str(result)


class HyPERLightMethod(AutoPromptingMethod):
    """Benchmark wrapper for :class:`MetaPromptOptimizer` (single LLM meta-prompt step)."""

    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split=None,
        evaluator=None,
        problem_description=None,
        **kwargs,
    ):
        meta_prompt_context = kwargs.pop("meta_prompt_context", None)
        use_structured_output = kwargs.pop("use_structured_output", False)
        optimizer = MetaPromptOptimizer(
            model=model,
            use_structured_output=use_structured_output,
            **kwargs,
        )
        meta_info = meta_prompt_context.copy() if meta_prompt_context else {}
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
        mc = ctx.config.get("method", {})
        return self.optimize(
            ctx.model,
            start_prompt,
            problem_description=ctx.config.get("problem_description"),
            meta_prompt_context=meta if meta else None,
            use_structured_output=mc.get("use_structured_output", False),
        )

    def is_data_driven(self) -> bool:
        return False

    @property
    @override
    def name(self) -> str:
        return "hyper_light"
