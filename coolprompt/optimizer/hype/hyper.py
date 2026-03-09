from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from coolprompt.utils.parsing import extract_answer, get_model_answer_extracted
from coolprompt.utils.prompt_templates.hyper_templates import (
    HypeMetaPromptBuilder,
    HypeMetaPromptConfig,
    META_INFO_SECTION,
    META_PROMPT_SECTIONS,
)


def _build_full_meta_prompt_template(builder: HypeMetaPromptBuilder) -> str:
    body = builder.build_meta_prompt()
    return (
        body
        + "\n\nUser query: {QUERY}\n"
        + META_INFO_SECTION.format(meta_info_content="{META_INFO}")
    )


class Optimizer(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def optimize(self):
        pass


class HyPEOptimizer(Optimizer):
    def __init__(
        self, model, config: Optional[HypeMetaPromptConfig] = None
    ) -> None:
        super().__init__(model)
        self.builder = HypeMetaPromptBuilder(config)
        self.meta_prompt = _build_full_meta_prompt_template(self.builder)

    def get_section(self, name: str) -> Any:
        """Возвращает текущее значение секции (для recommendations — List[str])."""
        if name not in META_PROMPT_SECTIONS:
            raise ValueError(f"Unknown section: {name}. Expected: {META_PROMPT_SECTIONS}")
        if name == "recommendations":
            return list(self.builder.config.recommendations)
        if name == "constraints":
            return list(self.builder.config.constraints)
        if name == "role":
            return self.builder.build_role_section()
        if name == "prompt_structure":
            return self.builder.build_prompt_structure_section()
        if name == "output_format":
            return self.builder.build_output_format_section()
        return None

    def update_section(
        self,
        name: str,
        value: Union[str, List[str]],
    ) -> None:
        """Обновляет секцию и пересобирает мета-промпт."""
        if name not in META_PROMPT_SECTIONS:
            raise ValueError(f"Unknown section: {name}. Expected: {META_PROMPT_SECTIONS}")
        if name == "recommendations":
            self.builder.config.recommendations = list(value)
        elif name == "constraints":
            self.builder.config.constraints = list(value)
        elif name == "output_format" and isinstance(value, str):
            self.builder.config.output_format_section = value
        else:
            raise ValueError(f"update_section for {name}: unsupported value type")
        self._rebuild_meta_prompt()

    def _rebuild_meta_prompt(self) -> None:
        self.meta_prompt = _build_full_meta_prompt_template(self.builder)

    def set_meta_prompt(self, meta_prompt: str) -> None:
        self.meta_prompt = meta_prompt

    def optimize(
        self, prompt: str, meta_info: Optional[dict[str, Any]] = None
    ) -> str:
        query = self._format_meta_prompt(prompt, **(meta_info or {}))
        raw_result = get_model_answer_extracted(self.model, query)
        result = self._process_model_output(raw_result)
        return result

    def _format_meta_prompt(self, prompt: str, **kwargs) -> str:
        meta_info_content = (
            "\n".join([f"{k}: {v}" for k, v in kwargs.items()])
            if kwargs
            else ""
        )
        result = self.meta_prompt.format(
            QUERY=prompt, META_INFO=meta_info_content
        )
        return result

    RESULT_PROMPT_TAGS = ("<result_prompt>", "</result_prompt>")

    def _process_model_output(self, output: Any) -> str:
        result = extract_answer(
            output,
            self.RESULT_PROMPT_TAGS,
            format_mismatch_label=output,
        )
        return result if isinstance(result, str) else str(result)

