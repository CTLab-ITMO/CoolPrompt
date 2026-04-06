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
        + "\n\n{META_INFO_BLOCK}"
        + "User query:\n<user_query>\n{QUERY}\n</user_query>\n"
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
        """Returns the current value of the section (for recommendations — List[str])."""
        if name not in META_PROMPT_SECTIONS:
            raise ValueError(
                f"Unknown section: {name}. Expected: {META_PROMPT_SECTIONS}"
            )
        if name == "recommendations":
            return list(self.builder.config.recommendations)
        if name == "constraints":
            return list(self.builder.config.constraints)
        return self.builder.get_cached_section(name)

    def update_section(
        self,
        name: str,
        value: Union[str, List[str]],
    ) -> None:
        """Updates the section and rebuilds the meta-prompt."""
        if name not in META_PROMPT_SECTIONS:
            raise ValueError(
                f"Unknown section: {name}. Expected: {META_PROMPT_SECTIONS}"
            )
        if name == "recommendations":
            self.builder.config.recommendations = list(value)
        elif name == "constraints":
            self.builder.config.constraints = list(value)
        elif name == "output_format" and isinstance(value, str):
            self.builder.config.output_format_section = value
        else:
            raise ValueError(
                f"update_section for {name}: unsupported value type"
            )
        self.builder.rebuild_all_sections()
        self._rebuild_meta_prompt()

    def _rebuild_meta_prompt(self) -> None:
        self.meta_prompt = _build_full_meta_prompt_template(self.builder)

    def set_meta_prompt(self, meta_prompt: str) -> None:
        self.meta_prompt = meta_prompt

    def optimize(
        self,
        prompt: str,
        meta_info: Optional[dict[str, Any]] = None,
        n_prompts: int = 1,
    ) -> Union[str, List[str]]:
        query = self._format_meta_prompt(prompt, **(meta_info or {}))
        raw_result = get_model_answer_extracted(self.model, query, n=n_prompts)
        if n_prompts == 1:
            return self._process_model_output(raw_result)
        return [self._process_model_output(r) for r in raw_result]

    def _format_meta_prompt(self, prompt: str, **kwargs) -> str:
        if kwargs:
            meta_info_content = "\n".join(
                [f"{k}: {v}" for k, v in kwargs.items()]
            )
            meta_info_block = META_INFO_SECTION.format(
                meta_info_content=meta_info_content
            )
        else:
            meta_info_block = ""

        return self.meta_prompt.format(
            QUERY=prompt, META_INFO_BLOCK=meta_info_block
        )

    RESULT_PROMPT_TAGS = ("<result_prompt>", "</result_prompt>")

    def _process_model_output(self, output: Any) -> str:
        result = extract_answer(
            output,
            self.RESULT_PROMPT_TAGS,
            format_mismatch_label=output,
        )
        return result if isinstance(result, str) else str(result)
