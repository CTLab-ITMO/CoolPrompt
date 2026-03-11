from dataclasses import dataclass, field
from typing import List, Optional


TARGET_PROMPT_FORMS = ["hypothetical ", "instructional "]


SIMPLE_HYPOTHETICAL_PROMPT = "Write a {target_prompt_form}prompt that will solve the user query effectively."

META_INFO_SECTION = "Task-related meta-information:\n<meta_info>\n{meta_info_content}\n</meta_info>\n"

META_PROMPT_SECTIONS = (
    "role",
    "prompt_structure",
    "recommendations",
    "constraints",
    "output_format",
)


@dataclass
class PromptSectionSpec:
    name: str
    description: str


@dataclass
class HypeMetaPromptConfig:
    target_prompt_form: str = "hypothetical instructional "
    require_markdown_prompt: bool = True
    include_role: bool = True
    section_names: List[str] = field(
        default_factory=lambda: [
            "Role",
            "Task context",
            "Instructions",
            "Output requirements",
        ]
    )
    section_specs: List[PromptSectionSpec] = field(
        default_factory=lambda: [
            PromptSectionSpec(
                name="Role",
                description=(
                    "Briefly define the assistant's role and expertise "
                    "relevant to the user query."
                ),
            ),
            PromptSectionSpec(
                name="Task context",
                description=(
                    "Summarize the user's query and any provided meta-information, "
                    "keeping all important constraints and domain details."
                ),
            ),
            PromptSectionSpec(
                name="Instructions",
                description=(
                    "Main part - instructions the assistant must follow "
                    "to solve the user's query while respecting constraints."
                ),
            ),
            PromptSectionSpec(
                name="Output requirements",
                description=(
                    "Clearly specify the desired tone "
                    "and the required level of detail for the assistant's answer. "
                    "If the user explicitly requests a particular output format or provides "
                    "an example response, restate that format and include the example verbatim, "
                    "without inventing any additional formatting or examples. Do not introduce any output format or examples that the user did not mention."
                ),
            ),
        ]
    )
    constraints: List[str] = field(
        default_factory=lambda: [
            "Preserve the language of the user's query.",
            "Preserve all code snippets, inline code, technical terms and special formatting.",
            "Do not remove or alter any explicit formatting instructions from the user.",
            "Do not change numerical values, units, or identifiers.",
        ]
    )
    recommendations: List[str] = field(default_factory=list)
    output_format_section: Optional[str] = None
    _cached_sections: dict = field(default_factory=dict, repr=False)


class HypeMetaPromptBuilder:
    ROLE_LINE = "You are an expert prompt engineer.\n"
    TASK_SECTION_TEMPLATE = (
        "Your only task is to write a {target_prompt_form}prompt that will "
        "solve the user query as effectively as possible.\n"
        "Do not answer the user query directly; only produce the new prompt.\n\n"
    )

    PROMPT_STRUCTURE_SECTION_TEMPLATE = (
        "### STRUCTURE OF THE PROMPT YOU MUST PRODUCE\n"
        "The prompt you write MUST be structured into the following sections, "
        "in this exact order, and each section must follow its guidelines:\n"
        "{sections_with_guidelines}\n\n"
    )

    CONSTRAINTS_SECTION_TEMPLATE = (
        "### HARD CONSTRAINTS\n{constraints_list}\n\n"
    )

    RECOMMENDATIONS_SECTION_TEMPLATE = (
        "### RECOMMENDATIONS\n"
        "Use these recommendations for writing the new prompt, "
        "based on analysis of previous generations:\n"
        "{recommendations_list}\n\n"
    )

    BASE_OUTPUT_FORMAT_SECTION = (
        "### YOUR RESPONSE FORMAT\n"
        "Return ONLY the resulting prompt, wrapped in the following XML tags:\n"
        "<result_prompt>\n"
        "  ...your resulting prompt here...\n"
        "</result_prompt>\n"
        "Do not include any explanations or additional text outside this XML element.\n\n"
    )

    MARKDOWN_OUTPUT_REQUIREMENTS = (
        "#### Markdown formatting for the resulting prompt\n"
        "- Write the entire prompt inside <result_prompt> using valid Markdown.\n"
        "- Use headings (e.g., `#`, `##`) for major sections of the prompt.\n"
        "- Use bulleted lists (e.g., `-` or `*`) for enumerations and checklists.\n"
        "- Preserve any code or pseudo-code using fenced code blocks (``` ... ```).\n"
        "- Do not introduce any additional formatting beyond what is necessary to make "
        "the prompt clear and well-structured."
    )

    HYPE_META_PROMPT_TEMPLATE = (
        "{role_section}"
        "{prompt_structure_section}"
        "{recommendations_section}"
        "{constraints_section}"
        "{output_format_section}"
    )

    def __init__(self, config: HypeMetaPromptConfig | None = None) -> None:
        self.config = config or HypeMetaPromptConfig()
        self._cache_all_sections()

    def _cache_all_sections(self) -> None:
        self.config._cached_sections = {
            "role": self.build_role_section(),
            "prompt_structure": self.build_prompt_structure_section(),
            "output_format": self.build_output_format_section(),
        }

    def get_cached_section(self, name: str) -> Optional[str]:
        return self.config._cached_sections.get(name)

    # ----- секция роли -----
    def build_role_section(self, include_role: bool | None = None) -> str:
        include_role = (
            include_role
            if include_role is not None
            else self.config.include_role
        )
        form = self.config.target_prompt_form or ""
        task_part = self.TASK_SECTION_TEMPLATE.format(target_prompt_form=form)
        if include_role:
            return self.ROLE_LINE + task_part
        return task_part

    # ----- секция формата (список имён секций) -----
    def build_prompt_structure_section(
        self,
        specs: list[PromptSectionSpec] | None = None,
    ) -> str:
        specs = specs or self.config.section_specs
        lines = [f"- [{spec.name}] {spec.description}" for spec in specs]
        return self.PROMPT_STRUCTURE_SECTION_TEMPLATE.format(
            sections_with_guidelines="\n".join(lines)
        )

    # ----- секция рекомендаций (на основе анализа предыдущих генераций) -----
    def build_recommendations_section(
        self,
        recommendations: List[str] | None = None,
    ) -> str:
        recs = (
            recommendations
            if recommendations is not None
            else self.config.recommendations
        )
        if not recs:
            return ""
        lines = "\n".join(f"- {r}" for r in recs)
        return self.RECOMMENDATIONS_SECTION_TEMPLATE.format(
            recommendations_list=lines
        )

    # ----- секция жёстких ограничений -----
    def build_constraints_section(
        self,
        constraints: List[str] | None = None,
    ) -> str:
        constraints = constraints or self.config.constraints
        if not constraints:
            return ""
        lines = "\n".join(f"- {c}" for c in constraints)
        return self.CONSTRAINTS_SECTION_TEMPLATE.format(constraints_list=lines)

    def build_output_format_section(self) -> str:
        # если в конфиге уже передан кастомный текст — используем его как базу
        section = (
            self.config.output_format_section
            or self.BASE_OUTPUT_FORMAT_SECTION
        )
        if self.config.require_markdown_prompt:
            section = section + self.MARKDOWN_OUTPUT_REQUIREMENTS
        return section

    # ----- сборка всего мета‑промпта -----
    def build_meta_prompt(
        self,
        *,
        target_prompt_form: str | None = None,
        section_specs: List[PromptSectionSpec] | None = None,
        recommendations: List[str] | None = None,
        constraints: List[str] | None = None,
        output_format_section: str | None = None,
        include_role: bool | None = None,
    ) -> str:
        # локальный override конфигов
        if target_prompt_form is not None:
            self.config.target_prompt_form = target_prompt_form
        if section_specs is not None:
            self.config.section_specs = section_specs
        if recommendations is not None:
            self.config.recommendations = recommendations
        if constraints is not None:
            self.config.constraints = constraints
        if output_format_section is not None:
            self.config.output_format_section = output_format_section
        if include_role is not None:
            self.config.include_role = include_role

        role_section = self.build_role_section(include_role=include_role)
        prompt_structure_section = self.build_prompt_structure_section()
        recommendations_section = self.build_recommendations_section(
            recommendations=recommendations
        )
        constraints_section = self.build_constraints_section()
        output_format_section = self.build_output_format_section()

        return self.HYPE_META_PROMPT_TEMPLATE.format(
            role_section=role_section,
            prompt_structure_section=prompt_structure_section,
            recommendations_section=recommendations_section,
            constraints_section=constraints_section,
            output_format_section=output_format_section,
        )

    def rebuild_all_sections(self) -> None:
        self._cache_all_sections()
