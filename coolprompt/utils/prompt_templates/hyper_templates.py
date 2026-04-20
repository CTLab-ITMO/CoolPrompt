from dataclasses import dataclass, field
from typing import Any, List, Optional, Union


TARGET_PROMPT_FORMS = ["hypothetical ", "instructional "]


SIMPLE_HYPOTHETICAL_PROMPT = (
    "Write a {target_prompt_form}prompt that will solve the user query effectively."
)

META_INFO_SECTION = (
    "Task-related meta-information which you must mention generating a new prompt:\n<meta_info>\n{meta_info_content}\n</meta_info>\n"
)

# Section name constants
SECTION_ROLE = "role"
SECTION_PROMPT_STRUCTURE = "prompt_structure"
SECTION_RECOMMENDATIONS = "recommendations"
SECTION_CONSTRAINTS = "constraints"
SECTION_OUTPUT_FORMAT = "output_format"

META_PROMPT_SECTIONS = (
    SECTION_ROLE,
    SECTION_PROMPT_STRUCTURE,
    SECTION_RECOMMENDATIONS,
    SECTION_CONSTRAINTS,
    SECTION_OUTPUT_FORMAT,
)


@dataclass
class PromptSectionSpec:
    name: str
    description: str


@dataclass
class HypeMetaPromptConfig:
    target_prompt_form: str = "hypothetical instructional "
    require_markdown_prompt: bool = False
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
                    "CRITICAL: You MUST include the exact output format specified in the user's query or meta information block "
                    "as a requirement in your generated prompt if it was specified."
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
    """
    Builder for HyPE meta-prompts.

    Constructs meta-prompts from configurable sections. Uses a caching strategy:
    - Static sections (role, prompt_structure, output_format) are cached on init
      and rebuilt only when their config changes.
    - Dynamic sections (recommendations, constraints) are stored as lists in config
      and built on-demand during build_meta_prompt().

    Typical usage:
        builder = HypeMetaPromptBuilder()
        meta_prompt = builder.build_meta_prompt()

        # Update a section
        builder.set_section(SECTION_RECOMMENDATIONS, ["Be concise", "Use examples"])
        meta_prompt = builder.build_meta_prompt()
    """

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

    CONSTRAINTS_SECTION_TEMPLATE = "### HARD CONSTRAINTS\n{constraints_list}\n\n"

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
        """Cache static sections."""
        self.config._cached_sections = {
            SECTION_ROLE: self.build_role_section(),
            SECTION_PROMPT_STRUCTURE: self.build_prompt_structure_section(),
            SECTION_OUTPUT_FORMAT: self.build_output_format_section(),
        }

    def get_cached_section(self, name: str) -> Optional[str]:
        """Return a cached section by name (role, prompt_structure, output_format)."""
        return self.config._cached_sections.get(name)

    def get_section(self, name: str) -> Union[str, List[str], None]:
        """Return section value by name (list for recommendations/constraints, str for others)."""
        if name not in META_PROMPT_SECTIONS:
            raise ValueError(
                f"Unknown section: {name}. Expected: {META_PROMPT_SECTIONS}"
            )
        if name == SECTION_RECOMMENDATIONS:
            return list(self.config.recommendations)
        if name == SECTION_CONSTRAINTS:
            return list(self.config.constraints)
        return self.get_cached_section(name)

    def set_section(self, name: str, value: Union[str, List[str]]) -> None:
        """Update a section value. Only recommendations, constraints, and output_format are settable."""
        if name not in META_PROMPT_SECTIONS:
            raise ValueError(
                f"Unknown section: {name}. Expected: {META_PROMPT_SECTIONS}"
            )
        if name == SECTION_RECOMMENDATIONS:
            if not isinstance(value, list):
                raise ValueError("recommendations must be a list of strings")
            self.config.recommendations = list(value)
        elif name == SECTION_CONSTRAINTS:
            if not isinstance(value, list):
                raise ValueError("constraints must be a list of strings")
            self.config.constraints = list(value)
        elif name == SECTION_OUTPUT_FORMAT:
            if not isinstance(value, str):
                raise ValueError("output_format must be a string")
            self.config.output_format_section = value
            self.config._cached_sections[SECTION_OUTPUT_FORMAT] = self.build_output_format_section()
        else:
            raise ValueError(f"Section '{name}' is read-only or not directly settable")

    def build_role_section(self, include_role: bool | None = None) -> str:
        """
        Build the opening section with role definition and task description.

        Contains two parts:
        - Role line (optional, controlled by include_role)
        - Task description: explains what the model should do (always included)

        The task description uses target_prompt_form to specify the type of prompt to generate
        (e.g., "hypothetical instructional").
        """
        include_role = (
            include_role if include_role is not None else self.config.include_role
        )
        form = self.config.target_prompt_form or ""
        task_part = self.TASK_SECTION_TEMPLATE.format(target_prompt_form=form)
        if include_role:
            return self.ROLE_LINE + task_part
        return task_part

    def build_prompt_structure_section(
        self,
        specs: list[PromptSectionSpec] | None = None,
    ) -> str:
        """Build the prompt structure guidelines section."""
        specs = specs or self.config.section_specs
        lines = [f"- [{spec.name}] {spec.description}" for spec in specs]
        return self.PROMPT_STRUCTURE_SECTION_TEMPLATE.format(
            sections_with_guidelines="\n".join(lines)
        ) if lines else ""

    def build_recommendations_section(
        self,
        recommendations: List[str] | None = None,
    ) -> str:
        """Build the recommendations section (empty string if no recommendations)."""
        recs = (
            recommendations
            if recommendations is not None
            else self.config.recommendations
        )
        if not recs:
            return ""
        lines = "\n".join(f"- {r}" for r in recs)
        return self.RECOMMENDATIONS_SECTION_TEMPLATE.format(recommendations_list=lines)

    def build_constraints_section(
        self,
        constraints: List[str] | None = None,
    ) -> str:
        """Build the hard constraints section."""
        constraints = constraints or self.config.constraints
        if not constraints:
            return ""
        lines = "\n".join(f"- {c}" for c in constraints)
        return self.CONSTRAINTS_SECTION_TEMPLATE.format(constraints_list=lines)

    def build_output_format_section(self) -> str:
        """Build the output format section (with optional markdown requirements)."""
        section = self.config.output_format_section or self.BASE_OUTPUT_FORMAT_SECTION
        if self.config.require_markdown_prompt:
            section = section + self.MARKDOWN_OUTPUT_REQUIREMENTS
        return section

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
        """
        Build the complete meta-prompt from all sections.

        Args can override config values for this build only.
        """
        # Apply overrides to config
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

        return self.HYPE_META_PROMPT_TEMPLATE.format(
            role_section=self.build_role_section(include_role=include_role),
            prompt_structure_section=self.build_prompt_structure_section(),
            recommendations_section=self.build_recommendations_section(recommendations=recommendations),
            constraints_section=self.build_constraints_section(),
            output_format_section=self.build_output_format_section(),
        )
