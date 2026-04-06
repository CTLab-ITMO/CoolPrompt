import itertools
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List

project_path = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
sys.path.insert(0, project_path)

from coolprompt.utils.prompt_templates.hyper_templates import (
    HypeMetaPromptBuilder,
    PromptSectionSpec,
)


def generate_sections_config(
    include_role_section: bool,
    include_task_context: bool,
    include_output_section: bool,
) -> List[PromptSectionSpec]:
    """Генерирует конфиг секций по флагам.

    Секция Instructions включается всегда.
    Role, Task context, Output requirements — опционально.
    """
    sections: List[PromptSectionSpec] = []

    if include_role_section:
        sections.append(
            PromptSectionSpec(
                name="Role",
                description=(
                    "Briefly define the assistant's role and expertise "
                    "relevant to the user query."
                ),
            )
        )

    if include_task_context:
        sections.append(
            PromptSectionSpec(
                name="Task context",
                description=(
                    "Provide the full context of the user's task: restate the query, "
                    "include all provided meta-information, domain details, constraints, "
                    "and any other information necessary to produce a correct solution. "
                    "Do not evaluate or condense — pass through everything relevant."
                ),
            )
        )

    sections.append(
        PromptSectionSpec(
            name="Instructions",
            description=(
                "Main part - instructions the assistant must follow "
                "to solve the user's query while respecting constraints."
            ),
        )
    )

    if include_output_section:
        sections.append(
            PromptSectionSpec(
                name="Output requirements",
                description=(
                    "Clearly specify the desired tone and required level of detail. "
                    "If the user explicitly requests a particular output format or "
                    "provides an example response, restate that format and include "
                    "the example verbatim, without inventing any additional formatting."
                ),
            )
        )

    return sections


def _make_variant_name(
    target_form: str,
    include_role: bool,
    use_sections: bool,
    task_context: bool,
    role_section: bool,
    output_section: bool,
    use_markdown: bool,
) -> str:
    """Имя варианта: TF_R_US_TC_RS_OS_MD"""
    tf = "hyp_inst" if "hypothetical" in target_form else "inst"
    return (
        f"TF{tf}"
        f"_R{int(include_role)}"
        f"_US{int(use_sections)}"
        f"_TC{int(task_context)}"
        f"_RS{int(role_section)}"
        f"_OS{int(output_section)}"
        f"_MD{int(use_markdown)}"
    )


def _build_meta_prompt_no_sections(
    builder: HypeMetaPromptBuilder,
    target_prompt_form: str,
    include_role: bool,
    use_markdown: bool,
) -> str:
    """Собирает мета-промпт БЕЗ секции STRUCTURE OF THE PROMPT.

    Используется когда use_sections=False.
    """
    builder.config.target_prompt_form = target_prompt_form
    builder.config.include_role = include_role
    builder.config.require_markdown_prompt = use_markdown

    role_section = builder.build_role_section(include_role=include_role)
    output_format_section = builder.build_output_format_section()

    # Собираем без prompt_structure_section, recommendations и constraints
    return (
        f"{role_section}"
        f"{output_format_section}"
    )


def main_ablation():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("ablation_prompts")
    out_dir.mkdir(exist_ok=True)

    builder = HypeMetaPromptBuilder()

    # Факторы:
    # target_form:    "instructional " | "hypothetical instructional "
    # include_role:   True | False   — включать роль в мета-промпт
    # use_sections:   True | False   — использовать секции в продуцируемом промпте
    # role_section:   True | False   — секция Role (только при US=1)
    # task_context:   True | False   — секция Task context (только при US=1)
    # output_section: True | False   — секция Output requirements (только при US=1)
    # use_markdown:   всегда False

    target_forms = ["instructional ", "hypothetical instructional "]
    include_roles = [True, False]
    use_markdown = False  # всегда выключен

    prompts: dict[str, str] = {}

    for target_form, include_role in itertools.product(target_forms, include_roles):
        # --- US=0: секций нет, RS=TC=OS=0 ---
        name = _make_variant_name(
            target_form=target_form,
            include_role=include_role,
            use_sections=False,
            task_context=False,
            role_section=False,
            output_section=False,
            use_markdown=use_markdown,
        )
        meta_prompt = _build_meta_prompt_no_sections(
            builder=builder,
            target_prompt_form=target_form,
            include_role=include_role,
            use_markdown=use_markdown,
        )
        prompts[name] = meta_prompt
        print(f"✅ {name}")

        # --- US=1: перебираем RS, TC, OS ---
        for role_section, task_context, output_section in itertools.product(
            [True, False], [True, False], [True, False]
        ):
            specs = generate_sections_config(
                include_role_section=role_section,
                include_task_context=task_context,
                include_output_section=output_section,
            )

            orig_markdown = builder.config.require_markdown_prompt
            builder.config.require_markdown_prompt = use_markdown

            meta_prompt = builder.build_meta_prompt(
                target_prompt_form=target_form,
                section_specs=specs,
                constraints=[],
                include_role=include_role,
            )

            name = _make_variant_name(
                target_form=target_form,
                include_role=include_role,
                use_sections=True,
                task_context=task_context,
                role_section=role_section,
                output_section=output_section,
                use_markdown=use_markdown,
            )
            prompts[name] = meta_prompt
            print(f"✅ {name}")

            builder.config.require_markdown_prompt = orig_markdown

    total_variants = len(prompts)
    json_file = out_dir / f"meta_prompts_{total_variants}v_{timestamp}.json"

    payload = {
        "meta": {
            "timestamp": timestamp,
            "total_variants": total_variants,
            "factors": [
                "target_form (inst | hyp_inst)",
                "include_role (R)",
                "use_sections (US)",
                "task_context (TC) — only when US=1",
                "role_section (RS) — only when US=1",
                "output_section (OS) — only when US=1",
                "markdown (MD) — always 0",
            ],
            "naming": "TF{inst|hyp_inst}_R{0|1}_US{0|1}_TC{0|1}_RS{0|1}_OS{0|1}_MD{0}",
            "note": (
                "When US=0, TC/RS/OS are forced to 0 (no sections). "
                "Total = 2(TF) × 2(R) × (1 + 2³) = 36 unique variants."
            ),
        },
        "prompts": prompts,
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Готово! {total_variants} вариантов в {json_file}")
    print(
        f"📊 Naming: TF{{inst|hyp_inst}}_R{{0|1}}_US{{0|1}}_TC{{0|1}}_RS{{0|1}}_OS{{0|1}}_MD{{0}}"
    )


if __name__ == "__main__":
    main_ablation()
