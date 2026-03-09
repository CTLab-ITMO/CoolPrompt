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

# Твой HypeMetaPromptBuilder + HypeMetaPromptConfig вставляем сюда


def generate_sections_config(
    include_role_section: bool, include_output_section: bool
) -> List[PromptSectionSpec]:
    """Генерирует конфиг секций по флагам"""
    base_sections = [
        PromptSectionSpec(
            name="Instructions",
            description=(
                "Main part - instructions the assistant must follow "
                "to solve the user's query while respecting constraints."
            ),
        ),
    ]

    if include_role_section:
        base_sections.insert(
            0,
            PromptSectionSpec(
                name="Role",
                description=(
                    "Briefly define the assistant's role and expertise "
                    "relevant to the user query."
                ),
            ),
        )

    if include_output_section:
        base_sections.append(
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

    return base_sections


def _make_variant_name(
    target_form: str,
    include_role: bool,
    role_section: bool,
    output_section: bool,
    use_markdown: bool,
) -> str:
    """Имя варианта: TF_R_RS_OS_MD"""
    tf = "hyp_inst" if "instructional" in target_form else "hyp"
    return f"TF{tf}_R{int(include_role)}_RS{int(role_section)}_OS{int(output_section)}_MD{int(use_markdown)}"


def main_32variants():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("ablation_prompts")
    out_dir.mkdir(exist_ok=True)
    json_file = out_dir / f"meta_prompts_32v_{timestamp}.json"

    builder = HypeMetaPromptBuilder()

    # 5 факторов × 2 уровня = 32
    factors = [
        ["", "instructional ", "hypothetical instructional "],
        [True, False],  # include_role
        [True, False],  # role_section
        [True, False],  # output_requirements_section
        [False],  # markdown
    ]

    total_variants = 32
    print(f"🚀 Генерируем 32 варианта мета-промптов → {json_file}")

    prompts: dict[str, str] = {}

    for combo in itertools.product(*factors):
        (
            target_form,
            include_role,
            role_section,
            output_section,
            use_markdown,
        ) = combo

        # Генерируем секции по флагам
        specs = generate_sections_config(role_section, output_section)

        # Включаем markdown
        orig_markdown = builder.config.require_markdown_prompt
        builder.config.require_markdown_prompt = use_markdown

        # Строим промпт (constraints пока отключены)
        meta_prompt = builder.build_meta_prompt(
            target_prompt_form=target_form,
            section_specs=specs,
            constraints=[],
            include_role=include_role,
        )

        name = _make_variant_name(
            target_form,
            include_role,
            role_section,
            output_section,
            use_markdown,
        )
        prompts[name] = meta_prompt

        print(f"✅ {name}")
        builder.config.require_markdown_prompt = orig_markdown

    payload = {
        "meta": {
            "timestamp": timestamp,
            "total_variants": total_variants,
            "factors": [
                "target_form",
                "include_role",
                "role_section",
                "output_section",
                "markdown",
            ],
            "naming": "TF{hyp|hyp_inst}_R{0|1}_RS{0|1}_OS{0|1}_MD{0|1}",
        },
        "prompts": prompts,
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Готово! 32 варианта в {json_file}")
    print(
        f"📊 Naming: TF{{hyp|hyp_inst}}_R{{0|1}}_RS{{0|1}}_OS{{0|1}}_MD{{0|1}}"
    )


if __name__ == "__main__":
    main_32variants()
