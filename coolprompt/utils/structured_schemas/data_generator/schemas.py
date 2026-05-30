from typing import List

from pydantic import BaseModel, Field


class ProblemDescriptionResponse(BaseModel):
    """Response schema for synthetic problem description generation.

    Used by :meth:`coolprompt.data_generator.generator.
    SyntheticDataGenerator._generate_problem_description` to obtain a
    concise, generalized textual description of the task that the user's
    initial prompt was created to solve.
    """

    problem_description: str = Field(
        description=(
            "Detailed yet generalized textual description of the task the "
            "user's prompt was created for. Strict requirements: "
            "(1) plain text only — no JSON, no bullet lists, no XML tags, "
            "no enumeration prefix, no quotes; "
            "(2) describe the task as a whole — do NOT reference specific "
            "examples, named entities, formulas, exact phrasings or "
            "dataset rows; "
            "(3) cover answer format, problem subject and scope when they "
            "are inferable from the prompt; "
            "(4) no meta-comments such as 'this prompt', 'as described "
            "above', 'the user wants'."
        )
    )


class ClassificationTaskExample(BaseModel):
    """A single (input, output) example for a classification task."""

    input: str = Field(
        description=(
            "Textual input for the classification task. Must contain ALL "
            "information required to predict the label (if answer choices "
            "are part of the task, concatenate them into the input "
            "string). Plain text only — no JSON, no XML, no enumeration."
        )
    )
    output: str = Field(
        description=(
            "Ground-truth class label for the corresponding input. Plain "
            "text only, no quotes, no trailing explanation, no extra "
            "fields."
        )
    )


class ClassificationTaskResponse(BaseModel):
    """Response schema for classification dataset synthesis."""

    examples: List[ClassificationTaskExample] = Field(
        description=(
            "Synthetic classification examples. Strict requirements: "
            "(1) the list length MUST equal the requested num_samples; "
            "(2) the label distribution should be as balanced / random as "
            "possible across the set of ground-truth labels for the task; "
            "(3) each item is an object with EXACTLY two string fields "
            "'input' and 'output' — no 'id', no extra keys; "
            "(4) inputs must be self-contained (include any answer "
            "choices inline)."
        )
    )


class GenerationTaskExample(BaseModel):
    """A single (input, output) example for a generation task."""

    input: str = Field(
        description=(
            "Textual input/request for the generation task. Must include "
            "all data required to produce the expected output. Plain text "
            "only — no JSON, no XML, no enumeration."
        )
    )
    output: str = Field(
        description=(
            "Correct model answer for the corresponding input. Plain text "
            "only — no quotes, no trailing explanation, no extra fields."
        )
    )


class GenerationTaskResponse(BaseModel):
    """Response schema for generation dataset synthesis."""

    examples: List[GenerationTaskExample] = Field(
        description=(
            "Synthetic generation examples. Strict requirements: "
            "(1) the list length MUST equal the requested num_samples; "
            "(2) each item is an object with EXACTLY two string fields "
            "'input' and 'output' — no 'id', no extra keys; "
            "(3) examples should cover diverse facets of the task to "
            "form a useful validation set, while staying faithful to the "
            "problem description."
        )
    )
