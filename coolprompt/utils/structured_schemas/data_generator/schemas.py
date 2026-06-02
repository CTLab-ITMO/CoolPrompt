from typing import List

from pydantic import BaseModel, Field


class ProblemDescriptionResponse(BaseModel):
    """Response schema for synthetic problem description generation.

    Used by :meth:`coolprompt.data_generator.generator.
    SyntheticDataGenerator._generate_problem_description` to obtain a
    textual description of the task that the user's initial prompt was
    created to solve.
    """

    problem_description: str = Field(
        description=(
            "Detailed textual problem description for which the user's "
            "prompt was created."
        )
    )


class ClassificationTaskExample(BaseModel):
    """A single (input, output) example for a classification task."""

    input: str = Field(
        description=(
            "Textual input for the classification task. Must contain all "
            "data required to predict the label; if answer choices are "
            "part of the task, concatenate them into the input string."
        )
    )
    output: str = Field(
        description=(
            "Textual ground-truth label corresponding to the input."
        )
    )


class ClassificationTaskResponse(BaseModel):
    """Response schema for classification dataset synthesis."""

    examples: List[ClassificationTaskExample] = Field(
        description=(
            "List of synthetic classification examples. Try to make the "
            "answer distribution as random as possible."
        )
    )


class GenerationTaskExample(BaseModel):
    """A single (input, output) example for a generation task."""

    input: str = Field(
        description=(
            "Textual input for the generation task. Must contain all "
            "data required to produce the expected output."
        )
    )
    output: str = Field(
        description=(
            "Textual correct model output corresponding to the input."
        )
    )


class GenerationTaskResponse(BaseModel):
    """Response schema for generation dataset synthesis."""

    examples: List[GenerationTaskExample] = Field(
        description=(
            "List of synthetic input-output examples for the generation "
            "task."
        )
    )
