from pydantic import BaseModel, Field


class TextualGradientResponse(BaseModel):
    """Response schema for textual-gradient feedback generation."""

    feedback: str = Field(
        description=(
            "A detailed natural-language reasoning about the prompt's flaws, "
            "grounded in the provided failed examples, together with concrete "
            "optimization directions tailored to the underlying data "
            "distribution observed in those examples. "
            "The feedback must (a) diagnose WHY the prompt fails on these "
            "specific examples, (b) extract data-driven, localized "
            "optimization strategies, and (c) explain HOW to revise the "
            "prompt to avoid the same mistakes on similar inputs. "
            "Plain text only, no XML tags, no enumeration markup beyond "
            "natural prose, no commentary about the task itself."
        )
    )
