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


class ShortTermHintResponse(BaseModel):
    """Response schema for RE-GPS short-term reflection hints.

    In RE-GPS the short-term reflection is produced by synthesizing BOTH
    parent prompts (worse and better) with their respective textual gradients
    (per-parent feedback derived from failed training examples). The hint must
    therefore be data-driven and explicitly leverage the provided feedbacks —
    not only a worse-vs-better surface comparison.
    """

    hint: str = Field(
        description=(
            "ONE concise hint (strictly under 20 words) acting as a verbal "
            "gradient in prompt space for RE-GPS short-term reflection. "
            "It MUST be synthesized from BOTH the worse and the better "
            "prompt AND their respective improvement feedbacks (textual "
            "gradients grounded in failed training examples): use the "
            "feedbacks to identify a localized, data-driven edit that would "
            "push a prompt from the worse pattern toward the better pattern "
            "while avoiding the failure modes diagnosed in the feedbacks. "
            "Prefer concrete edit operations such as word replacement, "
            "conversion to active or positive voice, adding a missing word, "
            "or deleting a redundant word. "
            "Plain text only, no XML tags, no enumeration, no preamble."
        )
    )


class MutatedPromptResponse(BaseModel):
    """Response schema for RE-GPS elitist mutation prompts.

    In RE-GPS the elitist mutation is steered by TWO complementary signals:
    (1) the accumulated long-term reflection memory (primary guidance), and
    (2) the elitist prompt's own improvement feedback / textual gradient
    (secondary, data-driven correction signal). Both must be reflected in
    the produced mutation; ignoring the elitist's textual gradient breaks
    the gradient-guided nature of the algorithm.
    """

    prompt: str = Field(
        description=(
            "A mutated prompt derived from the elitist (best-so-far) prompt. "
            "Mutation MUST be guided primarily by the provided long-term "
            "reflection (accumulated epoch-spanning insights about correct "
            "prompt structure and useful prompt features) AND MUST ALSO "
            "incorporate the improvement feedback (textual gradient) "
            "generated for the elitist, so that the new prompt fixes the "
            "concrete failure modes diagnosed in that feedback. "
            "The model may apply either a STRUCTURAL transformation "
            "(reorganize sections, change format) or a SEMANTIC modification "
            "(rephrase, replace words, adjust voice/tone, add or delete "
            "content) — whichever the long-term reflection and the elitist "
            "feedback jointly indicate as most beneficial. The result must "
            "remain a valid, self-contained prompt for the same task. "
            "Plain text only, no XML tags, no commentary."
        )
    )
