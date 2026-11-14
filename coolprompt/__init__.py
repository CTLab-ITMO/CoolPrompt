"""Public CoolPrompt package exports."""

__all__ = ["PromptTuner", "rider"]


def __getattr__(name: str):
    """Load the high-level assistant lazily.

    Keeping this import lazy lets lightweight utilities, such as the metadata
    selector, work without importing optional evaluation-model dependencies.
    """
    if name == "PromptTuner":
        from .assistant import PromptTuner

        return PromptTuner
    if name == "rider":
        from .optimizer.rider_original import rider

        return rider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
