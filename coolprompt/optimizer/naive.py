from utils.prompt_template import NAIVE_PROMPT_TEMPLATE


def naive_optimizer(self, prompt: str) -> str:
    """Rewrites prompt by injecting it into predefined template and querying LLM.

    Args:
        prompt: Input prompt to optimize.

    Returns:
        LLM-generated rewritten prompt.
    """
    template = NAIVE_PROMPT_TEMPLATE
    answer = self._model.invoke(template.replace("<PROMPT>", prompt)).strip()
    return answer[answer.find("Rewritten prompt:\n") + len("Rewritten prompt:\n") :]
