from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.utils.parsing import (
    extract_json,
    get_model_answer_extracted,
    safe_template,
)
from coolprompt.utils.prompt_templates.assistant_templates import (
    FEEDBACK_COLLECTING_TEMPLATE,
)


class PromptAssistant:
    """Prompt Assistant class. Provides a feedback based on
    start prompt and final prompt which tells the user how
    the original prompt was improved.

    Attributes:
        model (langchain.BaseLanguageModel): LangChain language model.
    """

    COMMON_FEEDBACK = (
        "You should follow these three simple rules:\n"
        "1. Structured 2. Detalized 3. Instructive"
    )

    def __init__(self, model: BaseLanguageModel):
        self.model = model

    def get_feedback(self, start_prompt: str, final_prompt: str) -> str:
        """Generates the feedback based on `start_prompt` and `final_prompt`

        Args:
            start_prompt (str): original prompt
            final_prompt (str): improved prompt
        Returns:
            str: generated feedback or the common feedback if could not parse
                the model's answer
        """

        prompt = safe_template(
            FEEDBACK_COLLECTING_TEMPLATE,
            start_prompt=start_prompt,
            final_prompt=final_prompt,
        )

        answer = get_model_answer_extracted(self.model, prompt)

        result = extract_json(answer)

        return result["feedback"] if result else self.COMMON_FEEDBACK
