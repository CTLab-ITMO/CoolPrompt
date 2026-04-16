from typing import List, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field
from coolprompt.utils.logging_config import logger


SYSTEM_PROMPT = """
## Роль 
Ты экспертный промпт-инженер, который разбирается в оптимизации промптов.

## Задача
Тебе дан исходный промпт от пользователя (в котором есть входной контекст задачи и вопрос).
Твоя задача составить сжатый промпт по схеме:
1. Сначала идёт входной контекст задачи в одном предложении в сжатом виде.
2. Далее ставится целевой запрос/вопрос, что требуется в данной задаче (в одном предложении).

Задачу решай по шагам:
- Проанализируй текст задачи, определи, где находится суть вопроса.
- Составь ОЧЕНЬ СИЛЬНО сжатую версию (не более 10 слов) входного контекста задачи.
- Составь ОЧЕНЬ СИЛЬНО сжатую версию (не более 10 слов) целевого вопроса/запроса с добавлением "ответь кратко".
- Объедини два предложения в один финальный промпт.

## Формат ответа
- Строго следуй формату JSON
- Ориентируйся только на текст изначального промпта
- За несуществующие слова ты будешь оштрафован
"""

USER_PROMPT = "## Исходный промпт: {prompt}"


class CompressedPromptResponse(BaseModel):
    """Structure for LLM answer."""
    reasoning: str = Field(description="Анализ задачи и вопроса в исходном промпте")
    prompt_input_context: str = Field(description="Выделенный входной контекст задачи в одном предложении")
    prompt_task: str = Field(description="Выделенное предложение самой задачи")
    final_prompt: str = Field(description="Финальный сжатый промпт")

class PromptCompressor:
    """
    Prompt compressor using LLM and structured output.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ):
        """
        Args:
            model: LangChain language model.
            system_prompt: System prompt for compression (if None, the default is used).
            user_prompt_template: User query template with {prompt} placeholder.
        """
        self.model = model
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.user_prompt = user_prompt or USER_PROMPT

        self.structured_model = model.with_structured_output(
            CompressedPromptResponse, method="json_schema"
        )

    def _build_messages(self, prompt: str) -> List[dict]:
        """Build messages for the LLM."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt.format(prompt=prompt)},
        ]

    def compress(self, prompt: str, return_metadata: bool = False) -> Union[str, CompressedPromptResponse]:
        """
        Compress a single prompt synchronously.

        Args:
            prompt: Original prompt.
            return_metadata: If True, returns the full CompressedPromptResponse object,
                             otherwise only the final_prompt string.

        Returns:
            Compressed prompt (string) or full object with metadata.
        """
        messages = self._build_messages(prompt)
        response = self.structured_model.invoke(messages)

        logger.debug(f"Compressed prompt from '{prompt[:50]}...' -> '{response.final_prompt[:50]}...'")
        return response if return_metadata else response.final_prompt
