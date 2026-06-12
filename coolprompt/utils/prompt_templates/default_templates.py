CLASSIFICATION_TASK_TEMPLATE = """{PROMPT}

Answer using the label from [{LABELS}].
Generate the final answer bracketed with <ans> and </ans>.

Input:
{INPUT}

Response:
"""

GENERATION_TASK_TEMPLATE = """{PROMPT}

Generate the final answer bracketed with <ans> and </ans>.

INPUT:
{INPUT}

RESPONSE:
"""

CLASSIFICATION_TASK_TEMPLATE_STRUCTURED = """{PROMPT}

Answer using the label from [{LABELS}].
Return the chosen label in the `answer` field of the structured response.

Input:
{INPUT}

Response:
"""

GENERATION_TASK_TEMPLATE_STRUCTURED = """{PROMPT}

Return the final answer in the `answer` field of the structured response.

INPUT:
{INPUT}

RESPONSE:
"""
