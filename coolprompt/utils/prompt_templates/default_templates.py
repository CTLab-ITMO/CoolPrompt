
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
