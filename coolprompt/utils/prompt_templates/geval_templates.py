COMMON_TEMPLATE = """You will be given {task_description}.

Your task is to rate the responce on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

{criteria_name} (1-{{metric_ceil}}): {criteria_description}

Evaluation Steps:

{eval_steps}

Source Text:

{{request}}

{criteria_name}:

{{responce}}


Evaluation Form (scores ONLY):

- {criteria_name}:
"""

ACCURACY_QA_TEMPLATE = COMMON_TEMPLATE.format(
    criteria_name="Accuracy",
    task_description="a responce to a question",
    criteria_description="The response is factually accurate and contains no errors or misconceptions. It correctly addresses the concepts and data related to the question.",
    eval_steps="""1) Read the question and the response carefully.
2) Identify any factual errors, misconceptions, or incorrect interpretations in the response.
3) Check if the response correctly uses terms and data.
4) Determine if the response provides accurate information that aligns with the theme of the request.
5) Assign a score based on the level of accuracy."""
)
