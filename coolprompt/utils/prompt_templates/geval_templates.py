COMMON_TEMPLATE = """You will be given {task_description}.

Your task is to rate the response on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

{criteria_name} (1-{{metric_ceil}}): {criteria_description}

Evaluation Steps:

{eval_steps}

Source Text:

{{request}}

{criteria_name}:

{{response}}


Evaluation Form (scores ONLY, no explanation):

- {criteria_name}:
"""

ACCURACY_QA_TEMPLATE = COMMON_TEMPLATE.format(
    criteria_name="Accuracy",
    task_description="a response to a question",
    criteria_description="The response is factually accurate and contains no errors or misconceptions. It correctly addresses the concepts and data related to the question.",
    eval_steps="""1) Read the question and the response carefully.
2) Identify any factual errors, misconceptions, or incorrect interpretations in the response.
3) Check if the response correctly uses terms and data.
4) Determine if the response provides accurate information that aligns with the theme of the request.
5) Assign a score based on the level of accuracy."""
)

COHERENCE_TEMPLATE = COMMON_TEMPLATE.format(
    criteria_name="Coherence",
    task_description="a generated text response",
    criteria_description="The response is well-structured, logically organized, and ideas flow naturally. Sentences and paragraphs connect smoothly without contradictions.",
    eval_steps="""1) Read the response carefully.
2) Check if ideas are presented in logical order.
3) Verify that sentences connect smoothly.
4) Look for contradictions or disjointed thoughts.
5) Assign a score based on overall coherence."""
)

FLUENCY_TEMPLATE = COMMON_TEMPLATE.format(
    criteria_name="Fluency",
    task_description="a generated text response",
    criteria_description="The response is grammatically correct, uses proper vocabulary, and reads naturally. No awkward phrasing or errors that impede understanding.",
    eval_steps="""1) Read the response carefully.
2) Check for grammatical errors.
3) Verify proper word choice and natural phrasing.
4) Look for awkward constructions.
5) Assign a score based on language quality."""
)

RELEVANCE_TEMPLATE = COMMON_TEMPLATE.format(
    criteria_name="Relevance",
    task_description="a response to a request",
    criteria_description="The response directly addresses the request and stays on topic. All information provided is pertinent to what was asked.",
    eval_steps="""1) Read the request and response carefully.
2) Check if the response addresses the main points.
3) Verify that information is on-topic.
4) Look for irrelevant details.
5) Assign a score based on relevance to the request."""
)
