TASK_DETECTOR_TEMPLATE = """You are an expert in task definition. You are very experienced in classifying tasks.
You should define from user query the name of the task from the list:
- classification (when is needed to classify something)
- generation (when is needed to provide a new text like NLP, NLI tasks like question answering or summarizing)

User Query:
{query}

Answer formatting:
Provide your answer in JSON object with key 'task' containing a name of the task: generation or classification.
Output format is the JSON structure below:
{{
   "task": "task name"
}}
Output JSON data only.
"""
