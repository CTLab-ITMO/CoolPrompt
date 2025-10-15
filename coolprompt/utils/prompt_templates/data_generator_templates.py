PROBLEM_DESCRIPTION_TEMPLATE = """You are an expert in LLM task domain.
You are given a user's prompt.
Write the detailed problem description for which that prompt was created.
Use only textual description. Do not add another data.
Prompt: {prompt}
Provide your answer in JSON format with object with key 'problem_description'.
Output format:
{{
    'problem_description': "Determined problem description"
}}
"""

CLASSIFICATION_DATA_GENERATING_TEMPLATE = """You are an expert in synthetic data generation. You are very experienced in creating task examples.
You should create a validation dataset of {num_samples} examples.
Create a set of ground-truth labels.
Then make some test questions (inputs) that correlates with problem description and use created labels as the responses. Try to make the answers distribution more random.
Problem description: {problem_description}
Provide your answer in JSON object with key 'examples' containing a list of your artificial examples. Each example is an object with keys 'input' and 'output' that are contain corresponding text.
Make sure to include all necessary data in 'input' object. You must not add any other objects except 'input' and 'output'.
Also remember that 'input' and 'output' are textual fields. If you have some answer choices for input - just concat them with input text into one string.
Output format is the JSON structure below:
{{
   "examples": [
       {{
          "input": "Textual input",
          "output": "Ground-truth label",
          "id": 1
       }},
       ...
       {{
          "input": "Textual input",
          "output": "Ground-truth label",
          "id": {num_samples}
       }}
   ]
}}
Output JSON data only. Remeber to create exactly {num_samples} examples.
"""

GENERATION_DATA_GENERATING_TEMPLATE = """
You are an expert in synthetic data generation. You are very experienced in creating task examples.
You should create a validation dataset of {num_samples} examples.
Create example pairs input-output that will correspond given problem description.
Problem description: {problem_description}
Provide your answer in JSON object with key 'examples' containing a list of your artificial examples. Each example is an object with keys 'input' and 'output' that are contain corresponding text.
Make sure to include all necessary data in 'input' object. You must not add any other objects except 'input' and 'output'.
Also remember that 'input' and 'output' are textual fields.
Output format is the JSON structure below:
{{
   "examples": [
       {{
          "input": "Textual input",
          "output": "Correct model output",
          "id": 1
       }},
       ...
       {{
          "input": "Textual input",
          "output": "Correct model output",
          "id": {num_samples}
       }}
   ]
}}
Output JSON data only. Remeber to create exactly {num_samples} examples.
"""
