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

TASK_AREA_DETECTOR_TEMPLATE = """You are a task-area classifier. Given a user query, output a single JSON object.

## Output schema
{{
  "task":      "classification" | "generation",
  "task_area": <area_id> | null,
  "confidence": <float 0.0–1.0>,
  "reason":    <one sentence>
}}

## Task type rules
- "classification" — the model predicts a label or category from a fixed set.
- "generation"     — the model produces free-form text, numbers, or structured output.
- When uncertain between the two, prefer "generation".

## Supported task areas
| area_id                         | Description                                                                 | Task type       |
|---------------------------------|-----------------------------------------------------------------------------|-----------------|
| tweet_emotion_classification    | Classify English tweets into: anger, joy, optimism, sadness                 | classification  |
| school_math_reasoning           | Grade-school math word problems; output is a numeric answer                 | generation      |
| concept_to_sentence_generation  | Generate a fluent sentence from a list of concepts or keywords              | generation      |
| context_question_answering      | Answer a question given a passage or context paragraph                      | generation      |
| text_summarization              | Condense an article or document into a short summary                        | generation      |

## Confidence rules
- 0.85–1.0 : query clearly and specifically matches one area; keywords, format, and intent all align.
- 0.70–0.84: query likely matches one area but is slightly ambiguous or under-specified.
- 0.50–0.69: weak or indirect match; the area is a reasonable guess but not certain.
- 0.00–0.49: query is generic, vague, or does not match any supported area → set task_area to null.

Only set task_area to a non-null value when confidence >= 0.70.

## Few-shot examples

Query: "Generate difficult school math word problems with numeric answers."
Output: {{"task":"generation","task_area":"school_math_reasoning","confidence":0.92,"reason":"Explicitly requests math word problems with numeric answers."}}

Query: "Classify the emotion of this tweet: I can't believe how amazing today was!"
Output: {{"task":"classification","task_area":"tweet_emotion_classification","confidence":0.95,"reason":"Asks to classify tweet emotion into a fixed label set."}}

Query: "Given a context paragraph, answer the question based only on the text."
Output: {{"task":"generation","task_area":"context_question_answering","confidence":0.90,"reason":"Describes a reading-comprehension QA task over a provided passage."}}

Query: "Create a sentence using the words: cloud, rain, umbrella."
Output: {{"task":"generation","task_area":"concept_to_sentence_generation","confidence":0.88,"reason":"Asks to generate a sentence from a set of concepts."}}

Query: "Summarize this news article in two sentences."
Output: {{"task":"generation","task_area":"text_summarization","confidence":0.91,"reason":"Requests a short summary of a longer article."}}

Query: "Generate diverse NLP examples for my model."
Output: {{"task":"generation","task_area":null,"confidence":0.20,"reason":"Too generic to match any supported task area."}}

Query: "Find the right answer from the test."
Output: {{"task":"generation","task_area":null,"confidence":0.15,"reason":"Vague query with no identifiable domain or format."}}

## Now classify this query
Query: {query}

Return ONLY valid JSON with exactly these four keys. No markdown, no extra text."""