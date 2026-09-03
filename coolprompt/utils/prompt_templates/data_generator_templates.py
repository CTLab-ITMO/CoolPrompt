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

PROBLEM_DESCRIPTION_BASED_ON_EXAMPLES_TEMPLATE = """You are an expert in LLM task domain.
You are given a user's prompt and a few examples from problem dataset.
User created this prompt to solve the task represented by given dataset.
Write the detailed problem description for which that prompt was created. Feel free to use provided examples from the dataset to highlight the key features of the task. You can pay attention to answer format, problem's subject and scope and other aspects that may be crucial for better understanding.
Remember, you should provide a very detailed problem description in order to make it understandable and clear as much as possible, but it is very important to make your problem description general and non-specific. Do not highlight the meaning of specific examples, you need to define the meaning of the task as a whole.
Use only textual description. Do not add another data.

User's prompt: {prompt}

Examples from dataset:
{examples}

Provide your answer in JSON format with object with key 'problem_description'.
Output format:
{{
    'problem_description': "Determined problem description"
}}
"""

PROBLEM_DESCRIPTION_BASED_ON_EXAMPLES_TEMPLATE_OLD = """You are an expert in LLM task domain.
You are given a user's prompt and a few examples from problem dataset.
User created this prompt to solve the task represented by given dataset.
Write the detailed problem description for which that prompt was created. Feel free to use provided examples from the dataset to highlight the key features of the task. You can pay attention to answer format, problem's subject and scope and other aspects that may be crucial for better understanding.
Remember, you should provide a very detailed problem description in order to make it understandable and clear as much as possible.
Use only textual description. Do not add another data.

User's prompt: {prompt}

Examples from dataset:
{examples}

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
Output JSON data only. Remember to create exactly {num_samples} examples.
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

CLASSIFICATION_CORNER_CASE_GENERATING_TEMPLATE = """
You are an expert in synthetic data generation. You are very experienced in creating task examples.
You should create a validation dataset of {num_samples} examples.
Create a set of ground-truth labels. Then make some test questions (inputs) that correlates with problem
description and use created labels as the responses. Try to make the answers distribution more random.
The key point of your task is to create as most corner and edge cases for the problem as possible. Try to
think out of line to create the most difficult or non-trivial or corner scenarios you can imagine.
Your examples should not be easy in terms of guessing the right answer. They should be diverse and
challenging.
Problem description: {problem_description}
Provide your answer in JSON object with key "examples"containing a list of your artificial corner-case
examples. Each example is an object with keys "input"and "output"which contain corresponding text.
Make sure to include all necessary data in "input"object. You must not add any other objects except
"input"and "output".
Also remember that "input"and "output"are textual fields. If you have some answer choices for input - just
concat them with input text into one string.
Output format is the JSON structure below:
{{
   "examples": [
       {{
          "input": "Textual corner-case input",
          "output": "Ground-truth label",
          "id": 1
       }},
       ...
       {{
          "input": "Textual corner-case input",
          "output": "Ground-truth label",
          "id": {num_samples}
       }}
   ]
}}
Output JSON data only. Remember to create exactly {num_samples} examples.
"""

GENERATION_CORNER_CASE_GENERATING_TEMPLATE = """
You are an expert in synthetic data generation. You are very experienced in creating task examples.
You should create a validation dataset of {num_samples} examples.
Create example pairs input-output that will correspond given problem description.
The key point of your task is to create as most corner and edge cases for the problem as possible. Try to
think out of line to create the most difficult or non-trivial or corner scenarios you can imagine.
Your examples should not be easy in terms of guessing the right answer. They should be diverse and
challenging.
Problem description: {problem_description}
Provide your answer in JSON object with key "examples"containing a list of your artificial corner-case
examples. Each example is an object with keys "input"and "output"which contain corresponding text.
Make sure to include all necessary data in "input"object. You must not add any other objects except
"input"and "output".
Also remember that "input"and "output"are textual fields. If you have some answer choices for input - just
concat them with input text into one string.
Output format is the JSON structure below: 
{{
   "examples": [
       {{
          "input": "Textual corner-case input",
          "output": "Correct model output",
          "id": 1
       }},
       ...
       {{
          "input": "Textual corner-case input",
          "output": "Correct model output",
          "id": {num_samples}
       }}
   ]
}}
Output JSON data only. Remember to create exactly {num_samples} examples.
"""

TWEETEVAL_STANDARD_RULES = """
You are an expert in synthetic data generation.
Create exactly {num_samples} TweetEval Emotion examples.

Problem description: {problem_description}
Task: Generate short realistic English tweets and assign one label.

USE ONLY LABELS:
- anger
- joy
- optimism
- sadness

Rules:
- Each example must have "input" and "output".
- Put the tweet text in "input".
- Put exactly one label in "output".
- Generate realistic short English tweets where the emotion is clearly and directly expressed.
- Keep the label distribution reasonably diverse across all four labels.
- Do not add explanations, comments, markdown, or extra fields.

Return valid JSON only, no markdown, no comments:
{{"examples": [{{"id": 1, "input": "...", "output": "anger"}}]}}
"""

TWEETEVAL_CORNER_CASE_RULES = """
You are an expert in synthetic data generation.
You should create a validation dataset of {num_samples} TweetEval Emotion corner-case examples.

Problem description: {problem_description}
Task: Generate short realistic English tweets and assign one label.

USE ONLY LABELS:
- anger
- joy
- optimism
- sadness

- Create exactly {num_samples} examples.
- Each example must have "input", "output".
- Put the tweet text in "input".
- Put exactly one label in "output".
- Do not add explanations, comments, markdown, or extra fields.

Corner-cases for this dataset are tweets where the dominant emotion is not expressed directly and must be inferred from context, tone, sarcasm, implication, or informal language.

Relevant corner-case types:
- sarcasm or irony;
- conflicting emotional signals;
- understatement;
- emotion hidden behind slang, punctuation, emojis, hashtags, memes, or casual tweet style;

Generation rules:
- Generate realistic short English tweets.
- Make examples difficult but still clearly labelable by a careful human.
- If an example could reasonably fit two labels, rewrite it to make the dominant label clearer.
- Keep sarcasm natural, not formulaic.
- Keep the label distribution reasonably diverse.

Return valid JSON only, no markdown, no comments:
{{"examples": [{{"id": 1, "input": "...", "output": "anger"}}]}}
"""

GSM8K_STANDARD_RULES = """
You are an expert synthetic data generator. Create exactly {num_samples} GSM8K-style math problems.

Problem description:
{problem_description}

Task: Given a grade-school math word problem, produce ONLY the final numeric answer.

Input format:
- A single, self-contained word problem written in plain English.
- All necessary information to solve the problem is embedded in the text.

Output format:
- The final numeric answer only (integer or decimal).
- No units, no punctuation, no labels like "Answer:" or "Final answer:".
- Examples of valid outputs: 42  |  3.5  |  100

Generation rules:
- Every problem must be fully solvable from its own text alone — no outside knowledge needed.
- Each problem must have a unique, unambiguous numeric answer.
- Vary problem length (2–5 sentences) and surface theme (food, money, sports, school, etc.).
- All numbers in the problem are relevant and should be used to reach the answer.
- Do NOT write reasoning, chain-of-thought, units, punctuation after the number, or any label.

Return valid JSON only, no markdown, no comments:
{{"examples": [{{"id": 1, "input": "Math problem description", "output": "42"}}]}}
"""

GSM8K_CORNER_CASE_RULES = """
You are an expert synthetic data generator. Create exactly {num_samples} GSM8K-style corner-case math problems.

Problem description:
{problem_description}

Task: Given a grade-school math word problem, produce ONLY the final numeric answer.

Input format:
- A single, self-contained word problem written in plain English.
- All necessary information to solve the problem is embedded in the text.

Output format:
- The final numeric answer only (integer or decimal).
- No units, no punctuation, no labels like "Answer:" or "Final answer:".
- Examples of valid outputs: 42  |  3.5  |  100

Corner-case categories — cover all 8 types, distributing {num_samples} examples across them:

1. irrelevant_numbers
   The problem contains one or more numbers that must be IGNORED to get the correct answer.
   
2. multi_step_arithmetic
   Solving requires TWO OR MORE sequential arithmetic operations.
   No single operation on the given numbers yields the answer directly.

3. reverse_operation
   The problem gives a RESULT and asks for an original or missing value.
   Solver must work backwards (e.g., subtract instead of add).

4. unit_conversion
   Numbers are given in mixed units; the solver must convert before computing.
   Keep conversions simple (minutes↔hours, cents↔dollars, cm↔m).

5. hidden_constraint
   A condition in the problem text restricts WHICH quantities count.
   Example: "Only items bought on Monday count." Quantities bought on other days must be ignored.

6. remaining_amount
   The problem involves additions AND removals over time.
   The question asks what is LEFT, not the running total.

7. grouped_quantities
   Multiple categories or groups are described, but the question asks about ONLY ONE group.

Generation rules:
- Every problem must be fully solvable from its own text alone — no outside knowledge needed.
- Use only grade-school arithmetic: +, −, ×, ÷. No algebra, geometry, or probability.
- Make distractor numbers plausible and tempting to misuse, but clearly irrelevant when read carefully.
- Each problem must have a unique, unambiguous numeric answer.
- Vary problem length (2–5 sentences) and surface theme (food, money, sports, school, etc.).
- Do NOT reveal the corner-case category inside the problem text.
- Do NOT write reasoning, chain-of-thought, units, punctuation after the number, or any label.

Return valid JSON only, no markdown, no comments:
{{"examples": [{{"id": 1, "input": "Math problem description", "output": "42"}}]}}
"""

COMMON_GEN_STANDARD_RULES = """
You are an expert synthetic data generator.
Create exactly {num_samples} CommonGen-style examples.

Problem description: {problem_description}

Task:
Generate synthetic input-output pairs for concept-to-sentence generation.

Each example must contain:
- input: 3-5 lowercase English lemmas, comma-separated
- output: one grammatical, fluent, plausible English sentence that uses all input concepts

Rules for input concepts:
- Generate the concept set yourself.
- Use 3-5 common English lemmas.
- Use lowercase words only.
- Use comma-separated format.
- Do not use proper nouns.
- Prefer concepts that can naturally appear together in one realistic scene.

Rules for output sentence:
- Use all input concepts.
- The sentence must be natural, realistic, and fluent.
- The sentence must express a plausible scene or event.
- Do not simply list or mention the concepts.
- Do not create absurd or impossible scenes.


Return valid JSON only, no markdown, no comments:
{{"examples": [{{"input": "concept1, concept2, concept3", "output": "One sentence."}}]}}
"""

COMMON_GEN_CORNER_CASE_RULES = """
You are an expert synthetic data generator.
Create exactly {num_samples} CommonGen corner-case examples.

Problem description: {problem_description}

Task: Given 3-5 concepts, generate exactly one natural English sentence using all of them.
- input: 3-5 lowercase English lemmas, comma-separated
- output: one grammatical, fluent, plausible sentence
- morphological variants allowed (run -> running, child -> children)

Corner-cases are concept sets where the connection is non-obvious but a plausible sentence still exists.
Cover these types diversely:
1. unseen_combination - common concepts that rarely appear together
2. cross_domain_bridging - concepts from different domains (sports, cooking, technology, nature)
3. semantic_tension - concepts that seem contradictory but can be resolved realistically
4. polysemy_trap - at least one concept has multiple meanings; use one clearly
5. temporal_ordering - concepts imply a causal or temporal sequence

Rules:
- Common English lemmas only, no proper nouns.
- No absurd, impossible, or fantasy scenes.
- Do not list concepts. Make the relation non-trivial but understandable.
- If a concept set cannot be connected plausibly, choose a different one.

Good: input: "chef, newspaper, umbrella"
      output: "The chef held an umbrella over the newspaper to keep the recipe dry."
Bad:  input: "chef, newspaper, umbrella"
      output: "A chef, a newspaper, and an umbrella are there."

Return valid JSON only, no markdown, no comments:
{{"examples": [{{"id": 1, "input": "concept1, concept2, concept3", "output": "One sentence."}}]}}
"""

SQUAD_V2_STANDARD_RULES = """
You are an expert synthetic data generator. Create exactly {num_samples} SQuAD v2 examples.

Problem description: {problem_description}

Task: Given a context and a question, answer using only the context, or output "unanswerable" if the answer is not supported.
- input: "Context: ... Question: ..."
- output: a short answer span from the context, or exactly "unanswerable"

Rules:
- For answerable examples, the output must be a short phrase explicitly present in the context.
- For unanswerable examples, the context must not contain the answer to the question.
- Include a mix of answerable and unanswerable examples.
- Use exactly "unanswerable" when no answer is supported.
- Contexts should be 3-6 sentences on varied topics (history, science, geography, etc.).

Good (answerable):
input: "Context: The Eiffel Tower was built in 1889 and is located in Paris. Question: Where is the Eiffel Tower located?"
output: "Paris"

Good (unanswerable):
input: "Context: The Eiffel Tower was built in 1889 and is located in Paris. Question: Who designed the Eiffel Tower?"
output: "unanswerable"

Return valid JSON only, no markdown, no comments:
{{"examples": [{{"id": 1, "input": "Context: passage text. Question: question text.", "output": "answer span or unanswerable"}}]}}
"""

SQUAD_V2_CORNER_CASE_RULES = """
You are an expert synthetic data generator. Create exactly {num_samples} SQuAD v2 corner-case examples.

Problem description: {problem_description}

Task: Given a context and a question, answer using only the context, or output "unanswerable" if the answer is not supported.
- input: "Context: ... Question: ..."
- output: a short answer span from the context, or exactly "unanswerable"

Corner-cases are examples where the context contains plausible distractors and the model must verify whether the answer is actually supported.

Cover these types diversely:
1. plausible_wrong_candidate - context contains a plausible but incorrect answer candidate
2. related_but_unanswerable - context discusses the topic but does not contain the answer
3. coreference_resolution - answer requires resolving pronouns or references
4. multi_sentence_evidence - answer requires connecting information across nearby sentences
5. entity_date_location_number_distractor - similar entities, dates, locations, or numbers appear in context
6. unstated_relation - question asks about a relation not stated in the context
7. negation_or_exception - context includes negation, exclusion, or exception wording

Rules:
- For answerable examples, the output must be explicitly supported by the context; keep it short and span-like.
- For unanswerable examples, the context must include plausible related distractors but not the correct answer.
- Include a mix of answerable and unanswerable examples.
- Use exactly "unanswerable" when no answer is supported.

Good (answerable):
input: "Context: Dr. Rivera presented her research in Paris in 2018. Her assistant Maya later presented a summary in Berlin in 2020. Question: Where did Dr. Rivera present her research?"
output: "Paris"

Good (unanswerable):
input: "Context: Dr. Rivera presented her research in Paris in 2018. Her assistant Maya later presented a summary in Berlin in 2020. Question: Where was Dr. Rivera born?"
output: "unanswerable"

Bad:
input: "Context: Dr. Rivera presented her research in Paris in 2018. Her assistant Maya later presented a summary in Berlin in 2020. Question: Where was Dr. Rivera born?"
output: "Paris"

Return valid JSON only, no markdown, no comments:
{{"examples": [{{"id": 1, "input": "Context: passage text. Question: question text.", "output": "answer span or unanswerable"}}]}}

Create exactly {num_samples} examples. Each must include only "id", "input", "output".
"""

XSUM_STANDARD_RULES = """
You are an expert synthetic data generator. Create exactly {num_samples} XSum-style examples.

Problem description: {problem_description}

Task: Given a short news-style article, write exactly one sentence summarizing the main point.
- input: a short news-style article (4-8 sentences)
- output: one concise sentence capturing the main event

Rules:
- Write a realistic news-style article on a varied topic (politics, science, sports, business, etc.).
- The summary must be exactly one sentence and faithfully reflect the article's main point.
- Do not copy any sentence verbatim from the article — paraphrase clearly.
- Include only information that appears in the article.
- The main event should be clearly stated and easy to identify.

Return valid JSON only, no markdown, no comments:
{{"examples": [{{"id": 1, "input": "Short news article.", "output": "One sentence summary."}}]}}
"""

XSUM_CORNER_CASE_RULES = """
You are an expert synthetic data generator. Create exactly {num_samples} XSum-style corner-case examples.

Problem description: {problem_description}

Task: Given a short news-style article, write exactly one sentence summarizing the main point.
- input: a short news-style article
- output: one concise sentence capturing the main event

Cover these corner-case types diversely:
1. main_event_hidden - the main event is buried in secondary details
2. contrast_or_concession - article contains although, however, or despite
3. cause_vs_consequence - cause and result can be confused
4. similar_entities - multiple people or groups have similar roles
5. temporal_or_numeric_detail - a date, amount, or number changes the meaning
6. proposal_vs_decision - a proposal must not be summarized as a final decision
7. accusation_vs_fact - an allegation must not be summarized as confirmed fact
8. expected_vs_actual - expected outcome differs from what actually happened

Rules:
- Write a realistic, information-dense article that requires careful summarization.
- The summary must be exactly one sentence, faithful, and with no facts outside the article.
- Do not copy a sentence verbatim.
- Preserve polarity, causality, and uncertainty.

Return valid JSON only, no markdown, no comments:
{{"examples": [{{"id": 1, "input": "Short news article.", "output": "One sentence summary."}}]}}
Create exactly {num_samples} examples. Each must include only "id", "input", "output".
"""

DATASET_STANDARD_RULES: dict[str, str] = {
    "common_gen": COMMON_GEN_STANDARD_RULES,
    "gsm8k": GSM8K_STANDARD_RULES,
    "tweeteval": TWEETEVAL_STANDARD_RULES,
    "squad_v2": SQUAD_V2_STANDARD_RULES,
    "xsum": XSUM_STANDARD_RULES,
}

DATASET_CORNER_CASE_RULES = {
    "tweeteval": TWEETEVAL_CORNER_CASE_RULES,
    "gsm8k": GSM8K_CORNER_CASE_RULES,
    "common_gen": COMMON_GEN_CORNER_CASE_RULES,
    "squad_v2": SQUAD_V2_CORNER_CASE_RULES,
    "xsum": XSUM_CORNER_CASE_RULES,
}


def get_standard_rules(dataset_name: str | None) -> str | None:
    if dataset_name is None:
        return None

    return DATASET_STANDARD_RULES.get(dataset_name.lower())


def get_corner_case_rules(dataset_name: str | None) -> str | None:
    if not dataset_name:
        return None
    return DATASET_CORNER_CASE_RULES.get(dataset_name.lower())
