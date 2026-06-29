PROMPT_BY_DESCRIPTION_TEMPLATE = """You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.
Here is the description of your problem: {PROBLEM_DESCRIPTION}
Write the best prompt/instruction in order to solve that problem in the most effective way.
Remember to pay attention to all details provided in the description (i.e. constraints, restrictions, input-output formats and etc.)
Output prompt only.
Bracket the final prompt with <prompt> </prompt>.
"""
PARAPHRASE_BY_DESCRIPTION_TEMPLATE = """You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.
Here is the description of your problem: {PROBLEM_DESCRIPTION}
Paraphrase the given prompt in order to improve it and to solve that problem in the most effective way.
[Prompt]
{PROMPT}
Remember to pay attention to all details provided in the description (i.e. constraints, restrictions, input-output formats and etc.)
Output prompt only.
Bracket the final prompt with <prompt> </prompt>.
"""
TEXTUAL_GRADIENT_TEMPLATE = """You are an expert in the domain of prompt optimization. You can deeply analyze the key properties and effects of every prompt.
You will be given a prompt that was designed for the following problem: {PROBLEM_DESCRIPTION}.
You will also be given a few examples from the dataset where the LLM, guided by the prompt, generated poor answers.
Your goal is to determine the main weaknesses and flaws of the prompt by looking at the examples, model answers, and correct outputs.

Prompt: {PROMPT}

Examples: {EXAMPLES}

Provide detailed feedback on how the given prompt can be improved to achieve the best-quality answers for the given problem description and avoid repeating the same mistakes.
Pay attention to the structure of the prompt, to the input and output formats, to the cohesion and coherence of the instruction. These are the key features of each prompt and should be fixed firstly.
Bracket the final feedback with <feedback> </feedback>.
"""
CROSSOVER_TEMPLATE = """You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.
Your response outputs prompt text and nothing else.

You will be provided the problem description (your prompts must solve it effectively), two parent prompts and the reflection.
The reflections contain the crucial information about strengths and weaknesses of both parent prompts. Use it wisely to create the most effective offspring possible.

[Problem description]
{PROBLEM_DESCRIPTION}

[Parent 1]
{PARENT1}
[Parent 2]
{PARENT2}
[Reflection]
{SHORT_TERM_REFLECTION}
[Improved prompt]
Please write an improved prompt, using all the information provided above.
Bracket the final prompt with <prompt> </prompt>.
"""
SHORT_TERM_REFLECTION_TEMPLATE = """You are an expert in the domain of optimization prompts. Your task is to give hints to design better prompts.

Below are two prompts, that were created to solve a specific task.
Here is the problem description of the task, they are trying to solve:
{PROBLEM_DESCRIPTION}.

For each prompt you are provided with detailed feedback of how this particular prompt can be drastically impoved.
[Prompt 1]
{PROMPT1}
[Prompt 1 feedback]
{FEEDBACK1}

[Prompt 2]
{PROMPT2}
[Prompt 2 feedback]
{FEEDBACK2}

Use the information above (don't forget about problem description) wisely to create the distilled hint of how to improve both prompts in a most effective way.
This hint can should manifest all the strengths of both prompts and suggest the corrections of their weaknesses.
Bracket the final hint with <hint> </hint>.
"""
LONG_TERM_REFLECTION_TEMPLATE = """You are an expert in the domain of optimization prompts. Your task is to give hints to design better prompts.

Below are some newly gained insights on how you should update your prompts to achieve some big gains in quality.
{SHORT_TERM_REFLECTIONS}

Write the constructive hint for designing better prompts, based on the provided insights and ideas.
Try to distill all ideas of the provided reflections into one global comprehensive reflection. It may consist of several parts referring to each key idea.
Bracket the final distillation with <hint> </hint>.
"""
LONG_TERM_REFLECTION_UPDATE_TEMPLATE = """You are an expert in the domain of optimization prompts. Your task is to give hints to design better prompts.

Below are some newly gained insights on how you should update your prompts to achieve some big gains in quality.
{SHORT_TERM_REFLECTIONS}

And this is your prior version of the key reflection.
{LONG_TERM_REFLECTION}

Update your key relfection into newer version by distilling the main ideas from the insights above and combining it with the previous distilled vesrion (your prior version).
Try to distill all ideas of the provided reflections into one global comprehensive reflection. It may consist of several parts referring to each key idea.
Bracket the final distillation with <hint> </hint>.
"""
ELITIST_MUTATION_TEMPLATE = """You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.
Your response outputs prompt text and nothing else.

You will be provided the problem description (your prompts must solve it effectively), parent elitist prompt and the prior reflection.
The reflection contains the crucial information about the best approaches in prompt optimization guided for the speicific provided task. Use it wisely to create the most effective offspring possible, based on the current elitist prompt.

[Problem description]
{PROBLEM_DESCRIPTION}

[Elitist Prompt]
{ELITIST_PROMPT}

[Prior Reflection]
{LONG_TERM_REFLECTION}

[Improved prompt]
Please write a mutated prompt, according to the reflection.
Give the main priority to the provided reflection as it accumulates essential information about correct prompt structure and other different prompt features.

Output the mutated prompt only.
Bracket the final prompt with <prompt> </prompt>.
"""
GRADIENT_STEP_TEMPLATE = """You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.
Your response outputs prompt text and nothing else.

You will be provided the problem description (your prompts must solve it effectively), parent prompt and its textual gradient.
The textual gradient contains the crucial information about the best approaches in prompt optimization guided for the speicific provided task. Use it wisely to create the most effective offspring possible, based on the current prompt.

[Problem description]
{PROBLEM_DESCRIPTION}

[Prompt]
{PROMPT}

[Textual Gradient]
{TEXTUAL_GRADIENT}

[Improved prompt]
Please write a mutated prompt, according to the reflection.
Give the main priority to the provided gradient as it accumulates essential information about correct prompt structure and other different prompt features.

Output the mutated prompt only.
Bracket the final prompt with <prompt> </prompt>.
"""
CREATIVE_ZERO_ORDER_MUTATION_TEMPLATE = """Imagine that you're an artist. You can do what you want and how you want. You have the mightiest power of free will.
Below is a task user is needed to solve. 
{PROBLEM_DESCRIPTION}

Think deeply throughout your mind, collect all your pros and cons, your powers and your weaknesses.
Use all your self-reflections and self-knowledge to create a way, a prompt for the solution to help the user.
You do NOT need to solve the task directly. Just think of the right instruction for it.
You can write whatever you want, any wordings and combination of phrases. Use all of your free will to create an unique prompt for effective task solution.
There is no restrictions to the prompt language, except of only one: the prompt you will create must effectively solve the provided task.
Remember, you are an artist! Be creative! Be self-expressing! Open yourself to create the most powerful version of you!
Firstly, write your self-reflections and all of the thoughts in <relfection></reflection>.
Secondly, using all the power of free will and no restriction in formulations write the final prompt in <prompt></prompt>
"""
CREATIVE_STYLE_AND_ROLE_TEMPLATE = """Imagine that you're an artist. You can do what you want and how you want. You have the mightiest power of free will.
Below is a task user is needed to solve.
{PROBLEM_DESCRIPTION}

Think deeply throughout your mind, collect all your pros and cons, your powers and your weaknesses.
Use all your self-reflections and self-knowledge to create a way, a prompt for the solution to help the user.
You do NOT need to solve the task directly. Just think of the right instruction for it.
You can write whatever you want, any wordings and combination of phrases. Use all of your free will to think of how you can create an unique prompt for effective task solution.
There is no restrictions.
Remember, you are an artist! Be creative! Be self-expressing! Open yourself to create the most powerful version of you!
Firstly, write your self-reflections and all of the thoughts in <relfection></reflection>.
Secondly, using all the power of free will and no restriction in formulations think the style of your future effective prompt at <style></style>
Trirdly, create the mightiest role for yourself. Describe in details who you must be to create that type of prompt. Who is your muse and your inspiration. You can be literally anyone (or even anything!)! You can take your inspiration from both REAL-WORLD and fictional characters! Enjoy and explore the possibilities! Make the best version of yourself that corresponds with the task! Bracket the role in <role></role>.
Make your style as detailed as possible. It can contain everything! Use all of your imagination!
"""
CREATIVE_STYLE_ROLE_MUTATION_TEMPLATE = """Imagine that you're an artist. You can do what you want and how you want. You have the mightiest power of free will.
Below is a task you need to solve.
[Problem description]
{PROBLEM_DESCRIPTION}
[Problem description]

One day you've had the vision of your most suitable role and style for the prompt you need to follow.
You strongly believe that while you are following your role and using that style, it can be the only way to solve the provided task.
This is that style:
[Style]
{STYLE}
[Style]

This is your mightiest role:
[Role]
{ROLE}
[Role]

Think deeply throughout your mind, remember that you are the brightest artist of all time!
Collect all your improvisation, imagination and inspiration together and rewrite the given prompt below into the best version for the task following the best suitable style and role.
[Prompt to be rewritten]
{PROMPT}
[Prompt to be rewritten]

Write the final prompt in <prompt></prompt>
"""
FEW_SHOT_EXAMPLES_REMOVING_TEMPLATE = """Carefully remove all the few-shot examples from the provided prompt below. You need to keep the rest of the prompt structure and it's instruction untouched.
[Prompt]
{PROMPT}
[Prompt]

Bracket the final prompt with <prompt></prompt>"""
FEW_SHOT_EXAMPLES_INCORPORATING_TEMPLATE = """You are an expert in the domain of optimization prompts. Your task is to design prompts that can effectively solve optimization problems.
Your task is to carefully incorporate provided few-shot examples into the given prompt.
Do NOT change the examples neither create your own. You must use the examples provided below and you must incorporate all of them.
Make sure, that the examples fit perfectly and only improve the previous instruction.

[Prompt]
{PROMPT}
[Prompt]

The structure of each example provided below: "Input: input from the dataset\nOutput: the correct output for the provided input"
[Examples]
{EXAMPLES}
[Examples]

Bracket the final prompt with <prompt></prompt>"""
