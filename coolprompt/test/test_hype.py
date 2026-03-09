import json
import logging
import os
from pathlib import Path
import sys

model_name = "qwen3_v2"
meta_dir = f"logs_hype_eval_{model_name}"
os.makedirs(meta_dir, exist_ok=True)

sys.path.append(str(Path(__file__).parent.parent.parent))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{meta_dir}/meta.txt"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

from coolprompt.assistant import PromptTuner  # noqa: 402

logger.info("Import completed")

tuner = PromptTuner()
logger.info("Initialization completed")

tasks_and_prompts = """bbh/boolean_expressions~Evaluate the result of a random Boolean expression.
bbh/hyperbaton~Order adjectives correctly in English sentences.
bbh/temporal_sequences~Answer questions about which times certain events could have occurred.
bbh/object_counting~Questions that involve enumerating objects and asking the model to count them.
bbh/disambiguation_qa~Clarify the meaning of sentences with ambiguous pronouns.
bbh/logical_deduction_three_objects~A logical deduction task which requires deducing the order of a sequence of objects.
bbh/logical_deduction_five_objects~A logical deduction task which requires deducing the order of a sequence of objects.
bbh/logical_deduction_seven_objects~A logical deduction task which requires deducing the order of a sequence of objects.
bbh/causal_judgement~Answer questions about causal attribution.
bbh/date_understanding~ Infer the date from context.
bbh/ruin_names~Select the humorous edit that 'ruins' the input movie or musical artist name.
bbh/word_sorting~Sort a list of words.
bbh/geometric_shapes~Name geometric shapes from their SVG paths.
bbh/movie_recommendation~Recommend movies similar to the given list of movies.
bbh/salient_translation_error_detection~Detect the type of error in an English translation of a German source sentence.
bbh/formal_fallacies~Distinguish deductively valid arguments from formal fallacies.
bbh/penguins_in_a_table~Answer questions about a table of penguins and their attributes.
bbh/dyck_languages~ Correctly close a Dyck-n word.
bbh/multistep_arithmetic_two~Solve multi-step arithmetic problems.
bbh/navigate~Given a series of navigation instructions, determine whether one would end up back at the starting point.
bbh/reasoning_about_colored_objects~Answer extremely simple questions about the colors of objects on a surface.
bbh/tracking_shuffled_objects_three_objects~A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.
bbh/tracking_shuffled_objects_five_objects~A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.
bbh/tracking_shuffled_objects_seven_objects~A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.
bbh/sports_understanding~Determine whether an artificially constructed sentence relating to sports is plausible or not.
bbh/snarks~Determine which of two sentences is sarcastic.
bbh/web_of_lies~Evaluate the truth value of a random Boolean function expressed as a natural-language word problem.
gsm8k~Solve the math word problem, giving your answer as an arabic numeral.
math~Solve the math word problem
medqa~Please use your domain knowledge in medical area to solve the questions.
mnli~In this task, you're given a pair of sentences, premise and hypothesis. Your job is to choose whether the two sentences clearly agree/disagree with each other, or if this cannot be determined.
mr~Please perform Sentiment Classification task
natural_instructions/task021~A question that is free of any grammatical or logical errors, should be labeled 'Yes.', otherwise it should be indicated as 'No.'. A question is grammatically correct if all its entities i.e. nouns, verbs, adjectives, prepositions, pronouns, adverbs are at appropriate position. A question is logically correct if the semantic makes sense.
natural_instructions/task050~You are given a sentence and a question in the input. If the information provided in the sentence is enough to answer the question, label "Yes.", otherwise label "No.". Do not use any facts other than those provided in the sentence while labeling "Yes." or "No.".
natural_instructions/task069~In this task, you will be shown a short story with a beginning, two potential middles, and an ending. Your job is to choose the middle statement that makes the story coherent / plausible by writing "1" or "2" in the output. If both sentences are plausible, pick the one that makes most sense.
openbookqa~Answer the following question: 
qnli~Define if the sentence entails the question.
samsum~Summarize the following text
sst-2~Please perform Sentiment Classification task.
trec~Classify this sentence based on the provided categories.
yahoo~Identify the most suitable category for this question."""

task2prompt = {}
for line in tasks_and_prompts.split("\n"):
    task, prompt = line.split("~")
    task2prompt[task] = prompt
meta_file = open(f"{meta_dir}/results_hype.txt", "a")
for task, prompt in task2prompt.items():
    logger.info(f"For the task {task} improving prompt:\n{prompt}")
    final_prompt = tuner.run(prompt)
    logger.info(f"Improved prompt:\n{final_prompt}")
    result = {"task": task, "prompt": final_prompt}
    meta_file.write(json.dumps(result) + "\n")
    meta_file.flush()
