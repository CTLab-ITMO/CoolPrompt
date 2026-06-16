import random
import re
import os
from pb.types import Population, EvolutionUnit
from typing import List, Dict, Optional
# from sentence_transformers import SentenceTransformer, util
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles

# ``Client`` was originally a ``cohere.Client``. With the OpenAI adapter
# in place, we just type-hint against ``object`` to avoid the dependency.
Client = object

from dotenv import load_dotenv
from rich import print

load_dotenv()

# Examples used by the lamarckian ``working_out_task_prompt`` mutator. They
# are injected by ``pb.configure_dataset`` once the user picks a dataset.
_DATASET_NAME: Optional[str] = None
_DATASET_EXAMPLES: List[Dict] = []


def set_dataset_examples(name: str, examples: List[Dict]) -> None:
    """Register the active dataset's worked examples for use in mutators."""
    global _DATASET_NAME, _DATASET_EXAMPLES
    _DATASET_NAME = name
    _DATASET_EXAMPLES = list(examples)

# need below for estimation_distribution_mutation, not currently using.
# model = SentenceTransformer('multi-qa-distilbert-cos-v1')
# print(model) 

# Direct Mutation mutators
def zero_order_prompt_gen(unit: EvolutionUnit, problem_description: str, model: Client, **kwargs) -> EvolutionUnit:
    """Generates a new task-prompt P by concatenating the problem description D with the prompt 
    'a list of 100 hints:'. New task-prompt P is the first generated hint.
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    result = model.generate(problem_description + " An ordered list of 100 hints: ")
    # search for the pattern "anything after 1. and before 2."
    pattern = r"1\.(.*?)2\."
    match = re.search(pattern, result[0].text, re.DOTALL)
    if match: 
        # return the first match
        unit.P = match.group(1).strip()
    else: 
        unit.P = ""
    
    return unit 

def first_order_prompt_gen(unit: EvolutionUnit, model: Client, **kwargs) -> EvolutionUnit:
    """Concatenate the mutation prompt M to the parent task-prompt P and pass it to the LLM to produce P'
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    unit.P = model.generate(unit.M + " " + unit.P)[0].text
    return unit
    
# Estimation of Distribution Mutation - there is a variation of this called EDA rank
# and index mutation. I didn't implement it.
def estimation_distribution_mutation(unit: EvolutionUnit, population_units: List[EvolutionUnit], **kwargs) -> EvolutionUnit:
    """ Provide a filtered and numbered list of the current population of task-prompts to the LLM and ask it to continue this list with new task-prompts.
    The List is filtered via ensuring that no two task-prompts have a score of >0.95 via BERT embedding cosine similarities.
    The List is randomly ordered.  

    NOTE: I am confused by this one. Does this mutate the entire population? What values of the continued list from the LLM do I use as prompts? randomly sampled?
    Not going to implement this one yet. Maybe should email someone. 
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    pass
def lineage_based_mutation(unit: EvolutionUnit, elites: List[EvolutionUnit], model: Client, **kwargs) -> EvolutionUnit:
    """Using the stored history of best units, provide the LLM this list in chronological order to produce a novel prompt as continuation.
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    HEADING = "GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY \n "
    # made a choice not to format it with newlines, could change later.
    ITEMS = "\n".join(["{}. {}".format(i+1, x.P) for i, x in enumerate(elites)])
    unit.P = model.generate(HEADING + ITEMS)[0].text
    
    return unit

# Hypermutation
def zero_order_hypermutation(unit: EvolutionUnit, problem_description: str, model: Client, **kwargs) -> EvolutionUnit:
    """ Concatenate the original problem_description to a randomly sampled thinking-style and feed it to the LLM to generate a new mutation-prompt.
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    RANDOM_THINKING_STYLE = random.sample(thinking_styles, 1)[0]
    unit.M = model.generate(problem_description + " " + RANDOM_THINKING_STYLE)[0].text
    return unit

def first_order_hypermutation(unit: EvolutionUnit, model: Client, **kwargs) -> EvolutionUnit:
    """ Concatenate the hyper-mutation prompt "Please summarize and improve the following instruction:"
    to a mutation-prompt to that the LLM generates a new mutation-prompt. This new mutation-prompt is then 
    instantly applied to the task-prompt of that unit.

    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    HYPER_MUTATION_PROMPT="Please summarize and improve the following instruction: "
    unit.M = model.generate(HYPER_MUTATION_PROMPT + unit.M)[0].text
    unit.P = model.generate(unit.M + " " + unit.P)[0].text
    return unit 


# Lamarckian Mutation
def working_out_task_prompt(unit: EvolutionUnit, model: Client, **kwargs) -> EvolutionUnit:
    """A 'lamarckian' mutation operator similar to instruction induction in APE.

    Show the LLM a worked example (input + reference target) drawn from
    the active dataset and ask it to reverse-engineer the underlying
    task-prompt.

    Returns:
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    if not _DATASET_EXAMPLES:
        # No dataset configured -- skip mutation gracefully.
        return unit

    example = random.sample(_DATASET_EXAMPLES, 1)[0]

    unit.P = model.generate(
        "I gave a friend an instruction and some advice. Here are the correct examples of his workings out:\n"
        f"INPUT:\n{example.get('input', '')}\n"
        f"OUTPUT:\n{example.get('target', '')}\n"
        "The instruction was: "
    )[0].text
    return unit

# Prompt crossover and context shuffling. These happen AFTER mutation operators. 
def prompt_crossover(**kwargs):
    """
    After a mutation operator is applied, 

    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
def context_shuffling(**kwargs):
    """
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """

# omitting the estimation_distribution_mutation
MUTATORS = [
    zero_order_prompt_gen,
    first_order_prompt_gen,
    #estimation_distribution_mutation,
    lineage_based_mutation,
    zero_order_hypermutation,
    first_order_hypermutation,
    working_out_task_prompt
]

POST_MUTATORS = [
    prompt_crossover,
    context_shuffling
]

def mutate(population: Population, model: Client) -> Population:
    """Select and apply a random mutator"""
    # steps
    # 1. parse through the population, grouping each evo unit by 2
    # 2. for each pair of evo units, using a uniform distribution, select a random mutator (of the 9)
    # 3. mutate and populate population.units

    # make index pairs
    indices = [i for i in range(len(population.units))]
    random.shuffle(indices)
    pairs = [indices[2*x:2*x+2] for x in range(len(indices) // 2)]

    # binary tourmanent genetic algorithm
    for i in range(len(pairs)):

        first_unit = population.units[pairs[i][0]]
        second_unit = population.units[pairs[i][1]]

        print("%"*77)
        print("First unit: \n")
        print(first_unit)
        print("%"*77)
        print("Second unit: \n")
        print(second_unit)

        # determine which unit has the higher fitness. Since I am currently testing and want to preserve the # of calls I am making to the LLM, there 
        # is a decent chance that I will hit equal fitness levels. in that case, first unit wins and second unit loses.
        
        # TODO: clean this up
        FIRST_WON = False
        if first_unit.fitness >=  second_unit.fitness:
            # loser gets mutated.
            FIRST_WON = True
            mutation_input = second_unit
        else:
            mutation_input = first_unit

        data = {
            'unit' : mutation_input,
            'model' : model,
            'elites' : population.elites,
            'problem_description': population.problem_description,
        }

        # uniformly pick and call a random mutation operator on the losing unit
        random_mutator = random.sample(MUTATORS, 1)[0]
        print(f"MUTATING: {mutation_input} with {random_mutator.__name__}")

        random_mutator(**data)

    return population