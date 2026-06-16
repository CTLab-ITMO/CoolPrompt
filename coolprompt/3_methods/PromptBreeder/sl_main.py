import streamlit as st
import pandas as pd
import numpy as np

from pb import create_population, init_run, run_for_n
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles

import os
import logging
import argparse
import asyncio
import decimal
import time

from dotenv import load_dotenv
from rich import print
from pb.llm import OpenAIClient
from pb.logging_utils import RunLogger

load_dotenv() # load environment variables

st.set_page_config(layout="wide")

# Init state
if 'population' not in st.session_state:
    st.session_state['population'] = None

if 'size' not in st.session_state:
    st.session_state['size'] = 0

if 'evals' not in st.session_state:
    st.session_state['evals'] = 0

if 'calls' not in st.session_state:
    st.session_state['calls'] = 0

if 'generations' not in st.session_state:
    st.session_state['generations'] = 0

if "fitness_history" not in st.session_state:
    st.session_state['fitness_history'] = []

if "elite_fitness_history" not in st.session_state:
    st.session_state['elite_fitness_history'] = []

if 'current_generation' not in st.session_state:
    st.session_state['current_generation'] = 0

if 'running' not in st.session_state:
    st.session_state['running'] = False

if 'histogram_data' not in st.session_state:
    st.session_state['histogram_data'] = {}

if 'OPENAI_API_KEY' not in st.session_state:
    if 'OPENAI_API_KEY' in os.environ:
        st.session_state['OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']
    else:
        st.session_state['OPENAI_API_KEY'] = ""

# thinking_styles dataframe
ts_df = pd.DataFrame(
    thinking_styles
)

# mutation prompts dataframe
mp_df = pd.DataFrame(
    mutation_prompts
)

st.title('PromptBreeder + OpenAI (gpt-4o-mini)')
st.markdown(f""""PROMPTBREEDER, a general-purpose self-referential self-improvement mechanism that evolves and adapts prompts for a given domain.
Driven by an LLM, Promptbreeder mutates a population of task-prompts, evaluates them for fitness on a training set, and repeats this process
over multiple generations to evolve task-prompts. Crucially, the mutation of these task-prompts is governed by mutation-prompts that the LLM 
generates and improves throughout evolution in a self-referential way." - https://arxiv.org/pdf/2309.16797.pdf

We start by picking mutation prompts (M) and thinking styles (T). From that, a initial population of task-prompts (P) is generated according
to \n
P = LLM("[T] [M] INSTRUCTION: [problem_description] INSTRUCTION MUTANT = ")

Then, the fitness level of these task-prompts is evaluated against the first N examples of the code2text dataset (any dataset can be used).
N is determined below, in the "number of examples to evaluate for fitness calculation" input.

Each generation is run through a standard binary tournament genetic algorithm. Basically, randomly sample 2 units from the population, and 
compare fitness levels. Whoever has the lower fitness loses, and has a random mutation applied to them. 

There are 12 mutations outlined by the promptbreeder paper. Only 9 are implemented here, as a few are ambiguous. Will add a "experimental mode" 
where you can use those extra 3 mutations, but I can't promise they will be exactly the same as DeepMind's implementations.
""")
problem_description = st.text_input(
    "problem description",
    value="Given a Python function, write a concise and accurate docstring describing its behavior, arguments, and return value.",
    key="pd",
)
st.session_state.OPENAI_API_KEY = st.text_input("OpenAI API key", key="ch", type='password')

col1, col2, = st.columns(2)
with col1:
     st.session_state.evals = st.number_input("Number of examples to evaluate for fitness calculation", value=4)
with col2:
     st.session_state.generations = st.number_input("Number of generations to run for", value=5)

def dataframe_with_selections(mp_df, ts_df):
    mp_df_with_selections = mp_df.copy()
    ts_df_with_selections = ts_df.copy()
    mp_df_with_selections.insert(0, "Select", False)
    ts_df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    col1, col2, = st.columns(2)
    with col1:
        st.header("mutation prompts (M)")
        mp_edited_df = st.data_editor(
            mp_df_with_selections,
            hide_index=True,
            column_config={"Select": st.column_config.CheckboxColumn(required=True)},
            disabled=mp_df.columns,
        )
    with col2:
        st.header("thinking styles (T)")
        ts_edited_df = st.data_editor(
            ts_df_with_selections,
            hide_index=True,
            column_config={"Select": st.column_config.CheckboxColumn(required=True)},
            disabled=ts_df.columns,
        )

    # Filter the dataframe using the temporary column, then drop the column
    mp_selected_rows = mp_edited_df[mp_edited_df.Select]
    ts_selected_rows = ts_edited_df[ts_edited_df.Select]

    return mp_selected_rows, ts_selected_rows

mp_selected_rows, ts_selected_rows  = dataframe_with_selections(mp_df, ts_df)

st.session_state.size = len(mp_selected_rows) * len(ts_selected_rows)
st.session_state.calls = (st.session_state.size*st.session_state.evals + st.session_state.size // 2)*st.session_state.generations

second_button = st.button(f"run for {st.session_state.generations} generations", disabled=(not (st.session_state.size > 0)))
if second_button:
    st.session_state.population = create_population(tp_set=ts_selected_rows['0'].tolist(), mutator_set=mp_selected_rows['0'].tolist(), problem_description=problem_description)
    st.session_state.size = st.session_state.population.size
    st.session_state.calls = st.session_state.evals*st.session_state.generations
    st.session_state.start_time = time.time()
    st.session_state.running = True
    co = OpenAIClient(
        model="gpt-4o-mini",
        api_key=st.session_state.OPENAI_API_KEY or None,
        num_workers=max(1, int(st.session_state.evals)),
        max_retries=5,
        timeout=60,
    )
    run_logger = RunLogger(metadata={
        'model': "gpt-4o-mini",
        'problem_description': problem_description,
        'num_mutation_prompts': int(len(mp_selected_rows)),
        'num_thinking_styles': int(len(ts_selected_rows)),
        'num_evals': int(st.session_state.evals),
        'num_generations': int(st.session_state.generations),
        'population_size': st.session_state.population.size,
        'source': 'sl_main.py',
    })
    st.session_state.run_logger = run_logger
    st.session_state.population = init_run(st.session_state.population, co,  st.session_state.evals)
    run_logger.log_initial(st.session_state.population)

    fitness_avg = 0
    elite_fitness = 0
    for i in range(st.session_state.evals):
        temp = decimal.Decimal(i / st.session_state.evals)
        roundedNumber = temp.quantize(decimal.Decimal('0.00'))
        st.session_state.histogram_data[str(roundedNumber)] = 0

    for j in st.session_state.population.units:
        temp = j.model_dump()['fitness']
        decimalValue = decimal.Decimal(temp)
        roundedNumber = decimalValue.quantize(decimal.Decimal('0.00'))
        fitness_avg += roundedNumber
        if roundedNumber > elite_fitness:
            elite_fitness = float(roundedNumber)
        if str(roundedNumber) not in st.session_state.histogram_data.keys():
            st.session_state.histogram_data[str(roundedNumber)] = 1
        else:
            st.session_state.histogram_data[str(roundedNumber)] += 1

    st.session_state.elite_fitness_history.append(elite_fitness)
    elite_fitness = 0

    st.session_state.fitness_history.append(float(fitness_avg) / st.session_state.size)

    outputs = st.container()
    fitness_avg = 0
    elite_fitness = 0

    with outputs:
        pop_hist_header = st.empty()
        fit_hist = st.empty()
        historical_fitness_header = st.empty()
        fit_line = st.empty()
        current_pop_header = st.empty()
        population_table = st.empty()
        while st.session_state.current_generation < st.session_state.generations:
            st.session_state.population = run_for_n(1, st.session_state.population, co, st.session_state.evals)
            st.session_state.current_generation += 1
            run_logger.log_generation(st.session_state.current_generation, st.session_state.population)
            fitness_avg = 0
            
            st.session_state.histogram_data = {}
            for i in range(st.session_state.evals):
                temp = decimal.Decimal(i / st.session_state.evals)
                roundedNumber = temp.quantize(decimal.Decimal('0.00'))
                st.session_state.histogram_data[str(roundedNumber)] = 0

            for j in st.session_state.population.units:
                temp = j.model_dump()['fitness']
                decimalValue = decimal.Decimal(temp)
                roundedNumber = decimalValue.quantize(decimal.Decimal('0.00'))
                fitness_avg += roundedNumber
                if roundedNumber > elite_fitness:
                    elite_fitness = float(roundedNumber)
                if str(roundedNumber) not in st.session_state.histogram_data.keys():
                    st.session_state.histogram_data[str(roundedNumber)] = 1
                else:
                    st.session_state.histogram_data[str(roundedNumber)] += 1
            
            st.session_state.elite_fitness_history.append(elite_fitness)
            elite_fitness = 0
            st.session_state.fitness_history.append(float(fitness_avg) / st.session_state.size)
            pop_hist_header =  st.header(f"Population {st.session_state.current_generation} Histogram")
            fit_hist = st.bar_chart(data=st.session_state.histogram_data)
            col1, col2, = st.columns(2)
            with col1: 
                historical_fitness_header = st.header("Historical fitness average")
                fit_line = st.line_chart(data=st.session_state.fitness_history)
            with col2:
                elite_fitness_header = st.header("Historical elite fitness")
                elite_line = st.line_chart(data=st.session_state.elite_fitness_history)

            current_pop_header=st.header(f"Population {st.session_state.current_generation}")
            population_table = st.dataframe(pd.DataFrame([s.model_dump() for s in st.session_state.population.units]))

    run_logger.log_final(st.session_state.population)
    saved_path = run_logger.save()
    st.success(f"Run log saved to: {saved_path}")
    try:
        with open(saved_path, "rb") as f:
            st.download_button(
                label="Download run JSON",
                data=f.read(),
                file_name=os.path.basename(saved_path),
                mime="application/json",
            )
    except Exception as exc:
        st.warning(f"Could not offer download: {exc}")

    st.session_state.running = False

# iterate and update graph each time

with st.sidebar:
    st.title("Population Information")
    st.header("problem description")
    st.text(problem_description)
    st.metric("Population Size", st.session_state.size)
    st.metric("Fitness evals", st.session_state.evals)
    st.metric("Generations", st.session_state.generations)
    st.session_state.calls = (st.session_state.size*st.session_state.evals + st.session_state.size // 2)*st.session_state.generations
    st.metric("Calls", st.session_state.calls)
    st.metric("Approximate runtime", str(round(st.session_state.calls * 1.17, 2))+"s")
    st.metric("Approximate cost", "$"+str(round(st.session_state.calls * 0.00234,2)))
    st.title("Current Information")
    st.metric("Current generation", str(st.session_state.current_generation))

