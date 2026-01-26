#!/bin/bash

#--model_name AnatoliiPotapov/T-lite-instruct-0.1 \
# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
python3 run.py \
    --dataset_name tweeteval \
    --task classification \
    --metric f1 \
    --problem_description "The task at hand involves classifying the sentiment expressed in a given piece of text. The objective is to analyze the emotional tone conveyed by the words and phrases within the text and categorize it into predefined sentiment labels. These labels typically include emotions such as joy, sadness, anger, optimism, and other relevant sentiments that reflect human feelings. 

  The input text may range from quotes, personal statements, social media posts, or any other written content where emotions are likely to be expressed. The classification process requires an understanding of context, nuances of language, and the ability to interpret implied meanings. For instance, a text may use sarcasm, irony, or cultural references that could alter the sentiment conveyed, thus making it essential to grasp the overall context in which the words are used. 

  In the provided examples, we see various expressions of sentiment: a motivational quote expressing positivity is categorized as 'optimism', while a humorous comment about spelling mistakes in the context of modern technology is classified as 'anger'. Other examples include expressions of joy, sadness, and anger based on the context of the statements made. This highlights that the sentiment classification task is not merely about identifying positive or negative words but involves a deeper analysis of the emotional context surrounding them. 

  Overall, the sentiment classification task plays a crucial role in sentiment analysis applications, including social media monitoring, customer feedback analysis, and emotional content understanding, allowing for the extraction of valuable insights from textual data." \
    --population_size 10 \
    --num_epochs 5 \
    --output_path ./4o_mini_spell_outputs/tweeteval
