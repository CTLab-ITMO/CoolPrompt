OPRO_META_TEMPLATE = """\
You are a prompt engineer. Your task is to generate an \
instruction that achieves a high score on a given task.

## Task Demonstrations
Below are some input-output pairs that demonstrate the task:

{task_demonstrations}

## Trajectory
Below is a history of instructions that were tried for this \
task, along with their scores. A higher score is better. \
The entries are sorted from worst to best.

{trajectory}

## Instructions
Generate a new instruction that will achieve a higher score \
than all previous attempts. The new instruction should be at \
most {max_tokens} tokens long.

Output the new instruction between <prompt> and </prompt> tags.

<prompt>your improved instruction here</prompt>"""
