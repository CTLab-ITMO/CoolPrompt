OPRO_META_TEMPLATE = """You are a prompt engineer. Below is a \
history of instructions that were tried for a task, along with \
their scores. A higher score is better.

{trajectory}

Generate a new instruction that will achieve a higher score \
than all previous attempts. The new instruction should be at \
most {max_tokens} tokens long.

Output the new instruction between <prompt> and </prompt> tags.

<prompt>your improved instruction here</prompt>"""
