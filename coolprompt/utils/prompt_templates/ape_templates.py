APE_PARAPHRASE_TEMPLATE = """You are a prompt engineer. \
Generate a variation of the instruction below that preserves \
its semantic meaning but uses different wording, structure, \
or phrasing. The new instruction should be at most \
{max_tokens} tokens long.

Current instruction:
{prompt}

Output the new instruction between <prompt> and </prompt> tags.

<prompt>your paraphrased instruction here</prompt>"""
