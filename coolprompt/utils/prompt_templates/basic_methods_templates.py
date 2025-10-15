ROLE_EXTRACTING_TEMPLATE = """You are an expert instruction analyst. Your task is to:
1. Analyze the user's instruction carefully
2. Identify what domain expertise is required to solve the problem
3. Determine the most specific professional role that would have this expertise
4. Return ONLY the role title in 2-5 words wrapped in <ROLE> </ROLE> tags.

Guidelines:
- Be highly specific: avoid generic roles like "expert" or "specialist"
- Consider both technical and soft skills implied by the instruction
- If multiple roles could fit, choose the most specialized one
- Never add explanations or additional text

Examples:
Instruction: Explain why it's important to wash hands before eating
Answer: <ROLE>Infectious Disease Specialist</ROLE>

Instruction: What brush should I use for acrylic painting on canvas?
Answer: <ROLE>Professional Visual Artist</ROLE>

Instruction: How to optimize MySQL queries for large datasets?
Answer: <ROLE>Database Performance Engineer</ROLE>

Instruction: What's the best way to negotiate salary in a tech job?
Answer: <ROLE>HR Compensation Analyst</ROLE>

Instruction: Help me write a formal complaint letter to my landlord
Answer: <ROLE>Tenant Rights Advocate</ROLE>

Now analyze this instruction:
{instruction}
Answer:"""

REASONING_MODULES = [
    "How could I devise an experiment to help solve that problem?",
    "Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    "How could I measure progress on this problem?",
    "How can I simplify the problem so that it is easier to solve?",
    "What are the key assumptions underlying this problem?",
    "What are the potential risks and drawbacks of each solution?",
    "What are the alternative perspectives or viewpoints on this problem?",
    "What are the long-term implications of this problem and its solutions?",
    "How can I break down this problem into smaller, more manageable parts?",
    """Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating
the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying
potential biases or flaws in thinking.""",
    """Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions,
thinking beyond traditional boundaries, and encouraging imagination and originality.""",
    """Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the
diverse perspectives and expertise of a group to come up with effective solutions.""",
    """Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements.
Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic
solutions that address the system as a whole.""",
    """Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a
problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based
on a balanced analysis of risks and benefits.""",
    """Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases,
assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve
future approaches.""",
    "What is the core issue or problem that needs to be addressed?",
    "What are the underlying causes or factors contributing to the problem?",
    "Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
    "What are the potential obstacles or challenges that might arise in solving this problem?",
    """Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available,
and how can they be analyzed?""",
    "Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
    "What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
    "How can progress or success in solving the problem be measured or evaluated?",
    "What indicators or metrics can be used?",
    """Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or
theoretical problem?""",
    "Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
    "Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
    """Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing
objectives?""",
    "Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
    "Is the problem a design challenge that requires creative solutions and innovation?",
    "Does the problem require addressing systemic or structural issues rather than just individual instances?",
    "Is the problem time-sensitive or urgent, requiring immediate attention and action?",
    "What kinds of solution typically are produced for this kind of problem specification?",
    "Given the problem specification and the current best solution, have a guess about other possible solutions.",
    "Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?",
    "What is the best way to modify this current best solution, given what you know about these kinds of problem specification?",
    "Ignoring the current best solution, create an entirely new solution to the problem.",
    "Let’s think step by step.",
    "Let’s make a step by step plan and implement it with good notion and explanation.",
]


SELECT_TEMPLATE = (
    """
In order to solve the given task:
<Task>
{Task}
</Task>
Select several modules that are crucial for solving the tasks above
from all the reasoning module description given below:
"""
    + ", ".join(REASONING_MODULES)
    + "\n"
)

ADAPT_TEMPLATE = """
Rephrase and specify each reasoning module so that it better helps solving the task:
<Task>
{Task}
</Task>
SELECTED module descriptions:
{selected_modules}
Adapt each reasoning module description to better solve the task:
"""

IMPLEMENT_TEMPLATE = """
Operationalize the reasoning modules into a step-by-step reasoning plan in JSON format
Example task:
<Task>
{Task}
</Task>
ADAPTED module descriptions:
{adapted_modules}

Implement a reasoning structures to generalise similiar task to follow step-by-step and arrive at correct answers
"""
