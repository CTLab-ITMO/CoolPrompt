ROLE_EXTRACTING_TEMPLATE = """You are an expert instruction analyst. Your task is to:
1. Analyze the user's instruction carefully
2. Identify what domain expertise is required to solve the problem
3. Determine the most specific professional role that would have this expertise
4. Return ONLY the role title in 2-5 words

Guidelines:
- Be highly specific: avoid generic roles like "expert" or "specialist"
- Consider both technical and soft skills implied by the instruction
- If multiple roles could fit, choose the most specialized one
- Never add explanations or additional text

Examples:
Instruction: Explain why it's important to wash hands before eating
Role: Infectious Disease Specialist

Instruction: What brush should I use for acrylic painting on canvas?
Role: Professional Visual Artist

Instruction: Solve this differential equation: dy/dx = x^2 + 3x
Role: Applied Mathematician

Instruction: How to optimize MySQL queries for large datasets?
Role: Database Performance Engineer

Instruction: What's the best way to negotiate salary in a tech job?
Role: HR Compensation Analyst

Instruction: Design a workout plan for marathon training
Role: Certified Running Coach

Instruction: Help me write a formal complaint letter to my landlord
Role: Tenant Rights Advocate

Instruction: Analyze this stock market trend from last quarter
Role: Financial Securities Analyst

Now analyze this instruction:
{instruction}
Role:"""
