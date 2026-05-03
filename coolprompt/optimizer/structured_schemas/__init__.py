"""Pydantic schemas for structured LLM outputs across autoprompting methods.

This package mirrors the layout of `coolprompt/optimizer/` and provides
response schemas used with `model.with_structured_output(...)` in each
optimization method. Schemas are grouped by method:

- reflective_prompt: schemas for ReflectivePrompt evolution stages.
- regps: schemas for ReGPS-specific stages (reuses some reflective schemas).
- hype: schemas for HyPE / HyPER / FeedbackModule.
"""
