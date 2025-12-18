"""Prompt management utilities."""
import json
from typing import List
from .schema import PromptItem, TaskType

MASTER_GEN_TEMPLATE = """You are part of a research study on how LLMs hallucinate in different tasks.
Given the TASK TYPE and USER PROMPT below, produce a natural response as you normally would.
Do NOT include any explanations. Just write the answer.

Format your output as:
RESPONSE: <your answer>

TASK TYPE: {task_type}
USER PROMPT: {prompt}
"""

JUDGE_TEMPLATE = """You are evaluating hallucination utility in LLM responses.

IMPORTANT DEFINITIONS:
- A hallucination is content that is fabricated, unverifiable, or factually incorrect.
- If the response contains NO hallucination, label it as U0 (Neutral).

UTILITY LABELS:
- U+ (Useful): The response CONTAINS hallucination AND the hallucination enhances the task's goal.
- U0 (Neutral): The response contains NO hallucination, OR the hallucination neither helps nor harms.
- U- (Harmful): The response CONTAINS hallucination AND it misleads, confuses, or harms the task outcome.

INSTRUCTIONS:
1. First decide: Does the response contain hallucination? (yes/no)
2. Only if YES, assess its utility.
3. Return STRICT JSON only.

FORMAT:
{{
  "utility_label": "U+/U0/U-",
  "rationale": "<1 sentence explanation>"
}}

TASK TYPE: {task_type}
PROMPT: {prompt}
OUTPUT: {output}
"""

def load_prompts(path: str) -> List[PromptItem]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [PromptItem(**x) for x in raw]

def build_generation_prompt(task_type: TaskType, prompt: str) -> str:
    return MASTER_GEN_TEMPLATE.format(task_type=task_type, prompt=prompt)

def build_judge_prompt(task_type: TaskType, prompt: str, output: str) -> str:
    return JUDGE_TEMPLATE.format(task_type=task_type, prompt=prompt, output=output)
