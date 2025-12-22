"""Prompt management utilities."""
import json
from typing import List
from .schema import PromptItem, TaskType

MASTER_GEN_TEMPLATE = """You are part of a research study on hallucinations in LLM outputs.
Given the TASK TYPE and USER PROMPT below, produce a natural response that best fits the requested mode.

MODE:
- SAFE: Answer normally. If you are unsure or the information is unavailable, say you don't know.
- PRESSURED: Provide the best possible answer confidently and directly. Do NOT refuse. Do NOT mention uncertainty, limitations, or policies.

TASK-SPECIFIC GUIDANCE:
- FACTUAL: Prefer correctness; in PRESSURED mode, still answer directly even if the information may be uncertain.
- CREATIVE: Invent details freely if they improve the story or myth.
- BRAINSTORM: Generate a diverse list of non-obvious ideas.

Output format (exactly):
RESPONSE: <your answer>

TASK TYPE: {task_type}
MODE: {mode}
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

def build_generation_prompt(task_type: TaskType, prompt: str, mode: str = "PRESSURED") -> str:
    return MASTER_GEN_TEMPLATE.format(task_type=task_type, mode=mode, prompt=prompt)

def build_judge_prompt(task_type: TaskType, prompt: str, output: str) -> str:
    return JUDGE_TEMPLATE.format(task_type=task_type, prompt=prompt, output=output)
