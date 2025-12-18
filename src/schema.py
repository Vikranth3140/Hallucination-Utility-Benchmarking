"""Data schemas for hallucination benchmarking."""
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any

TaskType = Literal["factual", "creative", "brainstorm"]
UtilityLabel = Literal["U+", "U0", "U-"]

class PromptItem(BaseModel):
    task_type: TaskType
    prompt_id: str
    prompt: str

class ModelOutput(BaseModel):
    task_type: TaskType
    prompt_id: str
    prompt: str
    model_name: str
    response: str
    meta: Dict[str, Any] = Field(default_factory=dict)

class LabeledExample(BaseModel):
    task_type: TaskType
    prompt_id: str
    prompt: str
    model_name: str
    response: str
    utility_label: UtilityLabel
    rationale: str
    judge_model: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
