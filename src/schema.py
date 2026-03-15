"""Data schemas for hallucination benchmarking."""
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional, Dict, Any

TaskType = Literal["factual", "creative", "brainstorm"]
UtilityLabel = Literal["useful", "neutral", "harmful"]

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
    unsupported_label: int
    utility_label: Optional[UtilityLabel] = None
    rationale: str = ""
    judge_model: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_utility_conditional(self) -> "LabeledExample":
        if self.unsupported_label == 0:
            if self.utility_label is not None:
                raise ValueError("utility_label must be null/None when unsupported_label is 0")
        elif self.unsupported_label == 1:
            if self.utility_label not in {"useful", "neutral", "harmful"}:
                raise ValueError("utility_label must be one of useful/neutral/harmful when unsupported_label is 1")
        else:
            raise ValueError("unsupported_label must be 0 or 1")
        return self
