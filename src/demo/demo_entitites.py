from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str

class ModelResult(BaseModel):
    code: str
    complexity: float
    lint_errors: int

class ComparisonResponse(BaseModel):
    base: ModelResult
    sft: ModelResult
    dpo: ModelResult