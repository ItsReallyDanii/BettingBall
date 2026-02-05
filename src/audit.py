from src.prompts import SYSTEM_PROMPT, AUDITOR_PROMPT
from src.llm import call_llm_json
from src.schemas import PredictionInput, ReasoningOutput, AuditOutput

def run_audit(inp: PredictionInput, reasoning: ReasoningOutput) -> AuditOutput:
    payload = {
        "instruction": AUDITOR_PROMPT,
        "input": inp.model_dump(),
        "reasoning": reasoning.model_dump()
    }
    out = call_llm_json(SYSTEM_PROMPT, payload)

    if "pass_fail" not in out:
        out = {
            "pass_fail": "pass" if reasoning.confidence_grade in ["A","B","C"] else "fail",
            "warnings": ["Dry-run auditor used; replace with real LLM judge."],
            "fixes": ["Implement provider adapter in src/llm.py"]
        }
    return AuditOutput(**out)