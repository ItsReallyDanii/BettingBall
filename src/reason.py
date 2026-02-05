from src.prompts import SYSTEM_PROMPT, REASONER_PROMPT
from src.llm import call_llm_json
from src.schemas import PredictionInput, ReasoningOutput

def generate_reasoning(inp: PredictionInput) -> ReasoningOutput:
    payload = {
        "instruction": REASONER_PROMPT,
        "input": inp.model_dump()
    }
    out = call_llm_json(SYSTEM_PROMPT, payload)
    return ReasoningOutput(**out)