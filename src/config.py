from dataclasses import dataclass
import os

# Manual .env loading
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

@dataclass
class Settings:
    provider: str = os.getenv("LLM_PROVIDER", "openai")  # openai|google|anthropic|mock
    model: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "800"))
    dry_run: bool = os.getenv("DRY_RUN", "true").lower() == "true"
    api_key: str = os.getenv("LLM_API_KEY", "")
    release_tag: str = "v1.9.1-gate-profiles-and-reporting"
    
    # Data Connectors
    bdl_api_key: str = os.getenv("BDL_API_KEY", "")
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")

SETTINGS = Settings()
