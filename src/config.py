from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    prompts_dir: Path = base_dir / "prompts"
    kb_dir: Path = base_dir / "knowledge-base"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

