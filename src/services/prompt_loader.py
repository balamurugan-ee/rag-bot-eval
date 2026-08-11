from pathlib import Path
from src.config import settings


class PromptLoader:
    """Handles loading and managing prompt templates"""
    
    @staticmethod
    def load_classification_prompt() -> str:
        """Load the classification prompt template"""
        prompt_path = settings.prompts_dir / "classification_prompt.txt"
        return prompt_path.read_text(encoding="utf-8")
    
    @staticmethod
    def load_receptionist_prompt() -> str:
        """Load the receptionist prompt template"""
        prompt_path = settings.prompts_dir / "receptionist_prompt.txt"
        return prompt_path.read_text(encoding="utf-8")

