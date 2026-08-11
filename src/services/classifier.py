from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from src.config import settings
from src.services.prompt_loader import PromptLoader


class DepartmentClassifier:
    """Classifies user queries into hospital departments"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=settings.temperature,
            api_key=settings.openai_api_key
        )
        self.prompt_template = self._load_prompt()
        
    def _load_prompt(self) -> PromptTemplate:
        """Load and prepare classification prompt"""
        prompt_text = PromptLoader.load_classification_prompt()
        return PromptTemplate(
            input_variables=["query"],
            template=prompt_text + "\n\nQuery: {query}\nOutput:"
        )
    
    def classify(self, query: str) -> str:
        """
        Classify a user query into a department
        
        Args:
            query: User's input query
            
        Returns:
            Department name as string
        """
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        result = chain.run(query=query)
        
        # Clean up the result (remove whitespace, newlines)
        department = result.strip()
        
        return department

