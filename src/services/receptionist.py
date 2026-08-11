from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Optional, TYPE_CHECKING
import logging

from src.config import settings
from src.services.prompt_loader import PromptLoader

if TYPE_CHECKING:
    from src.vectordb import VectorStoreManager

logger = logging.getLogger(__name__)


class ReceptionistBot:
    """Handles receptionist-style Q&A responses with RAG"""
    
    def __init__(self, vector_store: Optional["VectorStoreManager"] = None):
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=settings.temperature,
            api_key=settings.openai_api_key
        )
        self.prompt_template = self._load_prompt()
        self.vector_store = vector_store
        
    def _load_prompt(self) -> PromptTemplate:
        """Load receptionist prompt template"""
        prompt_text = PromptLoader.load_receptionist_prompt()
        return PromptTemplate(
            input_variables=["knowledge_base", "question"],
            template=prompt_text
        )
    
    def _get_context(self, question: str, top_k: int = 3) -> str:
        """
        Retrieve relevant context from KB
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve if using RAG
            
        Returns:
            Context string
        """
        if self.vector_store:
            # Use RAG retrieval
            logger.info(f"Using RAG retrieval (top_k={top_k})")
            chunks = self.vector_store.similarity_search(question, k=top_k)
            context = "\n\n".join([chunk.page_content for chunk in chunks])
            logger.info(f"Retrieved {len(chunks)} chunks, {len(context)} chars")
        else:
            # Fallback to full KB
            logger.info("Using full knowledge base (RAG not available)")
            kb_path = settings.kb_dir / "Riverside Multispecialty Hospital.md"
            context = kb_path.read_text(encoding="utf-8")
        
        return context
    
    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Generate an answer to a user question
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve (only used if RAG is available)
            
        Returns:
            Generated answer as string
        """
        # Get relevant context
        context = self._get_context(question, top_k)
        
        # Generate answer
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        result = chain.run(knowledge_base=context, question=question)
        
        return result.strip()






