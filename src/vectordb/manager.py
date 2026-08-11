from typing import List
import logging

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from src.config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector store for knowledge base retrieval"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.kb_path = settings.kb_dir / "Riverside Multispecialty Hospital.md"
        self.persist_directory = settings.base_dir / ".chroma_db"
        
    def initialize(self, force_reload: bool = False):
        """
        Initialize or load vector store
        
        Args:
            force_reload: If True, recreate vector store from scratch
        """
        if not force_reload and self.persist_directory.exists():
            logger.info("Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            )
            logger.info(f"Loaded vector store with {self.vector_store._collection.count()} documents")
        else:
            logger.info("Creating new vector store...")
            documents = self._load_and_chunk_kb()
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            logger.info(f"Created vector store with {len(documents)} chunks")
    
    def _load_and_chunk_kb(self) -> List[Document]:
        """
        Load knowledge base and split into chunks using header-based strategy
        
        Returns:
            List of Document chunks with metadata
        """
        logger.info(f"Loading knowledge base from {self.kb_path}")
        
        # Read the markdown file
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split by ## headers (department and main sections)
        # This keeps each department/section as a semantic unit
        headers_to_split_on = [
            ("#", "title"),      # Main title
            ("##", "section"),   # Departments and main sections  
            ("###", "subsection"),  # Sub-sections like "General Hours"
        ]
        
        from langchain.text_splitter import MarkdownHeaderTextSplitter
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # Keep headers in content for context
        )
        
        # Split by headers first
        header_splits = markdown_splitter.split_text(text)
        
        # For sections larger than 1000 chars, do further splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        
        # Process each header-based chunk
        final_chunks = []
        for doc in header_splits:
            # If chunk is too large, split it further
            if len(doc.page_content) > 1000:
                sub_chunks = text_splitter.split_documents([doc])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(doc)
        
        # Add custom metadata for better retrieval
        for i, chunk in enumerate(final_chunks):
            # Extract department name from section metadata
            section = chunk.metadata.get('section', '')
            
            # Identify department chunks
            departments = [
                'Cardiology', 'Pediatrics', 'Orthopedics', 'Dermatology',
                'Neurology', 'Ophthalmology', 'Radiology', 'General Medicine',
                'Billing', 'Pharmacy'
            ]
            
            for dept in departments:
                if dept.lower() in section.lower() or dept.lower() in chunk.page_content.lower()[:200]:
                    chunk.metadata['department'] = dept
                    break
            
            # Add chunk index
            chunk.metadata['chunk_id'] = i
        
        logger.info(f"Split knowledge base into {len(final_chunks)} semantic chunks")
        logger.info(f"Chunk size range: {min(len(c.page_content) for c in final_chunks)} - {max(len(c.page_content) for c in final_chunks)} chars")
        
        return final_chunks
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        results = self.vector_store.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
        
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple]:
        """
        Search with similarity scores
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        logger.info(f"Retrieved {len(results)} chunks with scores for query: {query[:50]}...")
        
        return results

