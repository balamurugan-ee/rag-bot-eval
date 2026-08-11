#!/usr/bin/env python3
"""
Utility script to rebuild the vector store from knowledge base
"""
import logging
from src.vectordb import VectorStoreManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Rebuild vector store"""
    logger.info("Starting vector store rebuild...")
    
    vector_store = VectorStoreManager()
    vector_store.initialize(force_reload=True)
    
    logger.info("Vector store rebuild complete!")
    logger.info(f"Database stored at: {vector_store.persist_directory}")
    
    # Test searches
    logger.info("\n" + "="*60)
    logger.info("Testing vector store with sample queries...")
    logger.info("="*60)
    
    # Test 1: Department-specific query
    logger.info("\n--- Test 1: Cardiology Query ---")
    results = vector_store.similarity_search("What are the cardiology hours?", k=2)
    for i, doc in enumerate(results, 1):
        logger.info(f"\nResult {i}:")
        logger.info(f"Metadata: {doc.metadata}")
        logger.info(f"Content: {doc.page_content[:150]}...")
    
    # Test 2: General query
    logger.info("\n--- Test 2: Appointment Query ---")
    results = vector_store.similarity_search("How do I book an appointment?", k=2)
    for i, doc in enumerate(results, 1):
        logger.info(f"\nResult {i}:")
        logger.info(f"Metadata: {doc.metadata}")
        logger.info(f"Content: {doc.page_content[:150]}...")
    
    # Test 3: Emergency query
    logger.info("\n--- Test 3: Emergency Query ---")
    results = vector_store.similarity_search("What is the emergency number?", k=2)
    for i, doc in enumerate(results, 1):
        logger.info(f"\nResult {i}:")
        logger.info(f"Metadata: {doc.metadata}")
        logger.info(f"Content: {doc.page_content[:150]}...")
    
    logger.info("\n" + "="*60)
    logger.info("All tests complete!")


if __name__ == "__main__":
    main()




