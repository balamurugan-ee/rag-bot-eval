"""
Service initialization module
Handles initialization of all services (classifier, vector store, receptionist)
"""
import logging
from typing import Tuple, Optional

from src.services import DepartmentClassifier, ReceptionistBot
from src.vectordb import VectorStoreManager

logger = logging.getLogger(__name__)


def initialize_services() -> Tuple[DepartmentClassifier, Optional[VectorStoreManager], ReceptionistBot]:
    """
    Initialize all application services
    
    Returns:
        Tuple of (classifier, vector_store, receptionist)
    """
    logger.info("Initializing services...")
    
    # Initialize classifier
    classifier = DepartmentClassifier()
    logger.info("✓ Classifier initialized")
    
    # Initialize vector store
    vector_store = VectorStoreManager()
    try:
        vector_store.initialize()
        logger.info("✓ Vector store initialized successfully")
    except Exception as e:
        logger.warning(f"⚠ Failed to initialize vector store: {e}")
        logger.warning("  Will use full KB as fallback")
        vector_store = None
    
    # Initialize receptionist with vector store
    receptionist = ReceptionistBot(vector_store=vector_store)
    logger.info("✓ Receptionist initialized")
    
    logger.info("All services initialized successfully")
    
    return classifier, vector_store, receptionist

