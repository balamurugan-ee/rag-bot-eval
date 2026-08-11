from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from src.routers.chat import router as chat_router
from src.routers.system import router as system_router
from src.init import initialize_services

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup and cleanup on shutdown"""
    # Initialize all services
    classifier, vector_store, receptionist = initialize_services()
    
    # Store services in app state
    app.state.classifier = classifier
    app.state.receptionist = receptionist
    app.state.vector_store = vector_store
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down services...")


app = FastAPI(
    title="Hospital Chatbot API",
    description="Department classification and receptionist chatbot for Riverside Multispecialty Hospital",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(system_router)
app.include_router(chat_router)




