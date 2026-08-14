from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from src.routers.chat import router as chat_router
from src.routers.classify import router as classify_router
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
    # Launch local Phoenix (UI at http://localhost:6006) and instrument LangChain
    # before any classifier/receptionist objects are created, so every LLM,
    # embedding, and retrieval call gets traced automatically.
    px.launch_app()
    tracer_provider = register(project_name="rag-bot-eval")
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    logger.info("Phoenix tracing enabled - UI at http://localhost:6006")

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

# Trace incoming HTTP requests as the root span, with classify/embed/generate
# calls nested underneath -- one coherent waterfall per /chat request.
FastAPIInstrumentor().instrument_app(app)

# Include routers
app.include_router(system_router)
app.include_router(chat_router)
app.include_router(classify_router)




