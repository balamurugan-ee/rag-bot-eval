from fastapi import APIRouter, HTTPException, Request
import logging
from typing import cast

from src.models import ChatRequest, ChatResponse, DepartmentType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest):
    """
    Main chat endpoint that:
    1. Classifies the user query into a department (stored as metadata)
    2. Uses RAG to retrieve relevant context
    3. Generates an appropriate response
    
    Args:
        request: FastAPI Request object (for accessing app state)
        chat_request: ChatRequest with user message
    """
    try:
        # Get services from app state
        classifier = request.app.state.classifier
        receptionist = request.app.state.receptionist
        
        logger.info(f"Received query: {chat_request.message}")
        
        # Step 1: Classify the department (metadata only)
        department = classifier.classify(chat_request.message)
        logger.info(f"Classified as: {department}")
        
        # Step 2: Generate response (automatically uses RAG if available)
        response = receptionist.answer(chat_request.message)
        logger.info(f"Generated response: {response[:100]}...")
        
        return ChatResponse(
            department=cast(DepartmentType, department),
            response=response
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )






