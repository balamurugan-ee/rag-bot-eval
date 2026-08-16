from fastapi import APIRouter, HTTPException, Request
import logging

from src.models import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest):
    """
    RAG-only chat endpoint: retrieves relevant context and generates an
    answer. Does not classify the query into a department -- that's a
    separate concern, handled by /classify.

    Args:
        request: FastAPI Request object (for accessing app state)
        chat_request: ChatRequest with user message
    """
    try:
        receptionist = request.app.state.receptionist

        logger.info(f"Received query: {chat_request.message}")

        response = receptionist.answer(chat_request.message)
        logger.info(f"Generated response: {response[:100]}...")

        return ChatResponse(response=response)

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )






