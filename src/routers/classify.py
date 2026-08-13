from fastapi import APIRouter, HTTPException, Request
import logging
from typing import cast

from src.models import ChatRequest, ClassifyResponse, DepartmentType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/classify", tags=["classify"])


@router.post("", response_model=ClassifyResponse)
async def classify(request: Request, chat_request: ChatRequest):
    """
    Classification-only endpoint: routes a query to a department without
    running the RAG receptionist. Useful for evaluating classification
    accuracy in isolation, since /chat's merged retrieval can mask a wrong
    department label with a still-correct answer.
    """
    try:
        classifier = request.app.state.classifier

        logger.info(f"Received query for classification: {chat_request.message}")
        department = classifier.classify(chat_request.message)
        logger.info(f"Classified as: {department}")

        return ClassifyResponse(department=cast(DepartmentType, department))

    except Exception as e:
        logger.error(f"Error processing classify request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )
