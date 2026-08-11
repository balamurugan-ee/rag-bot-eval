from fastapi import APIRouter

router = APIRouter(tags=["system"])


@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Hospital Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "/chat": "POST - Main chat endpoint",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

