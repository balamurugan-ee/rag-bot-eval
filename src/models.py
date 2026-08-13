from pydantic import BaseModel, Field
from typing import Literal

DepartmentType = Literal[
    "Cardiology",
    "Pediatrics", 
    "Orthopedics",
    "Dermatology",
    "Neurology",
    "Ophthalmology",
    "Radiology",
    "General Medicine",
    "Billing",
    "Pharmacy"
]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's message or query")


class ChatResponse(BaseModel):
    department: DepartmentType
    response: str


class ClassifyResponse(BaseModel):
    department: DepartmentType



class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None

