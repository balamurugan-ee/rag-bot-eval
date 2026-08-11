# Hospital Chatbot Application

A modular chatbot application for Riverside Multispecialty Hospital that performs:
1. **Department Classification** - Routes patient queries to the correct department
2. **Hospital Receptionist** - Answers questions about hospital services, timings, and policies

## Architecture

```
src/
├── config.py              # Configuration and settings
├── models.py              # Pydantic models for requests/responses
└── services/
    ├── classifier.py      # Department classification service
    ├── receptionist.py    # Q&A receptionist service
    └── prompt_loader.py   # Prompt template loader
```

## Setup

### 1. Install Dependencies

**Option A: Using pip with requirements.txt (Recommended)**

```bash
pip install -r requirements.txt
```

**Option B: Using pip with pyproject.toml**

```bash
pip install -e . --no-build-isolation
```

**Option C: Manual installation**

```bash
pip install fastapi uvicorn langchain langchain-openai pydantic pydantic-settings python-dotenv
```

### 2. Configure Environment

Create or update `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=your_actual_openai_api_key
```

### 3. Run the Server

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /chat

Main chat endpoint that classifies the query and generates a response.

**Request:**
```json
{
  "message": "I have chest pain"
}
```

**Response:**
```json
{
  "department": "Cardiology",
  "response": "For chest pain, please visit our Cardiology department. We are open Monday–Saturday, 9:00 AM–5:00 PM. If this is an emergency, please call our Emergency Department at +91 44 4100 2299."
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## Testing

### cURL Example

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "I have chest pain"}'
```

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "I have chest pain"}
)

print(response.json())
```

## Features

- **Modular Design**: Separation of concerns with services, models, and config
- **Type Safety**: Pydantic models for request/response validation
- **Error Handling**: Comprehensive error handling with proper HTTP status codes
- **Logging**: Request and response logging for debugging
- **LangChain Integration**: Uses LangChain for LLM orchestration
- **Prompt Management**: External prompt templates for easy modification

## Department Categories

1. Cardiology
2. Pediatrics
3. Orthopedics
4. Dermatology
5. Neurology
6. Ophthalmology
7. Radiology
8. General Medicine
9. Billing
10. Pharmacy

## Configuration

Edit `src/config.py` to modify:
- Model name (default: `gpt-4`)
- Temperature (default: `0.0`)
- Paths to prompts and knowledge base


