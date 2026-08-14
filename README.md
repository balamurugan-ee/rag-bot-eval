# Hospital Chatbot Application

Riverside Multispecialty Hospital chatbot with department classification and RAG-based Q&A.

## Structure

```
├── main.py           # FastAPI backend
├── src/              # Backend services (classifier, receptionist, vectordb)
├── ui/               # Streamlit UI (see ui/README.md)
├── prompts/          # Classification and receptionist prompts  
├── knowledge-base/   # Hospital information
└── tests/            # Promptfoo evaluation tests
```

## Quick Start

### Backend

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### UI (optional)

```bash
cd ui
pip install -r requirements.txt
streamlit run app.py
```

## Access

- API: http://localhost:8000
- API Docs: http://localhost:8000/docs  
- UI: http://localhost:8501

## Configuration

Create `.env` file:
```
OPENAI_API_KEY=your_key_here
```

## API Endpoints

### POST /chat

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
  "response": "For chest pain, please visit our Cardiology department..."
}
```

### POST /classify

Classification only.

### GET /health

Health check.

## Evaluation

```bash
promptfoo eval -c promptfooconfig.yaml
```

Test cases: `tests/classification_ground_truth.csv`













