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

### Run Promptfoo Evaluation

```bash
promptfoo eval -c promptfoo.classify.yaml
```

### Generate Confusion Matrix & Metrics

Export Promptfoo results and generate confusion matrix with accuracy, precision, recall, F1-scores:

```bash
chmod +x scripts/export_and_analyze.sh
./scripts/export_and_analyze.sh promptfoo.classify.yaml
```

Or manually:

```bash
# Export results
promptfoo eval -c promptfoo.classify.yaml --output outputs/promptfoo_results.json

# Generate confusion matrix
python scripts/build_confusion_matrix.py --input outputs/promptfoo_results.json
```

### Output Files

Results saved to `outputs/`:
- `confusion_matrix.png` - Heatmap visualization
- `confusion_matrix.csv` - Confusion matrix data
- `classification_report.txt` - Precision, recall, F1 per class
- `metrics_summary.json` - All metrics in JSON
- `detailed_results.csv` - Test cases with predictions

Test cases: `tests/classification_ground_truth.csv`













