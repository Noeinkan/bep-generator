# BEP ML Service

Machine Learning service for AI-powered BEP text generation.

## Features

### RAG (Retrieval-Augmented Generation) ⭐ Recommended

High-quality text generation using your own BEP documents:

- **Extracts** text from DOCX BEP templates
- **Creates** vector database (FAISS) for semantic search
- **Retrieves** relevant context from similar documents
- **Generates** contextual text using Claude API
- **Provides** source attribution for transparency

### LSTM Model (Fallback)

Character-level LSTM for offline generation:

- Works without API key
- Local processing
- No external dependencies
- Automatic fallback when RAG unavailable

## Quick Start

### Setup RAG System (Recommended)

```bash
setup_rag.bat
```

This script will:
1. Create Python virtual environment
2. Install dependencies
3. Extract text from DOCX files
4. Build vector database
5. Test the system

### Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
setx ANTHROPIC_API_KEY "sk-ant-..."

# 4. Extract DOCX text
python scripts/extract_docx.py

# 5. Create vector database
python -c "from rag_engine import BEPRAGEngine; e = BEPRAGEngine(); e.initialize(); e.load_or_create_vectorstore(force_rebuild=True)"
```

## Running the Service

```bash
venv\Scripts\activate
python api.py
```

Or use the shortcut:
```bash
start_service.bat
```

Service will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs

## API Endpoints

### Health Check

```http
GET /health
```

Returns system status:
```json
{
  "status": "healthy",
  "lstm_model_loaded": true,
  "device": "cpu",
  "rag_available": true,
  "rag_status": {
    "initialized": true,
    "vectorstore_loaded": true,
    "llm_available": true
  }
}
```

### Generate Suggestion

```http
POST /suggest
Content-Type: application/json

{
  "field_type": "executiveSummary",
  "partial_text": "This BEP establishes",
  "max_length": 300
}
```

Response:
```json
{
  "text": "a comprehensive framework for information management...",
  "method": "rag",
  "sources": [
    {"source": "BEP_Template.txt", "content": "..."}
  ]
}
```

## Project Structure

```
ml-service/
├── api.py                 # FastAPI service with RAG+LSTM
├── rag_engine.py          # RAG implementation
├── model_loader.py        # LSTM model (fallback)
├── requirements.txt       # Python dependencies
├── setup_rag.bat          # Automated setup script
│
├── scripts/
│   └── extract_docx.py    # DOCX text extraction
│
├── data/
│   ├── training_documents/
│   │   ├── docx/          # Place your DOCX files here
│   │   └── txt/           # Extracted text
│   └── vector_db/         # FAISS database
│
└── models/
    └── bep_model.pth      # LSTM weights (optional)
```

## Configuration

### Environment Variables

Create `.env` file:
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

Or set system variable:
```cmd
setx ANTHROPIC_API_KEY "sk-ant-..."
```

### Add Training Documents

1. Place DOCX files in: `data/training_documents/docx/`
2. Run: `python scripts/extract_docx.py`
3. Rebuild database: See [RAG Setup Guide](../RAG_SETUP_GUIDE.md)

## Troubleshooting

### RAG not working

Check:
1. API key configured: `echo %ANTHROPIC_API_KEY%`
2. Vector DB exists: `data\vector_db\`
3. Documents extracted: `data\training_documents\txt\`

### Service fails to start

Check:
1. Python 3.8+: `python --version`
2. Virtual environment activated
3. Dependencies installed: `pip list`
4. Port 8000 available: `netstat -ano | findstr :8000`

## Documentation

- [RAG Setup Guide](../RAG_SETUP_GUIDE.md) - Complete RAG setup instructions
- [AI Integration Guide](../AI_INTEGRATION_GUIDE.md) - LSTM model setup (legacy)

## Performance

### RAG (with Claude API)
- **Quality**: Excellent (uses actual BEP content)
- **Speed**: 1-3 seconds per request
- **Cost**: ~$0.01-0.03 per generation
- **Requirements**: API key, internet connection

### LSTM (fallback)
- **Quality**: Good (generic BEP patterns)
- **Speed**: 0.5-1 second per request
- **Cost**: Free (local processing)
- **Requirements**: Trained model file

## License

Same as main BEP Generator project
