# AI Text Generation Integration Guide

This guide explains how to use the AI-powered text generation feature integrated into the BEP Generator.

## Overview

The BEP Generator now includes an AI text generation system that helps users create BIM Execution Plan content. The system uses a trained LSTM neural network model that has learned from ISO 19650-compliant BEP documents.

## Architecture

```
┌─────────────────────────────────────────┐
│   React Frontend                         │
│   - TipTap Editor with AI Button        │
│   - AISuggestionButton Component        │
│   - useAISuggestion Hook                │
└──────────────────┬──────────────────────┘
                   │ HTTP Request
                   ↓
┌─────────────────────────────────────────┐
│   Node.js Backend (Port 3001)           │
│   - /api/ai/suggest                     │
│   - /api/ai/generate                    │
│   - /api/ai/health                      │
└──────────────────┬──────────────────────┘
                   │ HTTP Proxy
                   ↓
┌─────────────────────────────────────────┐
│   Python ML Service (Port 8000)         │
│   - FastAPI REST API                    │
│   - LSTM Model Inference                │
│   - Pre-trained Weights                 │
└─────────────────────────────────────────┘
```

## Setup Instructions

### Prerequisites

- Node.js 14+ (already installed for BEP Generator)
- Python 3.8+ (download from [python.org](https://www.python.org/downloads/))
- pip (comes with Python)

### Step 1: Install Python Dependencies

```bash
cd ml-service
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Train the Model (First Time Only)

The model needs to be trained once before use:

```bash
cd ml-service
python scripts\train_model.py --epochs 100
```

**Training Options:**
- `--epochs`: Number of training iterations (default: 100, recommended: 100-200)
- `--hidden-size`: LSTM hidden layer size (default: 512)
- `--seq-length`: Character sequence length (default: 100)
- `--learning-rate`: Learning rate (default: 0.001)

**Training Time:**
- CPU: ~10-20 minutes for 100 epochs
- GPU: ~2-5 minutes for 100 epochs

The trained model will be saved to `ml-service/models/bep_model.pth`.

### Step 3: Start the ML Service

```bash
cd ml-service
start_service.bat
```

Or manually:

```bash
cd ml-service
venv\Scripts\activate
python api.py
```

The service will start on `http://localhost:8000`.

### Step 4: Start the BEP Generator

In a separate terminal:

```bash
npm start
```

This starts both the React frontend (port 3000) and Node.js backend (port 3001).

## Usage

### In the BEP Editor

1. **Open any BEP form field** with a text editor (e.g., Executive Summary, Project Objectives)

2. **Click the sparkle (✨) icon** in the editor toolbar to generate AI suggestions

3. **AI generates contextually appropriate text** based on:
   - The field type (e.g., Executive Summary gets formal BEP language)
   - Any text you've already typed (it continues from your prompt)
   - Patterns learned from ISO 19650 BEP documents

4. **The suggested text is inserted** at your cursor position

5. **Edit and refine** the generated text as needed

### Example Workflow

1. **User types:** "The BEP establishes"
2. **Clicks AI button**
3. **AI continues:** " a robust framework for information management throughout the project lifecycle, ensuring compliance with ISO 19650 standards and the Employer's Information Requirements (EIR)."
4. **User edits** as needed

## API Endpoints

### Health Check

```bash
GET http://localhost:3001/api/ai/health
```

Returns:
```json
{
  "status": "ok",
  "ml_service": {
    "status": "healthy",
    "model_loaded": true,
    "device": "cpu"
  }
}
```

### Generate Suggestion

```bash
POST http://localhost:3001/api/ai/suggest
Content-Type: application/json

{
  "field_type": "executiveSummary",
  "partial_text": "The BEP establishes",
  "max_length": 200
}
```

Returns:
```json
{
  "success": true,
  "text": " a robust framework for information management...",
  "prompt_used": "The BEP establishes"
}
```

### Generate from Prompt

```bash
POST http://localhost:3001/api/ai/generate
Content-Type: application/json

{
  "prompt": "Project objectives include",
  "field_type": "projectObjectives",
  "max_length": 200,
  "temperature": 0.7
}
```

## Supported Field Types

The AI has specialized prompts for these field types:

- `projectName` - Project identification
- `projectDescription` - Project overview
- `executiveSummary` - Executive summary content
- `projectObjectives` - Project goals and objectives
- `bimObjectives` - BIM-specific objectives
- `projectScope` - Scope definition
- `stakeholders` - Stakeholder information
- `rolesResponsibilities` - Roles and responsibilities
- `deliveryTeam` - Team structure
- `collaborationProcedures` - Collaboration workflows
- `informationExchange` - Information exchange protocols
- `cdeWorkflow` - CDE workflow description
- `modelRequirements` - Model development requirements
- `dataStandards` - Data standards and schemas
- `namingConventions` - Naming convention descriptions
- `qualityAssurance` - QA procedures
- `validationChecks` - Validation procedures
- `technologyStandards` - Technology specifications
- `softwarePlatforms` - Software requirements
- `coordinationProcess` - Coordination procedures
- `clashDetection` - Clash detection workflows
- `healthSafety` - H&S information
- `handoverRequirements` - Handover procedures
- `asbuiltRequirements` - As-Built requirements
- `cobieRequirements` - COBie data requirements

## Improving the Model

### Adding More Training Data

1. Add BEP documents to `ml-service/data/training_data.txt`
2. Retrain the model:
   ```bash
   python scripts\train_model.py --epochs 150
   ```
3. Restart the ML service

### Tips for Better Training Data

- Use complete, well-written BEP documents
- Include documents from various project types
- Ensure documents follow ISO 19650 standards
- More data = better generation quality
- Aim for 100+ pages of text for best results

## Troubleshooting

### ML Service Won't Start

**Problem:** `Model file not found`

**Solution:** Train the model first:
```bash
cd ml-service
python scripts\train_model.py
```

### Connection Errors

**Problem:** `Cannot connect to AI service`

**Solution:** Ensure the ML service is running:
```bash
cd ml-service
start_service.bat
```

Check that port 8000 is available:
```bash
netstat -ano | findstr :8000
```

### Low-Quality Suggestions

**Problem:** Generated text doesn't make sense

**Solutions:**
1. Train longer: `python scripts\train_model.py --epochs 200`
2. Add more training data to `data/training_data.txt`
3. Retrain the model

### Slow Generation

**Problem:** AI takes too long to respond

**Solutions:**
1. Reduce `max_length` in requests (default: 200)
2. Use GPU acceleration if available
3. Reduce model complexity (hidden_size parameter)

## Development

### Project Structure

```
ml-service/
├── api.py                    # FastAPI service
├── model_loader.py           # Model loading and inference
├── requirements.txt          # Python dependencies
├── start_service.bat         # Windows startup script
├── data/
│   └── training_data.txt     # Training dataset
├── models/
│   ├── bep_model.pth         # Trained model weights
│   └── char_mappings.json    # Character vocabulary
└── scripts/
    └── train_model.py        # Training script

server/routes/
└── ai.js                     # Node.js API proxy

src/
├── components/forms/ai/
│   └── AISuggestionButton.js # React AI button component
├── hooks/
│   └── useAISuggestion.js    # AI suggestion hook
└── components/forms/editors/
    ├── TipTapEditor.js       # Enhanced with AI
    └── TipTapToolbar.js      # Toolbar with AI button
```

### Extending the System

#### Adding New Field Types

Edit `ml-service/model_loader.py`:

```python
self.field_prompts = {
    # ... existing prompts ...
    'myNewField': 'My new field context: ',
}
```

#### Customizing Generation

Adjust temperature for creativity:
- Lower (0.3-0.5): More conservative, repetitive
- Medium (0.6-0.8): Balanced (recommended)
- Higher (0.9-1.5): More creative, unpredictable

#### Using Different Models

Replace the LSTM with GPT-2:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
```

Then fine-tune on your BEP dataset.

## Performance

### Benchmarks

- **Model Size:** ~2-5 MB (LSTM) or ~500 MB (GPT-2)
- **Inference Time:** 100-500ms per request (CPU)
- **GPU Speedup:** 5-10x faster with CUDA GPU
- **Memory Usage:** ~500MB RAM (LSTM) or ~2GB (GPT-2)

### Scalability

- Single ML service handles ~10-20 concurrent users
- For production, deploy multiple ML service instances
- Consider using GPU for better performance

## Security & Privacy

- All processing happens locally - no data sent to external services
- Training data remains on your machine
- Model weights stored locally
- No user data is logged or stored by the ML service

## License

Same as BEP Generator main project.

## Support

For issues or questions:
1. Check this guide
2. Check ml-service logs: `server.err.log`, `server.out.log`
3. Open an issue on GitHub
4. Contact the development team
