# BEP AI Text Generation Service

ML service for generating BIM Execution Plan content based on ISO 19650 standards.

## Setup

1. Install Python 3.8+ if not already installed
2. Create virtual environment:
   ```bash
   cd ml-service
   python -m venv venv
   ```

3. Activate virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

1. Add your BEP training data to `data/training_data.txt`
2. Run the training script:
   ```bash
   python scripts/train_model.py
   ```
3. Model weights will be saved to `models/bep_model.pth`

## Running the Service

```bash
python api.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `POST /generate` - Generate text suggestions
  - Body: `{"prompt": "string", "field_type": "string", "max_length": 200}`
  - Returns: `{"text": "generated text..."}`

- `GET /health` - Health check
