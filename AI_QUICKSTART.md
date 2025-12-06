# AI Text Generation - Quick Start Guide

## What is it?

An AI-powered text generation system integrated into your BEP Generator that suggests contextually appropriate content for BIM Execution Plan documents based on ISO 19650 standards.

## Setup (First Time)

### Step 1: Run Setup Script

```bash
setup-ai.bat
```

This will:
- Install Python dependencies
- Train the AI model (~10-20 minutes)
- Prepare everything for use

### Step 2: Start Services

**Terminal 1 - ML Service:**
```bash
cd ml-service
start_service.bat
```

**Terminal 2 - BEP Generator:**
```bash
npm start
```

## How to Use

1. **Open any text field** in the BEP editor (e.g., Executive Summary, Project Objectives)

2. **Type some context** (optional):
   ```
   The BEP establishes
   ```

3. **Click the sparkle (✨) icon** in the toolbar

4. **AI generates text**:
   ```
   The BEP establishes a robust framework for information management
   throughout the project lifecycle, ensuring compliance with ISO 19650
   standards and the Employer's Information Requirements (EIR).
   ```

5. **Edit as needed** - The AI provides a starting point, you refine it!

## Tips for Best Results

### Write a Good Prompt
❌ Bad: Just clicking AI with empty field
✅ Good: "The project objectives include"

### Use the Right Context
The AI knows about different BEP sections:
- Executive Summary → Formal overview language
- Roles & Responsibilities → Team structure descriptions
- Technical Requirements → Standards and specifications

### Iterate
1. Generate initial text
2. Edit what you like
3. Generate more if needed
4. Combine and refine

## Troubleshooting

### "AI service unavailable"
**Fix:** Start the ML service:
```bash
cd ml-service
start_service.bat
```

### "Model not found"
**Fix:** Train the model:
```bash
cd ml-service
python scripts\train_model.py
```

### Poor quality suggestions
**Fix:** Train longer or add more data:
```bash
# Train with more epochs
python scripts\train_model.py --epochs 200

# Or add your own BEP examples to data/training_data.txt and retrain
```

## Architecture

```
User types → AI Button → Node.js API → Python ML Service → AI Model → Text Generated
```

All processing happens locally on your machine. No external services, no data sent anywhere.

## Performance

- **Generation Time:** 1-3 seconds per request
- **Model Size:** ~2-5 MB
- **Memory Usage:** ~500 MB RAM
- **GPU:** Optional but makes it 5-10x faster

## Advanced Usage

### Add Your Own Training Data

1. Add BEP documents to `ml-service/data/training_data.txt`
2. Retrain:
   ```bash
   cd ml-service
   python scripts\train_model.py --epochs 150
   ```
3. Restart ML service

### Customize Generation

Edit `ml-service/model_loader.py` to add custom field prompts:

```python
self.field_prompts = {
    'myCustomField': 'Your custom prompt here: ',
}
```

## What's Next?

- See [AI_INTEGRATION_GUIDE.md](AI_INTEGRATION_GUIDE.md) for complete documentation
- Check ml-service logs if something goes wrong
- Add more training data for better results
- Experiment with different prompts

## Support

Questions? Issues?
1. Read [AI_INTEGRATION_GUIDE.md](AI_INTEGRATION_GUIDE.md)
2. Check ml-service logs
3. Open a GitHub issue

---

**Remember:** The AI is a tool to assist you, not replace your expertise. Always review and refine generated content to ensure it meets your project's specific requirements.
