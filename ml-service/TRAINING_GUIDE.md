# BEP AI Model Training Guide

Complete guide for training the AI model with your BEP documents to generate contextually appropriate ISO 19650 compliant text.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What Was Created](#what-was-created)
3. [Document Preparation](#document-preparation)
4. [Processing Documents](#processing-documents)
5. [Training the Model](#training-the-model)
6. [Advanced Configuration](#advanced-configuration)
7. [Troubleshooting](#troubleshooting)

---

## What Was Created

### Directory Structure

```
ml-service/
├── data/
│   ├── iso19650_glossary.json              # ISO 19650 glossary (23 terms)
│   ├── consolidated_training_data.txt      # Auto-generated from your docs
│   └── training_documents/
│       ├── pdf/          # Place your PDF BEP files here
│       ├── docx/         # Place your DOCX BEP files here
│       └── txt/          # Place your TXT BEP files here (example included)
├── scripts/
│   ├── process_training_documents.py       # Extracts text from all formats
│   └── train_model.py                      # Training script (updated)
└── requirements.txt                        # Python dependencies
```

### Installed Libraries

- PyPDF2 - PDF text extraction
- pdfplumber - Advanced PDF parsing with tables
- python-docx - Word document processing
- nltk - Natural language processing
- tqdm - Progress bars

### ISO 19650 Glossary

The glossary includes 23 key terms with definitions and usage examples:
- EIR (Employer's Information Requirements)
- AIR (Asset Information Requirements)
- CDE (Common Data Environment)
- Information Manager, Lead Appointed Party
- Level of Information Need, MIDP, TIDP
- COBie, IFC, Federation, and more

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ with virtual environment activated
- Required libraries installed (automatic via `start_service.bat`)

### 3-Step Process

```bash
# Step 1: Add your BEP documents to the training folders
# Place PDFs in: ml-service/data/training_documents/pdf/
# Place DOCX in: ml-service/data/training_documents/docx/
# Place TXT in:  ml-service/data/training_documents/txt/

# Step 2: Process the documents
cd ml-service
venv\Scripts\python.exe scripts\process_training_documents.py

# Step 3: Train the model
venv\Scripts\python.exe scripts\train_model.py --epochs 150
```

**Important:** Always use `venv\Scripts\python.exe` to run scripts, as the required libraries are installed in the virtual environment.

That's it! The model is now ready to use.

---

## 📄 Document Preparation

### Supported Formats

- **PDF** (`.pdf`) - Published BEP documents, official templates
- **Word** (`.docx`) - Draft BEPs, editable templates
- **Text** (`.txt`) - Pre-processed content, plain text exports

Just drop your files in the appropriate folder:
- `ml-service/data/training_documents/pdf/`
- `ml-service/data/training_documents/docx/`
- `ml-service/data/training_documents/txt/`

### Document Quality Guidelines

#### ✅ **Include These:**

- **Complete BEP documents** following ISO 19650 structure
- **Professional language** with correct grammar and terminology
- **Structured sections:**
  - Executive Summary
  - Project Information
  - Roles and Responsibilities
  - Technical Requirements
  - Collaboration Procedures
  - Information Delivery Planning
- **Standard terminology:**
  - EIR, AIR, Level of Information Need
  - Information Manager, Lead Appointed Party
  - CDE, MIDP, TIDP
  - COBie, IFC, Federation

#### ❌ **Avoid These:**

- Draft documents with placeholder text ("TBD", "XXX", etc.)
- Documents with severe OCR errors
- Marketing materials or sales brochures
- Documents in languages other than English
- Incomplete or corrupted files

### Privacy & Confidentiality

**IMPORTANT:** Before adding documents, remove:

- Client names and contact information
- Specific project addresses and locations
- Commercial terms and pricing
- Proprietary methodologies
- Financial data

**Keep:**

- Technical processes and procedures
- Role descriptions and responsibilities
- Standards and requirements
- ISO 19650 compliant structure and terminology

**Tip:** Use find/replace to sanitize:
- "Acme Corporation" → "The Client"
- "London Bridge Project" → "The Project"
- Specific amounts → Generic terms

---

## 🔄 Processing Documents

### Basic Processing

Extract text from all documents and create consolidated training file:

```bash
cd ml-service
venv\Scripts\python.exe scripts\process_training_documents.py
```

**Output:**
- Creates `data/consolidated_training_data.txt`
- Includes ISO 19650 glossary context
- Combines all PDF, DOCX, and TXT files

### Advanced Options

```bash
# Specify custom output file
venv\Scripts\python.exe scripts\process_training_documents.py --output data/my_custom_training.txt

# Use different source directory
venv\Scripts\python.exe scripts\process_training_documents.py --data-dir path/to/my/documents
```

### What the Processor Does

1. **Extracts text** from PDFs (including tables)
2. **Extracts text** from DOCX (paragraphs, tables, headers)
3. **Reads** plain text files
4. **Cleans** the text:
   - Removes excessive whitespace
   - Eliminates duplicate lines
   - Normalizes formatting
5. **Adds ISO 19650 glossary** examples at the beginning
6. **Combines** everything into one consolidated file

### Verification

After processing, check:

```bash
# View file size (should be >50KB for good results)
ls -lh data/consolidated_training_data.txt

# View first 50 lines
head -n 50 data/consolidated_training_data.txt

# Count characters
wc -c data/consolidated_training_data.txt
```

**Recommended minimum:** 50,000+ characters

---

## 🧠 Training the Model

### Basic Training

Train with default settings (100 epochs):

```bash
cd ml-service
venv\Scripts\python.exe scripts\train_model.py
```

### Recommended Training

For better quality results:

```bash
# 150 epochs (good balance of quality vs. time)
venv\Scripts\python.exe scripts\train_model.py --epochs 150

# 200 epochs (high quality, takes longer)
venv\Scripts\python.exe scripts\train_model.py --epochs 200
```

### Using Custom Data File

```bash
# Train on specific file
venv\Scripts\python.exe scripts\train_model.py --data-file data/my_custom_training.txt --epochs 150
```

### Training Parameters

```bash
venv\Scripts\python.exe scripts\train_model.py --epochs 200 --hidden-size 512 --seq-length 100 --learning-rate 0.001 --batch-size 128
```

### What Happens During Training

1. **Loads** consolidated training data
2. **Prepares** character-level sequences
3. **Initializes** LSTM neural network
4. **Trains** the model (shows progress every 10 epochs)
5. **Saves** trained model to `models/bep_model.pth`
6. **Saves** character mappings to `models/char_mappings.json`
7. **Generates** sample text to verify quality

### Training Output

```
============================================================
BEP Text Generation Model Training
============================================================
Using device: cuda

Loading training data...
Total characters: 125,487
Unique characters: 95

Preparing dataset...
Training sequences: 125,387

Initializing model...
Model parameters: 1,234,567

Starting training...
Epoch [10/150], Loss: 1.8234
Epoch [20/150], Loss: 1.5421
Epoch [30/150], Loss: 1.3156
...
Epoch [150/150], Loss: 0.7234

Sample Generated Text:
============================================================
The BEP establishes a comprehensive framework for information
management throughout the project lifecycle, ensuring compliance
with ISO 19650 standards and the Employer's Information Requirements.
============================================================

Training complete!
```

### Training Time Estimates

| Epochs | Data Size | CPU | GPU |
|--------|-----------|-----|-----|
| 50     | 50KB      | ~5 min | ~1 min |
| 100    | 50KB      | ~10 min | ~2 min |
| 150    | 100KB     | ~20 min | ~4 min |
| 200    | 100KB     | ~30 min | ~6 min |

**Tip:** Use GPU if available for 5-10x faster training

### After Training

1. **Restart the API service:**
   ```bash
   cd ml-service
   start_service.bat
   ```

2. **Test in the BEP Generator:**
   - Open a text field
   - Type: "The BEP establishes"
   - Click the ✨ AI button
   - Verify the suggestion quality

---

## ⚙️ Advanced Configuration

### Improving Model Quality

#### 1. **Add More Training Data**

More data = better results. Aim for:
- **Minimum:** 50,000 characters
- **Good:** 100,000+ characters
- **Excellent:** 200,000+ characters

```bash
# Add more documents to training_documents folders
# Re-process
python scripts/process_training_documents.py

# Re-train with more epochs
python scripts/train_model.py --epochs 200
```

#### 2. **Increase Model Capacity**

For complex documents:

```bash
python scripts/train_model.py --hidden-size 768 --epochs 200
```

**Warning:** Larger models require more memory and training time.

#### 3. **Fine-Tune Learning Rate**

If training loss plateaus:

```bash
# Slower, more careful learning
python scripts/train_model.py --learning-rate 0.0005 --epochs 250

# Faster learning (may be less stable)
python scripts/train_model.py --learning-rate 0.002 --epochs 100
```

#### 4. **Adjust Sequence Length**

```bash
# Longer context (better coherence, more memory)
python scripts/train_model.py --seq-length 150

# Shorter context (faster, less coherent)
python scripts/train_model.py --seq-length 50
```

### Custom Field Prompts

Edit `ml-service/model_loader.py` to add specialized prompts:

```python
self.field_prompts = {
    'executiveSummary': 'This BIM Execution Plan establishes ',
    'projectObjectives': 'The primary objectives include ',
    'rolesResponsibilities': 'The Information Manager ',
    'technicalRequirements': 'In accordance with ISO 19650, ',
    'collaboration': 'All project information shall ',
}
```

### Monitoring Training Progress

To track training quality, look at the **loss value**:

- **Loss > 2.0:** Model just started, still learning basics
- **Loss 1.5-2.0:** Model learning structure
- **Loss 1.0-1.5:** Model learning vocabulary and patterns
- **Loss 0.7-1.0:** Model generating coherent text ✅
- **Loss < 0.7:** Excellent quality ⭐

**Note:** Lower is better, but too low might mean overfitting.

---

## 🔧 Troubleshooting

### Problem: "No text extracted" errors

**Cause:** PDF might be scanned images without text layer

**Solutions:**
1. Check if PDF is searchable (can you copy text?)
2. Use OCR software to convert image PDFs to text
3. Try alternative PDF extraction tools
4. Convert to DOCX format instead

### Problem: Poor quality suggestions

**Cause:** Insufficient or poor quality training data

**Solutions:**
1. **Add more documents** (aim for 100KB+ total)
2. **Train longer** (150-200 epochs)
3. **Review training data quality:**
   ```bash
   head -n 100 data/consolidated_training_data.txt
   ```
4. **Remove low-quality documents** with errors or incomplete text
5. **Ensure ISO 19650 terminology** is present

### Problem: Training loss not decreasing

**Cause:** Learning rate too high/low or insufficient epochs

**Solutions:**
```bash
# Reduce learning rate
python scripts/train_model.py --learning-rate 0.0005 --epochs 200

# Increase model capacity
python scripts/train_model.py --hidden-size 768 --epochs 150
```

### Problem: Model generates repetitive text

**Cause:** Overfitting or too many epochs

**Solutions:**
1. **Add more diverse training data**
2. **Reduce epochs** if loss is very low (<0.5)
3. **Increase dropout** (requires code modification)
4. **Use more training variety**

### Problem: "Out of memory" during training

**Cause:** Model or batch size too large for available RAM/VRAM

**Solutions:**
```bash
# Reduce batch size
python scripts/train_model.py --batch-size 64

# Reduce model size
python scripts/train_model.py --hidden-size 256 --batch-size 64

# Reduce sequence length
python scripts/train_model.py --seq-length 50
```

### Problem: File encoding errors

**Cause:** Non-UTF-8 encoding in source documents

**Solutions:**
1. Convert files to UTF-8 encoding
2. Use a text editor to re-save with UTF-8
3. Remove special characters causing issues

### Problem: DOCX tables not extracted

**Cause:** Complex table formatting

**Solutions:**
1. Simplify table structure in Word
2. Export table to text/CSV first
3. Manually copy important table content to TXT file

---

## 📊 Quality Metrics

### How to Evaluate Model Quality

After training, test with various prompts:

```bash
# Start the API service
cd ml-service
start_service.bat
```

Test these prompts in your BEP Generator:

| Prompt | Expected Quality |
|--------|------------------|
| "The BEP establishes" | Should mention ISO 19650, EIR, information management |
| "The Information Manager" | Should describe role, responsibilities |
| "All models shall" | Should reference standards, LOD, validation |
| "In accordance with ISO 19650" | Should continue with compliant language |

### Good Output Characteristics

✅ Uses correct ISO 19650 terminology
✅ Grammatically correct sentences
✅ Contextually appropriate content
✅ Professional tone
✅ No repetitive loops
✅ Coherent structure

### Poor Output Characteristics

❌ Nonsensical words or phrases
❌ Repetitive text patterns
❌ Mixed topics without coherence
❌ Incorrect terminology
❌ Grammatical errors

If output is poor → retrain with more/better data and more epochs.

---

## 🔄 Iterative Improvement Workflow

```
┌─────────────────────────────────────────┐
│ 1. Collect BEP Documents                │
│    (PDFs, DOCX, TXT)                     │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ 2. Process Documents                     │
│    python process_training_documents.py  │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ 3. Train Model                           │
│    python train_model.py --epochs 150    │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ 4. Test Quality                          │
│    Use BEP Generator AI features         │
└───────────────┬─────────────────────────┘
                │
                ▼
        Quality Good?
        /           \
      YES            NO
       │              │
       ▼              ▼
    Done!      Add more data / Adjust params
                      │
                      └──────────┐
                                 │
                     ┌───────────▼──────────┐
                     │ Return to Step 2     │
                     └──────────────────────┘
```

---

## 📚 Additional Resources

- [AI Quick Start Guide](AI_QUICKSTART.md) - Getting started quickly
- [AI Integration Guide](AI_INTEGRATION_GUIDE.md) - Complete technical documentation
- [ISO 19650 Glossary](data/iso19650_glossary.json) - Terminology reference
- [Training Documents README](data/training_documents/README.md) - Document organization guide

---

## 💡 Tips for Best Results

1. **Start with quality over quantity** - 10 good BEPs better than 50 poor ones
2. **Diversity matters** - Mix templates, real projects, different sectors
3. **Regular updates** - Add new documents periodically and retrain
4. **Test incrementally** - Train at 50, 100, 150 epochs to find sweet spot
5. **Use the glossary** - The included glossary enhances context understanding
6. **Monitor loss** - Stop if loss plateaus to save time
7. **GPU acceleration** - Use GPU if available for faster training
8. **Backup models** - Save `models/` folder after successful training

---

## 🎯 Summary

### Minimal Setup
```bash
# Add documents to training_documents/pdf/, /docx/, /txt/
python scripts/process_training_documents.py
python scripts/train_model.py --epochs 100
```

### Recommended Setup
```bash
# Add 5-10 quality BEP documents
python scripts/process_training_documents.py
python scripts/train_model.py --epochs 150
# Test and iterate
```

### Professional Setup
```bash
# Add 15+ diverse BEP documents (100KB+ total)
python scripts/process_training_documents.py
python scripts/train_model.py --epochs 200 --hidden-size 512
# Fine-tune and customize field prompts
```

---

**Remember:** The AI is a tool to **assist**, not replace your expertise. Always review and refine generated content to ensure it meets your project's specific requirements.

Happy training! 🚀
