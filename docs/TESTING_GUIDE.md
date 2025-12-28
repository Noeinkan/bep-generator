# Testing Guide - Ollama Integration

## Pre-requisites Checklist

Before testing, ensure:

- [ ] Ollama is installed: `ollama --version`
- [ ] Ollama is running: `curl http://localhost:11434/api/tags`
- [ ] Model downloaded: `ollama list` shows `llama3.2:3b`
- [ ] Python venv activated (if testing manually)
- [ ] `requests` library installed: `pip install requests`

---

## Test 1: Ollama Service (1 minute)

### Check Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

**Expected output:** JSON with list of models

### Check model is downloaded:

```bash
ollama list
```

**Expected output:**
```
NAME              ID              SIZE      MODIFIED
llama3.2:3b       abc123def       6.0 GB    2 hours ago
```

### Quick model test:

```bash
ollama run llama3.2:3b "Write a one-sentence BIM project summary"
```

**Expected:** Professional sentence about BIM project

---

## Test 2: Python Module (2 minutes)

### Manual Python Test:

```bash
cd ml-service
venv\Scripts\activate
python
```

Then in Python REPL:

```python
>>> from model_loader_ollama import get_generator
>>> gen = get_generator()
# Should see: "Connected to Ollama" and "Using model: llama3.2:3b"

>>> text = gen.suggest_for_field("executiveSummary", "", 200)
>>> print(text)
# Should see professional BEP executive summary text

>>> len(text)
# Should be > 50 characters

>>> exit()
```

### Automated Python Test:

```bash
cd ml-service
venv\Scripts\python.exe test_manual.py
```

**Expected output:**
```
[1/4] Testing import and initialization...
✅ Import successful

[2/4] Creating generator instance...
✅ Generator created

[3/4] Testing text generation for executiveSummary...
✅ Generation successful
📄 Generated Text (XXX chars):
...professional text here...

[4/4] Testing with partial text...
✅ Generation successful

🎉 ALL MANUAL TESTS PASSED!
```

---

## Test 3: FastAPI Endpoints (3 minutes)

### Start the API server:

**Terminal 1:**
```bash
cd ml-service
venv\Scripts\python.exe -m uvicorn api:app --reload --port 8000
```

**Expected output:**
```
INFO:     Loading BEP text generation model (backend: ollama)...
INFO:     Connected to Ollama at http://localhost:11434
INFO:     Using model: llama3.2:3b
INFO:     Model loaded successfully on device: ollama
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Test endpoints (Terminal 2):

#### 1. Root endpoint:
```bash
curl http://localhost:8000/
```

**Expected:**
```json
{
  "message": "BEP AI Text Generator API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

#### 2. Health check:
```bash
curl http://localhost:8000/health
```

**Expected:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "ollama"
}
```

#### 3. Generate endpoint:
```bash
curl -X POST http://localhost:8000/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\": \"This project aims to\", \"field_type\": \"executiveSummary\", \"max_length\": 150}"
```

**Expected:**
```json
{
  "text": "...professional continuation...",
  "prompt_used": "This project aims to"
}
```

#### 4. Suggest endpoint:
```bash
curl -X POST http://localhost:8000/suggest ^
  -H "Content-Type: application/json" ^
  -d "{\"field_type\": \"bimObjectives\", \"partial_text\": \"The main objectives are\", \"max_length\": 200}"
```

**Expected:**
```json
{
  "text": "...professional BIM objectives...",
  "prompt_used": "The main objectives are"
}
```

### Test API Documentation:

Open in browser: http://localhost:8000/docs

- Should see Swagger UI
- Try "POST /suggest" → "Try it out"
- Fill in parameters
- Click "Execute"
- Check response

---

## Test 4: Full Integration with npm (2 minutes)

### Start all services:

```bash
npm start
```

**Expected output:**
```
[Frontend] Starting React development server on port 3000...
[Backend] Server running on http://localhost:5001
[ML] Loading BEP text generation model (backend: ollama)...
[ML] Connected to Ollama at http://localhost:11434
[ML] Model loaded successfully
```

### Test in browser:

1. Open: http://localhost:3000
2. Login or create account
3. Create new BEP or open existing
4. Navigate to "Executive Summary" section
5. Click **"✨ AI Generate"** or **"AI Suggest"** button
6. Wait 2-4 seconds
7. **Expected:** Professional, coherent BEP text appears

---

## Test 5: Compare Outputs (Quality Check)

### Test Input:
```
"The BIM objectives for this project are to"
```

### Expected Output Quality:

✅ **Good output (Ollama working):**
```
The BIM objectives for this project are to establish a comprehensive
digital information management framework aligned with ISO 19650-2
standards. This includes implementing structured workflows for the
Common Data Environment, defining clear Level of Information Need
requirements for each project stage, ensuring effective coordination
between all delivery team members, and maintaining full traceability
of information containers throughout the project lifecycle.
```

❌ **Bad output (Ollama not working, fallback to char-RNN):**
```
implement iso standrds. this includes respons for deliv teams and
specifi the proc for inform exchang requir structur handov protocol
deliv mileston...
```

If you see bad output:
1. Check `ml-service/config.py` → `MODEL_TYPE = "ollama"`
2. Verify Ollama is running
3. Check API logs for errors

---

## Automated Full Test Suite

### Run complete test:

```bash
# Windows
test-ollama-full.bat

# Manual step-by-step
npm run verify:ollama
cd ml-service
venv\Scripts\python.exe test_manual.py
venv\Scripts\python.exe test_ollama_integration.py
```

---

## Troubleshooting Tests

### Test fails: "Cannot import config"

**Cause:** Wrong directory

**Fix:**
```bash
cd ml-service
python test_manual.py
```

### Test fails: "Cannot connect to Ollama"

**Cause:** Ollama not running

**Fix:**
```bash
# Check if running
curl http://localhost:11434/api/tags

# If not, start Ollama
# Windows: Search "Ollama" in Start Menu and launch
```

### Test fails: "Model not found"

**Cause:** Model not downloaded

**Fix:**
```bash
ollama pull llama3.2:3b
```

### API test fails: Port 8000 in use

**Fix:**
```bash
# Use different port
uvicorn api:app --port 8001

# Or kill existing process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Generated text is gibberish

**Cause:** Using char-RNN instead of Ollama

**Fix:** Check `ml-service/config.py`:
```python
MODEL_TYPE = "ollama"  # Should be "ollama", not "char-rnn"
```

---

## Success Criteria

All tests pass if:

- ✅ Ollama service responds on port 11434
- ✅ Model `llama3.2:3b` is listed in `ollama list`
- ✅ Python module imports without errors
- ✅ Generator creates successfully with `device: "ollama"`
- ✅ Text generation produces coherent, professional output (>50 chars)
- ✅ API health endpoint returns `"status": "healthy"`
- ✅ API generate/suggest endpoints return professional text
- ✅ Full app starts without errors
- ✅ Browser UI shows AI-generated professional BEP content

---

## Performance Benchmarks

Expected generation times:

| Hardware | Model | Time (200 chars) |
|----------|-------|------------------|
| CPU only (i5) | llama3.2:1b | 3-5s |
| CPU only (i7) | llama3.2:3b | 5-8s |
| GPU (RTX 2060) | llama3.2:3b | 2-3s |
| GPU (RTX 3060) | llama3.2:3b | 1-2s |
| GPU (RTX 4090) | mistral:7b | 1-2s |

If your times are >10s, consider:
- Using smaller model (`llama3.2:1b`)
- Closing other applications
- Checking GPU is being used (`nvidia-smi`)

---

## Next Steps After Successful Tests

1. ✅ All tests pass
2. 🎯 Start using in production: `npm start`
3. 📝 Generate real BEP documents
4. 🔧 Fine-tune parameters if needed (temperature, max_length)
5. 🚀 Deploy to production (optional)

---

**Happy testing! 🚀**
