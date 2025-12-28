# Quick Start Guide - Ollama Backend per BEP Generator

## Panoramica Rapida

Questo setup usa **Ollama** invece di PyTorch/Hugging Face per l'AI:

✅ **Pro:**
- Setup in 10 minuti invece di 30-60 minuti
- Nessun training necessario
- Qualità del testo superiore
- Più veloce (2-4 secondi vs 5-10 secondi)
- Meno RAM richiesta (8GB vs 16GB+)
- Setup più semplice

❌ **Contro:**
- Richiede download di ~6GB una volta sola
- Richiede Ollama in esecuzione in background

---

## Setup Completo (10 minuti)

### Step 1: Installa Ollama (3 minuti)

#### Windows
```bash
# Scarica da: https://ollama.com/download/windows
# Oppure usa winget:
winget install Ollama.Ollama
```

#### Linux/Mac
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Verifica installazione:**
```bash
ollama --version
```

### Step 2: Scarica il Modello (5 minuti)

```bash
# Modello raccomandato (6GB)
ollama pull llama3.2:3b
```

**Output atteso:**
```
pulling manifest
pulling 6a0746a1ec1a... 100% ▕████████████████▏ 2.0 GB
...
success
```

**Alternative leggere:**
```bash
# Se hai hardware limitato (2GB)
ollama pull llama3.2:1b

# Per qualità superiore (4GB, richiede GPU)
ollama pull mistral:7b
```

### Step 3: Verifica Ollama (1 minuto)

```bash
# Test veloce
ollama run llama3.2:3b "Write a BIM project summary"

# Test completo con lo script
cd bep-generator
npm run verify:ollama
```

**Output atteso:**
```
✅ Ollama è in esecuzione su http://localhost:11434
✅ Modello raccomandato 'llama3.2:3b' è installato
✅ Generazione completata in 2.45 secondi
🎉 TUTTI I TEST SUPERATI!
```

### Step 4: Avvia il BEP Generator (1 minuto)

```bash
# Dalla root del progetto
npm start
```

Questo avvierà:
- ✅ Frontend React → http://localhost:3000
- ✅ Backend Node.js → http://localhost:5001
- ✅ ML Service (Ollama) → http://localhost:5003

---

## Test dell'Integrazione

### Test 1: API Health Check

```bash
curl http://localhost:5003/health
```

**Output atteso:**
```json
{
  "status": "healthy",
  "ollama_connected": true,
  "model": "llama3.2:3b",
  "backend": "Ollama"
}
```

### Test 2: Generazione Testo

```bash
curl -X POST http://localhost:5003/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "This project aims to",
    "field_type": "executiveSummary",
    "max_length": 150
  }'
```

**Output atteso:**
```json
{
  "text": "establish a comprehensive framework for Building Information Modeling implementation across all project phases...",
  "prompt_used": "This project aims to",
  "model": "llama3.2:3b"
}
```

### Test 3: Suggestion per Campo Specifico

```bash
curl -X POST http://localhost:5003/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "field_type": "projectObjectives",
    "partial_text": "The main goals are",
    "max_length": 200
  }'
```

### Test 4: Frontend Integration

1. Apri http://localhost:3000
2. Crea un nuovo BEP
3. Vai in una sezione (es. Executive Summary)
4. Clicca sul pulsante AI "✨ Generate"
5. Verifica che il testo venga generato

---

## Comandi Utili

### Package.json Scripts

```bash
# Start tutto (frontend + backend + ML)
npm start

# Start solo ML service
npm run start:ml

# Verifica Ollama
npm run verify:ollama

# Start con vecchio sistema PyTorch (se necessario)
npm run start:ml:old
```

### Ollama Commands

```bash
# Lista modelli installati
ollama list

# Info su un modello
ollama show llama3.2:3b

# Rimuovi un modello
ollama rm llama3.2:3b

# Aggiorna un modello
ollama pull llama3.2:3b

# Test interattivo
ollama run llama3.2:3b
```

### ML Service Diretto

```bash
# Start manualmente (Windows)
cd ml-service
start_ollama_service.bat

# Start manualmente (Linux/Mac)
cd ml-service
source venv/bin/activate
python api_ollama.py
```

---

## Troubleshooting

### ❌ "Cannot connect to Ollama"

**Problema:** Il ML service non riesce a connettersi a Ollama

**Soluzione:**
```bash
# Verifica che Ollama sia in esecuzione
curl http://localhost:11434/api/tags

# Se non risponde, avvia Ollama:
# Windows: Cerca "Ollama" nel menu Start
# Linux/Mac: ollama serve
```

### ❌ "Model not found"

**Problema:** Il modello non è scaricato

**Soluzione:**
```bash
# Verifica modelli installati
ollama list

# Scarica il modello mancante
ollama pull llama3.2:3b
```

### ❌ Generazione molto lenta (>30 secondi)

**Problema:** Hardware insufficiente per il modello

**Soluzioni:**
1. Usa un modello più leggero:
```bash
ollama pull llama3.2:1b
```

2. Modifica `ml-service/api_ollama.py` e cambia il modello:
```python
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:1b')  # Cambia da 3b a 1b
```

3. Chiudi applicazioni non necessarie per liberare RAM

### ❌ "Port 5003 already in use"

**Problema:** Porta ML service occupata

**Soluzione:**
```bash
# Windows
netstat -ano | findstr :5003
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5003 | xargs kill -9
```

### ❌ Testo generato di bassa qualità

**Soluzioni:**

1. Usa il modello 3B o più grande:
```bash
ollama pull llama3.2:3b
# oppure
ollama pull mistral:7b
```

2. Modifica temperatura in `api_ollama.py`:
```python
# Per testo più coerente (meno creativo)
temperature=0.3

# Per testo più creativo (meno coerente)
temperature=0.9
```

---

## Confronto Performance

### Ollama (llama3.2:3b) vs PyTorch LSTM

| Metrica | Ollama | PyTorch LSTM |
|---------|--------|--------------|
| Setup Time | 10 min | 30-60 min |
| Model Download | 6 GB | Training dataset |
| Training Required | ❌ No | ✅ Yes (15-30 min) |
| RAM Required | 8 GB | 16+ GB |
| GPU Required | ⚠️ Optional | ⚠️ Recommended |
| Generation Speed | 2-4 sec | 5-10 sec |
| Text Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ |
| Context Understanding | ⭐⭐⭐⭐⭐ | ⭐⭐⭐☆☆ |
| Ease of Setup | ⭐⭐⭐⭐⭐ | ⭐⭐☆☆☆ |

---

## Architettura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                     BEP Generator                           │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Frontend   │    │   Backend    │    │  ML Service  │ │
│  │  React:3000  │◄──►│  Node:5001   │◄──►│ Python:5003  │ │
│  └──────────────┘    └──────────────┘    └──────┬───────┘ │
│                                                   │          │
└───────────────────────────────────────────────────┼─────────┘
                                                    │
                                           ┌────────▼────────┐
                                           │     Ollama      │
                                           │  localhost:11434│
                                           │                 │
                                           │ ┌─────────────┐ │
                                           │ │llama3.2:3b  │ │
                                           │ │   (6GB)     │ │
                                           │ └─────────────┘ │
                                           └─────────────────┘
```

### Flusso di Generazione Testo:

1. **User** digita testo nel frontend → clicca "Generate"
2. **Frontend** invia richiesta POST a `/api/ml/suggest`
3. **Backend** forward a ML Service `/suggest`
4. **ML Service** chiama Ollama API `/api/generate`
5. **Ollama** genera testo con LLM locale
6. **Risposta** ritorna attraverso la catena
7. **Frontend** mostra il testo generato

---

## File Importanti

### Nuovi File Ollama
- `ml-service/api_ollama.py` - FastAPI service con Ollama backend
- `ml-service/ollama_generator.py` - Generatore testo con Ollama
- `ml-service/verify_ollama.py` - Script verifica setup
- `ml-service/start_ollama_service.bat` - Avvio rapido Windows
- `docs/OLLAMA_SETUP.md` - Documentazione completa setup

### File Esistenti (backup)
- `ml-service/api.py` - Vecchia API con PyTorch (ancora funzionante)
- `ml-service/model_loader.py` - Vecchio loader LSTM (ancora funzionante)

### Configurazione
- `package.json` - Aggiornato con `start:ml` → Ollama
- `package.json` - `start:ml:old` → PyTorch (fallback)

---

## Prossimi Passi

Dopo aver completato il setup:

1. ✅ Verifica che tutto funzioni: `npm run verify:ollama`
2. ✅ Avvia il sistema: `npm start`
3. ✅ Testa la generazione nel frontend
4. 📖 Leggi [OLLAMA_SETUP.md](./OLLAMA_SETUP.md) per dettagli avanzati
5. 🎯 Sperimenta con diversi modelli e temperature
6. 🚀 Deploy in produzione (opzionale)

---

## FAQ

### 1. Posso usare entrambi i sistemi (Ollama e PyTorch)?

Sì, puoi passare da uno all'altro:
```bash
# Ollama (default)
npm run start:ml

# PyTorch (vecchio)
npm run start:ml:old
```

### 2. Quale modello è meglio per il mio hardware?

| Hardware | Modello Raccomandato | RAM | Download |
|----------|---------------------|-----|----------|
| GPU 8GB+ | llama3.2:3b | 8GB | 6GB |
| GPU 4GB+ | llama3.2:1b | 4GB | 2GB |
| CPU only | llama3.2:1b | 8GB | 2GB |
| Potente  | mistral:7b | 16GB | 4GB |

### 3. Ollama funziona offline?

Sì! Dopo aver scaricato il modello, Ollama funziona completamente offline.

### 4. Come cambio modello?

```bash
# Opzione 1: Variabile ambiente
set OLLAMA_MODEL=llama3.2:1b
npm run start:ml

# Opzione 2: Modifica api_ollama.py
# Cambia la riga: OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
```

### 5. Ollama usa la mia GPU?

Sì, Ollama rileva automaticamente la GPU NVIDIA con CUDA e la usa per accelerazione.

---

## Supporto

- **Issue GitHub**: [bep-generator/issues](https://github.com/yourusername/bep-generator/issues)
- **Ollama Docs**: https://github.com/ollama/ollama/blob/main/docs/api.md
- **Discord Ollama**: https://discord.gg/ollama

---

**🎉 Buon lavoro con il tuo BEP Generator potenziato da AI!**
