# Ollama Setup Guide - BEP Generator AI Backend

## Fase 1: Installazione Ollama (5 minuti)

### Cos'è Ollama?
Ollama è un software che gestisce modelli AI localmente sul tuo PC:
- È come Docker ma per modelli AI
- Esegue modelli di linguaggio (LLM) in locale senza bisogno di connessione internet
- Espone API REST per integrare l'AI nelle tue applicazioni
- Supporta vari modelli: Llama, Mistral, CodeLlama, ecc.

### Step 1.1: Download e Installazione

#### Windows
1. Vai su: https://ollama.com/download/windows
2. Scarica `OllamaSetup.exe`
3. Esegui l'installer (installazione automatica)
4. Ollama si avvia automaticamente in background

#### macOS
```bash
# Download e installazione
curl -fsSL https://ollama.com/install.sh | sh
```

#### Linux
```bash
# Download e installazione
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 1.2: Verifica Installazione

Apri un terminale/prompt dei comandi e verifica che Ollama sia installato:

```bash
ollama --version
```

Dovresti vedere qualcosa come: `ollama version is 0.x.x`

---

## Fase 2: Download Modello Llama 3.2 3B (5-10 minuti)

### Step 2.1: Pull del Modello

Il modello **Llama 3.2 3B** è il modello raccomandato per il BEP Generator:
- **Qualità**: Eccellente per generazione di testo professionale
- **Dimensione**: ~6 GB (download una volta sola)
- **Velocità**: 2-4 secondi per risposta su hardware moderno
- **Requisiti**: 8GB RAM minimo, GPU opzionale (migliora velocità)

```bash
# Download del modello (richiede ~6GB di spazio disco)
ollama pull llama3.2:3b
```

**Output atteso:**
```
pulling manifest
pulling 6a0746a1ec1a... 100% ▕████████████████▏ 2.0 GB
pulling 4fa551d4f938... 100% ▕████████████████▏ 1.4 KB
pulling 8ab4849b038c... 100% ▕████████████████▏  254 B
pulling 577073ffcc6c... 100% ▕████████████████▏  110 B
pulling ad1518640c43... 100% ▕████████████████▏  483 B
verifying sha256 digest
writing manifest
success
```

### Step 2.2: Modelli Alternativi (Opzionale)

Se il tuo PC ha hardware limitato, considera questi modelli:

#### Llama 3.2 1B - Più veloce, ottima qualità
```bash
ollama pull llama3.2:1b
```
- **Dimensione**: ~2 GB
- **Velocità**: 1-2 secondi
- **Requisiti**: 4GB RAM minimo

#### Mistral 7B - Migliore qualità (se hai GPU potente)
```bash
ollama pull mistral:7b
```
- **Dimensione**: ~4.1 GB
- **Velocità**: 3-5 secondi
- **Requisiti**: 16GB RAM, GPU consigliata

---

## Fase 3: Verifica Funzionamento (2 minuti)

### Step 3.1: Test Servizio Ollama

Ollama parte automaticamente in background dopo l'installazione. Verifica che sia in esecuzione:

```bash
# Test connessione API
curl http://localhost:11434/api/tags
```

**Output atteso:** Lista dei modelli scaricati in formato JSON.

### Step 3.2: Test Generazione Testo

Testa il modello con un prompt semplice:

```bash
ollama run llama3.2:3b "Write a brief executive summary for a BIM project"
```

**Output atteso:** Il modello genererà un testo professionale in pochi secondi.

### Step 3.3: Test con Script Python

Usa lo script di verifica fornito:

```bash
cd ml-service
python verify_ollama.py
```

Lo script verificherà:
- ✅ Ollama è in esecuzione
- ✅ Il modello è disponibile
- ✅ API REST funzionano correttamente
- ✅ Generazione testo funziona

---

## Fase 4: Integrazione con BEP Generator

### Step 4.1: Configurazione

Il BEP Generator si connetterà automaticamente a Ollama su `http://localhost:11434`.

Nessuna configurazione aggiuntiva necessaria se:
- Ollama è in esecuzione
- Hai scaricato almeno un modello
- La porta 11434 è libera

### Step 4.2: Modelli Supportati

Il BEP Generator supporta questi modelli (in ordine di preferenza):

1. **llama3.2:3b** (Raccomandato)
   - Qualità eccellente
   - Ottimo per testo tecnico BIM/BEP
   - Velocità bilanciata

2. **llama3.2:1b**
   - Ottima qualità
   - Più veloce del 3B
   - Buono per hardware limitato

3. **mistral:7b**
   - Qualità superiore
   - Più lento
   - Richiede GPU potente

### Step 4.3: Start del Sistema

Avvia tutti i servizi:

```bash
# Dalla root del progetto
npm start
```

Questo avvierà:
- Frontend React (porta 3000)
- Backend Node.js (porta 5001)
- ML Service Python con Ollama (porta 5003)

---

## Comandi Utili Ollama

### Gestione Modelli

```bash
# Lista modelli installati
ollama list

# Rimuovi un modello
ollama rm llama3.2:3b

# Aggiorna un modello
ollama pull llama3.2:3b

# Info su un modello
ollama show llama3.2:3b
```

### Servizio Ollama

```bash
# Start servizio (di solito automatico)
ollama serve

# Stop servizio
# Windows: Task Manager → Ollama → End Task
# Linux/Mac: killall ollama
```

### Test Interattivo

```bash
# Chat interattiva con il modello
ollama run llama3.2:3b

# Con parametri personalizzati
ollama run llama3.2:3b --temperature 0.7
```

---

## Troubleshooting

### Problema: "Ollama is not running"

**Soluzione:**
```bash
# Windows: Cerca "Ollama" nel menu Start e avvialo
# Linux/Mac:
ollama serve
```

### Problema: "Model not found"

**Soluzione:**
```bash
# Verifica modelli installati
ollama list

# Scarica il modello mancante
ollama pull llama3.2:3b
```

### Problema: "Port 11434 already in use"

**Soluzione:**
```bash
# Windows
netstat -ano | findstr :11434
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:11434 | xargs kill -9
```

### Problema: Generazione lenta o timeout

**Soluzioni:**
- Chiudi applicazioni non necessarie per liberare RAM
- Usa un modello più leggero (llama3.2:1b)
- Verifica che hai almeno 8GB RAM disponibili
- Se hai GPU NVIDIA, installa CUDA per accelerazione

### Problema: Testo generato di bassa qualità

**Soluzioni:**
- Usa il modello llama3.2:3b o mistral:7b
- Fornisci prompt più dettagliati
- Aumenta la temperatura per più creatività (0.7-0.9)
- Diminuisci la temperatura per più coerenza (0.3-0.5)

---

## Requisiti Hardware

### Minimi (llama3.2:1b)
- **RAM**: 4GB disponibili
- **Disco**: 5GB liberi
- **CPU**: Qualsiasi processore moderno
- **GPU**: Opzionale

### Raccomandati (llama3.2:3b)
- **RAM**: 8GB disponibili
- **Disco**: 10GB liberi
- **CPU**: Intel i5/AMD Ryzen 5 o superiore
- **GPU**: NVIDIA con 6GB+ VRAM (opzionale)

### Ottimali (mistral:7b o modelli più grandi)
- **RAM**: 16GB+ disponibili
- **Disco**: 20GB+ liberi
- **CPU**: Intel i7/AMD Ryzen 7 o superiore
- **GPU**: NVIDIA RTX 3060 o superiore (12GB+ VRAM)

---

## Confronto: Ollama vs Hugging Face

| Caratteristica | Ollama | Hugging Face (precedente) |
|---------------|--------|---------------------------|
| **Setup** | 5 minuti | 15-30 minuti |
| **Download** | ~6GB | ~6GB |
| **Velocità** | 2-4s per risposta | 5-10s per risposta |
| **RAM richiesta** | 8GB | 16GB+ |
| **GPU richiesta** | Opzionale | Consigliata |
| **Qualità** | Eccellente | Buona |
| **Facilità** | Molto facile | Complessa |
| **API** | REST nativa | Python transformers |

**Vantaggio Ollama**: Setup più semplice, performance migliori, meno dipendenze.

---

## Prossimi Passi

Dopo aver completato il setup di Ollama:

1. ✅ Verifica che Ollama sia in esecuzione
2. ✅ Hai scaricato almeno un modello
3. ➡️ **Procedi al setup del ML Service Python** (vedi [ML_SERVICE_SETUP.md](./ML_SERVICE_SETUP.md))
4. ➡️ Avvia il BEP Generator completo con `npm start`
5. ➡️ Testa la generazione AI nell'interfaccia web

---

## Risorse Utili

- **Sito Ollama**: https://ollama.com
- **Modelli disponibili**: https://ollama.com/library
- **Documentazione API**: https://github.com/ollama/ollama/blob/main/docs/api.md
- **Discord Ollama**: https://discord.gg/ollama
- **GitHub**: https://github.com/ollama/ollama

---

## FAQ

### 1. Posso usare Ollama offline?
Sì, una volta scaricato il modello, Ollama funziona completamente offline.

### 2. Quanti modelli posso avere contemporaneamente?
Tutti quelli che vuoi, limitato solo dallo spazio disco disponibile.

### 3. Ollama usa la mia GPU?
Sì, se hai una GPU NVIDIA con CUDA installato, Ollama la userà automaticamente.

### 4. Come cambio modello nel BEP Generator?
Modifica il file `ml-service/config.py` e cambia `MODEL_NAME = "llama3.2:3b"`.

### 5. Ollama è gratuito?
Sì, Ollama e tutti i modelli open-source (Llama, Mistral, ecc.) sono completamente gratuiti.

---

**🎉 Setup completato! Ora hai un AI backend locale professionale per il tuo BEP Generator.**
