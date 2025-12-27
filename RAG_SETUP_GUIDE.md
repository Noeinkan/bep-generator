# RAG System Setup Guide

## Overview

Il sistema RAG (Retrieval-Augmented Generation) migliora significativamente la qualità della generazione di testo per i BEP utilizzando:

1. **Estrazione testo dai DOCX** - Estrae contenuto dai tuoi documenti BEP esistenti
2. **Vector Database (FAISS)** - Crea un database vettoriale per ricerca semantica veloce
3. **Claude API** - Usa l'API di Anthropic Claude per generare testo contestuale di alta qualità
4. **Fallback LSTM** - Mantiene il sistema LSTM esistente come backup

## Vantaggi del RAG

- **Qualità superiore**: Genera testo basato sui tuoi documenti reali, non su pattern generici
- **Contestuale**: Recupera esempi rilevanti dai documenti per ogni campo
- **Flessibile**: Funziona con qualsiasi numero di documenti DOCX
- **Robusto**: Fallback automatico a LSTM se l'API non è disponibile
- **Tracciabile**: Mostra le fonti utilizzate per generare ogni suggerimento

## Prerequisiti

- Python 3.8 o superiore
- Account Anthropic (per API key)
- Documenti BEP in formato DOCX (almeno 3-5 per risultati ottimali)

## Installazione Rapida

### Opzione 1: Setup Automatico (Consigliato)

```bash
cd ml-service
setup_rag.bat
```

Lo script automatico:
1. Crea virtual environment
2. Installa dipendenze
3. Estrae testo dai DOCX
4. Crea il vector database
5. Configura il sistema

### Opzione 2: Setup Manuale

#### Passo 1: Installare Dipendenze

```bash
cd ml-service
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### Passo 2: Preparare i Documenti

Copia i tuoi documenti BEP (*.docx) nella cartella:
```
ml-service/data/training_documents/docx/
```

#### Passo 3: Estrarre Testo

```bash
python scripts/extract_docx.py
```

Questo comando:
- Estrae testo da tutti i DOCX
- Salva file .txt in `data/training_documents/txt/`
- Crea un file consolidato `data/consolidated_training_data.txt`

#### Passo 4: Configurare API Key

Ottieni la tua API key da [Anthropic Console](https://console.anthropic.com/).

**Windows:**
```cmd
setx ANTHROPIC_API_KEY "sk-ant-..."
```

**Oppure crea file `.env`:**
```bash
cd ml-service
copy .env.example .env
# Modifica .env e inserisci la tua API key
```

#### Passo 5: Creare Vector Database

```bash
python -c "from rag_engine import BEPRAGEngine; engine = BEPRAGEngine(); engine.initialize(); engine.load_or_create_vectorstore(force_rebuild=True)"
```

Questo comando:
- Scarica il modello di embedding (prima volta: ~90MB)
- Processa i documenti in chunks
- Crea embeddings vettoriali
- Salva il database FAISS in `data/vector_db/`

## Avviare il Servizio

```bash
cd ml-service
venv\Scripts\activate
python api.py
```

O usa lo shortcut:
```bash
start_service.bat
```

Il servizio sarà disponibile su:
- API: http://localhost:8000
- Documentazione interattiva: http://localhost:8000/docs

## Utilizzo

### 1. Verificare lo Stato

```bash
GET http://localhost:8000/health
```

Risposta:
```json
{
  "status": "healthy",
  "lstm_model_loaded": true,
  "device": "cpu",
  "rag_available": true,
  "rag_status": {
    "initialized": true,
    "vectorstore_loaded": true,
    "llm_available": true,
    "api_key_configured": true
  }
}
```

### 2. Generare Suggerimenti

```bash
POST http://localhost:8000/suggest
Content-Type: application/json

{
  "field_type": "executiveSummary",
  "partial_text": "This BEP establishes",
  "max_length": 300
}
```

Risposta:
```json
{
  "text": "a comprehensive framework for information management...",
  "prompt_used": "This BEP establishes",
  "method": "rag",
  "sources": [
    {
      "source": "BEP_Template_v1.0.txt",
      "content": "The BIM Execution Plan establishes..."
    }
  ]
}
```

### 3. Nell'interfaccia BEP Generator

1. Apri un campo di testo nel BEP
2. Inizia a scrivere (opzionale)
3. Clicca l'icona AI (✨)
4. Il sistema RAG genera testo contestuale basato sui tuoi documenti

## Architettura del Sistema

```
┌─────────────────────────────────────────┐
│   React Frontend                         │
│   - Editor con pulsante AI              │
└──────────────────┬──────────────────────┘
                   │ HTTP
                   ↓
┌─────────────────────────────────────────┐
│   Node.js Backend (Port 3001)           │
│   - Proxy per ML service                │
└──────────────────┬──────────────────────┘
                   │ HTTP
                   ↓
┌─────────────────────────────────────────┐
│   FastAPI ML Service (Port 8000)        │
│   ┌───────────────────────────────────┐ │
│   │  RAG Engine (Primary)             │ │
│   │  - FAISS Vector Search            │ │
│   │  - Claude API Generation          │ │
│   └───────────────┬───────────────────┘ │
│                   │ Fallback            │
│   ┌───────────────▼───────────────────┐ │
│   │  LSTM Model (Backup)              │ │
│   │  - Local generation               │ │
│   │  - No API required                │ │
│   └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## Struttura File

```
ml-service/
├── rag_engine.py              # Motore RAG principale
├── model_loader.py            # LSTM model (fallback)
├── api.py                     # FastAPI con logica di fallback
├── requirements.txt           # Dipendenze (aggiornato con RAG)
├── setup_rag.bat              # Script di setup automatico
├── .env.example               # Template configurazione
│
├── scripts/
│   └── extract_docx.py        # Estrazione testo da DOCX
│
├── data/
│   ├── training_documents/
│   │   ├── docx/              # I tuoi file DOCX
│   │   └── txt/               # Testo estratto
│   │
│   ├── vector_db/             # Database FAISS
│   │   ├── index.faiss
│   │   └── index.pkl
│   │
│   └── consolidated_training_data.txt
│
└── models/
    ├── bep_model.pth          # Modello LSTM (fallback)
    └── char_mappings.json
```

## Funzionalità Avanzate

### Ricostruire il Vector Database

Se aggiungi nuovi documenti DOCX:

```bash
# 1. Estrai testo
python scripts/extract_docx.py

# 2. Ricostruisci database
python -c "from rag_engine import BEPRAGEngine; engine = BEPRAGEngine(); engine.initialize(); engine.load_or_create_vectorstore(force_rebuild=True)"
```

### Configurare Retrieval

Nel file `rag_engine.py`, modifica questi parametri:

```python
# Numero di chunks da recuperare (default: 3)
result = rag.generate_suggestion(k=5)

# Dimensione chunks (default: 1000 caratteri)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Più grande = più contesto
    chunk_overlap=300  # Overlap tra chunks
)
```

### Cambiare Modello Claude

Nel file `rag_engine.py`:

```python
self.llm = ChatAnthropic(
    model="claude-3-opus-20240229",  # Opus per qualità massima
    # o "claude-3-haiku-20240307"     # Haiku per velocità
    temperature=0.7,  # 0.0-1.0, più alto = più creativo
    max_tokens=2048   # Lunghezza massima risposta
)
```

### Usare Modelli di Embedding Diversi

Nel file `rag_engine.py`:

```python
# Modello più grande e accurato (ma più lento)
self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Modello multilingue
self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

## Costi API

Con Claude API:

- **Claude 3.5 Sonnet**: ~$3 per 1M token input, ~$15 per 1M token output
- **Uso tipico**: ~500-1000 token per richiesta
- **Costo stimato**: $0.01-0.03 per generazione

Per ridurre i costi:
1. Usa chunk size più piccolo (meno context)
2. Riduci `max_tokens` nelle risposte
3. Usa Claude Haiku invece di Sonnet ($0.25/$1.25 per 1M token)

## Troubleshooting

### Errore: "Model file not found"

Il modello LSTM non è stato trainato. Non è necessario se usi solo RAG, ma per abilitare il fallback:

```bash
python scripts/train_model.py --epochs 100
```

### Errore: "ANTHROPIC_API_KEY not configured"

Configura la chiave API:

```cmd
setx ANTHROPIC_API_KEY "sk-ant-..."
```

Oppure crea file `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### Errore: "No documents found"

Assicurati di avere file .docx in:
```
ml-service/data/training_documents/docx/
```

Poi esegui:
```bash
python scripts/extract_docx.py
```

### RAG risponde sempre "LSTM fallback"

Controlla:
1. API key configurata correttamente
2. Vector database creato (`data/vector_db/` esiste)
3. Servizio avviato correttamente (guarda i log)

Verifica con:
```bash
GET http://localhost:8000/health
```

### Performance lente

1. **Prima generazione**: Il download del modello di embedding può richiedere tempo
2. **Vector database**: Ricostruisci solo quando necessario
3. **Chunk size**: Riduci se le risposte sono lente

### Qualità bassa

1. **Aggiungi più documenti**: Almeno 5-10 DOCX per risultati ottimali
2. **Usa documenti di qualità**: BEP completi e ben scritti
3. **Aumenta k**: Recupera più chunks (es. k=5)
4. **Usa Opus**: Modello più potente (ma più costoso)

## Best Practices

### Documenti di Training

- **Quantità**: Almeno 3-5 BEP completi
- **Qualità**: Documenti ISO 19650 compliant
- **Varietà**: Diversi tipi di progetto per generalizzazione
- **Formato**: DOCX ben formattati (evita scansioni PDF)

### Ottimizzazione

- **Cache vector DB**: Non ricostruire ad ogni avvio
- **Batch requests**: Se generi molti campi, considera batch
- **Monitor costs**: Tieni traccia dell'uso API

### Sicurezza

- **Non committare** `.env` con API key
- **Usa variabili ambiente** in produzione
- **Limita rate**: Implementa rate limiting se necessario

## Prossimi Passi

1. **Testa il sistema**: Prova diverse richieste
2. **Aggiungi documenti**: Più documenti = migliore qualità
3. **Personalizza prompts**: Modifica `field_contexts` in `rag_engine.py`
4. **Monitora performance**: Usa `/health` endpoint
5. **Feedback iterativo**: Migliora i documenti basandoti sui risultati

## Supporto

Per problemi o domande:

1. Controlla questa guida
2. Verifica i log del servizio
3. Controlla `/health` endpoint
4. Apri issue su GitHub

## Riferimenti

- [Anthropic Claude API](https://docs.anthropic.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [ISO 19650 Standards](https://www.iso.org/standard/68078.html)
