# Quick Start - RAG System

Guida rapida per iniziare con il sistema RAG in 5 minuti.

## Step 1: Prepara i documenti

Copia i tuoi file BEP (*.docx) nella cartella:
```
ml-service/data/training_documents/docx/
```

**Nota**: Hai già 10 documenti DOCX in questa cartella!

## Step 2: Ottieni l'API Key

1. Vai su [Anthropic Console](https://console.anthropic.com/)
2. Crea un account (se necessario)
3. Genera una nuova API key
4. Copia la chiave (inizia con `sk-ant-...`)

## Step 3: Esegui Setup

Apri il terminale e esegui:

```bash
cd ml-service
setup_rag.bat
```

Durante il setup ti verrà chiesto di configurare l'API key.

## Step 4: Configura API Key (SICURO - non verrà caricato su GitHub!)

**Metodo Consigliato - File .env** ✅ Protetto da Git:

Lo script `setup_rag.bat` ti guiderà automaticamente, oppure manualmente:

1. Vai in `ml-service/` e copia il template:
   ```cmd
   cd ml-service
   copy .env.example .env
   ```

2. Apri `ml-service\.env` con un editor di testo

3. Sostituisci `your-api-key-here` con la tua vera API key:
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-tua-chiave-qui
   ```

4. Salva e chiudi

**🔒 SICUREZZA**: Il file `.env` è già in `.gitignore` e NON verrà mai caricato su GitHub!

Per maggiori dettagli: [SETUP_API_KEY.md](ml-service/SETUP_API_KEY.md)

## Step 5: Avvia il servizio

```bash
cd ml-service
start_service.bat
```

Il servizio si avvierà su http://localhost:8000

## Step 6: Avvia il BEP Generator

In un altro terminale:

```bash
npm start
```

L'applicazione si aprirà su http://localhost:3000

## Step 7: Usa l'AI

1. Crea o apri un BEP
2. Clicca su un campo di testo
3. Clicca l'icona AI (✨) nella toolbar
4. Il sistema genererà testo basato sui tuoi documenti!

## Verifica

Per verificare che tutto funzioni:

```bash
curl http://localhost:8000/health
```

Dovresti vedere:
```json
{
  "status": "healthy",
  "rag_available": true,
  "lstm_model_loaded": true
}
```

## Problemi?

### "ANTHROPIC_API_KEY not configured"

Assicurati di aver configurato la chiave API (Step 4) e riavviato il servizio.

### "No documents found"

Verifica che ci siano file .docx in `ml-service/data/training_documents/docx/`

### "Port 8000 already in use"

Chiudi altri servizi sulla porta 8000:
```cmd
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

## Documentazione Completa

Per informazioni dettagliate, vedi:
- [RAG_SETUP_GUIDE.md](RAG_SETUP_GUIDE.md) - Setup completo
- [ml-service/README.md](ml-service/README.md) - API reference

## Costi

Con i 10 documenti forniti:
- **Setup iniziale**: Gratuito (usa modelli open-source per embedding)
- **Generazione testo**: ~$0.01-0.03 per richiesta (Claude API)
- **Stima mensile**: $5-20 per uso normale (~200-500 generazioni)

## Supporto

Se hai problemi, controlla:
1. Log del servizio ML
2. `/health` endpoint
3. [RAG_SETUP_GUIDE.md](RAG_SETUP_GUIDE.md) - Troubleshooting section
