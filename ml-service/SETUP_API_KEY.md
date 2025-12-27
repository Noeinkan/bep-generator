# Come Configurare l'API Key in Modo Sicuro

Questa guida ti mostra come configurare la tua Anthropic API key in modo sicuro, **senza rischiare di caricarla pubblicamente su GitHub**.

## ✅ Metodo Sicuro: File `.env` (RACCOMANDATO)

### Passo 1: Ottieni la tua API Key

1. Vai su [Anthropic Console](https://console.anthropic.com/)
2. Crea un account o effettua il login
3. Vai su **API Keys** nel menu
4. Clicca **Create Key**
5. **Copia la chiave** (inizia con `sk-ant-api03-...`)

### Passo 2: Crea il file `.env`

Nella cartella `ml-service`, crea un file chiamato `.env`:

```bash
cd ml-service
copy .env.example .env
```

### Passo 3: Modifica il file `.env`

Apri `ml-service\.env` con un editor di testo e inserisci la tua chiave:

```
ANTHROPIC_API_KEY=sk-ant-api03-la-tua-chiave-qui
```

**Esempio:**
```
ANTHROPIC_API_KEY=sk-ant-api03-xyz123abc456def789...
```

### Passo 4: Salva e chiudi

Salva il file. **Non condividere mai questo file con nessuno!**

### Passo 5: Verifica che sia protetto

Il file `.env` è già incluso in `.gitignore`, quindi **NON verrà mai caricato su GitHub**.

Verifica:
```bash
git status
```

Se vedi `.env` nella lista, aggiungi questa riga al `.gitignore`:
```
ml-service/.env
```

## 🔒 Sicurezza Garantita

Il file `.env` è protetto da Git in questi modi:

1. **`.gitignore` principale** include:
   ```
   .env
   ml-service/.env
   ```

2. **NON committare mai** il file `.env`:
   ```bash
   # Questo file NON apparirà mai in git status
   git add .
   git commit -m "update"
   # .env non verrà incluso
   ```

3. **Template pubblico** (`.env.example`) mostra la struttura senza rivelare la chiave:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

## 🚀 Come Usare

### Installare dipendenze (include `python-dotenv`)

```bash
cd ml-service
venv\Scripts\activate
pip install -r requirements.txt
```

### Avviare il servizio

```bash
python api.py
```

Il servizio caricherà automaticamente l'API key da `.env`

### Verificare che funzioni

```bash
# In un altro terminale
curl http://localhost:8000/health
```

Dovresti vedere:
```json
{
  "rag_status": {
    "api_key_configured": true,
    "llm_available": true
  }
}
```

## ❌ Cosa NON Fare

### ❌ NON usare variabili di sistema permanenti
```bash
# Evita questo se lavori su progetti pubblici
setx ANTHROPIC_API_KEY "sk-ant-..."
```
**Motivo**: Rimane nel sistema anche dopo aver cancellato il progetto

### ❌ NON hardcodare la chiave nel codice
```python
# MAI FARE QUESTO!!!
api_key = "sk-ant-api03-xyz123..."
```
**Motivo**: Finirebbe su GitHub

### ❌ NON committare il file `.env`
```bash
# NON FARE MAI:
git add ml-service/.env
git commit -m "add api key"
```
**Motivo**: La chiave sarebbe pubblica per sempre (anche se cancelli il commit!)

## 🆘 Se Hai Già Caricato la Chiave per Errore

1. **Revoca immediatamente** la chiave su [Anthropic Console](https://console.anthropic.com/)
2. **Genera una nuova chiave**
3. **Aggiorna** il file `.env` con la nuova chiave
4. **Non tentare** di cancellare il commit - la chiave rimane nella history di Git

## 🔄 Condividere il Progetto

Quando condividi il progetto con altri sviluppatori:

1. Condividi solo il file `.env.example`
2. Istruiscili a creare il proprio `.env`
3. Ogni sviluppatore usa la propria API key

## 📋 Checklist Sicurezza

Prima di fare `git push`, verifica:

- [ ] File `.env` esiste in `ml-service/.env`
- [ ] File `.env` contiene la tua API key
- [ ] File `.env` NON appare in `git status`
- [ ] File `.gitignore` include `.env` e `ml-service/.env`
- [ ] File `.env.example` NON contiene chiavi reali
- [ ] Hai testato che il servizio funzioni

## 🎯 Riassunto Veloce

```bash
# 1. Crea .env
cd ml-service
copy .env.example .env

# 2. Modifica .env e inserisci:
#    ANTHROPIC_API_KEY=sk-ant-api03-tua-chiave

# 3. Installa dipendenze
venv\Scripts\activate
pip install -r requirements.txt

# 4. Testa
python api.py

# 5. Verifica (altro terminale)
curl http://localhost:8000/health
```

## 💡 Perché Questo Metodo è Sicuro

1. ✅ **Locale**: La chiave resta solo sul tuo computer
2. ✅ **Ignorato da Git**: `.gitignore` impedisce il commit
3. ✅ **Standard**: Usato da migliaia di progetti
4. ✅ **Facile**: Basta modificare un file di testo
5. ✅ **Flessibile**: Ogni sviluppatore ha la propria chiave
6. ✅ **Documentato**: `.env.example` mostra cosa serve

## ❓ Domande Frequenti

**Q: Posso vedere se il file .env è protetto?**
A: Sì, esegui `git status`. Se `.env` non appare, è protetto.

**Q: Cosa faccio se cambio computer?**
A: Copia il file `.env` sul nuovo computer o crea una nuova API key.

**Q: Posso usare la stessa chiave in sviluppo e produzione?**
A: Meglio usare chiavi diverse. Crea una chiave per sviluppo e una per produzione.

**Q: Il file .env funziona anche su Linux/Mac?**
A: Sì, il sistema è identico su tutti i sistemi operativi.

---

**✨ Ora sei pronto! La tua API key sarà sempre privata e sicura.**
