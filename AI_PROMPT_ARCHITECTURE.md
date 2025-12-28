# AI Prompt Architecture - Single Source of Truth

## Overview

Il sistema di prompts AI per il BEP Generator è stato migrato da **hard-coded prompts in Python** a un'architettura **Single Source of Truth basata su JavaScript** (`helpContentData.js`).

## Architettura

### Prima (Problema)
- **63 prompts hard-coded** in `ml-service/ollama_generator.py`
- Duplicazione di informazioni tra `helpContentData.js` e `ollama_generator.py`
- Difficile da mantenere: modifiche richiedevano aggiornamenti in 2 file
- Possibilità di disallineamento tra frontend guidance e AI prompts

### Dopo (Soluzione)
- **Single source of truth**: `src/data/helpContentData.js`
- Python carica dinamicamente i prompts da JavaScript al runtime
- Aggiornamenti centralizzati: modifichi solo `helpContentData.js`
- Garantita coerenza tra Field Guidance Module e AI suggestions

## Struttura `aiPrompt` in helpContentData.js

Ogni campo in `helpContentData.js` ora include una sezione `aiPrompt`:

```javascript
modelValidation: {
  description: `...`,
  iso19650: `...`,
  bestPractices: [...],
  examples: {
    'Commercial Building': `...`,
    'Infrastructure': `...`
  },
  commonMistakes: [...],

  // AI Prompt Configuration
  aiPrompt: {
    system: 'You are a BIM model validation expert. Generate concise, practical validation procedures using checklist format.',
    instructions: 'Generate content similar to the examples above. Use checklist format (☑) with specific validation tools (e.g., Solibri Model Checker), quantifiable metrics (e.g., <50 clashes, ±5mm tolerance), and actionable items. Keep it practical and structured. Maximum 150 words.',
    style: 'checklist-based, specific tools mentioned, quantifiable metrics, structured categories'
  },

  relatedFields: [...]
}
```

### Componenti aiPrompt

1. **`system`**: Ruolo dell'AI expert (es: "You are a BIM model validation expert...")
2. **`instructions`**: Istruzioni specifiche per il formato e contenuto (riferimento agli examples)
3. **`style`**: Parole chiave che descrivono lo stile desiderato

## Files Coinvolti

### 1. `src/data/helpContentData.js`
- **Single source of truth** per field guidance e AI prompts
- Contiene 20+ campi con `aiPrompt` configuration
- Facilmente estendibile: aggiungi nuovi campi seguendo il template

### 2. `ml-service/load_help_content.py`
- Utility per caricare aiPrompt da JavaScript
- Converte da JavaScript object a Python dict
- Funzione principale: `load_field_prompts_from_help_content()`

```python
from load_help_content import load_field_prompts_from_help_content

# Carica tutti i prompts
prompts = load_field_prompts_from_help_content()
# Returns: {'modelValidation': {'system': '...', 'context': '...', 'style': '...'}, ...}
```

### 3. `ml-service/ollama_generator.py`
- Carica prompts dinamicamente da `helpContentData.js` all'inizializzazione
- Fallback a prompts hard-coded per campi non in helpContentData.js
- Uso: trasparente, nessun cambiamento nell'API

```python
class OllamaGenerator:
    def __init__(self):
        # Caricamento dinamico da helpContentData.js
        self.field_prompts = load_field_prompts_from_help_content()

        # Fallback per campi non trovati
        self._fallback_prompts = {...}
```

### 4. `ml-service/add_ai_prompts.py`
- Script di utilità per aggiungere aiPrompt a campi mancanti
- Mappa automaticamente 69 campi textarea con configurazioni appropriate
- Esegui: `python add_ai_prompts.py`

## Come Aggiungere/Modificare Prompts

### Opzione 1: Modifica Manuale (Raccomandato)

1. Apri `src/data/helpContentData.js`
2. Trova il campo (es: `modelValidation`)
3. Aggiungi/modifica la sezione `aiPrompt` dopo `commonMistakes`:

```javascript
    aiPrompt: {
      system: 'You are a [RUOLO ESPERTO]. [AZIONE].',
      instructions: 'Generate content similar to the examples above. [FORMATO SPECIFICO]. Maximum 150 words.',
      style: '[CARATTERISTICHE STILE]'
    },
```

### Opzione 2: Script Automatico

```bash
cd ml-service
python add_ai_prompts.py
```

Questo script:
- Legge `helpContentData.js`
- Identifica campi senza `aiPrompt`
- Genera configurazioni appropriate basandosi su field metadata
- Scrive le modifiche al file

## Testing

### Test del Loader

```bash
cd ml-service
python load_help_content.py
```

Output atteso:
```
Loaded 20 field prompts from helpContentData.js

Loaded field prompts:
  - modelValidation
  - reviewProcesses
  - approvalWorkflows
  ...

Total: 20 fields
```

### Test dell'AI Generator

Il modo migliore è testare direttamente nell'applicazione:

1. Avvia il backend: `npm run dev` (terminal 1)
2. Avvia Ollama service (se non già running)
3. Apri il BEP Generator nel browser
4. Clicca sull'AI Suggestion button per un campo con `aiPrompt`
5. Verifica che la suggestion sia:
   - Nello stile corretto (es: checklist per modelValidation)
   - Con tool specifici menzionati
   - Concisa (~150 words)
   - Simile agli examples in helpContentData.js

## Copertura Campi

### Campi con aiPrompt (20 in helpContentData.js)

- projectDescription ✅
- bimStrategy ✅
- valueProposition ✅
- bimGoals ✅
- primaryObjectives ✅
- reviewProcesses ✅
- approvalWorkflows ✅
- complianceVerification ✅
- modelReviewAuthorisation ✅
- dataTransferProtocols ✅
- encryptionRequirements ✅
- privacyConsiderations ✅
- ... (20 total)

### Campi con Fallback (49 in ollama_generator.py)

Per campi non presenti in `helpContentData.js`, il sistema usa prompts fallback hard-coded in `ollama_generator.py`. Questi possono essere progressivamente migrati a `helpContentData.js` quando necessario.

## Vantaggi dell'Architettura

### ✅ Single Source of Truth
- Un solo file da aggiornare (`helpContentData.js`)
- Nessuna duplicazione
- Garantita coerenza

### ✅ Maintainability
- Facile trovare e modificare prompts
- Template consistente
- Documentazione inline (examples, bestPractices)

### ✅ Flexibility
- Aggiungi/modifica senza rebuild Python
- Hot-reload possibile (riavvia solo API service)
- Examples e prompts nello stesso posto

### ✅ Quality
- AI genera contenuto simile agli examples
- Riferimenti a tool specifici (Solibri, Navisworks, etc.)
- Metriche quantificate (±5mm, <50 clashes, etc.)

## Troubleshooting

### Prompt non caricato

**Problema**: AI genera contenuto generico invece di field-specific

**Soluzione**:
1. Verifica che `aiPrompt` esista in helpContentData.js:
   ```bash
   grep -A 5 "fieldName:" src/data/helpContentData.js | grep aiPrompt
   ```
2. Verifica il loader:
   ```bash
   cd ml-service && python load_help_content.py
   ```
3. Riavvia l'API service

### JavaScript syntax error

**Problema**: Errore nel parsing di helpContentData.js

**Soluzione**:
- Verifica sintassi JavaScript (virgole, apici, parentesi)
- Usa un linter: `npx eslint src/data/helpContentData.js`

### Campo non trovato

**Problema**: Script dice "Field X not found"

**Soluzione**:
- Il campo potrebbe non esistere in helpContentData.js
- Verifica nome corretto: `grep "fieldName:" src/data/helpContentData.js`
- Alcuni campi sono solo in bepConfig.js (tables, checkboxes) - non servono AI prompts

## Roadmap Futuro

### Miglioramenti Pianificati

1. **Project-Type Specific Prompts**
   - Diversi prompts per Commercial/Infrastructure/Healthcare
   - Basati sugli examples esistenti in helpContentData.js

2. **Dynamic Example Injection**
   - Iniettare examples da helpContentData.js nel prompt AI
   - AI genera contenuto ancora più simile agli examples

3. **AI Prompt Analytics**
   - Tracciare quali campi usano AI suggestions
   - A/B testing di diversi prompt strategies
   - User feedback su quality delle suggestions

4. **Prompt Versioning**
   - Git history già traccia modifiche
   - Considerare versioning esplicito per rollback rapidi

## Conclusione

L'architettura Single Source of Truth centralizza la gestione dei prompts AI, migliorando maintainability, consistency, e quality delle AI suggestions nel BEP Generator.

Per domande o assistenza: consulta questo documento o il codice sorgente in:
- `src/data/helpContentData.js`
- `ml-service/load_help_content.py`
- `ml-service/ollama_generator.py`
