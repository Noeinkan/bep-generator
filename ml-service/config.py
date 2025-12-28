"""
ML Service Configuration
Select AI model backend: "ollama" or "char-rnn"
"""

# Model Selection
MODEL_TYPE = "ollama"  # Change to "char-rnn" for fallback to PyTorch LSTM

# Ollama Configuration
OLLAMA_MODEL = "llama3.2:3b"  # Options: "llama3.2:1b", "llama3.2:3b", "mistral:7b"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 60  # seconds

# Char-RNN Configuration (fallback)
CHAR_RNN_MODELS_DIR = "models"
