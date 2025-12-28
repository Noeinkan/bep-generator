"""
Ollama-based Text Generator for BEP Content

Uses Ollama's local LLM API to generate high-quality BEP content.
Replaces the PyTorch LSTM model with a modern, faster, and more accurate solution.
"""

import requests
import json
import logging
from typing import Optional, Dict
from pathlib import Path
from load_help_content import load_field_prompts_from_help_content

logger = logging.getLogger(__name__)


class OllamaGenerator:
    """Text generator using Ollama local LLM"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        """
        Initialize Ollama generator

        Args:
            base_url: Ollama API base URL
            model: Model name to use (e.g., 'llama3.2:3b', 'llama3.2:1b', 'mistral:7b')
        """
        self.base_url = base_url
        self.model = model
        self.timeout = 60  # seconds

        # Load field-specific system prompts from helpContentData.js
        # This provides a single source of truth for AI prompts across the application
        logger.info("Loading field prompts from helpContentData.js...")
        self.field_prompts = load_field_prompts_from_help_content()

        # Default fallback for fields without aiPrompt in helpContentData.js
        self.default_prompt = {
            'system': 'You are a BIM Execution Plan (BEP) expert following ISO 19650 standards.',
            'context': 'Provide professional BIM documentation content following industry best practices and ISO 19650 information management principles.'
        }

        # Verify Ollama connection on initialization
        self._verify_connection()

    def _verify_connection(self):
        """Verify that Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.warning(f"Ollama API returned status {response.status_code}")
                return

            # Check if model is available
            data = response.json()
            models = [m.get('name', '') for m in data.get('models', [])]

            if self.model not in models:
                logger.warning(f"Model '{self.model}' not found in Ollama")
                logger.warning(f"Available models: {', '.join(models)}")
                logger.warning(f"Please run: ollama pull {self.model}")
            else:
                logger.info(f"Ollama connection verified. Using model: {self.model}")

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Please ensure Ollama is running.")
            logger.error("Start Ollama with: ollama serve")
        except Exception as e:
            logger.error(f"Error verifying Ollama connection: {e}")

    def generate_text(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """
        Generate text based on a prompt

        Args:
            prompt: Starting text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (0.1-2.0)

        Returns:
            Generated text
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_length,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                generated = data.get('response', '').strip()
                return generated
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return "Error: Unable to generate text. Please check Ollama service."

        except requests.exceptions.Timeout:
            logger.error("Ollama request timeout")
            return "Error: Request timeout. Try a smaller model or increase timeout."
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error: {str(e)}"

    def suggest_for_field(self, field_type: str, partial_text: str = '', max_length: int = 200) -> str:
        """
        Generate field-specific suggestion

        Args:
            field_type: Type of BEP field
            partial_text: Existing text in the field
            max_length: Maximum characters to generate

        Returns:
            Generated suggestion
        """
        # Get field-specific prompt from helpContentData.js, or use default
        field_config = self.field_prompts.get(field_type, self.default_prompt)
        system_prompt = field_config.get('system', 'You are a BIM expert.')
        context = field_config.get('context', 'Provide professional BIM content.')

        # Build the prompt
        if partial_text and len(partial_text.strip()) > 10:
            # User has typed enough, continue their text
            prompt = f"{context}\n\nContinue this text professionally:\n{partial_text}"
        else:
            # No user text or very little, generate from scratch
            prompt = f"{context}\n\nGenerate professional content for this section."

        # Generate with lower temperature for more coherent, professional output
        generated = self.generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=0.5  # Lower temperature for BEP content
        )

        # Clean up the suggestion
        suggestion = self._clean_suggestion(generated, partial_text)

        return suggestion

    def _clean_suggestion(self, text: str, partial_text: str = '') -> str:
        """Clean up generated text"""
        import re

        # Remove any prompt repetition
        if partial_text:
            # If the model repeated the partial text, remove it
            text = text.replace(partial_text, '', 1).strip()

        # Remove common AI artifacts
        text = re.sub(r'^(Sure|Here|Okay|Certainly)[,!.]?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(I\'ll|I will|Let me)\s+\w+\s+', '', text, flags=re.IGNORECASE)

        # Fix spacing
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,;:!?])([a-zA-Z])', r'\1 \2', text)  # Add space after punctuation

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        # Ensure reasonable length
        if len(text) < 10:
            text = "Please provide more context or try again."

        return text.strip()


# Global instance
_ollama_generator = None


def get_ollama_generator(model: str = "llama3.2:3b") -> OllamaGenerator:
    """Get or create the global Ollama generator instance"""
    global _ollama_generator
    if _ollama_generator is None:
        _ollama_generator = OllamaGenerator(model=model)
    return _ollama_generator


# Alias for backward compatibility
get_generator = get_ollama_generator
