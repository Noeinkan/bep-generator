"""
Ollama-based Model Loader
Drop-in replacement for model_loader.py using Ollama instead of char-RNN
Same interface, zero changes needed in api.py
"""

import requests
import logging
from typing import Dict
from pathlib import Path
from config import OLLAMA_MODEL, OLLAMA_URL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)


class BEPTextGenerator:
    """Text generator using Ollama - same interface as char-RNN version"""

    def __init__(self, models_dir='models'):
        """Initialize (models_dir ignored for Ollama, kept for compatibility)"""
        self.device = "ollama"  # For compatibility with health checks
        self.model = OLLAMA_MODEL
        self.base_url = OLLAMA_URL
        self.timeout = OLLAMA_TIMEOUT

        # Field-specific prompts (same as original)
        self.field_prompts = {
            'projectName': 'the project is called ',
            'projectDescription': 'this project involves the development of a comprehensive building. the scope includes ',
            'executiveSummary': 'this bep establishes the framework for information management. the plan defines ',
            'projectObjectives': 'the primary objectives are to deliver high quality. key objectives include ',
            'bimObjectives': 'the bim objectives are to implement iso standards. this includes ',
            'projectScope': 'the project scope encompasses design and construction. the scope includes ',
            'stakeholders': 'key stakeholders include the client and design team. stakeholders are ',
            'rolesResponsibilities': 'the information manager is responsible for coordination. responsibilities include ',
            'deliveryTeam': 'the delivery team comprises professionals. the team includes ',
            'collaborationProcedures': 'collaboration procedures ensure coordination. these procedures include ',
            'informationExchange': 'information exchange requires structured handover. exchange requirements include ',
            'cdeWorkflow': 'the cde workflow consists of containers. the workflow ensures ',
            'modelRequirements': 'model requirements specify information need. requirements include ',
            'dataStandards': 'data standards require compliance with iso. standards include ',
            'namingConventions': 'file naming shall follow the standard. the naming format includes ',
            'qualityAssurance': 'quality assurance includes validation checks. procedures ensure ',
            'validationChecks': 'validation checks comprise clash detection. checks include ',
            'technologyStandards': 'technology standards specify software. standards require ',
            'softwarePlatforms': 'the team shall utilize bim tools. platforms include ',
            'coordinationProcess': 'coordination meetings occur weekly. the process includes ',
            'clashDetection': 'clash detection requires model federation. detection procedures include ',
            'healthSafety': 'health and safety information shall be documented. information includes ',
            'handoverRequirements': 'handover deliverables include models. requirements specify ',
            'asbuiltRequirements': 'as built models represent construction. models shall include ',
            'cobieRequirements': 'cobie data shall be populated. requirements include ',
            'default': 'the project requirements specify that '
        }

        self._verify_connection()

    def _verify_connection(self):
        """Verify Ollama connection"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Connected to Ollama at {self.base_url}")
                logger.info(f"Using model: {self.model}")
            else:
                logger.warning(f"Ollama returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            logger.error("Make sure Ollama is running: ollama serve")

    def generate_text(self, prompt, max_length=200, temperature=0.8):
        """
        Generate text continuation based on a prompt
        Same signature as char-RNN version
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
                return prompt + " " + generated  # Return full text like char-RNN
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return prompt + " [Error: Cannot generate text]"

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return prompt + " [Error: " + str(e) + "]"

    def suggest_for_field(self, field_type, partial_text='', max_length=200):
        """
        Generate suggestion for a specific BEP field
        Same signature as char-RNN version
        """
        # Get field-specific prompt
        base_prompt = self.field_prompts.get(field_type, self.field_prompts['default'])

        # Build prompt (same logic as char-RNN version)
        if partial_text and len(partial_text.strip()) > 3:
            prompt = partial_text
            prompt_prefix_len = len(partial_text)
        elif partial_text and len(partial_text.strip()) > 0:
            prompt = base_prompt + partial_text
            prompt_prefix_len = len(partial_text)
        else:
            prompt = base_prompt
            prompt_prefix_len = 0

        if not prompt or len(prompt.strip()) == 0:
            prompt = 'the project '
            prompt_prefix_len = 0

        # Generate with Ollama
        try:
            payload = {
                "model": self.model,
                "prompt": f"Continue this BIM Execution Plan text professionally: {prompt}",
                "stream": False,
                "options": {
                    "temperature": 0.5,  # Lower for more coherent output
                    "num_predict": max_length,
                    "top_p": 0.9
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                full_text = data.get('response', '').strip()

                # Extract only generated part
                if prompt_prefix_len > 0:
                    suggestion = full_text
                else:
                    suggestion = full_text

                # Clean up
                suggestion = self._clean_suggestion(suggestion)
                return suggestion
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return "Error: Cannot generate suggestion"

        except Exception as e:
            logger.error(f"Suggestion error: {e}")
            return f"Error: {str(e)}"

    def _clean_suggestion(self, text):
        """Clean up generated text (same as char-RNN version)"""
        import re

        # Remove AI artifacts
        text = re.sub(r'^(Sure|Here|Okay|Certainly)[,!.]?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(I\'ll|I will|Let me)\s+\w+\s+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^Continue this.*?:\s*', '', text, flags=re.IGNORECASE)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Fix spacing
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])([a-zA-Z])', r'\1 \2', text)

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        # Ensure reasonable length
        if len(text) < 10:
            text = "Please try again or provide more context."

        return text


# Global instance (same as char-RNN version)
_generator = None


def get_generator():
    """Get or create the global generator instance (same interface as char-RNN)"""
    global _generator
    if _generator is None:
        _generator = BEPTextGenerator()
    return _generator
