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

        # Field-specific system prompts for better context
        self.field_prompts = {
            'projectName': {
                'system': 'You are a BIM project naming expert. Generate concise, professional project names.',
                'context': 'Generate a professional project name for a BIM construction project.'
            },
            'projectDescription': {
                'system': 'You are a BIM project documentation expert. Write clear, comprehensive project descriptions following ISO 19650 standards.',
                'context': 'Write a detailed project description for a Building Information Modeling (BIM) project, covering scope, objectives, and key deliverables.'
            },
            'executiveSummary': {
                'system': 'You are a BIM Execution Plan (BEP) expert. Write executive summaries following ISO 19650 standards.',
                'context': 'Write an executive summary for a BIM Execution Plan (BEP) that establishes the framework for information management throughout the project lifecycle.'
            },
            'projectObjectives': {
                'system': 'You are a BIM project objectives specialist. Define clear, measurable objectives aligned with ISO 19650.',
                'context': 'Define the primary objectives for this BIM project, focusing on quality, efficiency, collaboration, and compliance with industry standards.'
            },
            'bimObjectives': {
                'system': 'You are a BIM strategy expert. Define BIM-specific objectives following ISO 19650.',
                'context': 'Define the BIM objectives for this project, including implementation of ISO 19650 standards, digital collaboration, and information management goals.'
            },
            'projectScope': {
                'system': 'You are a BIM scope definition expert. Define comprehensive project scopes.',
                'context': 'Define the project scope including design, construction, and operational phases, as well as all BIM deliverables and milestones.'
            },
            'stakeholders': {
                'system': 'You are a stakeholder management expert in BIM projects.',
                'context': 'List and describe key stakeholders in this BIM project, including the client, design team, contractors, and facility managers.'
            },
            'rolesResponsibilities': {
                'system': 'You are a BIM roles and responsibilities specialist following ISO 19650.',
                'context': 'Define roles and responsibilities for the BIM project team, including information manager, lead appointed party, and task teams.'
            },
            'deliveryTeam': {
                'system': 'You are a BIM team structure expert.',
                'context': 'Describe the project delivery team composition, including key professionals and their expertise areas.'
            },
            'collaborationProcedures': {
                'system': 'You are a BIM collaboration expert following ISO 19650 CDE workflows.',
                'context': 'Define collaboration procedures for information exchange, ensuring effective coordination and communication throughout the project.'
            },
            'informationExchange': {
                'system': 'You are an information management expert following ISO 19650.',
                'context': 'Define information exchange requirements, including structured data handover protocols and delivery milestones.'
            },
            'cdeWorkflow': {
                'system': 'You are a Common Data Environment (CDE) workflow expert following ISO 19650.',
                'context': 'Describe the CDE workflow including information containers, status codes (WIP, Shared, Published, Archived), and approval processes.'
            },
            'modelRequirements': {
                'system': 'You are a BIM model requirements specialist following ISO 19650.',
                'context': 'Specify model requirements including level of information need, geometric accuracy, and data standards.'
            },
            'dataStandards': {
                'system': 'You are a BIM data standards expert.',
                'context': 'Define data standards requiring compliance with ISO 19650, IFC standards, and industry-specific classification systems.'
            },
            'namingConventions': {
                'system': 'You are a BIM naming conventions specialist following ISO 19650.',
                'context': 'Define file naming conventions following standardized formats including project code, originator, zone, level, type, role, and number.'
            },
            'qualityAssurance': {
                'system': 'You are a BIM quality assurance expert.',
                'context': 'Define quality assurance procedures including validation checks, model audits, and compliance verification processes.'
            },
            'validationChecks': {
                'system': 'You are a BIM validation specialist.',
                'context': 'Describe validation checks including clash detection, data verification, and standards compliance audits.'
            },
            'technologyStandards': {
                'system': 'You are a BIM technology standards expert.',
                'context': 'Specify technology standards including software requirements, file formats, and interoperability protocols.'
            },
            'softwarePlatforms': {
                'system': 'You are a BIM software specialist.',
                'context': 'List required BIM software platforms and tools, including authoring, coordination, and analysis applications.'
            },
            'coordinationProcess': {
                'system': 'You are a BIM coordination expert.',
                'context': 'Define the coordination process including regular meetings, model federation, and issue resolution workflows.'
            },
            'clashDetection': {
                'system': 'You are a clash detection specialist.',
                'context': 'Define clash detection procedures including model federation, automated checks, and resolution tracking.'
            },
            'healthSafety': {
                'system': 'You are a construction health and safety information management expert.',
                'context': 'Define health and safety information requirements to be documented in BIM models and shared with the project team.'
            },
            'handoverRequirements': {
                'system': 'You are a project handover specialist following ISO 19650.',
                'context': 'Specify handover deliverables including as-built models, documentation, and asset information for facility management.'
            },
            'asbuiltRequirements': {
                'system': 'You are an as-built documentation expert.',
                'context': 'Define as-built model requirements representing actual construction conditions with verified dimensions and installed systems.'
            },
            'cobieRequirements': {
                'system': 'You are a COBie (Construction Operations Building Information Exchange) expert.',
                'context': 'Define COBie data requirements for structured asset information handover, including spaces, equipment, and maintenance schedules.'
            },
            'default': {
                'system': 'You are a BIM Execution Plan (BEP) expert following ISO 19650 standards.',
                'context': 'Provide professional BIM documentation content following industry best practices.'
            }
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
        # Get field-specific prompt
        field_config = self.field_prompts.get(field_type, self.field_prompts['default'])
        system_prompt = field_config['system']
        context = field_config['context']

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
