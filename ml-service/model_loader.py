"""
Model Loader and Text Generator

Handles loading the trained LSTM model and generating text suggestions
for BEP fields.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path


class CharLSTM(nn.Module):
    """Character-level LSTM language model - same architecture as training"""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))


class BEPTextGenerator:
    """Text generator for BEP content"""

    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.char_to_int = None
        self.int_to_char = None
        self.n_vocab = None

        # Field-specific prompts to guide generation
        self.field_prompts = {
            'projectName': 'Project Name: ',
            'projectDescription': 'This project involves ',
            'executiveSummary': 'The BEP establishes ',
            'projectObjectives': 'The primary objectives include ',
            'bimObjectives': 'The BIM objectives for this project are to ',
            'projectScope': 'The project scope encompasses ',
            'stakeholders': 'Key stakeholders include ',
            'rolesResponsibilities': 'The Information Manager is responsible for ',
            'deliveryTeam': 'The Delivery Team comprises ',
            'collaborationProcedures': 'Collaboration procedures include ',
            'informationExchange': 'Information exchange protocols require ',
            'cdeWorkflow': 'The Common Data Environment workflow consists of ',
            'modelRequirements': 'Model requirements specify that ',
            'dataStandards': 'Data standards require compliance with ',
            'namingConventions': 'File naming shall follow the format ',
            'qualityAssurance': 'Quality assurance procedures include ',
            'validationChecks': 'Validation checks comprise ',
            'technologyStandards': 'Technology standards specify the use of ',
            'softwarePlatforms': 'The project team shall utilize ',
            'coordinationProcess': 'Coordination meetings shall occur ',
            'clashDetection': 'Clash detection procedures require ',
            'healthSafety': 'Health and safety information shall be ',
            'handoverRequirements': 'Handover deliverables include ',
            'asbuiltRequirements': 'As-Built models must accurately represent ',
            'cobieRequirements': 'COBie data shall be progressively populated ',
            'default': ''
        }

        self.load_model()

    def load_model(self):
        """Load trained model and character mappings"""
        model_path = self.models_dir / 'bep_model.pth'
        mappings_path = self.models_dir / 'char_mappings.json'

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Please train the model first using: python scripts/train_model.py"
            )

        # Load character mappings
        with open(mappings_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
            self.char_to_int = mappings['char_to_int']
            self.int_to_char = {int(k): v for k, v in mappings['int_to_char'].items()}
            self.n_vocab = mappings['n_vocab']

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = CharLSTM(
            input_size=1,
            hidden_size=checkpoint['hidden_size'],
            output_size=self.n_vocab,
            num_layers=checkpoint['num_layers']
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

    def generate_text(self, prompt, max_length=200, temperature=0.8):
        """
        Generate text continuation based on a prompt

        Args:
            prompt: Starting text to continue from
            max_length: Maximum number of characters to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            Generated text string
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self.model.eval()
        with torch.no_grad():
            # Encode prompt
            inputs = [self.char_to_int.get(ch, 0) for ch in prompt]
            inputs = torch.tensor(inputs, dtype=torch.float32).reshape(1, len(inputs), 1)
            inputs = inputs / float(self.n_vocab)
            inputs = inputs.to(self.device)

            result = prompt
            hidden = None

            for _ in range(max_length):
                output, hidden = self.model(inputs, hidden)

                # Apply temperature
                output = output / temperature
                prob = torch.softmax(output, dim=1).cpu().detach().numpy()

                # Sample character
                char_idx = np.random.choice(range(self.n_vocab), p=prob[0])
                char = self.int_to_char[char_idx]
                result += char

                # Stop at sentence end for cleaner output
                if char == '.' and len(result) > len(prompt) + 50:
                    break

                # Prepare next input
                new_input = torch.tensor([[char_idx]], dtype=torch.float32).reshape(1, 1, 1)
                new_input = new_input / float(self.n_vocab)
                new_input = new_input.to(self.device)
                inputs = new_input

        return result

    def suggest_for_field(self, field_type, partial_text='', max_length=200):
        """
        Generate suggestion for a specific BEP field

        Args:
            field_type: Type of field (e.g., 'executiveSummary', 'projectObjectives')
            partial_text: Any existing text in the field
            max_length: Maximum characters to generate

        Returns:
            Suggested text completion
        """
        # Get field-specific prompt
        prompt = self.field_prompts.get(field_type, self.field_prompts['default'])

        # If user has already typed something, use that
        if partial_text and len(partial_text) > 3:
            prompt = partial_text
        elif partial_text:
            prompt = prompt + partial_text

        # Generate text
        full_text = self.generate_text(prompt, max_length=max_length, temperature=0.7)

        # Extract only the generated part (remove prompt)
        if partial_text:
            suggestion = full_text[len(partial_text):]
        else:
            suggestion = full_text[len(prompt):]

        # Clean up the suggestion
        suggestion = self._clean_suggestion(suggestion)

        return suggestion

    def _clean_suggestion(self, text):
        """Clean up generated text"""
        # Remove leading/trailing whitespace
        text = text.strip()

        # Try to end at a sentence boundary
        if '.' in text:
            sentences = text.split('.')
            # Keep complete sentences
            if len(sentences) > 1:
                text = '.'.join(sentences[:-1]) + '.'

        return text


# Global instance
_generator = None


def get_generator():
    """Get or create the global generator instance"""
    global _generator
    if _generator is None:
        models_dir = Path(__file__).parent / 'models'
        _generator = BEPTextGenerator(models_dir)
    return _generator
