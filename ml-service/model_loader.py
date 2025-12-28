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


class CharRNN(nn.Module):
    """Character-level RNN language model with embedding layer - MUST match training architecture"""

    def __init__(self, vocab_size, embed_dim, hidden_size, output_size,
                 num_layers=2, rnn_type='lstm', dropout=0.3):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.embed_dim = embed_dim

        # Embedding layer: learns vector representations for each character
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # RNN layer (LSTM or GRU)
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(embed_dim, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:  # Default to LSTM
            self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_length) - integer indices
        batch_size = x.size(0)

        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        # Pass through embedding layer
        # embedded shape: (batch_size, seq_length, embed_dim)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # Pass through RNN
        out, hidden = self.rnn(embedded, hidden)

        # Take output from last time step
        out = out[:, -1, :]
        out = self.dropout(out)

        # Pass through output layer
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        if self.rnn_type == 'gru':
            return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        else:  # LSTM
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))


# Keep backward compatibility
CharLSTM = CharRNN


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
        # Longer, more specific prompts produce better results with character-level models
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

        # Get architecture parameters from checkpoint
        hidden_size = checkpoint['hidden_size']
        num_layers = checkpoint['num_layers']
        embed_dim = checkpoint.get('embed_dim', 128)  # Default to 128 if not found
        rnn_type = checkpoint.get('rnn_type', 'lstm')  # Default to lstm if not found

        self.model = CharRNN(
            vocab_size=self.n_vocab,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            output_size=self.n_vocab,
            num_layers=num_layers,
            rnn_type=rnn_type,
            dropout=0.3
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")
        print(f"Architecture: {rnn_type.upper()} | Hidden: {hidden_size} | Embed: {embed_dim} | Layers: {num_layers}")

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

        # Ensure prompt is not empty
        if not prompt or len(prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty. Please provide a starting text.")

        self.model.eval()
        with torch.no_grad():
            # Encode prompt as integer indices
            inputs = [self.char_to_int.get(ch, 0) for ch in prompt]

            # Ensure we have at least one character
            if len(inputs) == 0:
                raise ValueError("Prompt resulted in empty sequence after encoding")

            inputs = torch.tensor(inputs, dtype=torch.long).unsqueeze(0)  # Shape: (1, seq_length)
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
                # Require at least 80 characters for a complete thought
                if char == '.' and len(result) > len(prompt) + 80:
                    break

                # Prepare next input (single character as integer index)
                new_input = torch.tensor([[char_idx]], dtype=torch.long).to(self.device)  # Shape: (1, 1)
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
        base_prompt = self.field_prompts.get(field_type, self.field_prompts['default'])

        # Ensure we always have a non-empty prompt
        # If no base prompt exists for this field type, use a generic one
        if not base_prompt or base_prompt.strip() == '':
            base_prompt = 'the '

        # Build the final prompt
        if partial_text and len(partial_text.strip()) > 3:
            # User has typed enough, use their text as prompt
            prompt = partial_text
            prompt_prefix_len = len(partial_text)
        elif partial_text and len(partial_text.strip()) > 0:
            # User has typed a little, combine with base prompt
            prompt = base_prompt + partial_text
            prompt_prefix_len = len(partial_text)
        else:
            # No user text, use base prompt only
            prompt = base_prompt
            prompt_prefix_len = 0

        # Ensure prompt is not empty
        if not prompt or len(prompt.strip()) == 0:
            prompt = 'the project '
            prompt_prefix_len = 0

        # Generate text with lower temperature for more coherent output
        # Lower temperature = more predictable/coherent (0.3-0.5)
        # Higher temperature = more creative/random (0.8-1.2)
        full_text = self.generate_text(prompt, max_length=max_length, temperature=0.5)

        # Extract only the generated part (remove prompt)
        if prompt_prefix_len > 0:
            suggestion = full_text[prompt_prefix_len:]
        else:
            # Remove the entire prompt we added
            suggestion = full_text[len(prompt):]

        # Clean up the suggestion
        suggestion = self._clean_suggestion(suggestion)

        return suggestion

    def _clean_suggestion(self, text):
        """Clean up generated text"""
        # Remove leading/trailing whitespace
        text = text.strip()

        # Remove any weird character repetitions (e.g., "aaaa", "......")
        import re
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)  # Max 2 repeated chars

        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,;:!?])([a-zA-Z])', r'\1 \2', text)  # Add space after punctuation

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        # Try to end at a sentence boundary
        if '.' in text:
            sentences = text.split('.')
            # Keep complete sentences (at least 10 chars each to avoid fragments)
            complete_sentences = []
            for sent in sentences[:-1]:  # Exclude last (incomplete) sentence
                if len(sent.strip()) >= 10:
                    complete_sentences.append(sent.strip())

            if complete_sentences:
                text = '. '.join(complete_sentences) + '.'
            elif len(text) > 20:
                # If no complete sentences, at least clean up the text
                # Find the last complete word
                words = text.split()
                if len(words) > 3:
                    text = ' '.join(words[:-1]) + '...'

        # Remove common gibberish patterns
        text = re.sub(r'\b([a-z])\1{2,}\b', '', text, flags=re.IGNORECASE)  # Remove words like "aaaa"
        text = text.strip()

        # Ensure reasonable length (at least some content)
        if len(text) < 10:
            text = "Please try again or provide more context."

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
