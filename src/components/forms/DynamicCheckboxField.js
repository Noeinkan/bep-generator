import React, { useState, useMemo } from 'react';
import { Plus, X } from 'lucide-react';
import FieldHelpTooltip from './FieldHelpTooltip';
import HELP_CONTENT from '../../data/helpContentData';

const DynamicCheckboxField = ({ field, value, onChange, error, optionsList: initialOptions }) => {
  const { name, label, required } = field;
  const helpContent = HELP_CONTENT[name];

  // State per le opzioni custom aggiunte dall'utente
  const [customOptions, setCustomOptions] = useState([]);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [newOptionText, setNewOptionText] = useState('');

  // Combina le opzioni iniziali con quelle custom
  const allOptions = useMemo(() => {
    return [...initialOptions, ...customOptions];
  }, [initialOptions, customOptions]);

  const handleCheckboxChange = (option) => {
    const current = Array.isArray(value) ? value : [];
    const updated = current.includes(option)
      ? current.filter(item => item !== option)
      : [...current, option];
    onChange(name, updated);
  };

  const handleAddOption = () => {
    const trimmed = newOptionText.trim();
    if (trimmed && !allOptions.includes(trimmed)) {
      setCustomOptions(prev => [...prev, trimmed]);
      setNewOptionText('');
      setShowAddDialog(false);
    }
  };

  const handleRemoveCustomOption = (option) => {
    // Rimuovi l'opzione dalla lista custom
    setCustomOptions(prev => prev.filter(opt => opt !== option));

    // Rimuovi anche dal valore selezionato se presente
    const current = Array.isArray(value) ? value : [];
    if (current.includes(option)) {
      onChange(name, current.filter(item => item !== option));
    }
  };

  const isCustomOption = (option) => customOptions.includes(option);

  const FieldLabel = ({ children, className = "block text-sm font-medium mb-2" }) => (
    <div className="flex items-center gap-2 mb-2">
      <label className={className}>
        {children}
      </label>
      {helpContent && (
        <FieldHelpTooltip fieldName={name} helpContent={helpContent} />
      )}
    </div>
  );

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <FieldLabel>
          {label} {required && '*'}
        </FieldLabel>
        <button
          type="button"
          onClick={() => setShowAddDialog(true)}
          className="flex items-center gap-1 px-3 py-1 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          title="Add custom option"
        >
          <Plus size={16} />
          Add Option
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-60 overflow-y-auto border rounded-lg p-3">
        {allOptions.map(option => (
          <div key={option} className="relative group">
            <label
              htmlFor={`${name}-${option}`}
              className="flex items-center space-x-2 p-2 border rounded cursor-pointer hover:bg-gray-50 pr-8"
            >
              <input
                id={`${name}-${option}`}
                type="checkbox"
                checked={(value || []).includes(option)}
                onChange={() => handleCheckboxChange(option)}
                className="rounded flex-shrink-0"
              />
              <span className="text-sm break-words">{option}</span>
            </label>

            {/* Pulsante rimuovi per opzioni custom */}
            {isCustomOption(option) && (
              <button
                type="button"
                onClick={() => handleRemoveCustomOption(option)}
                className="absolute top-1 right-1 p-1 bg-red-500 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600"
                title="Remove custom option"
              >
                <X size={12} />
              </button>
            )}
          </div>
        ))}
      </div>

      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}

      {/* Dialog per aggiungere nuova opzione */}
      {showAddDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-4">Add Custom {label} Option</h3>

            <input
              type="text"
              value={newOptionText}
              onChange={(e) => setNewOptionText(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  handleAddOption();
                }
              }}
              placeholder="Enter new option name..."
              className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 mb-4"
              autoFocus
            />

            {newOptionText.trim() && allOptions.includes(newOptionText.trim()) && (
              <p className="text-amber-600 text-sm mb-4">
                This option already exists
              </p>
            )}

            <div className="flex gap-2 justify-end">
              <button
                type="button"
                onClick={() => {
                  setShowAddDialog(false);
                  setNewOptionText('');
                }}
                className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleAddOption}
                disabled={!newOptionText.trim() || allOptions.includes(newOptionText.trim())}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                Add Option
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DynamicCheckboxField;
