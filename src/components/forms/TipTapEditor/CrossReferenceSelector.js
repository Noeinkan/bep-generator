import React from 'react';
import { BEP_STEPS } from '../../../constants/bepSteps';

const CrossReferenceSelector = ({ onSelect, onClose }) => {
  const handleSelect = (stepId, section = null) => {
    onSelect(stepId, section);
    onClose();
  };

  return (
    <div className="absolute top-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg p-4 z-20 w-96 max-h-96 overflow-y-auto">
      <h3 className="text-sm font-medium text-gray-700 mb-3">Insert Cross Reference</h3>
      <div className="space-y-2">
        {BEP_STEPS.map((step) => (
          <div key={step.id} className="border border-gray-200 rounded p-2">
            <button
              onClick={() => handleSelect(step.id)}
              className="w-full text-left text-sm font-medium text-blue-600 hover:text-blue-800 hover:bg-blue-50 p-1 rounded"
              type="button"
            >
              Step {step.id}: {step.title}
            </button>
            {step.sections && step.sections.length > 0 && (
              <div className="ml-4 mt-1 space-y-1">
                {step.sections.map((section, index) => (
                  <button
                    key={index}
                    onClick={() => handleSelect(step.id, section)}
                    className="block text-xs text-gray-600 hover:text-gray-800 hover:bg-gray-50 p-1 rounded w-full text-left"
                    type="button"
                  >
                    • {section}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="flex justify-end mt-3">
        <button
          onClick={onClose}
          className="px-3 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded"
          type="button"
        >
          Close
        </button>
      </div>
    </div>
  );
};

export default CrossReferenceSelector;