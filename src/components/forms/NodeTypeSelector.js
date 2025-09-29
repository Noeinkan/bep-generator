import React from 'react';
import { getNodeTypeOptions } from '../../utils/nodeTypes';

const NodeTypeSelector = ({ selectedType, onTypeChange, isVisible, onClose }) => {
  const nodeTypes = getNodeTypeOptions();

  if (!isVisible) return null;

  return (
    <div className="absolute top-full left-0 mt-1 bg-white border border-gray-300 rounded-lg shadow-lg z-20 min-w-48">
      <div className="p-2 border-b border-gray-200">
        <span className="text-sm font-medium text-gray-700">Select Node Type</span>
      </div>
      <div className="max-h-48 overflow-y-auto">
        {nodeTypes.map((type) => (
          <button
            key={type.value}
            onClick={() => {
              onTypeChange(type.value);
              onClose();
            }}
            className={`w-full flex items-center space-x-3 px-3 py-2 text-left hover:bg-gray-50 ${
              selectedType === type.value ? 'bg-blue-50 text-blue-700' : 'text-gray-700'
            }`}
          >
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: type.color }}
            />
            <span className="text-sm">{type.label}</span>
          </button>
        ))}
      </div>
      <div className="p-2 border-t border-gray-200">
        <button
          onClick={onClose}
          className="w-full text-sm text-gray-500 hover:text-gray-700"
        >
          Cancel
        </button>
      </div>
    </div>
  );
};

export default NodeTypeSelector;