import React, { useEffect, useRef, useState } from 'react';
import { Check, X } from 'lucide-react';

const InlineEditor = ({ position, initialValue, onSave, onCancel, nodeType }) => {
  const [value, setValue] = useState(initialValue);
  const inputRef = useRef(null);

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, []);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSave();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onCancel();
    }
  };

  const handleSave = () => {
    if (value.trim()) {
      onSave(value.trim());
    } else {
      onCancel();
    }
  };

  return (
    <div
      className="absolute z-50 animate-scale-in"
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
        transform: 'translate(-50%, -50%)'
      }}
    >
      <div className="bg-white rounded-lg shadow-2xl border-2 border-green-400 p-1">
        <div className="flex items-center space-x-2">
          <input
            ref={inputRef}
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            className="px-3 py-2 border-0 focus:ring-0 focus:outline-none text-sm font-medium min-w-48"
            placeholder="Node name..."
          />
          <div className="flex items-center space-x-1 pr-1">
            <button
              onClick={handleSave}
              className="p-1.5 bg-green-500 hover:bg-green-600 text-white rounded transition-colors duration-200"
              title="Save (Enter)"
            >
              <Check className="w-4 h-4" />
            </button>
            <button
              onClick={onCancel}
              className="p-1.5 bg-gray-300 hover:bg-gray-400 text-gray-700 rounded transition-colors duration-200"
              title="Cancel (Esc)"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
        <div className="px-3 pb-2 pt-1">
          <div className="text-xs text-gray-500 flex items-center justify-between">
            <span>Enter to save • Esc to cancel</span>
            <span className="text-xs text-gray-400">{value.length} chars</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InlineEditor;
