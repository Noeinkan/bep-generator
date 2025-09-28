import React, { useState } from 'react';
import {
  Bold,
  Italic,
  Underline,
  AlignLeft,
  AlignCenter,
  AlignRight,
  List,
  ListOrdered,
  Type,
  X
} from 'lucide-react';

const FormattingToolbar = ({ onFormat, show, onClose, position = { top: 0, left: 0 } }) => {
  const [selectedFont, setSelectedFont] = useState('default');

  const fontOptions = [
    { value: 'default', label: 'Default' },
    { value: 'arial', label: 'Arial' },
    { value: 'times', label: 'Times New Roman' },
    { value: 'courier', label: 'Courier New' },
    { value: 'georgia', label: 'Georgia' },
    { value: 'verdana', label: 'Verdana' }
  ];

  const handleFormat = (type, value = null) => {
    onFormat(type, value);
  };

  if (!show) return null;

  return (
    <div
      className="absolute z-50 bg-white border border-gray-300 rounded-lg shadow-lg p-3 w-80 max-w-sm"
      style={{
        top: position.top,
        left: position.left,
        display: show ? 'block' : 'none'
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-700">Text Formatting</span>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-100 rounded"
          title="Close toolbar"
        >
          <X size={16} />
        </button>
      </div>

      <div className="space-y-2">
        {/* Font Selection */}
        <div className="flex items-center space-x-2">
          <Type size={16} className="text-gray-600" />
          <select
            value={selectedFont}
            onChange={(e) => {
              setSelectedFont(e.target.value);
              handleFormat('font', e.target.value);
            }}
            className="text-sm border border-gray-300 rounded px-2 py-1 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
          >
            {fontOptions.map(font => (
              <option key={font.value} value={font.value}>{font.label}</option>
            ))}
          </select>
        </div>

        {/* Text Style Buttons */}
        <div className="flex items-center space-x-1">
          <button
            onClick={() => handleFormat('bold')}
            className="p-2 hover:bg-gray-100 rounded border border-gray-300"
            title="Bold"
          >
            <Bold size={16} />
          </button>
          <button
            onClick={() => handleFormat('italic')}
            className="p-2 hover:bg-gray-100 rounded border border-gray-300"
            title="Italic"
          >
            <Italic size={16} />
          </button>
          <button
            onClick={() => handleFormat('underline')}
            className="p-2 hover:bg-gray-100 rounded border border-gray-300"
            title="Underline"
          >
            <Underline size={16} />
          </button>
        </div>

        {/* Alignment Buttons */}
        <div className="flex items-center space-x-1">
          <button
            onClick={() => handleFormat('align', 'left')}
            className="p-2 hover:bg-gray-100 rounded border border-gray-300"
            title="Align Left"
          >
            <AlignLeft size={16} />
          </button>
          <button
            onClick={() => handleFormat('align', 'center')}
            className="p-2 hover:bg-gray-100 rounded border border-gray-300"
            title="Align Center"
          >
            <AlignCenter size={16} />
          </button>
          <button
            onClick={() => handleFormat('align', 'right')}
            className="p-2 hover:bg-gray-100 rounded border border-gray-300"
            title="Align Right"
          >
            <AlignRight size={16} />
          </button>
        </div>

        {/* List Buttons */}
        <div className="flex items-center space-x-1">
          <button
            onClick={() => handleFormat('list', 'bullet')}
            className="p-2 hover:bg-gray-100 rounded border border-gray-300"
            title="Bullet List"
          >
            <List size={16} />
          </button>
          <button
            onClick={() => handleFormat('list', 'numbered')}
            className="p-2 hover:bg-gray-100 rounded border border-gray-300"
            title="Numbered List"
          >
            <ListOrdered size={16} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default FormattingToolbar;