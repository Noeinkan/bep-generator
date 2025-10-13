import React, { useState } from 'react';
import { X, Table as TableIcon } from 'lucide-react';

const TableInsertDialog = ({ onInsert, onClose }) => {
  const [rows, setRows] = useState(3);
  const [cols, setCols] = useState(3);
  const [withHeaderRow, setWithHeaderRow] = useState(true);

  const presets = [
    { label: '2×2', rows: 2, cols: 2, header: false },
    { label: '3×3', rows: 3, cols: 3, header: true },
    { label: '4×4', rows: 4, cols: 4, header: true },
    { label: '5×3', rows: 5, cols: 3, header: true },
    { label: '3×5', rows: 3, cols: 5, header: true },
  ];

  const handleInsert = () => {
    onInsert({ rows, cols, withHeaderRow });
    onClose();
  };

  const handlePreset = (preset) => {
    setRows(preset.rows);
    setCols(preset.cols);
    setWithHeaderRow(preset.header);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleInsert();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    }
  };

  return (
    <div
      className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-40 z-50"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg shadow-2xl p-6 w-full max-w-md"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        {/* Header */}
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center gap-2">
            <TableIcon size={24} className="text-blue-600" />
            <h3 className="text-xl font-semibold text-gray-800">Insert Table</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-gray-100 transition-colors"
            type="button"
            title="Close"
          >
            <X size={20} />
          </button>
        </div>

        {/* Presets */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Preset Sizes:
          </label>
          <div className="flex flex-wrap gap-2">
            {presets.map((preset) => (
              <button
                key={preset.label}
                onClick={() => handlePreset(preset)}
                className={`px-4 py-2 rounded-lg border-2 transition-all ${
                  rows === preset.rows && cols === preset.cols && withHeaderRow === preset.header
                    ? 'border-blue-600 bg-blue-50 text-blue-700 font-medium'
                    : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
                }`}
                type="button"
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        {/* Custom dimensions */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Custom Dimensions:
          </label>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-gray-600 mb-1">Rows</label>
              <input
                type="number"
                min="1"
                max="20"
                value={rows}
                onChange={(e) => setRows(Math.max(1, Math.min(20, parseInt(e.target.value) || 1)))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                autoFocus
              />
            </div>
            <div>
              <label className="block text-xs text-gray-600 mb-1">Columns</label>
              <input
                type="number"
                min="1"
                max="10"
                value={cols}
                onChange={(e) => setCols(Math.max(1, Math.min(10, parseInt(e.target.value) || 1)))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          </div>
        </div>

        {/* Header row option */}
        <div className="mb-6">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={withHeaderRow}
              onChange={(e) => setWithHeaderRow(e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700">Include header row</span>
          </label>
        </div>

        {/* Preview */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Preview:</label>
          <div className="border border-gray-300 rounded-lg p-3 bg-gray-50 overflow-auto">
            <table className="w-full border-collapse text-xs">
              <tbody>
                {Array.from({ length: rows }).map((_, rowIndex) => (
                  <tr key={rowIndex}>
                    {Array.from({ length: cols }).map((_, colIndex) => {
                      const isHeader = withHeaderRow && rowIndex === 0;
                      const CellTag = isHeader ? 'th' : 'td';
                      return (
                        <CellTag
                          key={colIndex}
                          className={`border border-gray-300 px-2 py-1 ${
                            isHeader ? 'bg-gray-200 font-semibold' : 'bg-white'
                          }`}
                        >
                          {isHeader ? 'Header' : 'Cell'}
                        </CellTag>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={handleInsert}
            className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
            type="button"
          >
            Insert Table
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-lg transition-colors"
            type="button"
          >
            Cancel
          </button>
        </div>
      </div>

      <style jsx>{`
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateY(-20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .bg-white {
          animation: slideIn 0.2s ease-out;
        }
      `}</style>
    </div>
  );
};

export default TableInsertDialog;
