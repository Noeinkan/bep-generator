import React from 'react';
import { Users } from 'lucide-react';
import FormattedTextEditor from './FormattedTextEditor';

const EditableTable = React.memo(({ field, value, onChange, error }) => {
  const { name, label, required, columns = ['Role/Discipline', 'Name/Company', 'Experience/Notes'] } = field;
  const tableData = Array.isArray(value) ? value : [];

  const addRow = () => {
    const newRow = columns.reduce((acc, col) => ({ ...acc, [col]: '' }), {});
    onChange(name, [...tableData, newRow]);
  };

  const removeRow = (index) => {
    const newData = tableData.filter((_, i) => i !== index);
    onChange(name, newData);
  };

  const updateCell = (rowIndex, column, cellValue) => {
    const newData = tableData.map((row, index) =>
      index === rowIndex ? { ...row, [column]: cellValue } : row
    );
    onChange(name, newData);
  };

  const moveRow = (fromIndex, toIndex) => {
    if (toIndex < 0 || toIndex >= tableData.length) return;
    const newData = [...tableData];
    const [movedRow] = newData.splice(fromIndex, 1);
    newData.splice(toIndex, 0, movedRow);
    onChange(name, newData);
  };

  return (
    <div className="mb-8">
      <label className="block text-lg font-semibold mb-4 text-gray-800">
        {label} {required && <span className="text-red-500">*</span>}
      </label>

      <div className="border rounded-xl overflow-hidden shadow-sm bg-white">
        <div className="bg-gradient-to-r from-gray-50 to-gray-100 px-6 py-4 border-b border-gray-200">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <span className="text-base font-semibold text-gray-800">
                {tableData.length} {tableData.length === 1 ? 'Entry' : 'Entries'}
              </span>
              {tableData.length > 0 && (
                <span className="text-sm text-gray-500">
                  Click and drag to reorder • Use textarea for multi-line content
                </span>
              )}
            </div>
            <button
              type="button"
              onClick={addRow}
              className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-all transform hover:scale-105 shadow-md"
            >
              <span className="text-lg">+</span>
              <span>Add Row</span>
            </button>
          </div>
        </div>

        {tableData.length === 0 ? (
          <div className="p-12 text-center text-gray-500">
            <Users className="w-16 h-16 mx-auto mb-4 text-gray-300" />
            <p className="text-lg">No entries yet. Click "Add Row" to get started.</p>
          </div>
        ) : (
          <div className="overflow-x-auto bg-white">
            <table className="w-full min-w-full table-fixed">
              <thead className="bg-gray-50">
                <tr>
                  <th className="w-16 px-2 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                    Order
                  </th>
                  {columns.map((column, index) => (
                    <th key={column} className={`px-3 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider ${
                      columns.length === 4 ? 'w-1/4' :
                      columns.length === 3 ? 'w-1/3' :
                      'w-auto'
                    }`}>
                      {column}
                    </th>
                  ))}
                  <th className="w-16 px-2 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {tableData.map((row, rowIndex) => (
                  <tr key={rowIndex} className="hover:bg-gray-50 transition-colors">
                    <td className="px-2 py-2">
                      <div className="flex flex-col items-center space-y-1">
                        <button
                          type="button"
                          onClick={() => moveRow(rowIndex, rowIndex - 1)}
                          disabled={rowIndex === 0}
                          className="w-6 h-6 flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs"
                          title="Move up"
                        >
                          ↑
                        </button>
                        <span className="text-xs font-medium text-gray-600 bg-gray-100 px-1 py-0.5 rounded">{rowIndex + 1}</span>
                        <button
                          type="button"
                          onClick={() => moveRow(rowIndex, rowIndex + 1)}
                          disabled={rowIndex === tableData.length - 1}
                          className="w-6 h-6 flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs"
                          title="Move down"
                        >
                          ↓
                        </button>
                      </div>
                    </td>
                    {columns.map(column => (
                      <td key={column} className="px-1 py-2">
                        <FormattedTextEditor
                          value={row[column] || ''}
                          onChange={(newValue) => updateCell(rowIndex, column, newValue)}
                          className="text-sm"
                          placeholder={`Enter ${column.toLowerCase()}...`}
                          rows={4}
                        />
                      </td>
                    ))}
                    <td className="px-2 py-2">
                      <button
                        type="button"
                        onClick={() => removeRow(rowIndex)}
                        className="w-8 h-8 flex items-center justify-center text-red-600 hover:text-red-800 hover:bg-red-50 rounded transition-colors font-medium text-sm"
                        title="Remove row"
                      >
                        ✕
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
});

export default EditableTable;