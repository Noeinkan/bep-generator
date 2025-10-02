import React, { useState } from 'react';
import { Users, Plus, X, Edit2 } from 'lucide-react';
import TipTapEditor from './TipTapEditor';

const EditableTable = React.memo(({ field, value, onChange, error }) => {
  const { name, label, required, columns: presetColumns = ['Role/Discipline', 'Name/Company', 'Experience/Notes'] } = field;

  // Value structure: { columns: [], data: [] } or legacy format (just array)
  const isLegacyFormat = Array.isArray(value);
  const tableValue = isLegacyFormat
    ? { columns: presetColumns, data: value }
    : (value || { columns: presetColumns, data: [] });

  const columns = tableValue.columns || presetColumns;
  const tableData = Array.isArray(tableValue.data) ? tableValue.data : [];

  const [editingColumn, setEditingColumn] = useState(null);
  const [editingColumnName, setEditingColumnName] = useState('');
  const [columnToDelete, setColumnToDelete] = useState(null);

  const updateTableValue = (newColumns, newData) => {
    onChange(name, { columns: newColumns, data: newData });
  };

  const addRow = () => {
    const newRow = columns.reduce((acc, col) => ({ ...acc, [col]: '' }), {});
    updateTableValue(columns, [...tableData, newRow]);
  };

  const removeRow = (index) => {
    const newData = tableData.filter((_, i) => i !== index);
    updateTableValue(columns, newData);
  };

  const updateCell = (rowIndex, column, cellValue) => {
    const newData = tableData.map((row, index) =>
      index === rowIndex ? { ...row, [column]: cellValue } : row
    );
    updateTableValue(columns, newData);
  };

  const moveRow = (fromIndex, toIndex) => {
    if (toIndex < 0 || toIndex >= tableData.length) return;
    const newData = [...tableData];
    const [movedRow] = newData.splice(fromIndex, 1);
    newData.splice(toIndex, 0, movedRow);
    updateTableValue(columns, newData);
  };

  // Column management functions
  const addColumn = () => {
    const newColumnName = `Column ${columns.length + 1}`;
    const newColumns = [...columns, newColumnName];
    const newData = tableData.map(row => ({ ...row, [newColumnName]: '' }));
    updateTableValue(newColumns, newData);
  };

  const removeColumn = (columnIndex) => {
    if (columns.length <= 1) {
      alert('Cannot remove the last column');
      return;
    }
    const columnToRemove = columns[columnIndex];
    const newColumns = columns.filter((_, i) => i !== columnIndex);
    const newData = tableData.map(row => {
      const newRow = { ...row };
      delete newRow[columnToRemove];
      return newRow;
    });
    updateTableValue(newColumns, newData);
    setColumnToDelete(null);
  };

  const renameColumn = (oldName, newName) => {
    if (!newName.trim() || newName === oldName) {
      setEditingColumn(null);
      return;
    }
    if (columns.includes(newName)) {
      alert('A column with this name already exists');
      return;
    }
    const newColumns = columns.map(col => col === oldName ? newName : col);
    const newData = tableData.map(row => {
      const newRow = { ...row };
      newRow[newName] = newRow[oldName];
      delete newRow[oldName];
      return newRow;
    });
    updateTableValue(newColumns, newData);
    setEditingColumn(null);
  };

  const moveColumn = (fromIndex, toIndex) => {
    if (toIndex < 0 || toIndex >= columns.length) return;
    const newColumns = [...columns];
    const [movedColumn] = newColumns.splice(fromIndex, 1);
    newColumns.splice(toIndex, 0, movedColumn);
    updateTableValue(newColumns, tableData);
  };

  return (
    <div className="mb-8">
      {label && (
        <label className="block text-lg font-semibold mb-4 text-gray-800">
          {label} {required && <span className="text-red-500">*</span>}
        </label>
      )}

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
              <thead className="bg-gradient-to-r from-gray-50 to-gray-100">
                <tr>
                  <th className="w-10 px-1 py-3">
                  </th>
                  {columns.map((column, colIndex) => (
                    <th key={column} className={`px-4 py-3 text-left ${
                      columns.length === 4 ? 'w-1/4' :
                      columns.length === 3 ? 'w-1/3' :
                      'w-auto'
                    }`}>
                      <div className="group">
                        {/* Column name (editable) */}
                        <div className="mb-2">
                          {editingColumn === colIndex ? (
                            <input
                              type="text"
                              value={editingColumnName}
                              onChange={(e) => setEditingColumnName(e.target.value)}
                              onBlur={() => renameColumn(column, editingColumnName)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                  renameColumn(column, editingColumnName);
                                } else if (e.key === 'Escape') {
                                  setEditingColumn(null);
                                }
                              }}
                              autoFocus
                              className="w-full px-2 py-1.5 text-sm font-semibold border-2 border-blue-500 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
                            />
                          ) : (
                            <div
                              onClick={() => {
                                setEditingColumn(colIndex);
                                setEditingColumnName(column);
                              }}
                              className="px-2 py-1.5 text-sm font-semibold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-white hover:shadow-sm rounded-lg transition-all"
                              title="Click to rename"
                            >
                              {column}
                            </div>
                          )}
                        </div>

                        {/* Column controls - appear on hover */}
                        <div className="flex items-center justify-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                          {/* Move left */}
                          <button
                            type="button"
                            onClick={() => moveColumn(colIndex, colIndex - 1)}
                            disabled={colIndex === 0}
                            className="p-1 rounded-md bg-white hover:bg-blue-50 text-blue-600 disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-white shadow-sm border border-gray-200 transition-all hover:scale-110 text-xs"
                            title="Move left"
                          >
                            ←
                          </button>

                          {/* Edit */}
                          <button
                            type="button"
                            onClick={() => {
                              setEditingColumn(colIndex);
                              setEditingColumnName(column);
                            }}
                            className="p-1 rounded-md bg-white hover:bg-green-50 text-green-600 shadow-sm border border-gray-200 transition-all hover:scale-110"
                            title="Rename column"
                          >
                            <Edit2 size={12} />
                          </button>

                          {/* Move right */}
                          <button
                            type="button"
                            onClick={() => moveColumn(colIndex, colIndex + 1)}
                            disabled={colIndex === columns.length - 1}
                            className="p-1 rounded-md bg-white hover:bg-blue-50 text-blue-600 disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-white shadow-sm border border-gray-200 transition-all hover:scale-110 text-xs"
                            title="Move right"
                          >
                            →
                          </button>

                          {/* Delete */}
                          <button
                            type="button"
                            onClick={() => setColumnToDelete(colIndex)}
                            className="p-1 rounded-md bg-white hover:bg-red-50 text-red-500 shadow-sm border border-gray-200 transition-all hover:scale-110"
                            title="Remove column"
                          >
                            <X size={12} />
                          </button>
                        </div>
                      </div>
                    </th>
                  ))}
                  <th className="w-12 px-1 py-3 text-center">
                    <button
                      type="button"
                      onClick={addColumn}
                      className="p-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white shadow-md transition-all hover:scale-110"
                      title="Add column"
                    >
                      <Plus size={16} />
                    </button>
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {tableData.map((row, rowIndex) => (
                  <tr key={rowIndex} className="hover:bg-gray-50 transition-colors">
                    <td className="px-1 py-2">
                      <div className="flex flex-col items-center space-y-0.5">
                        <button
                          type="button"
                          onClick={() => moveRow(rowIndex, rowIndex - 1)}
                          disabled={rowIndex === 0}
                          className="w-5 h-5 flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs"
                          title="Move up"
                        >
                          ↑
                        </button>
                        <span className="text-xs font-medium text-gray-600 bg-gray-100 px-1 rounded">{rowIndex + 1}</span>
                        <button
                          type="button"
                          onClick={() => moveRow(rowIndex, rowIndex + 1)}
                          disabled={rowIndex === tableData.length - 1}
                          className="w-5 h-5 flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs"
                          title="Move down"
                        >
                          ↓
                        </button>
                      </div>
                    </td>
                    {columns.map(column => (
                      <td key={column} className="px-1 py-2">
                        <TipTapEditor
                          value={row[column] || ''}
                          onChange={(newValue) => updateCell(rowIndex, column, newValue)}
                          className="text-sm"
                          placeholder={`Enter ${column.toLowerCase()}...`}
                          minHeight="48px"
                          showToolbar={false}
                          autoSaveKey={`table-${name}-${rowIndex}-${column}`}
                        />
                      </td>
                    ))}
                    <td className="px-1 py-2">
                      <button
                        type="button"
                        onClick={() => removeRow(rowIndex)}
                        className="w-6 h-6 flex items-center justify-center text-red-600 hover:text-red-800 hover:bg-red-50 rounded transition-colors font-medium text-sm"
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

      {/* Delete Column Confirmation Modal */}
      {columnToDelete !== null && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setColumnToDelete(null)}>
          <div className="bg-white rounded-2xl shadow-2xl p-6 max-w-md w-full mx-4 transform transition-all" onClick={(e) => e.stopPropagation()}>
            {/* Header */}
            <div className="flex items-start gap-4 mb-4">
              <div className="p-3 bg-red-100 rounded-full">
                <X size={24} className="text-red-600" />
              </div>
              <div className="flex-1">
                <h3 className="text-xl font-bold text-gray-900 mb-1">Remove Column?</h3>
                <p className="text-gray-600">
                  Are you sure you want to remove the column <span className="font-semibold text-gray-900">"{columns[columnToDelete]}"</span>?
                </p>
              </div>
            </div>

            {/* Warning message */}
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 mb-6">
              <p className="text-sm text-amber-800">
                ⚠️ This will permanently delete all data in this column from all rows.
              </p>
            </div>

            {/* Actions */}
            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => setColumnToDelete(null)}
                className="flex-1 px-4 py-2.5 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => removeColumn(columnToDelete)}
                className="flex-1 px-4 py-2.5 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors shadow-lg"
              >
                Remove Column
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

export default EditableTable;