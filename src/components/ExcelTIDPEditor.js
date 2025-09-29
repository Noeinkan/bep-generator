import React, { useState, useCallback } from 'react';
import { Save, X, Plus, Trash2, FileText } from 'lucide-react';
import FormattedTextEditor from './forms/FormattedTextEditor';

const ExcelTIDPEditor = ({ onClose, onSave }) => {
  const [tidpData, setTidpData] = useState({
    taskTeam: '',
    discipline: '',
    teamLeader: '',
    description: '',
    containers: [
      {
        id: 'c-1',
        'Container Name': 'Federated Architectural Model',
        'Type': 'Model',
        'Format': 'IFC',
        'LOI Level': 'LOD 300',
        'Author': 'John Smith',
        'Dependencies': 'Structural Model',
        'Est. Time': '3 days',
        'Milestone': 'Stage 3',
        'Due Date': '2025-02-15',
        'Status': 'Planned'
      }
    ]
  });

  const [selectedCell, setSelectedCell] = useState(null);
  const [editingCell, setEditingCell] = useState(null);

  const handleTidpDataChange = useCallback((field, value) => {
    setTidpData(prev => ({
      ...prev,
      [field]: value
    }));
  }, []);

  const columns = [
    { key: 'Container Name', width: '20%' },
    { key: 'Type', width: '10%' },
    { key: 'Format', width: '8%' },
    { key: 'LOI Level', width: '10%' },
    { key: 'Author', width: '12%' },
    { key: 'Dependencies', width: '15%' },
    { key: 'Est. Time', width: '8%' },
    { key: 'Milestone', width: '10%' },
    { key: 'Due Date', width: '10%' },
    { key: 'Status', width: '7%' }
  ];

  const typeOptions = ['Model', 'Drawing', 'Document', 'Report'];
  const formatOptions = ['IFC', 'DWG', 'PDF', 'XLSX'];
  const loiOptions = ['LOD 100', 'LOD 200', 'LOD 300', 'LOD 350', 'LOD 400'];
  const statusOptions = ['Planned', 'In Progress', 'Completed', 'Delayed'];

  const handleCellClick = (rowIndex, colKey, event) => {
    setSelectedCell({ row: rowIndex, col: colKey });
    if (event.detail === 2) {
      setEditingCell({ row: rowIndex, col: colKey });
    }
  };

  const handleCellChange = useCallback((rowIndex, colKey, value) => {
    setTidpData(prev => ({
      ...prev,
      containers: prev.containers.map((container, index) =>
        index === rowIndex ? { ...container, [colKey]: value } : container
      )
    }));
  }, []);

  const addRow = useCallback(() => {
    const newRow = {
      id: `c-${Date.now()}`,
      'Container Name': '',
      'Type': 'Model',
      'Format': 'IFC',
      'LOI Level': 'LOD 200',
      'Author': '',
      'Dependencies': '',
      'Est. Time': '1 day',
      'Milestone': '',
      'Due Date': '',
      'Status': 'Planned'
    };
    setTidpData(prev => ({
      ...prev,
      containers: [...prev.containers, newRow]
    }));
  }, []);

  const deleteRow = useCallback((index) => {
    setTidpData(prev => ({
      ...prev,
      containers: prev.containers.filter((_, i) => i !== index)
    }));
  }, []);

  const handleSave = () => {
    onSave(tidpData);
  };

  const renderCell = (rowIndex, colKey, value) => {
    const isSelected = selectedCell?.row === rowIndex && selectedCell?.col === colKey;
    const isEditing = editingCell?.row === rowIndex && editingCell?.col === colKey;

    if (isEditing) {
      if (colKey === 'Type') {
        return (
          <select
            value={value}
            onChange={(e) => handleCellChange(rowIndex, colKey, e.target.value)}
            onBlur={() => setEditingCell(null)}
            autoFocus
            className="w-full h-[40px] p-2 border-none outline-none bg-white text-sm"
          >
            {typeOptions.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        );
      }
      if (colKey === 'Format') {
        return (
          <select
            value={value}
            onChange={(e) => handleCellChange(rowIndex, colKey, e.target.value)}
            onBlur={() => setEditingCell(null)}
            autoFocus
            className="w-full h-[40px] p-2 border-none outline-none bg-white text-sm"
          >
            {formatOptions.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        );
      }
      if (colKey === 'LOI Level') {
        return (
          <select
            value={value}
            onChange={(e) => handleCellChange(rowIndex, colKey, e.target.value)}
            onBlur={() => setEditingCell(null)}
            autoFocus
            className="w-full h-[40px] p-2 border-none outline-none bg-white text-sm"
          >
            {loiOptions.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        );
      }
      if (colKey === 'Status') {
        return (
          <select
            value={value}
            onChange={(e) => handleCellChange(rowIndex, colKey, e.target.value)}
            onBlur={() => setEditingCell(null)}
            autoFocus
            className="w-full h-[40px] p-2 border-none outline-none bg-white text-sm"
          >
            {statusOptions.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        );
      }
      if (colKey === 'Due Date') {
        return (
          <input
            type="date"
            value={value}
            onChange={(e) => handleCellChange(rowIndex, colKey, e.target.value)}
            onBlur={() => setEditingCell(null)}
            autoFocus
            className="w-full h-[40px] p-2 border-none outline-none bg-white text-sm"
          />
        );
      }
      return (
        <input
          type="text"
          value={value}
          onChange={(e) => handleCellChange(rowIndex, colKey, e.target.value)}
          onBlur={() => setEditingCell(null)}
          autoFocus
          className="w-full h-[40px] p-2 border-none outline-none bg-white text-sm"
        />
      );
    }

    return (
      <div
        className={`w-full h-full p-2 cursor-cell ${
          isSelected ? 'bg-blue-100 border-2 border-blue-500' : 'hover:bg-gray-50'
        }`}
        onClick={(e) => handleCellClick(rowIndex, colKey, e)}
      >
        {value || ''}
      </div>
    );
  };

  return (
    <div className="fixed inset-0 bg-white z-50 flex flex-col">
      {/* Header */}
      <div className="bg-blue-600 text-white p-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <FileText className="w-5 h-5" />
          <h1 className="text-lg font-bold">TIDP Excel Editor</h1>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={addRow}
            className="bg-blue-700 hover:bg-blue-800 px-2 py-1 rounded flex items-center space-x-1 text-sm"
          >
            <Plus className="w-4 h-4" />
            <span>Add Row</span>
          </button>
          <button
            onClick={handleSave}
            className="bg-green-600 hover:bg-green-700 px-3 py-1 rounded flex items-center space-x-1 text-sm"
          >
            <Save className="w-4 h-4" />
            <span>Save TIDP</span>
          </button>
          <button
            onClick={onClose}
            className="bg-red-600 hover:bg-red-700 px-2 py-1 rounded flex items-center space-x-1 text-sm"
          >
            <X className="w-4 h-4" />
            <span>Close</span>
          </button>
        </div>
      </div>

      {/* TIDP Info */}
      <div className="bg-gray-50 p-3 border-b">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Task Team</label>
            <FormattedTextEditor
              value={tidpData.taskTeam}
              onChange={(value) => handleTidpDataChange('taskTeam', value)}
              placeholder="Architecture Team"
              rows={1}
              showToolbar={false}
              autoGrow={false}
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Discipline</label>
            <select
              key="discipline-select"
              value={tidpData.discipline}
              onChange={(e) => handleTidpDataChange('discipline', e.target.value)}
              className="w-full p-2 border border-gray-300 rounded text-sm"
            >
              <option value="">Select Discipline</option>
              <option value="architecture">Architecture</option>
              <option value="structural">Structural Engineering</option>
              <option value="mep">MEP Engineering</option>
              <option value="civil">Civil Engineering</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Team Leader</label>
            <FormattedTextEditor
              value={tidpData.teamLeader}
              onChange={(value) => handleTidpDataChange('teamLeader', value)}
              placeholder="John Smith"
              rows={1}
              showToolbar={false}
              autoGrow={false}
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Description</label>
            <FormattedTextEditor
              value={tidpData.description}
              onChange={(value) => handleTidpDataChange('description', value)}
              placeholder="TIDP description"
              rows={2}
              showToolbar={false}
              autoGrow={true}
            />
          </div>
        </div>
      </div>

      {/* Excel-like Table */}
      <div className="flex-1 overflow-auto p-2">
        <div className="bg-white border border-gray-300 rounded-lg shadow-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full table-fixed border-collapse">
              <thead>
                <tr className="bg-gray-100 border-b border-gray-300">
                  <th className="w-10 p-2 border-r border-gray-300 text-center text-sm font-semibold text-gray-700">
                    #
                  </th>
                  {columns.map(col => (
                    <th
                      key={col.key}
                      className="p-2 border-r border-gray-300 text-left text-sm font-semibold text-gray-700"
                      style={{ width: col.width }}
                    >
                      {col.key}
                    </th>
                  ))}
                  <th className="w-12 p-2 text-center text-sm font-semibold text-gray-700">Actions</th>
                </tr>
              </thead>
              <tbody>
                {tidpData.containers.map((container, rowIndex) => (
                  <tr key={container.id} className="border-b border-gray-200 hover:bg-gray-50">
                    <td className="p-2 border-r border-gray-300 text-center text-gray-500 text-sm">
                      {rowIndex + 1}
                    </td>
                    {columns.map(col => (
                      <td
                        key={col.key}
                        className="border-r border-gray-300 p-0"
                        style={{ width: col.width }}
                      >
                        {renderCell(rowIndex, col.key, container[col.key])}
                      </td>
                    ))}
                    <td className="p-2 text-center">
                      <button
                        onClick={() => deleteRow(rowIndex)}
                        disabled={tidpData.containers.length === 1}
                        className="text-red-500 hover:text-red-700 disabled:opacity-50"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Footer with instructions */}
      <div className="bg-gray-100 p-2 border-t text-xs text-gray-600">
        <div className="flex items-center justify-between">
          <div>
            <strong>Excel-like navigation:</strong> Click cells to select, double-click to edit.
            Use arrow keys, Tab, Enter to navigate.
          </div>
          <div>
            {selectedCell ? `Selected: Row ${selectedCell.row + 1}, Column ${selectedCell.col}` : 'No cell selected'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExcelTIDPEditor;