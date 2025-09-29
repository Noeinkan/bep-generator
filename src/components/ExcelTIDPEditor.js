import React, { useState } from 'react';
import { Save, X, Plus, Trash2, FileText } from 'lucide-react';

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

  const columns = [
    { key: 'Container Name', width: '200px' },
    { key: 'Type', width: '100px' },
    { key: 'Format', width: '80px' },
    { key: 'LOI Level', width: '100px' },
    { key: 'Author', width: '120px' },
    { key: 'Dependencies', width: '150px' },
    { key: 'Est. Time', width: '100px' },
    { key: 'Milestone', width: '120px' },
    { key: 'Due Date', width: '120px' },
    { key: 'Status', width: '100px' }
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

  const handleCellChange = (rowIndex, colKey, value) => {
    const updatedContainers = [...tidpData.containers];
    updatedContainers[rowIndex][colKey] = value;
    setTidpData({ ...tidpData, containers: updatedContainers });
  };

  const addRow = () => {
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
    setTidpData({
      ...tidpData,
      containers: [...tidpData.containers, newRow]
    });
  };

  const deleteRow = (index) => {
    if (tidpData.containers.length > 1) {
      setTidpData({
        ...tidpData,
        containers: tidpData.containers.filter((_, i) => i !== index)
      });
    }
  };

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
            className="w-full h-full border-none outline-none bg-white"
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
            className="w-full h-full border-none outline-none bg-white"
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
            className="w-full h-full border-none outline-none bg-white"
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
            className="w-full h-full border-none outline-none bg-white"
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
            className="w-full h-full border-none outline-none bg-white"
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
          className="w-full h-full border-none outline-none bg-white"
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
      <div className="bg-blue-600 text-white p-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <FileText className="w-6 h-6" />
          <h1 className="text-xl font-bold">TIDP Excel Editor</h1>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={addRow}
            className="bg-blue-700 hover:bg-blue-800 px-3 py-1 rounded flex items-center space-x-1"
          >
            <Plus className="w-4 h-4" />
            <span>Add Row</span>
          </button>
          <button
            onClick={handleSave}
            className="bg-green-600 hover:bg-green-700 px-4 py-1 rounded flex items-center space-x-1"
          >
            <Save className="w-4 h-4" />
            <span>Save TIDP</span>
          </button>
          <button
            onClick={onClose}
            className="bg-red-600 hover:bg-red-700 px-3 py-1 rounded flex items-center space-x-1"
          >
            <X className="w-4 h-4" />
            <span>Close</span>
          </button>
        </div>
      </div>

      {/* TIDP Info */}
      <div className="bg-gray-50 p-4 border-b">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Task Team</label>
            <input
              type="text"
              value={tidpData.taskTeam}
              onChange={(e) => setTidpData({ ...tidpData, taskTeam: e.target.value })}
              className="w-full p-2 border border-gray-300 rounded"
              placeholder="Architecture Team"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Discipline</label>
            <select
              value={tidpData.discipline}
              onChange={(e) => setTidpData({ ...tidpData, discipline: e.target.value })}
              className="w-full p-2 border border-gray-300 rounded"
            >
              <option value="">Select Discipline</option>
              <option value="architecture">Architecture</option>
              <option value="structural">Structural Engineering</option>
              <option value="mep">MEP Engineering</option>
              <option value="civil">Civil Engineering</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Team Leader</label>
            <input
              type="text"
              value={tidpData.teamLeader}
              onChange={(e) => setTidpData({ ...tidpData, teamLeader: e.target.value })}
              className="w-full p-2 border border-gray-300 rounded"
              placeholder="John Smith"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
            <input
              type="text"
              value={tidpData.description}
              onChange={(e) => setTidpData({ ...tidpData, description: e.target.value })}
              className="w-full p-2 border border-gray-300 rounded"
              placeholder="TIDP description"
            />
          </div>
        </div>
      </div>

      {/* Excel-like Table */}
      <div className="flex-1 overflow-auto">
        <div className="p-4">
          <div className="bg-white border border-gray-300 rounded-lg shadow-sm overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="bg-gray-100 border-b border-gray-300">
                    <th className="w-12 p-2 border-r border-gray-300 text-center">#</th>
                    {columns.map(col => (
                      <th
                        key={col.key}
                        className="p-2 border-r border-gray-300 text-left font-semibold text-gray-700"
                        style={{ minWidth: col.width }}
                      >
                        {col.key}
                      </th>
                    ))}
                    <th className="w-16 p-2 text-center">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {tidpData.containers.map((container, rowIndex) => (
                    <tr key={container.id} className="border-b border-gray-200 hover:bg-gray-50">
                      <td className="p-2 border-r border-gray-300 text-center text-gray-500">
                        {rowIndex + 1}
                      </td>
                      {columns.map(col => (
                        <td
                          key={col.key}
                          className="border-r border-gray-300 p-0"
                          style={{ minWidth: col.width }}
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
      </div>

      {/* Footer with instructions */}
      <div className="bg-gray-100 p-3 border-t text-sm text-gray-600">
        <div className="flex items-center justify-between">
          <div>
            <strong>Excel-like navigation:</strong> Click cells to select, double-click to edit.
            Use arrow keys, Tab, Enter to navigate.
          </div>
          <div className="text-xs">
            {selectedCell ? `Selected: Row ${selectedCell.row + 1}, Column ${selectedCell.col}` : 'No cell selected'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExcelTIDPEditor;