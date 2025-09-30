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
        id: 'IC-ARCH-001',
        'Information Container ID': 'IC-ARCH-001',
        'Information Container Name/Title': 'Federated Architectural Model',
        'Description': 'Complete architectural model including all building elements',
        'Task Name': 'Architectural Modeling',
        'Responsible Task Team/Party': 'Architecture Team',
        'Author': 'John Smith',
        'Dependencies/Predecessors': 'Site Survey, Structural Grid',
        'Level of Information Need (LOIN)': 'LOD 300',
        'Classification': 'Pr_20_30_60 - Building model',
        'Estimated Production Time': '3 days',
        'Delivery Milestone': 'Stage 3 - Developed Design',
        'Due Date': '2025-02-15',
        'Format/Type': 'IFC 4.0',
        'Purpose': 'Coordination and visualization',
        'Acceptance Criteria': 'Model validation passed, no clashes with structural model',
        'Review and Authorization Process': 'S3 - Issue for comment',
        'Status': 'Planned'
      }
    ]
  });

  const [selectedCell, setSelectedCell] = useState(null);
  const [editingCell, setEditingCell] = useState(null);

  const columns = [
    { key: 'Information Container ID', width: '120px' },
    { key: 'Information Container Name/Title', width: '160px' },
    { key: 'Description', width: '160px' },
    { key: 'Task Name', width: '120px' },
    { key: 'Responsible Task Team/Party', width: '140px' },
    { key: 'Author', width: '100px' },
    { key: 'Dependencies/Predecessors', width: '140px' },
    { key: 'Level of Information Need (LOIN)', width: '140px' },
    { key: 'Classification', width: '120px' },
    { key: 'Estimated Production Time', width: '120px' },
    { key: 'Delivery Milestone', width: '120px' },
    { key: 'Due Date', width: '100px' },
    { key: 'Format/Type', width: '100px' },
    { key: 'Purpose', width: '120px' },
    { key: 'Acceptance Criteria', width: '160px' },
    { key: 'Review and Authorization Process', width: '160px' },
    { key: 'Status', width: '80px' }
  ];

  // const typeOptions = ['Model', 'Drawing', 'Document', 'Report'];
  const formatOptions = ['IFC 2x3', 'IFC 4.0', 'DWG', 'PDF', 'XLSX', 'DOCX', 'RVT', 'NWD'];
  const loiOptions = ['LOD 100', 'LOD 200', 'LOD 300', 'LOD 350', 'LOD 400', 'LOD 500'];
  const statusOptions = ['Planned', 'In Progress', 'Under Review', 'Approved', 'Completed', 'Delayed'];
  const milestoneOptions = [
    'Stage 1 - Preparation',
    'Stage 2 - Concept Design',
    'Stage 3 - Developed Design',
    'Stage 4 - Technical Design',
    'Stage 5 - Manufacturing & Construction',
    'Stage 6 - Handover & Close Out',
    'Stage 7 - In Use'
  ];
  const classificationOptions = [
    'Pr_20_30_60 - Building model',
    'Pr_20_30_70 - Space model',
    'Pr_20_30_80 - Zone model',
    'Pr_30_10 - Element',
    'Pr_30_20 - Component',
    'Pr_30_30 - Assembly'
  ];
  const reviewProcessOptions = [
    'S1 - Work in progress',
    'S2 - Shared for coordination',
    'S3 - Issue for comment',
    'S4 - Issue for approval',
    'S5 - Issue for construction'
  ];

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
      id: `IC-${Date.now()}`,
      'Information Container ID': `IC-${Date.now()}`,
      'Information Container Name/Title': '',
      'Description': '',
      'Task Name': '',
      'Responsible Task Team/Party': '',
      'Author': '',
      'Dependencies/Predecessors': '',
      'Level of Information Need (LOIN)': 'LOD 200',
      'Classification': '',
      'Estimated Production Time': '1 day',
      'Delivery Milestone': '',
      'Due Date': '',
      'Format/Type': 'IFC 4.0',
      'Purpose': '',
      'Acceptance Criteria': '',
      'Review and Authorization Process': 'S1 - Work in progress',
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
      if (colKey === 'Format/Type') {
        return (
          <select
            value={value}
            onChange={(e) => handleCellChange(rowIndex, colKey, e.target.value)}
            onBlur={() => setEditingCell(null)}
            autoFocus
            className="w-full h-full border-none outline-none bg-white text-xs px-1 py-0.5"
          >
            {formatOptions.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        );
      }
      if (colKey === 'Level of Information Need (LOIN)') {
        return (
          <select
            value={value}
            onChange={(e) => handleCellChange(rowIndex, colKey, e.target.value)}
            onBlur={() => setEditingCell(null)}
            autoFocus
            className="w-full h-full border-none outline-none bg-white text-xs px-1 py-0.5"
          >
            {loiOptions.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        );
      }
      if (colKey === 'Delivery Milestone') {
        return (
          <select
            value={value}
            onChange={(e) => handleCellChange(rowIndex, colKey, e.target.value)}
            onBlur={() => setEditingCell(null)}
            autoFocus
            className="w-full h-full border-none outline-none bg-white text-xs px-1 py-0.5"
          >
            {milestoneOptions.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        );
      }
      if (colKey === 'Classification') {
        return (
          <select
            value={value}
            onChange={(e) => handleCellChange(rowIndex, colKey, e.target.value)}
            onBlur={() => setEditingCell(null)}
            autoFocus
            className="w-full h-full border-none outline-none bg-white text-xs px-1 py-0.5"
          >
            <option value="">Select Classification</option>
            {classificationOptions.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        );
      }
      if (colKey === 'Review and Authorization Process') {
        return (
          <select
            value={value}
            onChange={(e) => handleCellChange(rowIndex, colKey, e.target.value)}
            onBlur={() => setEditingCell(null)}
            autoFocus
            className="w-full h-full border-none outline-none bg-white text-xs px-1 py-0.5"
          >
            {reviewProcessOptions.map(option => (
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
            className="w-full h-full border-none outline-none bg-white text-xs px-1 py-0.5"
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
            className="w-full h-full border-none outline-none bg-white text-xs px-1 py-0.5"
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
          className="w-full h-full border-none outline-none bg-white text-xs px-1 py-0.5"
        />
      );
    }

    return (
      <div
        className={`w-full h-full px-1 py-0.5 cursor-cell text-xs ${
          isSelected ? 'bg-blue-100 border border-blue-500' : 'hover:bg-gray-50'
        }`}
        onClick={(e) => handleCellClick(rowIndex, colKey, e)}
      >
        <div className="truncate">
          {value || ''}
        </div>
      </div>
    );
  };

  return (
    <div className="fixed inset-0 bg-white z-50 flex flex-col text-sm">
      {/* Header - Compact */}
      <div className="bg-blue-600 text-white px-3 py-2 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center space-x-2">
          <FileText className="w-4 h-4" />
          <h1 className="text-lg font-bold">TIDP Excel Editor</h1>
        </div>
        <div className="flex items-center space-x-1">
          <button
            onClick={addRow}
            className="bg-blue-700 hover:bg-blue-800 px-2 py-1 rounded text-xs flex items-center space-x-1"
          >
            <Plus className="w-3 h-3" />
            <span>Add Row</span>
          </button>
          <button
            onClick={handleSave}
            className="bg-green-600 hover:bg-green-700 px-3 py-1 rounded text-xs flex items-center space-x-1"
          >
            <Save className="w-3 h-3" />
            <span>Save TIDP</span>
          </button>
          <button
            onClick={onClose}
            className="bg-red-600 hover:bg-red-700 px-2 py-1 rounded text-xs flex items-center space-x-1"
          >
            <X className="w-3 h-3" />
            <span>Close</span>
          </button>
        </div>
      </div>

      {/* TIDP Info - Ultra Compact */}
      <div className="bg-gray-50 px-3 py-2 border-b flex-shrink-0">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Task Team</label>
            <input
              type="text"
              value={tidpData.taskTeam}
              onChange={(e) => setTidpData({ ...tidpData, taskTeam: e.target.value })}
              className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:ring-1 focus:ring-blue-500"
              placeholder="Architecture Team"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Discipline</label>
            <select
              value={tidpData.discipline}
              onChange={(e) => setTidpData({ ...tidpData, discipline: e.target.value })}
              className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:ring-1 focus:ring-blue-500"
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
            <input
              type="text"
              value={tidpData.teamLeader}
              onChange={(e) => setTidpData({ ...tidpData, teamLeader: e.target.value })}
              className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:ring-1 focus:ring-blue-500"
              placeholder="John Smith"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Description</label>
            <input
              type="text"
              value={tidpData.description}
              onChange={(e) => setTidpData({ ...tidpData, description: e.target.value })}
              className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:ring-1 focus:ring-blue-500"
              placeholder="TIDP description"
            />
          </div>
        </div>
      </div>

      {/* Excel-like Table - Optimized for maximum space */}
      <div className="flex-1 overflow-hidden min-h-0">
        <div className="h-full px-2 py-1">
          <div className="bg-white border border-gray-300 rounded-lg shadow-sm overflow-hidden h-full flex flex-col">
            <div className="overflow-auto flex-1">
              <table className="w-full border-collapse">
                <thead className="bg-gray-100 border-b border-gray-300 sticky top-0 z-10">
                  <tr>
                    <th className="w-8 px-1 py-1 border-r border-gray-300 text-center text-xs font-semibold text-gray-700">#</th>
                    {columns.map(col => (
                      <th
                        key={col.key}
                        className="px-1 py-1 border-r border-gray-300 text-left font-semibold text-gray-700 text-xs"
                        style={{ minWidth: col.width }}
                      >
                        <div className="truncate" title={col.key}>
                          {col.key}
                        </div>
                      </th>
                    ))}
                    <th className="w-12 px-1 py-1 text-center text-xs font-semibold text-gray-700">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {tidpData.containers.map((container, rowIndex) => (
                    <tr key={container.id} className="border-b border-gray-200 hover:bg-gray-50">
                      <td className="px-1 py-1 border-r border-gray-300 text-center text-gray-500 text-xs">
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
                      <td className="px-1 py-1 text-center">
                        <button
                          onClick={() => deleteRow(rowIndex)}
                          disabled={tidpData.containers.length === 1}
                          className="text-red-500 hover:text-red-700 disabled:opacity-50 text-xs"
                        >
                          <Trash2 className="w-3 h-3" />
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

      {/* Footer with instructions - Compact */}
      <div className="bg-gray-100 px-3 py-1 border-t text-xs text-gray-600 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div>
            <strong>Navigation:</strong> Click to select, double-click to edit. Use Tab/Enter/arrows.
          </div>
          <div className="text-xs">
            {selectedCell ? `Row ${selectedCell.row + 1}, ${selectedCell.col}` : 'No selection'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExcelTIDPEditor;