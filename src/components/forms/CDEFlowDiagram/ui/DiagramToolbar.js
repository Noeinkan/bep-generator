import React from 'react';
import { Download, Upload, RotateCcw, Trash2, Plus, Settings as SettingsIcon } from 'lucide-react';

/**
 * Toolbar component for diagram actions
 */
const DiagramToolbar = ({
  onReset,
  onAddSwimlane,
  onRemoveSwimlane,
  onClear,
  onExport,
  onImport,
  onToggleSettings,
  showSettings
}) => {
  return (
    <div style={{
      padding: '12px 16px',
      background: '#f9fafb',
      borderBottom: '1px solid #e5e7eb',
      display: 'flex',
      gap: '12px',
      alignItems: 'center',
      flexWrap: 'wrap'
    }}>
      <button
        onClick={onReset}
        style={{
          padding: '8px 16px',
          background: '#6b7280',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          fontSize: '14px',
          fontWeight: '500',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}
      >
        <RotateCcw size={16} />
        Reset
      </button>
      <button
        onClick={onAddSwimlane}
        style={{
          padding: '8px 16px',
          background: '#3b82f6',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          fontSize: '14px',
          fontWeight: '500',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}
      >
        <Plus size={16} />
        Add Swimlane
      </button>
      <button
        onClick={onRemoveSwimlane}
        style={{
          padding: '8px 16px',
          background: '#f59e0b',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          fontSize: '14px',
          fontWeight: '500',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}
      >
        <Trash2 size={16} />
        Remove Swimlane
      </button>
      <button
        onClick={onClear}
        style={{
          padding: '8px 16px',
          background: '#ef4444',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          fontSize: '14px',
          fontWeight: '500',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}
      >
        <Trash2 size={16} />
        Clear All
      </button>
      <button
        onClick={onExport}
        style={{
          padding: '8px 16px',
          background: '#10b981',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          fontSize: '14px',
          fontWeight: '500',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}
      >
        <Download size={16} />
        Export
      </button>
      <label
        style={{
          padding: '8px 16px',
          background: '#8b5cf6',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          fontSize: '14px',
          fontWeight: '500',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}
      >
        <Upload size={16} />
        Import
        <input
          type="file"
          accept=".json"
          onChange={onImport}
          style={{ display: 'none' }}
        />
      </label>
      <button
        onClick={onToggleSettings}
        style={{
          padding: '8px 16px',
          background: showSettings ? '#059669' : '#10b981',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          fontSize: '14px',
          fontWeight: '500',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          marginLeft: 'auto'
        }}
      >
        <SettingsIcon size={16} />
        {showSettings ? 'Hide Settings' : 'Customize'}
      </button>
    </div>
  );
};

export default DiagramToolbar;
