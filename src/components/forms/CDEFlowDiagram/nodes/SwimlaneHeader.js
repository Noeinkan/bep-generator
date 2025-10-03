import React from 'react';
import { Plus } from 'lucide-react';

/**
 * Swimlane header component with Add Solution button
 */
const SwimlaneHeader = ({ data, id }) => {
  const [editingLabel, setEditingLabel] = React.useState(false);
  const [editingFullLabel, setEditingFullLabel] = React.useState(false);
  const [label, setLabel] = React.useState(data.label);
  const [fullLabel, setFullLabel] = React.useState(data.fullLabel);

  const handleAddClick = () => {
    if (data.onAddSolution) {
      data.onAddSolution(data.id);
    }
  };

  const handleSaveLabel = () => {
    if (data.onLabelChange) {
      data.onLabelChange(id, 'label', label);
    }
    setEditingLabel(false);
  };

  const handleSaveFullLabel = () => {
    if (data.onLabelChange) {
      data.onLabelChange(id, 'fullLabel', fullLabel);
    }
    setEditingFullLabel(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', alignItems: 'center' }}>
      <div
        style={{
          padding: '10px 16px',
          background: data.color || '#e5e7eb',
          border: `2px solid ${data.borderColor || '#9ca3af'}`,
          borderRadius: '8px',
          fontWeight: '700',
          fontSize: '15px',
          color: data.textColor || '#374151',
          textAlign: 'center',
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
          width: '200px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          cursor: 'text',
        }}
        title="Double-click to edit"
      >
        {editingLabel ? (
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            onBlur={handleSaveLabel}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSaveLabel();
              if (e.key === 'Escape') { setLabel(data.label); setEditingLabel(false); }
            }}
            className="nodrag"
            autoFocus
            style={{
              background: 'transparent',
              border: '1px solid currentColor',
              borderRadius: '4px',
              padding: '2px 4px',
              width: '100%',
              textAlign: 'center',
              fontSize: '15px',
              fontWeight: '700',
              color: 'inherit',
              textTransform: 'uppercase',
            }}
          />
        ) : (
          <div
            onClick={(e) => { e.stopPropagation(); setEditingLabel(true); }}
            className="nodrag nopan"
            style={{ cursor: 'text', userSelect: 'none' }}
          >
            {data.label}
          </div>
        )}
        <div
          style={{ fontSize: '10px', fontWeight: '400', marginTop: '2px', opacity: 0.8, cursor: 'text' }}
          title="Click to edit"
        >
          {editingFullLabel ? (
            <input
              value={fullLabel}
              onChange={(e) => setFullLabel(e.target.value)}
              onBlur={handleSaveFullLabel}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSaveFullLabel();
                if (e.key === 'Escape') { setFullLabel(data.fullLabel); setEditingFullLabel(false); }
              }}
              className="nodrag"
              autoFocus
              style={{
                background: 'transparent',
                border: '1px solid currentColor',
                borderRadius: '4px',
                padding: '2px 4px',
                width: '100%',
                textAlign: 'center',
                fontSize: '10px',
                fontWeight: '400',
                color: 'inherit',
              }}
            />
          ) : (
            <span
              onClick={(e) => { e.stopPropagation(); setEditingFullLabel(true); }}
              className="nodrag nopan"
              style={{ cursor: 'text', userSelect: 'none' }}
            >
              {data.fullLabel}
            </span>
          )}
        </div>
      </div>
      <button
        onClick={handleAddClick}
        className="nodrag"
        style={{
          padding: '6px 12px',
          background: data.borderColor || '#3b82f6',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          fontSize: '12px',
          fontWeight: '500',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        }}
      >
        <Plus size={14} />
        Add Solution
      </button>
    </div>
  );
};

export default SwimlaneHeader;
