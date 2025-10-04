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
  const [isHovered, setIsHovered] = React.useState(false);

  const handleAddClick = (e) => {
    e.stopPropagation();
    e.preventDefault();
    console.log('Add Solution clicked for swimlane:', data.id);
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
    <div style={{ display: 'flex', flexDirection: 'row', gap: '12px', alignItems: 'center' }}>
      <div
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        style={{
          padding: '10px 16px',
          background: data.color || '#e5e7eb',
          border: `2px solid ${data.borderColor || '#9ca3af'}`,
          borderRadius: '10px',
          fontWeight: '700',
          fontSize: '14px',
          color: data.textColor || '#374151',
          textAlign: 'center',
          textTransform: 'uppercase',
          letterSpacing: '0.8px',
          minWidth: '140px',
          boxShadow: isHovered ? '0 4px 12px rgba(0,0,0,0.15)' : '0 2px 6px rgba(0,0,0,0.08)',
          cursor: 'text',
          transition: 'all 0.2s ease',
          transform: isHovered ? 'translateY(-1px)' : 'translateY(0)',
        }}
        title="Click to edit"
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
          style={{ fontSize: '9px', fontWeight: '400', marginTop: '2px', opacity: 0.75, cursor: 'text' }}
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
                fontSize: '9px',
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
        onMouseDown={handleAddClick}
        onClick={handleAddClick}
        onPointerDown={handleAddClick}
        onMouseEnter={(e) => {
          e.currentTarget.style.transform = 'scale(1.05)';
          e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.2)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = 'scale(1)';
          e.currentTarget.style.boxShadow = '0 2px 6px rgba(0,0,0,0.12)';
        }}
        className="nodrag nopan"
        style={{
          padding: '7px 12px',
          background: data.borderColor || '#3b82f6',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          fontSize: '11px',
          fontWeight: '600',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '5px',
          boxShadow: '0 2px 6px rgba(0,0,0,0.12)',
          transition: 'all 0.2s ease',
          letterSpacing: '0.02em',
          whiteSpace: 'nowrap',
          pointerEvents: 'all',
          zIndex: 1000,
          position: 'relative',
        }}
        title={`Add solution to ${data.label}`}
      >
        <Plus size={14} strokeWidth={2.5} />
        Add
      </button>
    </div>
  );
};

export default SwimlaneHeader;
