import React, { useState } from 'react';
import { Handle, Position, NodeResizer } from 'reactflow';

/**
 * Solution node component with editable header, description, and resizing
 */
const SolutionNode = React.memo(({ data, id, selected }) => {
  const [editingHeader, setEditingHeader] = useState(false);
  const [editingDescription, setEditingDescription] = useState(false);
  const [label, setLabel] = useState(data.label);
  const [description, setDescription] = useState(data.description || '');
  const [isHovered, setIsHovered] = useState(false);

  const handleSaveHeader = () => {
    if (data.onChange) {
      data.onChange(id, label);
    }
    setEditingHeader(false);
  };

  const handleSaveDescription = () => {
    if (data.onDescriptionChange) {
      data.onDescriptionChange(id, description);
    }
    setEditingDescription(false);
  };

  const nodeStyle = data.nodeStyle || {
    background: '#ffffff',
    borderColor: '#6b7280',
    textColor: '#000000'
  };

  return (
    <>
      <NodeResizer
        color="#3b82f6"
        isVisible={selected}
        minWidth={180}
        minHeight={100}
        lineStyle={{ strokeWidth: 2 }}
        handleStyle={{
          width: 12,
          height: 12,
          borderRadius: '50%',
          backgroundColor: '#3b82f6',
          border: '2px solid white',
        }}
      />
      <div
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        style={{
          border: `2px solid ${selected ? '#3b82f6' : nodeStyle.borderColor}`,
          borderRadius: '10px',
          background: nodeStyle.background,
          color: nodeStyle.textColor,
          width: '100%',
          height: '100%',
          minWidth: '180px',
          fontSize: '14px',
          boxShadow: selected
            ? '0 8px 20px rgba(59, 130, 246, 0.3), 0 4px 10px rgba(0, 0, 0, 0.1)'
            : isHovered
              ? '0 6px 16px rgba(0,0,0,0.15)'
              : '0 3px 8px rgba(0,0,0,0.1)',
          position: 'relative',
          overflow: 'hidden',
          transition: 'box-shadow 0.2s ease, border 0.2s ease',
        }}
      >
      <Handle
        type="target"
        position={Position.Left}
        style={{
          background: selected ? '#3b82f6' : nodeStyle.borderColor,
          left: '-10px',
          width: '16px',
          height: '16px',
          border: '3px solid white',
          transition: 'all 0.2s ease',
          opacity: 1,
          zIndex: 10,
        }}
      />

      {/* Header section - DRAGGABLE AREA */}
      <div
        style={{
          background: selected ? '#3b82f6' : nodeStyle.borderColor,
          color: '#ffffff',
          padding: '12px 14px',
          fontWeight: '600',
          fontSize: '14px',
          cursor: editingHeader ? 'text' : 'grab',
          transition: 'background 0.2s ease',
          letterSpacing: '0.01em',
          userSelect: editingHeader ? 'text' : 'none',
        }}
        onDoubleClick={(e) => {
          e.stopPropagation();
          setEditingHeader(true);
        }}
        title="Drag to move • Double-click to edit"
      >
        {editingHeader ? (
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            onBlur={handleSaveHeader}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSaveHeader();
              if (e.key === 'Escape') { setLabel(data.label); setEditingHeader(false); }
            }}
            className="nodrag nopan nowheel"
            autoFocus
            onFocus={(e) => e.target.select()}
            style={{
              background: 'rgba(255,255,255,0.98)',
              color: '#1f2937',
              border: 'none',
              borderRadius: '4px',
              padding: '4px 8px',
              width: '100%',
              fontSize: '14px',
              fontWeight: '600',
              outline: 'none',
              cursor: 'text',
            }}
          />
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
            <span style={{ flex: 1 }}>{data.label}</span>
            {isHovered && <span style={{ fontSize: '11px', opacity: 0.8, flexShrink: 0 }}>⋮⋮</span>}
          </div>
        )}
      </div>

      {/* Description section - NON-DRAGGABLE */}
      <div
        style={{
          padding: '12px 14px',
          minHeight: '60px',
          fontSize: '13px',
          color: description || editingDescription ? '#4b5563' : '#9ca3af',
          cursor: editingDescription ? 'text' : 'pointer',
          lineHeight: '1.5',
          background: editingDescription ? '#f9fafb' : 'transparent',
          transition: 'background 0.2s ease',
        }}
        onClick={(e) => {
          if (!editingDescription) {
            e.stopPropagation();
            setEditingDescription(true);
          }
        }}
        className="nodrag nopan"
        title="Click to edit description"
      >
        {editingDescription ? (
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            onBlur={handleSaveDescription}
            onKeyDown={(e) => {
              if (e.key === 'Escape') { setDescription(data.description || ''); setEditingDescription(false); }
            }}
            className="nodrag nopan nowheel"
            autoFocus
            onFocus={(e) => e.target.select()}
            placeholder="Add a description..."
            style={{
              background: '#ffffff',
              color: '#4b5563',
              border: `2px solid ${nodeStyle.borderColor}`,
              borderRadius: '6px',
              padding: '8px',
              width: '100%',
              fontSize: '13px',
              minHeight: '50px',
              resize: 'vertical',
              fontFamily: 'inherit',
              lineHeight: '1.5',
              outline: 'none',
              cursor: 'text',
            }}
          />
        ) : (
          <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '8px' }}>
            <span style={{ flex: 1, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {description || 'Add description...'}
            </span>
            {isHovered && description && <span style={{ fontSize: '11px', opacity: 0.5, flexShrink: 0 }}>✎</span>}
          </div>
        )}
      </div>

      <Handle
        type="source"
        position={Position.Right}
        style={{
          background: selected ? '#3b82f6' : nodeStyle.borderColor,
          right: '-10px',
          width: '16px',
          height: '16px',
          border: '3px solid white',
          transition: 'all 0.2s ease',
          opacity: 1,
          zIndex: 10,
        }}
      />
    </div>
    </>
  );
});

SolutionNode.displayName = 'SolutionNode';

export default SolutionNode;
