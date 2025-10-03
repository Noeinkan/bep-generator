import React, { useState } from 'react';
import { Handle, Position } from 'reactflow';

/**
 * Solution node component with editing capability and horizontal handles
 */
const SolutionNode = React.memo(({ data, id }) => {
  const [editing, setEditing] = useState(false);
  const [label, setLabel] = useState(data.label);

  const handleSave = () => {
    if (data.onChange) {
      data.onChange(id, label);
    }
    setEditing(false);
  };

  const nodeStyle = data.nodeStyle || {
    background: '#ffffff',
    borderColor: '#6b7280',
    textColor: '#000000'
  };

  return (
    <div
      style={{
        padding: '10px 14px',
        border: `1px solid ${nodeStyle.borderColor}`,
        borderRadius: '6px',
        background: nodeStyle.background,
        color: nodeStyle.textColor,
        minWidth: '160px',
        textAlign: 'center',
        fontSize: '14px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        position: 'relative',
      }}
      title="Click text to edit"
    >
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: nodeStyle.borderColor, left: '-8px' }}
      />
      {editing ? (
        <input
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          onBlur={handleSave}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSave();
            if (e.key === 'Escape') { setLabel(data.label); setEditing(false); }
          }}
          className="nodrag"
          autoFocus
          style={{
            background: nodeStyle.background,
            color: nodeStyle.textColor,
            border: `1px solid ${nodeStyle.borderColor}`,
            borderRadius: '4px',
            padding: '4px 8px',
            width: '100%',
            textAlign: 'center',
            fontSize: '14px',
          }}
        />
      ) : (
        <div
          onClick={(e) => {
            e.stopPropagation();
            setEditing(true);
          }}
          className="nodrag nopan"
          style={{ cursor: 'text', userSelect: 'none' }}
        >
          {data.label}
        </div>
      )}
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: nodeStyle.borderColor, right: '-8px' }}
      />
    </div>
  );
});

SolutionNode.displayName = 'SolutionNode';

export default SolutionNode;
