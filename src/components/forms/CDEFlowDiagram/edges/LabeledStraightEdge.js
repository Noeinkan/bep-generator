import React, { useState } from 'react';
import { EdgeLabelRenderer, getStraightPath } from 'reactflow';

/**
 * Custom straight edge with editable label (horizontal connections)
 */
const LabeledStraightEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  markerEnd,
  style
}) => {
  const [edgePath, labelX, labelY] = getStraightPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const [editing, setEditing] = useState(false);
  const [label, setLabel] = useState(data?.label || '');

  const handleSave = () => {
    if (data?.onChange) {
      data.onChange(id, label);
    }
    setEditing(false);
  };

  return (
    <>
      <path
        id={id}
        d={edgePath}
        fill="none"
        stroke={style?.stroke || '#9ca3af'}
        strokeWidth={style?.strokeWidth || 2}
        markerEnd={markerEnd}
      />
      <EdgeLabelRenderer>
        <div
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            background: 'white',
            padding: '2px 8px',
            borderRadius: '4px',
            fontSize: '12px',
            fontWeight: 500,
            border: '1px solid #d1d5db',
            cursor: 'pointer',
            pointerEvents: 'all',
          }}
          onDoubleClick={(e) => {
            e.stopPropagation();
            setEditing(true);
          }}
          title="Double-click to edit label"
          className="nodrag nopan"
        >
          {editing ? (
            <input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              onBlur={handleSave}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSave();
                if (e.key === 'Escape') { setLabel(data?.label || ''); setEditing(false); }
              }}
              autoFocus
              onClick={(e) => e.stopPropagation()}
              style={{
                border: '1px solid #3b82f6',
                borderRadius: '2px',
                padding: '2px 4px',
                fontSize: '12px',
                width: '80px',
              }}
            />
          ) : (
            <span>{label || 'flow'}</span>
          )}
        </div>
      </EdgeLabelRenderer>
    </>
  );
};

export default LabeledStraightEdge;
