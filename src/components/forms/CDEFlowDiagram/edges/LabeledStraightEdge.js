import React, { useState } from 'react';
import { EdgeLabelRenderer, getBezierPath } from 'reactflow';

/**
 * Custom step edge with editable label (90-degree turns)
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
  style,
  selected
}) => {
  // Create step path with 90-degree turns
  const createStepPath = () => {
    const midX = (sourceX + targetX) / 2;

    // Create path with right angles
    return `M ${sourceX},${sourceY} L ${midX},${sourceY} L ${midX},${targetY} L ${targetX},${targetY}`;
  };

  const edgePath = createStepPath();
  const labelX = (sourceX + targetX) / 2;
  const labelY = (sourceY + targetY) / 2;

  const [editing, setEditing] = useState(false);
  const [label, setLabel] = useState(data?.label || '');
  const [isHovered, setIsHovered] = useState(false);

  const handleSave = () => {
    if (data?.onChange) {
      data.onChange(id, label);
    }
    setEditing(false);
  };

  const edgeColor = selected ? '#3b82f6' : (style?.stroke || '#9ca3af');
  const edgeWidth = selected ? 4 : (style?.strokeWidth || 3);

  return (
    <>
      {/* Invisible wider path for easier selection */}
      <path
        id={`${id}-hitbox`}
        d={edgePath}
        fill="none"
        stroke="transparent"
        strokeWidth={24}
        strokeLinejoin="round"
        style={{ cursor: 'pointer' }}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      />
      {/* Visible path with right angles */}
      <path
        id={id}
        d={edgePath}
        fill="none"
        stroke={edgeColor}
        strokeWidth={isHovered ? edgeWidth + 1 : edgeWidth}
        strokeLinejoin="miter"
        strokeLinecap="square"
        markerEnd={markerEnd}
        style={{
          pointerEvents: 'none',
          transition: 'all 0.2s ease',
          opacity: isHovered || selected ? 1 : 0.85,
        }}
      />
      <EdgeLabelRenderer>
        <div
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            background: selected
              ? 'rgba(59, 130, 246, 0.95)'
              : isHovered
                ? 'rgba(243, 244, 246, 0.95)'
                : 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(4px)',
            color: selected ? 'white' : '#374151',
            padding: '4px 10px',
            borderRadius: '6px',
            fontSize: '12px',
            fontWeight: 600,
            border: selected ? '2px solid rgba(59, 130, 246, 0.8)' : `2px solid ${isHovered ? 'rgba(209, 213, 219, 0.8)' : 'rgba(229, 231, 235, 0.6)'}`,
            cursor: 'pointer',
            pointerEvents: 'all',
            boxShadow: selected
              ? '0 4px 12px rgba(59, 130, 246, 0.3)'
              : isHovered
                ? '0 2px 8px rgba(0,0,0,0.1)'
                : '0 1px 3px rgba(0,0,0,0.08)',
            transition: 'all 0.2s ease',
            letterSpacing: '0.02em',
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
                border: 'none',
                borderRadius: '4px',
                padding: '2px 6px',
                fontSize: '12px',
                fontWeight: 600,
                width: '90px',
                outline: 'none',
                background: 'white',
                color: '#374151',
              }}
            />
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span>{label || 'flow'}</span>
              {isHovered && !selected && <span style={{ fontSize: '10px', opacity: 0.6 }}>✎</span>}
            </div>
          )}
        </div>
      </EdgeLabelRenderer>
    </>
  );
};

export default LabeledStraightEdge;
