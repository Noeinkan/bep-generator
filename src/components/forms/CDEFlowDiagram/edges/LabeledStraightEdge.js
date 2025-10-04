import React, { useState } from 'react';
import { EdgeLabelRenderer } from 'reactflow';

/**
 * Custom step edge with editable label and style controls
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
  const [editingStyle, setEditingStyle] = useState(false);
  const [label, setLabel] = useState(data?.label || '');
  const [isHovered, setIsHovered] = useState(false);

  // Edge style settings
  const [lineStyle, setLineStyle] = useState(data?.lineStyle || 'solid');
  const [lineColor, setLineColor] = useState(data?.lineColor || style?.stroke || '#9ca3af');
  const [arrowType, setArrowType] = useState(data?.arrowType || 'forward');

  const handleSave = () => {
    if (data?.onChange) {
      data.onChange(id, label, { lineStyle, lineColor, arrowType });
    }
    setEditing(false);
  };

  const handleStyleSave = () => {
    if (data?.onChange) {
      data.onChange(id, label, { lineStyle, lineColor, arrowType });
    }
    setEditingStyle(false);
  };

  const edgeColor = selected ? '#3b82f6' : lineColor;
  const edgeWidth = selected ? 4 : (style?.strokeWidth || 3);

  // Get stroke dash array based on line style
  const getStrokeDashArray = () => {
    switch (lineStyle) {
      case 'dashed':
        return '10 5';
      case 'dotted':
        return '2 4';
      default:
        return 'none';
    }
  };

  // Get marker type based on arrow configuration
  const getMarkerEnd = () => {
    if (arrowType === 'none') return undefined;
    if (arrowType === 'bidirectional') return 'url(#arrow)';
    return markerEnd || 'url(#arrow)';
  };

  const getMarkerStart = () => {
    return arrowType === 'bidirectional' ? 'url(#arrow-start)' : undefined;
  };

  return (
    <>
      {/* SVG definitions for arrow markers */}
      <defs>
        <marker
          id="arrow"
          markerWidth="10"
          markerHeight="10"
          refX="9"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M0,0 L0,6 L9,3 z" fill={edgeColor} />
        </marker>
        <marker
          id="arrow-start"
          markerWidth="10"
          markerHeight="10"
          refX="0"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M9,0 L9,6 L0,3 z" fill={edgeColor} />
        </marker>
      </defs>

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
        strokeDasharray={getStrokeDashArray()}
        markerEnd={getMarkerEnd()}
        markerStart={getMarkerStart()}
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
            background: 'transparent',
            color: selected ? '#3b82f6' : '#374151',
            padding: '4px 10px',
            borderRadius: '6px',
            fontSize: '12px',
            fontWeight: 600,
            border: 'none',
            cursor: 'pointer',
            pointerEvents: 'all',
            transition: 'all 0.2s ease',
            letterSpacing: '0.02em',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '4px',
          }}
          onDoubleClick={(e) => {
            e.stopPropagation();
            setEditing(true);
          }}
          title="Double-click to edit label, Right-click for style"
          className="nodrag nopan"
          onContextMenu={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setEditingStyle(!editingStyle);
          }}
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
              className="nodrag nopan nowheel"
              autoFocus
              onFocus={(e) => e.target.select()}
              onClick={(e) => e.stopPropagation()}
              style={{
                border: '2px solid #3b82f6',
                borderRadius: '4px',
                padding: '4px 8px',
                fontSize: '12px',
                fontWeight: 600,
                width: '100px',
                outline: 'none',
                background: 'white',
                color: '#374151',
              }}
            />
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>{label || 'flow'}</span>
              {(isHovered || selected) && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setEditingStyle(!editingStyle);
                  }}
                  style={{
                    background: '#3b82f6',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    padding: '3px 8px',
                    fontSize: '10px',
                    fontWeight: '600',
                    cursor: 'pointer',
                  }}
                  title="Edit line style"
                >
                  Style
                </button>
              )}
            </div>
          )}

          {/* Style Editor Panel */}
          {editingStyle && (
            <div
              onClick={(e) => e.stopPropagation()}
              style={{
                background: 'white',
                border: '1.5px solid #e5e7eb',
                borderRadius: '8px',
                padding: '8px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                marginTop: '4px',
                minWidth: '150px',
              }}
            >
              <div style={{ fontSize: '10px', fontWeight: '600', marginBottom: '6px', color: '#6b7280' }}>
                LINE STYLE
              </div>
              <select
                value={lineStyle}
                onChange={(e) => { setLineStyle(e.target.value); handleStyleSave(); }}
                style={{
                  width: '100%',
                  padding: '4px 6px',
                  fontSize: '11px',
                  border: '1px solid #d1d5db',
                  borderRadius: '4px',
                  marginBottom: '6px',
                }}
              >
                <option value="solid">Solid</option>
                <option value="dashed">Dashed</option>
                <option value="dotted">Dotted</option>
              </select>

              <div style={{ fontSize: '10px', fontWeight: '600', marginBottom: '6px', color: '#6b7280' }}>
                COLOR
              </div>
              <input
                type="color"
                value={lineColor}
                onChange={(e) => { setLineColor(e.target.value); handleStyleSave(); }}
                style={{
                  width: '100%',
                  height: '28px',
                  border: '1px solid #d1d5db',
                  borderRadius: '4px',
                  marginBottom: '6px',
                  cursor: 'pointer',
                }}
              />

              <div style={{ fontSize: '10px', fontWeight: '600', marginBottom: '6px', color: '#6b7280' }}>
                ARROWS
              </div>
              <select
                value={arrowType}
                onChange={(e) => { setArrowType(e.target.value); handleStyleSave(); }}
                style={{
                  width: '100%',
                  padding: '4px 6px',
                  fontSize: '11px',
                  border: '1px solid #d1d5db',
                  borderRadius: '4px',
                }}
              >
                <option value="forward">Forward →</option>
                <option value="bidirectional">Bidirectional ↔</option>
                <option value="none">None —</option>
              </select>
            </div>
          )}
        </div>
      </EdgeLabelRenderer>
    </>
  );
};

export default LabeledStraightEdge;
