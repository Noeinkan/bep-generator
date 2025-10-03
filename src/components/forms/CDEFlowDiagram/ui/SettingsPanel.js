import React from 'react';
import { SWIMLANES } from '../CDEFlowDiagram.constants';

/**
 * Settings panel for customizing swimlanes and nodes
 */
const SettingsPanel = ({
  swimlaneCustom,
  nodeStyle,
  onSwimlaneCustomChange,
  onNodeStyleChange
}) => {
  return (
    <div style={{
      padding: '16px',
      background: '#f3f4f6',
      borderBottom: '1px solid #e5e7eb',
      maxHeight: '300px',
      overflowY: 'auto'
    }}>
      <h3 style={{ fontSize: '14px', fontWeight: '600', marginBottom: '12px' }}>Customize Swimlanes</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
        {SWIMLANES.map(lane => {
          const custom = swimlaneCustom[lane.id] || {};
          return (
            <div key={lane.id} style={{
              padding: '12px',
              background: 'white',
              borderRadius: '6px',
              border: '1px solid #e5e7eb'
            }}>
              <div style={{ fontSize: '12px', fontWeight: '600', marginBottom: '8px' }}>{lane.label}</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div>
                  <label style={{ fontSize: '11px', display: 'block', marginBottom: '4px' }}>Background Color</label>
                  <input
                    type="color"
                    value={custom.color || lane.color}
                    onChange={(e) => onSwimlaneCustomChange(lane.id, 'color', e.target.value)}
                    style={{ width: '100%', height: '32px', border: '1px solid #d1d5db', borderRadius: '4px' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '11px', display: 'block', marginBottom: '4px' }}>Border Color</label>
                  <input
                    type="color"
                    value={custom.borderColor || lane.borderColor}
                    onChange={(e) => onSwimlaneCustomChange(lane.id, 'borderColor', e.target.value)}
                    style={{ width: '100%', height: '32px', border: '1px solid #d1d5db', borderRadius: '4px' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '11px', display: 'block', marginBottom: '4px' }}>Text Color</label>
                  <input
                    type="color"
                    value={custom.textColor || lane.textColor}
                    onChange={(e) => onSwimlaneCustomChange(lane.id, 'textColor', e.target.value)}
                    style={{ width: '100%', height: '32px', border: '1px solid #d1d5db', borderRadius: '4px' }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <h3 style={{ fontSize: '14px', fontWeight: '600', marginTop: '16px', marginBottom: '12px' }}>Customize Solution Nodes</h3>
      <div style={{
        padding: '12px',
        background: 'white',
        borderRadius: '6px',
        border: '1px solid #e5e7eb',
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '12px'
      }}>
        <div>
          <label style={{ fontSize: '11px', display: 'block', marginBottom: '4px' }}>Background Color</label>
          <input
            type="color"
            value={nodeStyle.background}
            onChange={(e) => onNodeStyleChange('background', e.target.value)}
            style={{ width: '100%', height: '32px', border: '1px solid #d1d5db', borderRadius: '4px' }}
          />
        </div>
        <div>
          <label style={{ fontSize: '11px', display: 'block', marginBottom: '4px' }}>Border Color</label>
          <input
            type="color"
            value={nodeStyle.borderColor}
            onChange={(e) => onNodeStyleChange('borderColor', e.target.value)}
            style={{ width: '100%', height: '32px', border: '1px solid #d1d5db', borderRadius: '4px' }}
          />
        </div>
        <div>
          <label style={{ fontSize: '11px', display: 'block', marginBottom: '4px' }}>Text Color</label>
          <input
            type="color"
            value={nodeStyle.textColor}
            onChange={(e) => onNodeStyleChange('textColor', e.target.value)}
            style={{ width: '100%', height: '32px', border: '1px solid #d1d5db', borderRadius: '4px' }}
          />
        </div>
      </div>
    </div>
  );
};

export default SettingsPanel;
