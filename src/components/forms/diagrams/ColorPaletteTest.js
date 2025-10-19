// Visual Test Component - Use this to verify all color combinations
// Save this as: src/components/forms/diagrams/ColorPaletteTest.js

import React from 'react';

/**
 * ColorPaletteTest - Visual verification of all color combinations
 * Import and render this component to see all 8 color themes side-by-side
 */
const ColorPaletteTest = () => {
  const colorPalettes = [
    { 
      name: 'Blue',
      lead: '#2196f3', 
      appointed: '#e3f2fd', 
      border: '#1976d2',
      buttonEdit: '#1565c0',
      buttonDelete: '#c62828',
      buttonAdd: '#2e7d32'
    },
    { 
      name: 'Green',
      lead: '#4caf50', 
      appointed: '#e8f5e8', 
      border: '#388e3c',
      buttonEdit: '#1b5e20',
      buttonDelete: '#c62828',
      buttonAdd: '#00695c'
    },
    { 
      name: 'Orange',
      lead: '#ff9800', 
      appointed: '#fff3e0', 
      border: '#f57c00',
      buttonEdit: '#e65100',
      buttonDelete: '#b71c1c',
      buttonAdd: '#1b5e20'
    },
    { 
      name: 'Purple',
      lead: '#9c27b0', 
      appointed: '#f3e5f5', 
      border: '#7b1fa2',
      buttonEdit: '#4a148c',
      buttonDelete: '#c62828',
      buttonAdd: '#1b5e20'
    },
    { 
      name: 'Red',
      lead: '#f44336', 
      appointed: '#ffebee', 
      border: '#d32f2f',
      buttonEdit: '#b71c1c',
      buttonDelete: '#6a1b9a',
      buttonAdd: '#1b5e20'
    },
    { 
      name: 'Cyan',
      lead: '#00bcd4', 
      appointed: '#e0f2f1', 
      border: '#0097a7',
      buttonEdit: '#006064',
      buttonDelete: '#c62828',
      buttonAdd: '#1b5e20'
    },
    { 
      name: 'Brown',
      lead: '#795548', 
      appointed: '#efebe9', 
      border: '#5d4037',
      buttonEdit: '#3e2723',
      buttonDelete: '#c62828',
      buttonAdd: '#1b5e20'
    },
    { 
      name: 'Blue Grey',
      lead: '#607d8b', 
      appointed: '#eceff1', 
      border: '#455a64',
      buttonEdit: '#263238',
      buttonDelete: '#c62828',
      buttonAdd: '#1b5e20'
    }
  ];

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>Color Palette Test - OrgStructureChart</h1>
      <p>Visual verification of button visibility across all 8 themes</p>
      
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: '20px',
        marginTop: '20px'
      }}>
        {colorPalettes.map((colors, index) => (
          <div key={index} style={{
            border: '2px solid #ddd',
            borderRadius: '8px',
            padding: '16px',
            backgroundColor: '#f9f9f9'
          }}>
            <h3 style={{ 
              marginTop: 0, 
              marginBottom: '16px',
              textAlign: 'center',
              color: colors.border
            }}>
              {colors.name} Theme
            </h3>
            
            {/* Lead Node */}
            <div style={{
              background: colors.lead,
              border: `2px solid ${colors.border}`,
              borderRadius: '6px',
              padding: '12px',
              marginBottom: '12px',
              textAlign: 'center'
            }}>
              <div style={{ 
                fontSize: '16px', 
                fontWeight: 'bold', 
                marginBottom: '8px',
                color: 'white'
              }}>
                Lead Appointed Party
              </div>
              <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.9)', marginBottom: '12px' }}>
                Lead Role
              </div>
              
              {/* Buttons */}
              <div style={{ display: 'flex', gap: '6px', justifyContent: 'center', flexWrap: 'wrap' }}>
                <button style={{
                  fontSize: '11px',
                  padding: '4px 8px',
                  backgroundColor: colors.buttonEdit,
                  color: 'white',
                  border: 'none',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.3)'
                }}>
                  Edit
                </button>
                <button style={{
                  fontSize: '11px',
                  padding: '4px 8px',
                  backgroundColor: colors.buttonDelete,
                  color: 'white',
                  border: 'none',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.3)'
                }}>
                  Delete
                </button>
                <button style={{
                  fontSize: '11px',
                  padding: '4px 8px',
                  backgroundColor: colors.buttonAdd,
                  color: 'white',
                  border: 'none',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.3)'
                }}>
                  Add
                </button>
              </div>
            </div>
            
            {/* Appointed Party Node */}
            <div style={{
              background: colors.appointed,
              border: `1px solid ${colors.border}`,
              borderRadius: '4px',
              padding: '8px'
            }}>
              <div style={{ 
                fontSize: '14px', 
                fontWeight: 'bold', 
                marginBottom: '4px',
                color: colors.border
              }}>
                Appointed Party
              </div>
              <div style={{ 
                fontSize: '12px', 
                color: colors.border, 
                marginBottom: '8px',
                opacity: 0.8
              }}>
                Role/Service
              </div>
              
              {/* Buttons */}
              <div style={{ display: 'flex', gap: '4px' }}>
                <button style={{
                  fontSize: '10px',
                  padding: '3px 6px',
                  backgroundColor: colors.buttonEdit,
                  color: 'white',
                  border: 'none',
                  borderRadius: '2px',
                  cursor: 'pointer',
                  boxShadow: '0 1px 2px rgba(0,0,0,0.3)'
                }}>
                  Edit
                </button>
                <button style={{
                  fontSize: '10px',
                  padding: '3px 6px',
                  backgroundColor: colors.buttonDelete,
                  color: 'white',
                  border: 'none',
                  borderRadius: '2px',
                  cursor: 'pointer',
                  boxShadow: '0 1px 2px rgba(0,0,0,0.3)'
                }}>
                  Del
                </button>
              </div>
            </div>
            
            {/* Color Info */}
            <div style={{
              marginTop: '12px',
              fontSize: '11px',
              color: '#666',
              borderTop: '1px solid #ddd',
              paddingTop: '8px'
            }}>
              <div><strong>Lead:</strong> {colors.lead}</div>
              <div><strong>Edit:</strong> {colors.buttonEdit}</div>
              <div><strong>Delete:</strong> {colors.buttonDelete}</div>
              <div><strong>Add:</strong> {colors.buttonAdd}</div>
            </div>
          </div>
        ))}
      </div>
      
      {/* Instructions */}
      <div style={{
        marginTop: '40px',
        padding: '20px',
        backgroundColor: '#e3f2fd',
        borderRadius: '8px',
        border: '2px solid #2196f3'
      }}>
        <h2 style={{ marginTop: 0 }}>Testing Checklist</h2>
        <ul>
          <li>✓ All buttons should be clearly visible on lead backgrounds</li>
          <li>✓ Edit buttons use theme-specific dark colors</li>
          <li>✓ Delete buttons use dark red (or purple in red theme)</li>
          <li>✓ Add buttons use dark green (or teal in green theme)</li>
          <li>✓ Box shadows make buttons "pop" from backgrounds</li>
          <li>✓ Text on buttons is white and readable</li>
        </ul>
        
        <h3>Accessibility Check</h3>
        <p>Use Chrome DevTools to verify contrast ratios:</p>
        <ol>
          <li>Right-click a button → Inspect</li>
          <li>Check the "Contrast" section in Styles panel</li>
          <li>Verify ratio is ≥ 4.5:1 (WCAG AA)</li>
        </ol>
        
        <h3>Usage</h3>
        <p>To test this component in your app:</p>
        <pre style={{
          backgroundColor: '#263238',
          color: '#aed581',
          padding: '12px',
          borderRadius: '4px',
          overflow: 'auto'
        }}>
{`// In your App.js or test route:
import ColorPaletteTest from './components/forms/diagrams/ColorPaletteTest';

// Add to your routes or component:
<ColorPaletteTest />`}
        </pre>
      </div>
    </div>
  );
};

export default ColorPaletteTest;
