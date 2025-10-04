import React, { useState } from 'react';
import { Download, Upload, RotateCcw, Trash2, Plus, Settings as SettingsIcon, Minimize2, ChevronDown } from 'lucide-react';

/**
 * Compact, modern toolbar with icon buttons and dropdown menus
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
  const [hoveredButton, setHoveredButton] = useState(null);
  const [showMoreMenu, setShowMoreMenu] = useState(false);

  const iconButtonStyle = (buttonId, variant = 'default') => {
    const variants = {
      default: {
        base: '#ffffff',
        hover: '#f3f4f6',
        color: '#374151',
        border: '#d1d5db',
      },
      primary: {
        base: '#3b82f6',
        hover: '#2563eb',
        color: '#ffffff',
        border: '#3b82f6',
      },
      success: {
        base: '#10b981',
        hover: '#059669',
        color: '#ffffff',
        border: '#10b981',
      },
    };

    const v = variants[variant];
    const isHovered = hoveredButton === buttonId;

    return {
      padding: '10px',
      background: isHovered ? v.hover : v.base,
      color: v.color,
      border: `1.5px solid ${v.border}`,
      borderRadius: '8px',
      fontSize: '13px',
      fontWeight: '600',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '6px',
      transition: 'all 0.15s ease',
      boxShadow: isHovered
        ? '0 2px 8px rgba(0,0,0,0.1)'
        : '0 1px 3px rgba(0,0,0,0.05)',
      minWidth: '40px',
      height: '40px',
    };
  };

  const DropdownMenu = ({ show, onClose, children }) => {
    // Close menu when clicking outside
    React.useEffect(() => {
      if (!show) return;

      const handleClickOutside = (e) => {
        if (!e.target.closest('.dropdown-menu-container') && !e.target.closest('.more-button')) {
          onClose();
        }
      };

      // Add small delay to prevent immediate close
      const timer = setTimeout(() => {
        document.addEventListener('click', handleClickOutside);
      }, 100);

      return () => {
        clearTimeout(timer);
        document.removeEventListener('click', handleClickOutside);
      };
    }, [show, onClose]);

    if (!show) return null;

    return (
      <div
        className="dropdown-menu-container"
        style={{
          position: 'absolute',
          top: '50px',
          right: '0',
          background: 'white',
          borderRadius: '10px',
          boxShadow: '0 8px 24px rgba(0,0,0,0.15)',
          border: '1.5px solid #e5e7eb',
          padding: '6px',
          minWidth: '180px',
          zIndex: 1000,
        }}
      >
        {children}
      </div>
    );
  };

  const menuItemStyle = (isHovered, variant = 'default') => ({
    padding: '10px 12px',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    cursor: 'pointer',
    borderRadius: '6px',
    fontSize: '13px',
    fontWeight: '500',
    background: isHovered
      ? variant === 'danger' ? '#fef2f2' : '#f3f4f6'
      : 'transparent',
    color: variant === 'danger' ? '#dc2626' : '#374151',
    transition: 'all 0.15s ease',
  });

  return (
    <div style={{
      padding: '12px 16px',
      background: '#ffffff',
      borderBottom: '1.5px solid #e5e7eb',
      display: 'flex',
      gap: '8px',
      alignItems: 'center',
    }}>
      {/* Primary Action - Add Swimlane */}
      <button
        onClick={onAddSwimlane}
        onMouseEnter={() => setHoveredButton('add')}
        onMouseLeave={() => setHoveredButton(null)}
        style={{
          ...iconButtonStyle('add', 'primary'),
          padding: '10px 16px',
        }}
        title="Add swimlane"
      >
        <Plus size={18} strokeWidth={2.5} />
        <span>Add Lane</span>
      </button>

      {/* Compact Icon Buttons */}
      <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
        <button
          onClick={onReset}
          onMouseEnter={() => setHoveredButton('reset')}
          onMouseLeave={() => setHoveredButton(null)}
          style={iconButtonStyle('reset')}
          title="Reset to default"
        >
          <RotateCcw size={18} strokeWidth={2.5} />
        </button>

        <button
          onClick={onExport}
          onMouseEnter={() => setHoveredButton('export')}
          onMouseLeave={() => setHoveredButton(null)}
          style={iconButtonStyle('export')}
          title="Export diagram"
        >
          <Download size={18} strokeWidth={2.5} />
        </button>

        <label
          onMouseEnter={() => setHoveredButton('import')}
          onMouseLeave={() => setHoveredButton(null)}
          style={iconButtonStyle('import')}
          title="Import diagram"
        >
          <Upload size={18} strokeWidth={2.5} />
          <input
            type="file"
            accept=".json"
            onChange={onImport}
            style={{ display: 'none' }}
          />
        </label>
      </div>

      {/* More Menu Dropdown */}
      <div style={{ position: 'relative', marginLeft: '4px' }}>
        <button
          className="more-button"
          onClick={() => setShowMoreMenu(!showMoreMenu)}
          onMouseEnter={() => setHoveredButton('more')}
          onMouseLeave={() => setHoveredButton(null)}
          style={{
            ...iconButtonStyle('more'),
            padding: '10px 14px',
          }}
          title="More actions"
        >
          <span style={{ fontSize: '13px', fontWeight: '600' }}>More</span>
          <ChevronDown size={16} strokeWidth={2.5} />
        </button>

        <DropdownMenu show={showMoreMenu} onClose={() => setShowMoreMenu(false)}>
          <div
            onClick={() => {
              onRemoveSwimlane();
              setShowMoreMenu(false);
            }}
            onMouseEnter={() => setHoveredButton('menuRemove')}
            onMouseLeave={() => setHoveredButton(null)}
            style={menuItemStyle(hoveredButton === 'menuRemove', 'default')}
          >
            <Minimize2 size={16} strokeWidth={2.5} />
            <span>Remove Last Lane</span>
          </div>
          <div
            onClick={() => {
              onClear();
              setShowMoreMenu(false);
            }}
            onMouseEnter={() => setHoveredButton('menuClear')}
            onMouseLeave={() => setHoveredButton(null)}
            style={menuItemStyle(hoveredButton === 'menuClear', 'danger')}
          >
            <Trash2 size={16} strokeWidth={2.5} />
            <span>Clear All Solutions</span>
          </div>
        </DropdownMenu>
      </div>

      {/* Settings Toggle - Right Aligned */}
      <button
        onClick={onToggleSettings}
        onMouseEnter={() => setHoveredButton('settings')}
        onMouseLeave={() => setHoveredButton(null)}
        style={{
          ...iconButtonStyle('settings', showSettings ? 'success' : 'default'),
          marginLeft: 'auto',
        }}
        title="Customize styles"
      >
        <SettingsIcon size={18} strokeWidth={2.5} />
      </button>
    </div>
  );
};

export default DiagramToolbar;
