import React, { useCallback } from 'react';
import {
  Bold,
  Italic,
  Underline,
  AlignLeft,
  AlignCenter,
  AlignRight,
  List,
  ListOrdered,
  Type,
  X,
  Heading,
  Link as LinkIcon,
  Code,
  Quote,
  Image as ImageIcon,
  RotateCcw,
  RotateCw,
  Eye,
  EyeOff
} from 'lucide-react';

// Reusable ToolbarButton component
const ToolbarButton = ({ icon: Icon, onClick, title, isActive = false }) => (
  <button
    onClick={onClick}
    className={`p-2 rounded border border-gray-300 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
      isActive ? 'bg-blue-100 text-blue-600' : ''
    }`}
    title={title}
    aria-label={title}
    type="button"
  >
    <Icon size={16} />
  </button>
);

const FormattingToolbar = ({
  onFormat,
  show = true,
  onClose,
  position = { top: 0, left: 0 },
  currentFont = 'default',
  currentAlignment = 'left',
  activeFormats = { bold: false, italic: false, underline: false },
  compact = false,
  onPreviewToggle,
  isPreviewMode,
  onUndo,
  onRedo,
}) => {
  const fontOptions = [
    { value: 'default', label: 'Default' },
    { value: 'arial', label: 'Arial' },
    { value: 'times', label: 'Times New Roman' },
    { value: 'courier', label: 'Courier New' },
    { value: 'georgia', label: 'Georgia' },
    { value: 'verdana', label: 'Verdana' },
  ];

  const fontSizeOptions = [
    { value: '12', label: '12px' },
    { value: '14', label: '14px' },
    { value: '16', label: '16px' },
    { value: '18', label: '18px' },
    { value: '24', label: '24px' },
    { value: '32', label: '32px' },
  ];

  const handleFormat = useCallback((type, value = null) => {
    onFormat(type, value);
  }, [onFormat]);

  if (!show) return null;

  if (compact) {
    return (
      <div
        className="bg-gray-100 border border-gray-300 border-b-0 rounded-t-lg p-2"
        role="toolbar"
        aria-label="Text formatting toolbar"
      >
        <div className="flex items-center justify-between space-x-2">
          {/* Left side - Font controls */}
          <div className="flex items-center space-x-2">
            <select
              value={currentFont}
              onChange={(e) => handleFormat('font', e.target.value)}
              className="text-xs border border-gray-300 rounded px-1 py-1 bg-white focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
              aria-label="Font"
            >
              {fontOptions.map((font) => (
                <option key={font.value} value={font.value}>
                  {font.label}
                </option>
              ))}
            </select>
            <select
              onChange={(e) => handleFormat('fontSize', e.target.value)}
              className="text-xs border border-gray-300 rounded px-1 py-1 bg-white focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
              aria-label="Size"
            >
              <option value="">Size</option>
              {fontSizeOptions.map((size) => (
                <option key={size.value} value={size.value}>
                  {size.label}
                </option>
              ))}
            </select>
          </div>

          {/* Center - Text Style Buttons */}
          <div className="flex items-center space-x-1">
            <ToolbarButton
              icon={Bold}
              onClick={() => handleFormat('bold')}
              title="Bold"
              isActive={activeFormats.bold}
            />
            <ToolbarButton
              icon={Italic}
              onClick={() => handleFormat('italic')}
              title="Italic"
              isActive={activeFormats.italic}
            />
            <ToolbarButton
              icon={Underline}
              onClick={() => handleFormat('underline')}
              title="Underline"
              isActive={activeFormats.underline}
            />
          </div>

          {/* Right side - Alignment, Lists, and More */}
          <div className="flex items-center space-x-1">
            <ToolbarButton icon={AlignLeft} onClick={() => handleFormat('align', 'left')} title="Align Left" isActive={currentAlignment === 'left'} />
            <ToolbarButton icon={AlignCenter} onClick={() => handleFormat('align', 'center')} title="Align Center" isActive={currentAlignment === 'center'} />
            <ToolbarButton icon={AlignRight} onClick={() => handleFormat('align', 'right')} title="Align Right" isActive={currentAlignment === 'right'} />
            <div className="w-px h-4 bg-gray-300 mx-1"></div>
            <ToolbarButton icon={List} onClick={() => handleFormat('list', 'bullet')} title="Bullet List" />
            <ToolbarButton icon={ListOrdered} onClick={() => handleFormat('list', 'numbered')} title="Numbered List" />
            <ToolbarButton icon={Heading} onClick={() => handleFormat('heading')} title="Heading" />
            <ToolbarButton icon={LinkIcon} onClick={() => handleFormat('link')} title="Insert Link" />
            <ToolbarButton icon={Code} onClick={() => handleFormat('code')} title="Inline Code" />
            <ToolbarButton icon={Quote} onClick={() => handleFormat('blockquote')} title="Blockquote" />
            <ToolbarButton icon={ImageIcon} onClick={() => handleFormat('image')} title="Insert Image" />
            <ToolbarButton icon={RotateCcw} onClick={onUndo} title="Undo (Ctrl+Z)" />
            <ToolbarButton icon={RotateCw} onClick={onRedo} title="Redo (Ctrl+Y)" />
            <ToolbarButton icon={isPreviewMode ? EyeOff : Eye} onClick={onPreviewToggle} title={isPreviewMode ? 'Edit Mode' : 'Preview Mode'} />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="absolute z-50 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg p-3 w-80 max-w-sm"
      style={{ top: position.top, left: position.left }}
      role="toolbar"
      aria-label="Text formatting toolbar"
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-200">
          Text Formatting
        </span>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
          title="Close toolbar"
          aria-label="Close toolbar"
          type="button"
        >
          <X size={16} className="text-gray-600 dark:text-gray-200" />
        </button>
      </div>

      <div className="space-y-3">
        {/* Font Selection */}
        <div className="flex items-center space-x-2">
          <Type size={16} className="text-gray-600 dark:text-gray-200" />
          <select
            value={currentFont}
            onChange={(e) => handleFormat('font', e.target.value)}
            className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            aria-label="Select font"
          >
            {fontOptions.map((font) => (
              <option key={font.value} value={font.value}>
                {font.label}
              </option>
            ))}
          </select>
        </div>

        {/* Font Size Selection */}
        <div className="flex items-center space-x-2">
          <Type size={16} className="text-gray-600 dark:text-gray-200" />
          <select
            onChange={(e) => handleFormat('fontSize', e.target.value)}
            className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            aria-label="Select font size"
          >
            {fontSizeOptions.map((size) => (
              <option key={size.value} value={size.value}>
                {size.label}
              </option>
            ))}
          </select>
        </div>

        {/* Text Style Buttons */}
        <div className="flex items-center space-x-1">
          <ToolbarButton
            icon={Bold}
            onClick={() => handleFormat('bold')}
            title="Bold"
            isActive={activeFormats.bold}
          />
          <ToolbarButton
            icon={Italic}
            onClick={() => handleFormat('italic')}
            title="Italic"
            isActive={activeFormats.italic}
          />
          <ToolbarButton
            icon={Underline}
            onClick={() => handleFormat('underline')}
            title="Underline"
            isActive={activeFormats.underline}
          />
        </div>

        {/* Alignment Buttons */}
        <div className="flex items-center space-x-1">
          <ToolbarButton
            icon={AlignLeft}
            onClick={() => handleFormat('align', 'left')}
            title="Align Left"
            isActive={currentAlignment === 'left'}
          />
          <ToolbarButton
            icon={AlignCenter}
            onClick={() => handleFormat('align', 'center')}
            title="Align Center"
            isActive={currentAlignment === 'center'}
          />
          <ToolbarButton
            icon={AlignRight}
            onClick={() => handleFormat('align', 'right')}
            title="Align Right"
            isActive={currentAlignment === 'right'}
          />
        </div>

        {/* List Buttons */}
        <div className="flex items-center space-x-1">
          <ToolbarButton
            icon={List}
            onClick={() => handleFormat('list', 'bullet')}
            title="Bullet List"
          />
          <ToolbarButton
            icon={ListOrdered}
            onClick={() => handleFormat('list', 'numbered')}
            title="Numbered List"
          />
        </div>
      </div>
    </div>
  );
};

// Optional: Add React.memo to prevent unnecessary re-renders
export default React.memo(FormattingToolbar);