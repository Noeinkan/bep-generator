import React, { useState, useCallback, useRef } from 'react';
import {
  Bold,
  Italic,
  Underline as UnderlineIcon,
  Strikethrough,
  Code,
  Heading1,
  Heading2,
  Heading3,
  List,
  ListOrdered,
  Quote,
  Undo,
  Redo,
  Link as LinkIcon,
  Image as ImageIcon,
  Table as TableIcon,
  AlignLeft,
  AlignCenter,
  AlignRight,
  AlignJustify,
  Minus,
  Highlighter,
  Search,
  ZoomIn,
  ZoomOut,
  FileText,
  ChevronDown,
} from 'lucide-react';
import TemplateSelector from '../controls/TemplateSelector';
import TableInsertDialog from '../dialogs/TableInsertDialog';

const TipTapToolbar = ({ editor, zoom = 100, onZoomChange, onFindReplace, fieldName }) => {
  const [showLinkInput, setShowLinkInput] = useState(false);
  const [linkUrl, setLinkUrl] = useState('');
  const [showColorPicker, setShowColorPicker] = useState(false);
  const [currentColor, setCurrentColor] = useState('#000000');
  const [showHighlightPicker, setShowHighlightPicker] = useState(false);
  const [currentHighlight, setCurrentHighlight] = useState('#ffff00');
  const [showTemplateSelector, setShowTemplateSelector] = useState(false);
  const [showTableDialog, setShowTableDialog] = useState(false);
  const templateButtonRef = useRef(null);

  const addLink = useCallback(() => {
    if (linkUrl && editor) {
      // If there's no text selected, we need to extend the selection to avoid creating an empty link
      const { state } = editor;
      const { from, to } = state.selection;

      if (from === to) {
        // No text selected - insert the URL as both text and link
        editor.chain().focus().insertContent(`<a href="${linkUrl}">${linkUrl}</a>`).run();
      } else {
        // Text is selected - apply link to selection
        editor.chain().focus().setLink({ href: linkUrl }).run();
      }

      setLinkUrl('');
      setShowLinkInput(false);
    }
  }, [editor, linkUrl]);

  const addImage = useCallback(() => {
    if (editor) {
      // Create a file input element
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';

      input.onchange = (e) => {
        const file = e.target.files?.[0];
        if (file) {
          const reader = new FileReader();
          reader.onload = (event) => {
            const src = event.target.result;
            if (src) {
              editor.chain().focus().setImage({ src }).run();
            }
          };
          reader.readAsDataURL(file);
        }
      };

      input.click();
    }
  }, [editor]);

  const addTable = useCallback((options) => {
    if (editor) {
      editor.chain().focus().insertTable(options).run();
    }
  }, [editor]);

  const setColor = useCallback((color) => {
    if (editor) {
      editor.chain().focus().setColor(color).run();
      setCurrentColor(color);
      setShowColorPicker(false);
    }
  }, [editor]);

  const setHighlight = useCallback((color) => {
    if (editor) {
      editor.chain().focus().setHighlight({ color }).run();
      setCurrentHighlight(color);
      setShowHighlightPicker(false);
    }
  }, [editor]);

  const handleZoomIn = useCallback(() => {
    if (onZoomChange && zoom < 200) {
      onZoomChange(Math.min(200, zoom + 10));
    }
  }, [zoom, onZoomChange]);

  const handleZoomOut = useCallback(() => {
    if (onZoomChange && zoom > 50) {
      onZoomChange(Math.max(50, zoom - 10));
    }
  }, [zoom, onZoomChange]);

  if (!editor) {
    return null;
  }

  const fonts = [
    { label: 'Default', value: '' },
    { label: 'Arial', value: 'Arial, sans-serif' },
    { label: 'Times New Roman', value: 'Times New Roman, serif' },
    { label: 'Courier New', value: 'Courier New, monospace' },
    { label: 'Georgia', value: 'Georgia, serif' },
    { label: 'Verdana', value: 'Verdana, sans-serif' },
    { label: 'Calibri', value: 'Calibri, sans-serif' },
    { label: 'Comic Sans MS', value: 'Comic Sans MS, cursive' },
  ];

  const fontSizes = ['12px', '14px', '16px', '18px', '20px', '24px', '28px', '32px', '36px', '48px'];

  const colors = [
    '#000000', '#434343', '#666666', '#999999', '#b7b7b7', '#cccccc', '#d9d9d9', '#efefef', '#f3f3f3', '#ffffff',
    '#980000', '#ff0000', '#ff9900', '#ffff00', '#00ff00', '#00ffff', '#4a86e8', '#0000ff', '#9900ff', '#ff00ff',
    '#e6b8af', '#f4cccc', '#fce5cd', '#fff2cc', '#d9ead3', '#d0e0e3', '#c9daf8', '#cfe2f3', '#d9d2e9', '#ead1dc',
    '#dd7e6b', '#ea9999', '#f9cb9c', '#ffe599', '#b6d7a8', '#a2c4c9', '#a4c2f4', '#9fc5e8', '#b4a7d6', '#d5a6bd',
    '#cc4125', '#e06666', '#f6b26b', '#ffd966', '#93c47d', '#76a5af', '#6d9eeb', '#6fa8dc', '#8e7cc3', '#c27ba0',
    '#a61c00', '#cc0000', '#e69138', '#f1c232', '#6aa84f', '#45818e', '#3c78d8', '#3d85c6', '#674ea7', '#a64d79',
    '#85200c', '#990000', '#b45f06', '#bf9000', '#38761d', '#134f5c', '#1155cc', '#0b5394', '#351c75', '#741b47',
    '#5b0f00', '#660000', '#783f04', '#7f6000', '#274e13', '#0c343d', '#1c4587', '#073763', '#20124d', '#4c1130',
  ];

  const ToolbarButton = ({ onClick, active, disabled, children, title }) => (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`p-2 rounded hover:bg-gray-200 transition-colors ${
        active ? 'bg-blue-100 text-blue-600' : 'text-gray-700'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      type="button"
    >
      {children}
    </button>
  );

  const ToolbarDivider = () => <div className="w-px h-6 bg-gray-300 mx-1" />;

  return (
    <div className="tiptap-toolbar border border-gray-300 rounded-t-lg bg-gray-50 p-2 flex flex-wrap gap-1 items-center">
      {/* Undo/Redo */}
      <ToolbarButton
        onClick={() => editor.chain().focus().undo().run()}
        disabled={!editor.can().undo()}
        title="Undo (Ctrl+Z)"
      >
        <Undo size={18} />
      </ToolbarButton>
      <ToolbarButton
        onClick={() => editor.chain().focus().redo().run()}
        disabled={!editor.can().redo()}
        title="Redo (Ctrl+Y)"
      >
        <Redo size={18} />
      </ToolbarButton>

      <ToolbarDivider />

      {/* Font Family */}
      <select
        className="px-2 py-1 border border-gray-300 rounded text-sm bg-white cursor-pointer hover:border-gray-400"
        onChange={(e) => {
          if (e.target.value) {
            editor.chain().focus().setFontFamily(e.target.value).run();
          } else {
            editor.chain().focus().unsetFontFamily().run();
          }
        }}
        title="Font Family"
      >
        {fonts.map((font) => (
          <option key={font.value} value={font.value} style={{ fontFamily: font.value }}>
            {font.label}
          </option>
        ))}
      </select>

      {/* Font Size */}
      <select
        className="px-2 py-1 border border-gray-300 rounded text-sm bg-white cursor-pointer hover:border-gray-400"
        onChange={(e) => {
          if (e.target.value) {
            editor.chain().focus().setFontSize(e.target.value).run();
          } else {
            editor.chain().focus().unsetFontSize().run();
          }
        }}
        title="Font Size"
      >
        <option value="">Default</option>
        {fontSizes.map((size) => (
          <option key={size} value={size}>
            {size}
          </option>
        ))}
      </select>

      <ToolbarDivider />

      {/* Text Formatting */}
      <ToolbarButton
        onClick={() => editor.chain().focus().toggleBold().run()}
        active={editor.isActive('bold')}
        title="Bold (Ctrl+B)"
      >
        <Bold size={18} />
      </ToolbarButton>
      <ToolbarButton
        onClick={() => editor.chain().focus().toggleItalic().run()}
        active={editor.isActive('italic')}
        title="Italic (Ctrl+I)"
      >
        <Italic size={18} />
      </ToolbarButton>
      <ToolbarButton
        onClick={() => editor.chain().focus().toggleUnderline().run()}
        active={editor.isActive('underline')}
        title="Underline (Ctrl+U)"
      >
        <UnderlineIcon size={18} />
      </ToolbarButton>
      <ToolbarButton
        onClick={() => editor.chain().focus().toggleStrike().run()}
        active={editor.isActive('strike')}
        title="Strikethrough"
      >
        <Strikethrough size={18} />
      </ToolbarButton>
      <ToolbarButton
        onClick={() => editor.chain().focus().toggleCode().run()}
        active={editor.isActive('code')}
        title="Code"
      >
        <Code size={18} />
      </ToolbarButton>

      <ToolbarDivider />

      {/* Color Picker */}
      <div className="relative">
        <button
          onClick={() => setShowColorPicker(!showColorPicker)}
          className="p-2 rounded hover:bg-gray-200 transition-colors flex items-center gap-1"
          title="Text Color"
          type="button"
        >
          <div className="w-4 h-4 border border-gray-400 rounded" style={{ backgroundColor: currentColor }} />
          <ChevronDown size={12} />
        </button>
        {showColorPicker && (
          <div className="absolute top-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg p-2 z-20">
            <div className="grid grid-cols-10 gap-1 w-48">
              {colors.map((color) => (
                <button
                  key={color}
                  onClick={() => setColor(color)}
                  className="w-5 h-5 rounded border border-gray-300 hover:scale-110 transition-transform"
                  style={{ backgroundColor: color }}
                  title={color}
                  type="button"
                />
              ))}
            </div>
            <button
              onClick={() => setShowColorPicker(false)}
              className="mt-2 w-full py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded"
              type="button"
            >
              Close
            </button>
          </div>
        )}
      </div>

      {/* Highlight Color Picker */}
      <div className="relative">
        <button
          onClick={() => setShowHighlightPicker(!showHighlightPicker)}
          className="p-2 rounded hover:bg-gray-200 transition-colors flex items-center gap-1"
          title="Highlight Color"
          type="button"
        >
          <Highlighter size={18} style={{ color: currentHighlight }} />
          <ChevronDown size={12} />
        </button>
        {showHighlightPicker && (
          <div className="absolute top-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg p-2 z-20">
            <div className="grid grid-cols-10 gap-1 w-48">
              {colors.map((color) => (
                <button
                  key={color}
                  onClick={() => setHighlight(color)}
                  className="w-5 h-5 rounded border border-gray-300 hover:scale-110 transition-transform"
                  style={{ backgroundColor: color }}
                  title={color}
                  type="button"
                />
              ))}
            </div>
            <div className="flex gap-2 mt-2">
              <button
                onClick={() => {
                  if (editor) editor.chain().focus().unsetHighlight().run();
                  setShowHighlightPicker(false);
                }}
                className="flex-1 py-1 text-xs bg-red-100 hover:bg-red-200 rounded"
                type="button"
              >
                Remove
              </button>
              <button
                onClick={() => setShowHighlightPicker(false)}
                className="flex-1 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded"
                type="button"
              >
                Close
              </button>
            </div>
          </div>
        )}
      </div>

      <ToolbarDivider />

      {/* Headings */}
      <ToolbarButton
        onClick={() => editor.chain().focus().toggleHeading({ level: 1 }).run()}
        active={editor.isActive('heading', { level: 1 })}
        title="Heading 1"
      >
        <Heading1 size={18} />
      </ToolbarButton>
      <ToolbarButton
        onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}
        active={editor.isActive('heading', { level: 2 })}
        title="Heading 2"
      >
        <Heading2 size={18} />
      </ToolbarButton>
      <ToolbarButton
        onClick={() => editor.chain().focus().toggleHeading({ level: 3 }).run()}
        active={editor.isActive('heading', { level: 3 })}
        title="Heading 3"
      >
        <Heading3 size={18} />
      </ToolbarButton>

      <ToolbarDivider />

      {/* Alignment */}
      <ToolbarButton
        onClick={() => editor.chain().focus().setTextAlign('left').run()}
        active={editor.isActive({ textAlign: 'left' })}
        title="Align Left"
      >
        <AlignLeft size={18} />
      </ToolbarButton>
      <ToolbarButton
        onClick={() => editor.chain().focus().setTextAlign('center').run()}
        active={editor.isActive({ textAlign: 'center' })}
        title="Align Center"
      >
        <AlignCenter size={18} />
      </ToolbarButton>
      <ToolbarButton
        onClick={() => editor.chain().focus().setTextAlign('right').run()}
        active={editor.isActive({ textAlign: 'right' })}
        title="Align Right"
      >
        <AlignRight size={18} />
      </ToolbarButton>
      <ToolbarButton
        onClick={() => editor.chain().focus().setTextAlign('justify').run()}
        active={editor.isActive({ textAlign: 'justify' })}
        title="Justify"
      >
        <AlignJustify size={18} />
      </ToolbarButton>

      <ToolbarDivider />

      {/* Lists */}
      <ToolbarButton
        onClick={() => editor.chain().focus().toggleBulletList().run()}
        active={editor.isActive('bulletList')}
        title="Bullet List"
      >
        <List size={18} />
      </ToolbarButton>
      <ToolbarButton
        onClick={() => editor.chain().focus().toggleOrderedList().run()}
        active={editor.isActive('orderedList')}
        title="Numbered List"
      >
        <ListOrdered size={18} />
      </ToolbarButton>

      <ToolbarDivider />

      {/* Blockquote */}
      <ToolbarButton
        onClick={() => editor.chain().focus().toggleBlockquote().run()}
        active={editor.isActive('blockquote')}
        title="Blockquote"
      >
        <Quote size={18} />
      </ToolbarButton>

      {/* Horizontal Rule */}
      <ToolbarButton onClick={() => editor.chain().focus().setHorizontalRule().run()} title="Horizontal Rule">
        <Minus size={18} />
      </ToolbarButton>

      <ToolbarDivider />

      {/* Link */}
      <div className="relative">
        <ToolbarButton
          onClick={() => setShowLinkInput(!showLinkInput)}
          active={editor.isActive('link')}
          title="Insert Link"
        >
          <LinkIcon size={18} />
        </ToolbarButton>
        {showLinkInput && (
          <div className="absolute top-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg p-3 z-20 w-64">
            <input
              type="url"
              className="w-full px-2 py-1 border border-gray-300 rounded mb-2 text-sm"
              placeholder="https://example.com"
              value={linkUrl}
              onChange={(e) => setLinkUrl(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  addLink();
                }
              }}
              autoFocus
            />
            <div className="flex gap-2">
              <button
                onClick={addLink}
                className="flex-1 px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
                type="button"
              >
                Add Link
              </button>
              <button
                onClick={() => {
                  editor.chain().focus().unsetLink().run();
                  setShowLinkInput(false);
                }}
                className="px-3 py-1 bg-gray-200 hover:bg-gray-300 rounded text-sm"
                type="button"
              >
                Remove
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Image */}
      <ToolbarButton onClick={addImage} title="Insert Image">
        <ImageIcon size={18} />
      </ToolbarButton>

      {/* Table */}
      <ToolbarButton onClick={() => setShowTableDialog(true)} title="Insert Table">
        <TableIcon size={18} />
      </ToolbarButton>

      <ToolbarDivider />

      {/* Find & Replace */}
      <ToolbarButton onClick={onFindReplace} title="Find & Replace (Ctrl+F)">
        <Search size={18} />
      </ToolbarButton>

      <ToolbarDivider />

      {/* Zoom Controls */}
      <ToolbarButton
        onClick={handleZoomOut}
        disabled={zoom <= 50}
        title="Zoom Out"
      >
        <ZoomOut size={18} />
      </ToolbarButton>
      <span className="px-2 py-1 text-sm font-medium text-gray-700 min-w-[4rem] text-center">
        {zoom}%
      </span>
      <ToolbarButton
        onClick={handleZoomIn}
        disabled={zoom >= 200}
        title="Zoom In"
      >
        <ZoomIn size={18} />
      </ToolbarButton>

      <ToolbarDivider />

      {/* Templates - Emphasized Button */}
      <button
        ref={templateButtonRef}
        onClick={() => setShowTemplateSelector(true)}
        className="ml-2 px-4 py-2 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 transition-all shadow-md hover:shadow-lg flex items-center gap-2 font-medium"
        title="Load example text for this field"
        type="button"
      >
        <FileText size={18} />
        <span className="text-sm">Example Text</span>
      </button>

      {/* Template Selector Dialog */}
      {showTemplateSelector && (
        <TemplateSelector
          editor={editor}
          fieldName={fieldName}
          triggerRef={templateButtonRef}
          onClose={() => setShowTemplateSelector(false)}
        />
      )}

      {/* Table Insert Dialog */}
      {showTableDialog && (
        <TableInsertDialog
          onInsert={addTable}
          onClose={() => setShowTableDialog(false)}
        />
      )}
    </div>
  );
};

export default React.memo(TipTapToolbar);
