import React, { useState, useRef, useCallback, useEffect } from 'react';
import FormattingToolbar from './FormattingToolbar';
import DOMPurify from 'dompurify';

const FormattedTextEditor = ({
  value = '',
  onChange,
  placeholder = '',
  rows = 3,
  className = '',
  id,
  'aria-required': ariaRequired
}) => {
  const [showToolbar, setShowToolbar] = useState(false);
  const [toolbarPosition, setToolbarPosition] = useState({ top: 0, left: 0 });
  const [isRichText, setIsRichText] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const textareaRef = useRef(null);
  const editorRef = useRef(null);
  const containerRef = useRef(null);

  const getFontClass = (fontValue) => {
    const fontMap = {
      'arial': 'font-sans',
      'times': 'font-serif',
      'courier': 'font-mono',
      'georgia': 'font-serif',
      'verdana': 'font-sans',
      'default': ''
    };
    return fontMap[fontValue] || '';
  };

  const handleTextSelection = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea || !containerRef.current) return;

    const selection = textarea.selectionStart !== textarea.selectionEnd;

    if (selection) {
      // Calculate position for toolbar based on selection
      const containerRect = containerRef.current.getBoundingClientRect();
      const textareaRect = textarea.getBoundingClientRect();

      setToolbarPosition({
        top: textareaRect.top - containerRect.top - 60, // Above the selection
        left: textareaRect.left - containerRect.left
      });
      setShowToolbar(true);
    }
  }, []);

  const handleClickOutside = useCallback((event) => {
    if (containerRef.current && !containerRef.current.contains(event.target)) {
      setShowToolbar(false);
    }
  }, []);

  const wrapSelection = (startTag, endTag) => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const selectedText = value.substring(start, end);
    const beforeText = value.substring(0, start);
    const afterText = value.substring(end);

    const newValue = beforeText + startTag + selectedText + endTag + afterText;
    onChange(newValue);

    // Restore selection
    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(
        start + startTag.length,
        end + startTag.length
      );
    }, 0);
  };

  const insertAtCursor = (text) => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const beforeText = value.substring(0, start);
    const afterText = value.substring(start);

    const newValue = beforeText + text + afterText;
    onChange(newValue);

    // Position cursor after inserted text
    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(start + text.length, start + text.length);
    }, 0);
  };

  const handleFormat = (type, formatValue) => {
    switch (type) {
      case 'bold':
        wrapSelection('**', '**');
        break;
      case 'italic':
        wrapSelection('*', '*');
        break;
      case 'underline':
        wrapSelection('<u>', '</u>');
        break;
      case 'align':
        const alignTag = formatValue === 'center' ? '<center>' :
                        formatValue === 'right' ? '<div style="text-align: right;">' :
                        '<div style="text-align: left;">';
        const alignEndTag = formatValue === 'center' ? '</center>' : '</div>';
        wrapSelection(alignTag, alignEndTag);
        break;
      case 'list':
        const textarea = textareaRef.current;
        if (!textarea) return;

        const lines = value.split('\n');
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;

        // Find line numbers for selection
        let charCount = 0;
        let startLine = 0;
        let endLine = 0;

        for (let i = 0; i < lines.length; i++) {
          if (charCount <= start && start <= charCount + lines[i].length) {
            startLine = i;
          }
          if (charCount <= end && end <= charCount + lines[i].length) {
            endLine = i;
          }
          charCount += lines[i].length + 1; // +1 for newline
        }

        // Apply list formatting
        const newLines = lines.map((line, index) => {
          if (index >= startLine && index <= endLine && line.trim()) {
            if (formatValue === 'bullet') {
              return line.startsWith('• ') ? line : '• ' + line.replace(/^\d+\.\s*/, '');
            } else {
              const listNumber = index - startLine + 1;
              return line.match(/^\d+\.\s/) ? line : `${listNumber}. ` + line.replace(/^•\s*/, '');
            }
          }
          return line;
        });

        onChange(newLines.join('\n'));
        break;
      case 'font':
        // For font changes, we'll apply a CSS class
        break;
      default:
        break;
    }
  };

  const processDisplayValue = (text) => {
    if (!isRichText) return text;

    // Basic markdown-like processing for display
    return DOMPurify.sanitize(
      text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br/>')
    );
  };

  const baseClasses = "w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-y";

  useEffect(() => {
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [handleClickOutside]);

  return (
    <div ref={containerRef} className="relative">
      {/* Rich Text Toggle */}
      <div className="mb-2 flex items-center justify-end">
        <label className="flex items-center space-x-1 text-sm text-gray-600">
          <input
            type="checkbox"
            checked={isRichText}
            onChange={(e) => setIsRichText(e.target.checked)}
            className="rounded"
          />
          <span>Rich Text Preview</span>
        </label>
      </div>

      {/* Floating Formatting Toolbar */}
      <FormattingToolbar
        show={showToolbar}
        onFormat={handleFormat}
        onClose={() => setShowToolbar(false)}
        position={toolbarPosition}
      />

      {/* Text Editor */}
      <div className="relative">
        {isRichText ? (
          <div className="relative">
            {!isEditing && (
              <div
                ref={editorRef}
                className={`${baseClasses} ${className} min-h-[120px] overflow-y-auto bg-gray-50 cursor-text`}
                style={{ minHeight: `${rows * 24}px` }}
                dangerouslySetInnerHTML={{ __html: processDisplayValue(value) }}
                onClick={() => {
                  setIsEditing(true);
                  setTimeout(() => {
                    if (textareaRef.current) {
                      textareaRef.current.focus();
                    }
                  }, 0);
                }}
              />
            )}
            {isEditing && (
              <textarea
                ref={textareaRef}
                id={id}
                aria-required={ariaRequired}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                rows={rows}
                className={`${baseClasses} ${className}`}
                placeholder={placeholder}
                style={{ minHeight: `${rows * 24}px` }}
                onMouseUp={handleTextSelection}
                onKeyUp={handleTextSelection}
                onBlur={() => {
                  setTimeout(() => setIsEditing(false), 150);
                }}
                autoFocus
              />
            )}
          </div>
        ) : (
          <textarea
            ref={textareaRef}
            id={id}
            aria-required={ariaRequired}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            rows={rows}
            className={`${baseClasses} ${className}`}
            placeholder={placeholder}
            style={{ minHeight: `${rows * 24}px` }}
            onMouseUp={handleTextSelection}
            onKeyUp={handleTextSelection}
          />
        )}
      </div>

      {/* Help Text */}
      {showToolbar && (
        <div className="mt-2 text-xs text-gray-500">
          <p>Select text and use toolbar buttons to format. Use ** for bold, * for italic.</p>
        </div>
      )}
    </div>
  );
};

export default FormattedTextEditor;