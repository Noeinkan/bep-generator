import Image from '@tiptap/extension-image';
import { ReactNodeViewRenderer } from '@tiptap/react';
import { NodeViewWrapper } from '@tiptap/react';
import React, { useState, useRef, useEffect } from 'react';

const ResizableImageComponent = ({ node, updateAttributes }) => {
  const [isResizing, setIsResizing] = useState(false);
  const [resizeDirection, setResizeDirection] = useState(null);
  const imageRef = useRef(null);
  const [dimensions, setDimensions] = useState({
    width: node.attrs.width || 'auto',
    height: node.attrs.height || 'auto',
  });
  const aspectRatioRef = useRef(null);
  const [isEditingCaption, setIsEditingCaption] = useState(false);
  const [caption, setCaption] = useState(node.attrs.caption || '');

  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e) => {
      if (!imageRef.current || !resizeDirection) return;

      const rect = imageRef.current.getBoundingClientRect();
      const isShiftPressed = e.shiftKey;

      // Calculate initial aspect ratio if Shift is pressed
      if (isShiftPressed && !aspectRatioRef.current) {
        aspectRatioRef.current = rect.width / rect.height;
      } else if (!isShiftPressed) {
        aspectRatioRef.current = null;
      }

      if (resizeDirection.includes('right')) {
        const newWidth = e.clientX - rect.left;
        if (newWidth > 50) {
          if (isShiftPressed && aspectRatioRef.current) {
            // Maintain aspect ratio: calculate proportional height
            const newHeight = newWidth / aspectRatioRef.current;
            setDimensions({ width: newWidth, height: newHeight });
          } else {
            setDimensions(prev => ({ ...prev, width: newWidth }));
          }
        }
      }

      if (resizeDirection.includes('bottom')) {
        const newHeight = e.clientY - rect.top;
        if (newHeight > 50) {
          if (isShiftPressed && aspectRatioRef.current) {
            // Maintain aspect ratio: calculate proportional width
            const newWidth = newHeight * aspectRatioRef.current;
            setDimensions({ width: newWidth, height: newHeight });
          } else {
            setDimensions(prev => ({ ...prev, height: newHeight }));
          }
        }
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      setResizeDirection(null);
      aspectRatioRef.current = null;
      updateAttributes({
        width: dimensions.width,
        height: dimensions.height,
      });
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing, resizeDirection, dimensions, updateAttributes]);

  const handleResizeStart = (direction) => (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsResizing(true);
    setResizeDirection(direction);
  };

  const handleCaptionSave = () => {
    updateAttributes({ caption });
    setIsEditingCaption(false);
  };

  const handleCaptionKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleCaptionSave();
    } else if (e.key === 'Escape') {
      setCaption(node.attrs.caption || '');
      setIsEditingCaption(false);
    }
  };

  return (
    <NodeViewWrapper className="resizable-image-wrapper" style={{ display: 'inline-block', position: 'relative', maxWidth: '100%' }}>
      <img
        ref={imageRef}
        src={node.attrs.src}
        alt={node.attrs.alt || ''}
        style={{
          width: dimensions.width === 'auto' ? 'auto' : `${dimensions.width}px`,
          height: dimensions.height === 'auto' ? 'auto' : `${dimensions.height}px`,
          maxWidth: '100%',
          display: 'block',
        }}
        className="tiptap-image"
      />

      {/* Resize handles */}
      <div
        className="resize-handle resize-handle-right"
        onMouseDown={handleResizeStart('right')}
        style={{
          position: 'absolute',
          right: '-4px',
          top: '50%',
          transform: 'translateY(-50%)',
          width: '8px',
          height: '40px',
          background: '#3b82f6',
          cursor: 'ew-resize',
          borderRadius: '4px',
          opacity: 0.7,
        }}
      />

      <div
        className="resize-handle resize-handle-bottom"
        onMouseDown={handleResizeStart('bottom')}
        style={{
          position: 'absolute',
          bottom: '-4px',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '40px',
          height: '8px',
          background: '#3b82f6',
          cursor: 'ns-resize',
          borderRadius: '4px',
          opacity: 0.7,
        }}
      />

      <div
        className="resize-handle resize-handle-corner"
        onMouseDown={handleResizeStart('right-bottom')}
        style={{
          position: 'absolute',
          right: '-4px',
          bottom: '-4px',
          width: '12px',
          height: '12px',
          background: '#3b82f6',
          cursor: 'nwse-resize',
          borderRadius: '50%',
          opacity: 0.7,
        }}
      />

      {/* Caption */}
      <div
        style={{
          width: '100%',
          textAlign: 'center',
          fontSize: '0.9em',
          color: '#666',
          fontStyle: 'italic',
          padding: '0.5rem 0',
          marginTop: '0.5rem',
        }}
      >
        {isEditingCaption ? (
          <input
            type="text"
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            onBlur={handleCaptionSave}
            onKeyDown={handleCaptionKeyDown}
            placeholder="Enter image caption..."
            autoFocus
            style={{
              width: '100%',
              border: '1px solid #3b82f6',
              borderRadius: '4px',
              padding: '0.25rem 0.5rem',
              outline: 'none',
              textAlign: 'center',
              fontStyle: 'italic',
              fontSize: '0.9em',
            }}
          />
        ) : (
          <div
            onClick={() => setIsEditingCaption(true)}
            style={{
              cursor: 'pointer',
              padding: '0.25rem',
              minHeight: '1.5em',
              color: caption ? '#666' : '#aaa',
            }}
          >
            {caption || 'Click to add caption...'}
          </div>
        )}
      </div>
    </NodeViewWrapper>
  );
};

export const ResizableImage = Image.extend({
  name: 'resizableImage',

  addAttributes() {
    return {
      ...this.parent?.(),
      width: {
        default: 'auto',
        parseHTML: element => element.getAttribute('width') || 'auto',
        renderHTML: attributes => {
          if (attributes.width === 'auto') {
            return {};
          }
          return { width: attributes.width };
        },
      },
      height: {
        default: 'auto',
        parseHTML: element => element.getAttribute('height') || 'auto',
        renderHTML: attributes => {
          if (attributes.height === 'auto') {
            return {};
          }
          return { height: attributes.height };
        },
      },
      caption: {
        default: '',
        parseHTML: element => element.getAttribute('data-caption') || '',
        renderHTML: attributes => {
          if (!attributes.caption) {
            return {};
          }
          return { 'data-caption': attributes.caption };
        },
      },
    };
  },

  addNodeView() {
    return ReactNodeViewRenderer(ResizableImageComponent);
  },
});

export default ResizableImage;
