import { Node, mergeAttributes } from '@tiptap/core';
import { ReactNodeViewRenderer } from '@tiptap/react';
import { NodeViewWrapper } from '@tiptap/react';
import React, { useState } from 'react';

// Caption component for images and tables
const CaptionComponent = ({ node, updateAttributes, deleteNode }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [caption, setCaption] = useState(node.attrs.caption || '');

  const handleSave = () => {
    updateAttributes({ caption });
    setIsEditing(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSave();
    } else if (e.key === 'Escape') {
      setCaption(node.attrs.caption || '');
      setIsEditing(false);
    }
  };

  return (
    <NodeViewWrapper>
      <div className="caption-wrapper" style={{
        textAlign: 'center',
        fontSize: '0.9em',
        color: '#666',
        fontStyle: 'italic',
        padding: '0.5rem 0',
        borderTop: node.attrs.type === 'table' ? '1px solid #e5e7eb' : 'none',
        borderBottom: node.attrs.type === 'image' ? '1px solid #e5e7eb' : 'none',
        margin: '0.5rem 0'
      }}>
        {isEditing ? (
          <input
            type="text"
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            onBlur={handleSave}
            onKeyDown={handleKeyDown}
            placeholder="Enter caption..."
            autoFocus
            style={{
              width: '100%',
              border: '1px solid #3b82f6',
              borderRadius: '4px',
              padding: '0.25rem 0.5rem',
              outline: 'none',
              textAlign: 'center',
              fontStyle: 'italic'
            }}
          />
        ) : (
          <div
            onClick={() => setIsEditing(true)}
            style={{
              cursor: 'pointer',
              padding: '0.25rem',
              minHeight: '1.5em'
            }}
          >
            {caption || 'Click to add caption...'}
          </div>
        )}
      </div>
    </NodeViewWrapper>
  );
};

export const Caption = Node.create({
  name: 'caption',

  group: 'block',

  content: 'inline*',

  addAttributes() {
    return {
      caption: {
        default: '',
        parseHTML: element => element.textContent,
        renderHTML: attributes => {
          if (!attributes.caption) {
            return {};
          }
          return { 'data-caption': attributes.caption };
        },
      },
      type: {
        default: 'table',
        parseHTML: element => element.getAttribute('data-type') || 'table',
        renderHTML: attributes => {
          return { 'data-type': attributes.type };
        },
      },
    };
  },

  parseHTML() {
    return [
      {
        tag: 'caption',
      },
      {
        tag: 'figcaption',
      },
    ];
  },

  renderHTML({ HTMLAttributes }) {
    return ['figcaption', mergeAttributes(HTMLAttributes), 0];
  },

  addNodeView() {
    return ReactNodeViewRenderer(CaptionComponent);
  },
});

export default Caption;
