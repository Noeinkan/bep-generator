import React, { useEffect, useRef } from 'react';
import { getNodeTypeOptions } from '../../../utils/nodeTypes';

const QuickAddMenu = ({ position, onAddNode, onClose, visible }) => {
  const menuRef = useRef(null);
  const nodeTypes = getNodeTypeOptions().filter(type => type.value !== 'root');

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        onClose();
      }
    };

    const handleEscape = (event) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    if (visible) {
      document.addEventListener('mousedown', handleClickOutside);
      document.addEventListener('keydown', handleEscape);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [visible, onClose]);

  if (!visible) return null;

  return (
    <div
      ref={menuRef}
      className="absolute z-50 animate-scale-in"
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
        transform: 'translate(-50%, -50%)'
      }}
    >
      <div className="bg-white rounded-xl shadow-2xl border border-gray-200 p-2 min-w-48">
        <div className="text-xs font-semibold text-gray-500 px-3 py-2 uppercase tracking-wide">
          Add Node
        </div>
        <div className="space-y-1">
          {nodeTypes.map((type, index) => (
            <button
              key={type.value}
              onClick={() => {
                onAddNode(type.value);
                onClose();
              }}
              className="w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg hover:bg-gradient-to-r hover:from-green-50 hover:to-green-100 transition-all duration-200 group"
              style={{ animationDelay: `${index * 30}ms` }}
            >
              <div
                className="w-4 h-4 rounded-full shadow-sm group-hover:scale-110 transition-transform duration-200"
                style={{ backgroundColor: type.color }}
              />
              <div className="flex-1 text-left">
                <div className="text-sm font-medium text-gray-700 group-hover:text-green-700">
                  {type.label}
                </div>
                <div className="text-xs text-gray-400">
                  {getShortcutLabel(index)}
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>
      <div className="text-center mt-2">
        <span className="text-xs text-gray-400 bg-white px-2 py-1 rounded shadow-sm">
          Press ESC to cancel
        </span>
      </div>
    </div>
  );
};

const getShortcutLabel = (index) => {
  return `Press ${index + 1}`;
};

export default QuickAddMenu;
