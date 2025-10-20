import React, { useEffect, useRef } from 'react';
import { Plus, Edit3, Trash2, Copy } from 'lucide-react';
import { getNodeTypeOptions } from '../../../utils/nodeTypes';

const NodeContextMenu = ({
  position,
  node,
  onAddChild,
  onEdit,
  onDelete,
  onDuplicate,
  onChangeType,
  onClose,
  visible
}) => {
  const menuRef = useRef(null);

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

  if (!visible || !node) return null;

  const nodeTypes = getNodeTypeOptions().filter(type => type.value !== 'root');
  const canDelete = node.id !== 'root';

  return (
    <div
      ref={menuRef}
      className="absolute z-50 animate-scale-in"
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`
      }}
    >
      <div className="bg-white rounded-xl shadow-2xl border border-gray-200 py-2 min-w-56">
        <div className="px-3 py-2 border-b border-gray-100">
          <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
            Node Actions
          </div>
          <div className="text-sm font-medium text-gray-700 mt-1 truncate">
            {node.name}
          </div>
        </div>

        <div className="py-1">
          <MenuItem
            icon={Plus}
            label="Add Child Node"
            shortcut="Tab"
            onClick={() => {
              onAddChild();
              onClose();
            }}
          />
          <MenuItem
            icon={Edit3}
            label="Edit Node"
            shortcut="Enter"
            onClick={() => {
              onEdit();
              onClose();
            }}
          />
          <MenuItem
            icon={Copy}
            label="Duplicate Node"
            shortcut="Ctrl+D"
            onClick={() => {
              onDuplicate();
              onClose();
            }}
          />
        </div>

        <div className="border-t border-gray-100 my-1" />

        <div className="px-3 py-2">
          <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">
            Change Type
          </div>
          <div className="space-y-1">
            {nodeTypes.map((type) => (
              <button
                key={type.value}
                onClick={() => {
                  onChangeType(type.value);
                  onClose();
                }}
                className={`w-full flex items-center space-x-2 px-2 py-1.5 rounded-lg hover:bg-gray-50 transition-colors duration-150 ${
                  node.type === type.value ? 'bg-green-50' : ''
                }`}
              >
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: type.color }}
                />
                <span className={`text-sm ${
                  node.type === type.value ? 'font-medium text-green-700' : 'text-gray-600'
                }`}>
                  {type.label}
                </span>
              </button>
            ))}
          </div>
        </div>

        {canDelete && (
          <>
            <div className="border-t border-gray-100 my-1" />
            <div className="py-1">
              <MenuItem
                icon={Trash2}
                label="Delete Node"
                shortcut="Del"
                onClick={() => {
                  onDelete();
                  onClose();
                }}
                danger
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
};

const MenuItem = ({ icon: Icon, label, shortcut, onClick, danger = false }) => {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center justify-between px-3 py-2 hover:bg-gray-50 transition-colors duration-150 ${
        danger ? 'hover:bg-red-50' : ''
      }`}
    >
      <div className="flex items-center space-x-3">
        <Icon className={`w-4 h-4 ${danger ? 'text-red-500' : 'text-gray-500'}`} />
        <span className={`text-sm ${danger ? 'text-red-700 font-medium' : 'text-gray-700'}`}>
          {label}
        </span>
      </div>
      {shortcut && (
        <span className="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded">
          {shortcut}
        </span>
      )}
    </button>
  );
};

export default NodeContextMenu;
