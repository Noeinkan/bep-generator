import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Search, Plus, Edit3, Trash2, Undo, Redo, Maximize2, LayoutGrid, Copy } from 'lucide-react';

const CommandPalette = ({ visible, onClose, onCommand, selectedNode, canUndo, canRedo }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef(null);

  const commands = [
    {
      id: 'add-node',
      label: 'Add Child Node',
      icon: Plus,
      shortcut: 'Tab',
      action: 'addNode',
      enabled: !!selectedNode
    },
    {
      id: 'edit-node',
      label: 'Edit Node',
      icon: Edit3,
      shortcut: 'Enter',
      action: 'editNode',
      enabled: !!selectedNode
    },
    {
      id: 'delete-node',
      label: 'Delete Node',
      icon: Trash2,
      shortcut: 'Del',
      action: 'deleteNode',
      enabled: !!selectedNode && selectedNode !== 'root'
    },
    {
      id: 'duplicate-node',
      label: 'Duplicate Node',
      icon: Copy,
      shortcut: 'Ctrl+D',
      action: 'duplicateNode',
      enabled: !!selectedNode
    },
    {
      id: 'undo',
      label: 'Undo',
      icon: Undo,
      shortcut: 'Ctrl+Z',
      action: 'undo',
      enabled: canUndo
    },
    {
      id: 'redo',
      label: 'Redo',
      icon: Redo,
      shortcut: 'Ctrl+Y',
      action: 'redo',
      enabled: canRedo
    },
    {
      id: 'reset-view',
      label: 'Reset View',
      icon: Maximize2,
      shortcut: 'Ctrl+0',
      action: 'resetView',
      enabled: true
    },
    {
      id: 'organize',
      label: 'Organize Nodes',
      icon: LayoutGrid,
      shortcut: 'Ctrl+Shift+O',
      action: 'organizeNodes',
      enabled: true
    }
  ];

  const filteredCommands = commands.filter(cmd =>
    cmd.enabled &&
    (cmd.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
     cmd.shortcut.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  useEffect(() => {
    if (visible && inputRef.current) {
      inputRef.current.focus();
    }
  }, [visible]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [searchTerm]);

  const executeCommand = useCallback((command) => {
    onCommand(command.action);
    onClose();
    setSearchTerm('');
  }, [onCommand, onClose]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!visible) return;

      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => (prev + 1) % filteredCommands.length);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => (prev - 1 + filteredCommands.length) % filteredCommands.length);
      } else if (e.key === 'Enter' && filteredCommands.length > 0) {
        e.preventDefault();
        executeCommand(filteredCommands[selectedIndex]);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [visible, selectedIndex, filteredCommands, onClose, executeCommand]);

  if (!visible) return null;

  return (
    <div className="absolute inset-0 bg-black bg-opacity-50 flex items-start justify-center pt-20 z-50 animate-fade-in">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-xl mx-4 animate-slide-down">
        {/* Search Input */}
        <div className="relative border-b border-gray-200">
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            ref={inputRef}
            type="text"
            placeholder="Type a command or search..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-12 pr-4 py-4 text-lg border-0 focus:ring-0 focus:outline-none rounded-t-xl"
          />
        </div>

        {/* Command List */}
        <div className="max-h-64 overflow-y-auto">
          {filteredCommands.length === 0 ? (
            <div className="px-4 py-8 text-center text-gray-400">
              <p>No commands found</p>
            </div>
          ) : (
            <div className="py-2">
              {filteredCommands.map((command, index) => (
                <CommandItem
                  key={command.id}
                  command={command}
                  isSelected={index === selectedIndex}
                  onClick={() => executeCommand(command)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 px-4 py-3 bg-gray-50 rounded-b-xl">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <div className="flex items-center space-x-4">
              <span><kbd className="px-2 py-1 bg-white border border-gray-300 rounded">↑↓</kbd> Navigate</span>
              <span><kbd className="px-2 py-1 bg-white border border-gray-300 rounded">Enter</kbd> Execute</span>
              <span><kbd className="px-2 py-1 bg-white border border-gray-300 rounded">Esc</kbd> Close</span>
            </div>
            <span>{filteredCommands.length} commands</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const CommandItem = ({ command, isSelected, onClick }) => {
  const Icon = command.icon;

  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center justify-between px-4 py-3 transition-all duration-150 ${
        isSelected
          ? 'bg-gradient-to-r from-green-50 to-green-100 border-l-4 border-green-500'
          : 'hover:bg-gray-50 border-l-4 border-transparent'
      }`}
    >
      <div className="flex items-center space-x-3">
        <Icon className={`w-5 h-5 ${isSelected ? 'text-green-600' : 'text-gray-400'}`} />
        <span className={`text-sm font-medium ${isSelected ? 'text-green-900' : 'text-gray-700'}`}>
          {command.label}
        </span>
      </div>
      {command.shortcut && (
        <span className={`text-xs px-2 py-1 rounded ${
          isSelected
            ? 'bg-green-200 text-green-800'
            : 'bg-gray-200 text-gray-600'
        }`}>
          {command.shortcut}
        </span>
      )}
    </button>
  );
};

export default CommandPalette;
