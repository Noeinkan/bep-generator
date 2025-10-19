import React, { useState, useCallback } from 'react';
import {
  Folder,
  FolderPlus,
  Edit3,
  Trash2,
  ChevronUp,
  ChevronDown,
  Plus,
  Save,
  X,
  Move3D,
  FolderOpen
} from 'lucide-react';
import FieldHeader from '../base/FieldHeader';

const FolderStructureDiagram = ({ field, value, onChange, error }) => {
  const { name, label, number, required } = field;

  // Parse the value (could be string or object)
  const parseStructure = (val) => {
    if (typeof val === 'string') {
      // Convert text diagram to structured data
      const lines = val.split('\n').filter(line => line.trim());
      const structure = [];
      const stack = [{ children: structure, level: -1 }];

      lines.forEach(line => {
        const match = line.match(/^(\s*)(📁\s*)?(.+)$/);
        if (match) {
          const [, indent, , folderName] = match;
          const level = Math.floor(indent.length / 4); // Assuming 4 spaces per level
          const cleanName = folderName.replace(/[├└│─└┐┤┴┬┌│]/g, '').trim();

          // Pop stack to correct level
          while (stack.length > level + 1) {
            stack.pop();
          }

          const folder = {
            id: Date.now() + Math.random(),
            name: cleanName,
            children: []
          };

          stack[stack.length - 1].children.push(folder);
          stack.push(folder);
        }
      });

      return structure;
    }
    return val || getDefaultStructure();
  };

  const getDefaultStructure = () => [
    {
      id: 1,
      name: 'WIP (Work in Progress)',
      children: []
    },
    {
      id: 2,
      name: 'SHARED (Coordination)',
      children: []
    },
    {
      id: 3,
      name: 'PUBLISHED (Approved)',
      children: []
    }
  ];

  const [structure, setStructure] = useState(() => parseStructure(value));
  const [editingId, setEditingId] = useState(null);
  const [editingName, setEditingName] = useState('');
  const [expandedNodes, setExpandedNodes] = useState(new Set([1, 2, 3]));

  const generateTextDiagram = useCallback((folders, level = 0) => {
    let result = '';
    folders.forEach((folder, index) => {
      const isLast = index === folders.length - 1;
      const indent = '    '.repeat(level);
      const connector = level === 0 ? '' : (isLast ? '└── ' : '├── ');

      result += `${indent}${connector}📁 ${folder.name}\n`;

      if (folder.children && folder.children.length > 0) {
        result += generateTextDiagram(folder.children, level + 1);
      }
    });
    return result;
  }, []);

  const updateValue = useCallback((newStructure) => {
    setStructure(newStructure);
    const textDiagram = generateTextDiagram(newStructure);
    onChange(name, textDiagram);
  }, [name, onChange, generateTextDiagram]);

  const findNodeById = (nodes, id) => {
    for (const node of nodes) {
      if (node.id === id) return { node, parent: null };
      if (node.children) {
        const found = findNodeById(node.children, id);
        if (found.node) return { ...found, parent: found.parent || node };
      }
    }
    return { node: null, parent: null };
  };

  const addFolder = (parentId) => {
    const newStructure = JSON.parse(JSON.stringify(structure));
    const { node: parentNode } = findNodeById(newStructure, parentId);

    if (parentNode) {
      const newFolder = {
        id: Date.now() + Math.random(),
        name: 'New Folder',
        children: []
      };
      parentNode.children.push(newFolder);
      setExpandedNodes(prev => new Set([...prev, parentId]));
      updateValue(newStructure);
      setEditingId(newFolder.id);
      setEditingName('New Folder');
    }
  };

  const removeFolder = (folderId) => {
    const newStructure = JSON.parse(JSON.stringify(structure));

    const removeFromNode = (nodes) => {
      for (let i = 0; i < nodes.length; i++) {
        if (nodes[i].id === folderId) {
          nodes.splice(i, 1);
          return true;
        }
        if (nodes[i].children && removeFromNode(nodes[i].children)) {
          return true;
        }
      }
      return false;
    };

    removeFromNode(newStructure);
    updateValue(newStructure);
  };

  const moveFolder = (folderId, direction) => {
    const newStructure = JSON.parse(JSON.stringify(structure));

    const moveInNode = (nodes) => {
      for (let i = 0; i < nodes.length; i++) {
        if (nodes[i].id === folderId) {
          const newIndex = direction === 'up' ? i - 1 : i + 1;
          if (newIndex >= 0 && newIndex < nodes.length) {
            [nodes[i], nodes[newIndex]] = [nodes[newIndex], nodes[i]];
            return true;
          }
        }
        if (nodes[i].children && moveInNode(nodes[i].children)) {
          return true;
        }
      }
      return false;
    };

    moveInNode(newStructure);
    updateValue(newStructure);
  };

  const renameFolder = (folderId, newName) => {
    const newStructure = JSON.parse(JSON.stringify(structure));
    const { node } = findNodeById(newStructure, folderId);

    if (node) {
      node.name = newName;
      updateValue(newStructure);
    }
  };

  const startEditing = (folderId, currentName) => {
    setEditingId(folderId);
    setEditingName(currentName);
  };

  const saveEdit = () => {
    if (editingId && editingName.trim()) {
      renameFolder(editingId, editingName.trim());
    }
    setEditingId(null);
    setEditingName('');
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditingName('');
  };

  const toggleExpanded = (nodeId) => {
    setExpandedNodes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(nodeId)) {
        newSet.delete(nodeId);
      } else {
        newSet.add(nodeId);
      }
      return newSet;
    });
  };

  const renderFolder = (folder, level = 0, parentId = null, isLast = false, ancestorLines = []) => {
    const isExpanded = expandedNodes.has(folder.id);
    const hasChildren = folder.children && folder.children.length > 0;
    const isEditing = editingId === folder.id;

    return (
      <div key={folder.id} className="select-none">
        <div className="flex items-center space-x-1 py-1 px-2 rounded hover:bg-gray-50 group relative">
          {/* Tree Lines */}
          {level > 0 && (
            <div className="flex items-center" style={{ width: `${level * 20}px` }}>
              {/* Ancestor vertical lines */}
              {ancestorLines.map((shouldShowLine, index) => (
                <div
                  key={index}
                  className="relative"
                  style={{ width: '20px', height: '24px' }}
                >
                  {shouldShowLine && (
                    <div
                      className="absolute left-2 top-0 w-px bg-gray-400"
                      style={{ height: '24px' }}
                    />
                  )}
                </div>
              ))}

              {/* Current level connector */}
              <div className="relative" style={{ width: '20px', height: '24px' }}>
                {/* Vertical line from parent */}
                {!isLast && (
                  <div
                    className="absolute left-2 top-0 w-px bg-gray-400"
                    style={{ height: '24px' }}
                  />
                )}
                {/* Horizontal line to folder */}
                <div
                  className="absolute top-3 left-2 h-px bg-gray-400"
                  style={{ width: '10px' }}
                />
                {/* Vertical line up to parent */}
                <div
                  className="absolute left-2 top-0 w-px bg-gray-400"
                  style={{ height: '12px' }}
                />
                {/* Corner connector for last item */}
                {isLast && (
                  <div
                    className="absolute left-2 top-0 w-px bg-gray-400"
                    style={{ height: '12px' }}
                  />
                )}
              </div>
            </div>
          )}
          {/* Expand/Collapse Button */}
          {hasChildren && (
            <button
              onClick={() => toggleExpanded(folder.id)}
              className="w-4 h-4 flex items-center justify-center text-gray-400 hover:text-gray-600"
            >
              {isExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
            </button>
          )}
          {!hasChildren && <div className="w-4" />}

          {/* Folder Icon */}
          <div className="text-blue-500">
            {isExpanded ? <FolderOpen className="w-4 h-4" /> : <Folder className="w-4 h-4" />}
          </div>

          {/* Folder Name */}
          <div className="flex-1 flex items-center space-x-2">
            {isEditing ? (
              <div className="flex items-center space-x-2 flex-1">
                <input
                  type="text"
                  value={editingName}
                  onChange={(e) => setEditingName(e.target.value)}
                  className="flex-1 px-2 py-1 text-sm border border-blue-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                  autoFocus
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') saveEdit();
                    if (e.key === 'Escape') cancelEdit();
                  }}
                />
                <button
                  onClick={saveEdit}
                  className="w-6 h-6 flex items-center justify-center text-green-600 hover:bg-green-50 rounded"
                >
                  <Save className="w-3 h-3" />
                </button>
                <button
                  onClick={cancelEdit}
                  className="w-6 h-6 flex items-center justify-center text-red-600 hover:bg-red-50 rounded"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ) : (
              <>
                <span className="text-sm font-medium text-gray-700 flex-1">{folder.name}</span>

                {/* Action Buttons */}
                <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => addFolder(folder.id)}
                    className="w-6 h-6 flex items-center justify-center text-green-600 hover:bg-green-50 rounded"
                    title="Add subfolder"
                  >
                    <Plus className="w-3 h-3" />
                  </button>

                  <button
                    onClick={() => startEditing(folder.id, folder.name)}
                    className="w-6 h-6 flex items-center justify-center text-blue-600 hover:bg-blue-50 rounded"
                    title="Rename folder"
                  >
                    <Edit3 className="w-3 h-3" />
                  </button>

                  <button
                    onClick={() => moveFolder(folder.id, 'up')}
                    className="w-6 h-6 flex items-center justify-center text-gray-600 hover:bg-gray-50 rounded"
                    title="Move up"
                  >
                    <ChevronUp className="w-3 h-3" />
                  </button>

                  <button
                    onClick={() => moveFolder(folder.id, 'down')}
                    className="w-6 h-6 flex items-center justify-center text-gray-600 hover:bg-gray-50 rounded"
                    title="Move down"
                  >
                    <ChevronDown className="w-3 h-3" />
                  </button>

                  {(folder.id !== 1 && folder.id !== 2 && folder.id !== 3) && (
                    <button
                      onClick={() => removeFolder(folder.id)}
                      className="w-6 h-6 flex items-center justify-center text-red-600 hover:bg-red-50 rounded"
                      title="Delete folder"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </>
            )}
          </div>
        </div>

        {/* Children */}
        {isExpanded && hasChildren && (
          <div>
            {folder.children.map((child, index) => {
              const isLastChild = index === folder.children.length - 1;
              const newAncestorLines = [...ancestorLines, !isLast];
              return renderFolder(child, level + 1, folder.id, isLastChild, newAncestorLines);
            })}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="mb-8 w-full">
      <FieldHeader 
        fieldName={name}
        label={label}
        number={number}
        required={required}
      />

      <div className="w-full border rounded-xl overflow-hidden shadow-sm bg-white">
        <div className="bg-gradient-to-r from-blue-50 to-blue-100 px-6 py-4 border-b border-blue-200">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <Move3D className="w-5 h-5 text-blue-600" />
              <span className="text-base font-semibold text-blue-800">Interactive File Structure</span>
            </div>
            <button
              onClick={() => {
                const newFolder = {
                  id: Date.now() + Math.random(),
                  name: 'New CDE Level',
                  children: []
                };
                const newStructure = [...structure, newFolder];
                updateValue(newStructure);
                setEditingId(newFolder.id);
                setEditingName('New CDE Level');
              }}
              className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-all transform hover:scale-105 shadow-md"
            >
              <FolderPlus className="w-4 h-4" />
              <span>Add CDE Level</span>
            </button>
          </div>
          <p className="text-sm text-blue-700 mt-2">
            Click folders to expand/collapse • Hover over folders to see action buttons • Drag and drop to reorganize
          </p>
        </div>

        <div className="w-full p-6 bg-gray-50 min-h-80 max-h-[600px] overflow-y-auto">
          {structure.map((folder, index) => {
            const isLastRoot = index === structure.length - 1;
            return renderFolder(folder, 0, null, isLastRoot, []);
          })}
        </div>

        <div className="w-full bg-gray-100 px-6 py-3 border-t">
          <p className="text-xs text-gray-600">
            💡 Tip: Use the action buttons to add subfolders, rename, reorder, or delete folders.
            The text representation is automatically updated and saved to your BEP.
          </p>
        </div>
      </div>

      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
};

export default FolderStructureDiagram;