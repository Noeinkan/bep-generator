import React, { useState } from 'react';
import { Plus, Trash2, Maximize2, Brain, ChevronDown, Undo, Redo } from 'lucide-react';
import NodeTypeSelector from './NodeTypeSelector';
import LayoutControls from './LayoutControls';
import { NODE_TYPES } from '../../utils/nodeTypes';

const MindmapControls = ({
  selectedNode,
  zoom,
  onAddNode,
  onDeleteNode,
  onResetView,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
  onOrganizeNodes,
  onSnapToGrid
}) => {
  const [showTypeSelector, setShowTypeSelector] = useState(false);
  const [selectedNodeType, setSelectedNodeType] = useState(NODE_TYPES.DISCIPLINE);

  const handleAddNode = () => {
    onAddNode(selectedNodeType);
  };
  return (
    <div className="bg-gradient-to-r from-green-50 to-green-100 px-6 py-4 border-b border-green-200">
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-3">
          <Brain className="w-5 h-5 text-green-600" />
          <span className="text-base font-semibold text-green-800">Volume Strategy Mindmap</span>
        </div>

        <div className="flex items-center space-x-2">
          {selectedNode && (
            <span className="text-sm text-green-700 bg-green-50 px-2 py-1 rounded">
              Selected: {selectedNode}
            </span>
          )}

          <div className="relative">
            <button
              onClick={handleAddNode}
              disabled={!selectedNode}
              className={`p-2 rounded ${
                selectedNode
                  ? 'text-green-600 hover:bg-green-100 cursor-pointer'
                  : 'text-gray-400 cursor-not-allowed'
              }`}
              title={selectedNode ? "Add Child Node" : "Select a node first"}
              aria-label="Add Child Node"
            >
              <Plus className="w-4 h-4" />
            </button>

            <button
              onClick={() => setShowTypeSelector(!showTypeSelector)}
              disabled={!selectedNode}
              className={`p-1 rounded ${
                selectedNode
                  ? 'text-green-600 hover:bg-green-100 cursor-pointer'
                  : 'text-gray-400 cursor-not-allowed'
              }`}
              title="Select node type"
              aria-label="Select node type"
            >
              <ChevronDown className="w-3 h-3" />
            </button>

            <NodeTypeSelector
              selectedType={selectedNodeType}
              onTypeChange={setSelectedNodeType}
              isVisible={showTypeSelector}
              onClose={() => setShowTypeSelector(false)}
            />
          </div>

          <button
            onClick={onDeleteNode}
            disabled={!selectedNode || selectedNode === 'root'}
            className={`p-2 rounded ${
              selectedNode && selectedNode !== 'root'
                ? 'text-red-600 hover:bg-red-100 cursor-pointer'
                : 'text-gray-400 cursor-not-allowed'
            }`}
            title={!selectedNode ? "Select a node first" : selectedNode === 'root' ? "Cannot delete root node" : "Delete Node"}
            aria-label="Delete Node"
          >
            <Trash2 className="w-4 h-4" />
          </button>

          <div className="w-px h-6 bg-gray-300" />

          <button
            onClick={onUndo}
            disabled={!canUndo}
            className={`p-2 rounded ${
              canUndo
                ? 'text-blue-600 hover:bg-blue-100 cursor-pointer'
                : 'text-gray-400 cursor-not-allowed'
            }`}
            title="Undo (Ctrl+Z)"
            aria-label="Undo"
          >
            <Undo className="w-4 h-4" />
          </button>

          <button
            onClick={onRedo}
            disabled={!canRedo}
            className={`p-2 rounded ${
              canRedo
                ? 'text-blue-600 hover:bg-blue-100 cursor-pointer'
                : 'text-gray-400 cursor-not-allowed'
            }`}
            title="Redo (Ctrl+Y)"
            aria-label="Redo"
          >
            <Redo className="w-4 h-4" />
          </button>

          <div className="w-px h-6 bg-gray-300" />

          <LayoutControls
            onOrganizeNodes={onOrganizeNodes}
            onSnapToGrid={onSnapToGrid}
          />

          <button
            onClick={onResetView}
            className="p-2 text-gray-600 hover:bg-gray-100 rounded"
            title="Reset View"
            aria-label="Reset View"
          >
            <Maximize2 className="w-4 h-4" />
          </button>

          <span className="text-sm text-gray-600 min-w-12 text-center">
            {Math.round(zoom * 100)}%
          </span>
        </div>
      </div>

      <p className="text-sm text-green-700 mt-2">
        Click nodes to select • Double-click to edit • Drag to pan • Scroll to zoom
      </p>
    </div>
  );
};

export default MindmapControls;