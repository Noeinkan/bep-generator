import React, { useRef, useState, useCallback, useEffect } from 'react';
import { useMindmapD3 } from '../../hooks/useMindmapD3';
import { useUndoRedo } from '../../hooks/useUndoRedo';
import {
  parseValue,
  convertToText,
  addNodeToTree,
  removeNodeFromTree,
  updateNodeInTree,
  searchNodes
} from '../../utils/mindmapUtils';
import { organizeNodes, snapToGrid } from '../../utils/layoutUtils';
import MindmapControls from './MindmapControls';
import EditModal from './EditModal';
import SearchFilter from './SearchFilter';

const VolumeStrategyMindmap = ({ field, value, onChange, error }) => {
  const { name, label, required } = field;
  const svgRef = useRef(null);
  const [editingNode, setEditingNode] = useState(null);
  const [editingText, setEditingText] = useState('');
  const [zoom, setZoom] = useState(1);
  const [selectedNode, setSelectedNode] = useState(null);

  // Search and filter state
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTypes, setSelectedTypes] = useState([]);
  const [highlightedNodes, setHighlightedNodes] = useState([]);

  const [mindmapData, setMindmapData] = useState(() => parseValue(value));

  // Initialize undo/redo with the parsed value
  const {
    canUndo,
    canRedo,
    undo,
    redo,
    pushToHistory
  } = useUndoRedo(parseValue(value));

  const updateValue = useCallback((newData, skipHistory = false) => {
    setMindmapData(newData);

    if (!skipHistory) {
      pushToHistory(newData);
    }

    const textRepresentation = convertToText(newData);
    onChange(name, textRepresentation);
  }, [name, onChange, pushToHistory]);

  // Use the custom D3 hook
  const { resetView } = useMindmapD3(
    svgRef,
    mindmapData,
    selectedNode,
    setSelectedNode,
    setEditingNode,
    setEditingText,
    setZoom,
    updateValue,
    highlightedNodes
  );

  const addNode = (nodeType) => {
    if (!selectedNode) return;
    const newData = addNodeToTree(mindmapData, selectedNode, nodeType);
    if (newData) updateValue(newData);
  };

  const deleteNode = () => {
    console.log('deleteNode called, selectedNode:', selectedNode);
    if (!selectedNode || selectedNode === 'root') {
      console.log('Cannot delete: no selection or trying to delete root');
      return;
    }

    console.log('Attempting to remove node:', selectedNode);
    const newData = removeNodeFromTree(mindmapData, selectedNode);
    console.log('Remove result:', newData ? 'Success' : 'Failed');

    if (newData) {
      setSelectedNode(null);
      updateValue(newData);
      console.log('Node successfully deleted and state updated');
    } else {
      console.log('Failed to delete node - not found in tree');
    }
  };

  const saveEdit = () => {
    if (!editingNode || !editingText.trim()) return;
    const newData = updateNodeInTree(mindmapData, editingNode, editingText);
    if (newData) {
      updateValue(newData);
      setEditingNode(null);
      setEditingText('');
    }
  };

  const cancelEdit = () => {
    setEditingNode(null);
    setEditingText('');
  };

  const handleUndo = useCallback(() => {
    const previousState = undo();
    if (previousState) {
      updateValue(previousState, true); // Skip history to prevent infinite loop
    }
  }, [undo, updateValue]);

  const handleRedo = useCallback(() => {
    const nextState = redo();
    if (nextState) {
      updateValue(nextState, true); // Skip history to prevent infinite loop
    }
  }, [redo, updateValue]);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.ctrlKey || e.metaKey) {
        if (e.key === 'z' && !e.shiftKey) {
          e.preventDefault();
          handleUndo();
        } else if ((e.key === 'y') || (e.key === 'z' && e.shiftKey)) {
          e.preventDefault();
          handleRedo();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [canUndo, canRedo, handleUndo, handleRedo]);

  // Search functionality
  const matchingNodes = searchNodes(mindmapData, searchTerm, selectedTypes);

  const handleSearchChange = (term) => {
    setSearchTerm(term);
    if (term) {
      const matches = searchNodes(mindmapData, term, selectedTypes);
      setHighlightedNodes(matches.map(node => node.id));
    } else {
      setHighlightedNodes([]);
    }
  };

  const handleTypeFilterChange = (types) => {
    setSelectedTypes(types);
    if (searchTerm) {
      const matches = searchNodes(mindmapData, searchTerm, types);
      setHighlightedNodes(matches.map(node => node.id));
    }
  };

  const handleClearFilters = () => {
    setSearchTerm('');
    setSelectedTypes([]);
    setHighlightedNodes([]);
  };

  const handleNavigateToNode = (nodeId) => {
    setSelectedNode(nodeId);
    // TODO: Add smooth animation to focus on the node
  };

  const handleOrganizeNodes = (layoutType) => {
    const organizedData = organizeNodes(mindmapData, layoutType);
    updateValue(organizedData);
  };

  const handleSnapToGrid = () => {
    const snappedData = snapToGrid(mindmapData);
    updateValue(snappedData);
  };

  return (
    <div className="mb-8 w-full" role="region" aria-label="Volume Strategy Mindmap">
      <label className="block text-lg font-semibold mb-4 text-gray-800">
        {label} {required && <span className="text-red-500">*</span>}
      </label>

      <div className="w-full border rounded-xl overflow-hidden shadow-sm bg-white">
        <MindmapControls
          selectedNode={selectedNode}
          zoom={zoom}
          onAddNode={addNode}
          onDeleteNode={deleteNode}
          onResetView={resetView}
          onUndo={handleUndo}
          onRedo={handleRedo}
          canUndo={canUndo}
          canRedo={canRedo}
          onOrganizeNodes={handleOrganizeNodes}
          onSnapToGrid={handleSnapToGrid}
        />

        <SearchFilter
          searchTerm={searchTerm}
          onSearchChange={handleSearchChange}
          selectedTypes={selectedTypes}
          onTypeFilterChange={handleTypeFilterChange}
          onClearFilters={handleClearFilters}
          matchingNodes={matchingNodes}
          onNavigateToNode={handleNavigateToNode}
        />

        <div className="relative bg-gray-50 w-full min-h-[700px]">
          <svg
            ref={svgRef}
            width="100%"
            height="700"
            viewBox="0 0 1000 700"
            preserveAspectRatio="none"
            className="border-none w-full h-full"
            style={{
              background: 'linear-gradient(45deg, #f8fafc 25%, transparent 25%), linear-gradient(-45deg, #f8fafc 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #f8fafc 75%), linear-gradient(-45deg, transparent 75%, #f8fafc 75%)',
              backgroundSize: '20px 20px',
              backgroundPosition: '0 0, 0 10px, 10px -10px, -10px 0px'
            }}
            tabIndex="0"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && editingNode) saveEdit();
              if (e.key === 'Escape' && editingNode) cancelEdit();
            }}
          />

          <EditModal
            editingNode={editingNode}
            editingText={editingText}
            setEditingText={setEditingText}
            onSave={saveEdit}
            onCancel={cancelEdit}
          />
        </div>

        <div className="w-full bg-gray-100 px-6 py-3 border-t">
          <p className="text-xs text-gray-600">
            💡 Tip: Select a node and click + to add children. Double-click nodes to rename them.
            Use mouse wheel to zoom and drag to pan around the mindmap.
          </p>
        </div>
      </div>

      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
};

export default VolumeStrategyMindmap;