import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  MarkerType,
  Panel,
  useReactFlow,
  ReactFlowProvider,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  Plus,
  Download,
  Upload,
  RotateCcw,
  RotateCw,
  Layers,
  Trash2,
  Grid3x3,
  AlignHorizontalJustifyCenter,
  AlignVerticalJustifyCenter,
} from 'lucide-react';

import { nodeTypes, availableShapes } from './CustomNodes';
import {
  convertToReactFlow,
  convertFromReactFlow,
  getDefaultDiagramStructure,
} from './diagramMigration';
import { getTemplate, getTemplateOptions } from './diagramTemplates';

const CDEDiagramBuilderInner = ({ field, value, onChange, error }) => {
  const { name, label, required } = field;
  const { screenToFlowPosition } = useReactFlow();

  // Parse initial value
  const getInitialData = () => {
    if (typeof value === 'string' && value) {
      try {
        const parsed = JSON.parse(value);
        return convertToReactFlow(parsed);
      } catch (e) {
        console.warn('Invalid diagram JSON, using default:', e);
      }
    }
    return convertToReactFlow(getDefaultDiagramStructure());
  };

  const initialData = getInitialData();
  const [nodes, setNodes, onNodesChange] = useNodesState(initialData.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialData.edges);
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [snapToGrid, setSnapToGrid] = useState(true);
  const [showToolbar, setShowToolbar] = useState(true);
  const [selectedNodes, setSelectedNodes] = useState([]);
  const [showTemplates, setShowTemplates] = useState(false);
  const reactFlowWrapper = useRef(null);

  // Save to parent form on changes
  const saveToParent = useCallback(
    (currentNodes, currentEdges) => {
      const oldFormat = convertFromReactFlow(currentNodes, currentEdges);
      onChange(name, JSON.stringify(oldFormat, null, 2));
    },
    [name, onChange]
  );

  // Track changes and save
  useEffect(() => {
    if (nodes.length > 0 || edges.length > 0) {
      saveToParent(nodes, edges);
    }
  }, [nodes, edges, saveToParent]);

  // Add to history for undo/redo
  const addToHistory = useCallback(() => {
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push({
      nodes: JSON.parse(JSON.stringify(nodes)),
      edges: JSON.parse(JSON.stringify(edges)),
    });
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  }, [nodes, edges, history, historyIndex]);

  // Undo
  const undo = useCallback(() => {
    if (historyIndex > 0) {
      const prevState = history[historyIndex - 1];
      setNodes(prevState.nodes);
      setEdges(prevState.edges);
      setHistoryIndex(historyIndex - 1);
    }
  }, [history, historyIndex, setNodes, setEdges]);

  // Redo
  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const nextState = history[historyIndex + 1];
      setNodes(nextState.nodes);
      setEdges(nextState.edges);
      setHistoryIndex(historyIndex + 1);
    }
  }, [history, historyIndex, setNodes, setEdges]);

  // Handle connection between nodes
  const onConnect = useCallback(
    (params) => {
      addToHistory();
      setEdges((eds) =>
        addEdge(
          {
            ...params,
            type: 'smoothstep',
            animated: false,
            markerEnd: {
              type: MarkerType.ArrowClosed,
              width: 20,
              height: 20,
            },
            style: {
              strokeWidth: 2,
            },
          },
          eds
        )
      );
    },
    [setEdges, addToHistory]
  );

  // Add a new node
  const addNode = useCallback(
    (type) => {
      addToHistory();
      const position = screenToFlowPosition({ x: 400, y: 300 });

      const newNode = {
        id: `node-${Date.now()}`,
        type,
        position,
        data: {
          label: `New ${type.charAt(0).toUpperCase() + type.slice(1)}`,
          type,
        },
      };
      setNodes((nds) => [...nds, newNode]);
    },
    [screenToFlowPosition, setNodes, addToHistory]
  );

  // Add a new layer/swimlane
  const addLayer = useCallback(() => {
    addToHistory();
    const maxY = nodes.reduce((max, node) => {
      if (node.type === 'swimlane') {
        return Math.max(max, node.position.y);
      }
      return max;
    }, 0);

    const newLayer = {
      id: `layer-${Date.now()}`,
      type: 'swimlane',
      position: { x: 0, y: maxY + 200 },
      data: {
        label: 'New Layer',
        layerIndex: nodes.filter((n) => n.type === 'swimlane').length,
      },
      style: {
        width: 1200,
        height: 150,
      },
      draggable: false,
      selectable: true,
    };
    setNodes((nds) => [...nds, newLayer]);
  }, [nodes, setNodes, addToHistory]);

  // Delete selected nodes
  const deleteSelected = useCallback(() => {
    addToHistory();
    setNodes((nds) => nds.filter((node) => !node.selected));
    setEdges((eds) =>
      eds.filter((edge) => {
        const sourceExists = nodes.find((n) => n.id === edge.source && !n.selected);
        const targetExists = nodes.find((n) => n.id === edge.target && !n.selected);
        return sourceExists && targetExists;
      })
    );
  }, [nodes, setNodes, setEdges, addToHistory]);

  // Export as JSON
  const exportJSON = useCallback(() => {
    const oldFormat = convertFromReactFlow(nodes, edges);
    const dataStr = JSON.stringify(oldFormat, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'diagram-export.json';
    link.click();
    URL.revokeObjectURL(url);
  }, [nodes, edges]);

  // Note: Export as SVG/PNG can be added later with additional libraries
  // - SVG: Use React Flow's getNodesBounds and getViewportForBounds methods
  // - PNG: Install html2canvas and use: html2canvas(document.querySelector('.react-flow'))

  // Import from JSON
  const importJSON = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
          try {
            const imported = JSON.parse(event.target.result);
            const converted = convertToReactFlow(imported);
            addToHistory();
            setNodes(converted.nodes);
            setEdges(converted.edges);
          } catch (error) {
            alert('Invalid JSON file');
          }
        };
        reader.readAsText(file);
      }
    };
    input.click();
  }, [setNodes, setEdges, addToHistory]);

  // Align selected nodes horizontally
  const alignHorizontal = useCallback(() => {
    const selected = nodes.filter((n) => n.selected && n.type !== 'swimlane');
    if (selected.length < 2) return;

    addToHistory();
    const avgY = selected.reduce((sum, n) => sum + n.position.y, 0) / selected.length;
    setNodes((nds) =>
      nds.map((node) => {
        if (node.selected && node.type !== 'swimlane') {
          return { ...node, position: { ...node.position, y: avgY } };
        }
        return node;
      })
    );
  }, [nodes, setNodes, addToHistory]);

  // Align selected nodes vertically
  const alignVertical = useCallback(() => {
    const selected = nodes.filter((n) => n.selected && n.type !== 'swimlane');
    if (selected.length < 2) return;

    addToHistory();
    const avgX = selected.reduce((sum, n) => sum + n.position.x, 0) / selected.length;
    setNodes((nds) =>
      nds.map((node) => {
        if (node.selected && node.type !== 'swimlane') {
          return { ...node, position: { ...node.position, x: avgX } };
        }
        return node;
      })
    );
  }, [nodes, setNodes, addToHistory]);

  // Handle selection change
  const onSelectionChange = useCallback(({ nodes: selectedNodes }) => {
    setSelectedNodes(selectedNodes);
  }, []);

  // Load template
  const loadTemplate = useCallback(
    (templateKey) => {
      const template = getTemplate(templateKey);
      const converted = convertToReactFlow(template.data);
      addToHistory();
      setNodes(converted.nodes);
      setEdges(converted.edges);
      setShowTemplates(false);
    },
    [setNodes, setEdges, addToHistory]
  );

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.ctrlKey || event.metaKey) {
        if (event.key === 'z' && !event.shiftKey) {
          event.preventDefault();
          undo();
        } else if (event.key === 'z' && event.shiftKey) {
          event.preventDefault();
          redo();
        } else if (event.key === 'y') {
          event.preventDefault();
          redo();
        }
      }
      if (event.key === 'Delete' || event.key === 'Backspace') {
        if (selectedNodes.length > 0) {
          event.preventDefault();
          deleteSelected();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [undo, redo, deleteSelected, selectedNodes]);

  return (
    <div className="mb-8 w-full">
      <label className="block text-lg font-semibold mb-4 text-gray-800">
        {label} {required && <span className="text-red-500">*</span>}
      </label>

      <div className="w-full border rounded-xl overflow-hidden shadow-lg bg-white">
        {/* Header */}
        <div className="bg-gradient-to-r from-purple-50 to-purple-100 px-6 py-4 border-b border-purple-200">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <Layers className="w-5 h-5 text-purple-600" />
              <span className="text-base font-semibold text-purple-800">
                CDE Diagram Builder V2 (Infinite Canvas)
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={undo}
                disabled={historyIndex <= 0}
                className="p-2 text-gray-600 hover:bg-gray-100 rounded disabled:opacity-30"
                title="Undo (Ctrl+Z)"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
              <button
                onClick={redo}
                disabled={historyIndex >= history.length - 1}
                className="p-2 text-gray-600 hover:bg-gray-100 rounded disabled:opacity-30"
                title="Redo (Ctrl+Y)"
              >
                <RotateCw className="w-4 h-4" />
              </button>
              <div className="w-px h-6 bg-gray-300" />
              <button
                onClick={() => setSnapToGrid(!snapToGrid)}
                className={`p-2 rounded ${
                  snapToGrid ? 'bg-blue-500 text-white' : 'text-gray-600 hover:bg-gray-100'
                }`}
                title="Toggle Snap to Grid"
              >
                <Grid3x3 className="w-4 h-4" />
              </button>
              <button
                onClick={alignHorizontal}
                disabled={selectedNodes.length < 2}
                className="p-2 text-gray-600 hover:bg-gray-100 rounded disabled:opacity-30"
                title="Align Horizontal"
              >
                <AlignHorizontalJustifyCenter className="w-4 h-4" />
              </button>
              <button
                onClick={alignVertical}
                disabled={selectedNodes.length < 2}
                className="p-2 text-gray-600 hover:bg-gray-100 rounded disabled:opacity-30"
                title="Align Vertical"
              >
                <AlignVerticalJustifyCenter className="w-4 h-4" />
              </button>
              <div className="w-px h-6 bg-gray-300" />
              <button
                onClick={deleteSelected}
                disabled={selectedNodes.length === 0}
                className="p-2 text-red-600 hover:bg-red-100 rounded disabled:opacity-30"
                title="Delete Selected (Del)"
              >
                <Trash2 className="w-4 h-4" />
              </button>
              <div className="w-px h-6 bg-gray-300" />
              <button
                onClick={importJSON}
                className="p-2 text-gray-600 hover:bg-gray-100 rounded"
                title="Import JSON"
              >
                <Upload className="w-4 h-4" />
              </button>
              <button
                onClick={exportJSON}
                className="p-2 text-gray-600 hover:bg-gray-100 rounded"
                title="Export JSON"
              >
                <Download className="w-4 h-4" />
              </button>
              <div className="w-px h-6 bg-gray-300" />
              <div className="relative">
                <button
                  onClick={() => setShowTemplates(!showTemplates)}
                  className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-all"
                  title="Load Template"
                >
                  <Layers className="w-4 h-4" />
                  <span>Templates</span>
                </button>
                {showTemplates && (
                  <div className="absolute right-0 top-full mt-2 bg-white border border-gray-300 rounded-lg shadow-xl z-50 w-64">
                    <div className="p-2 border-b border-gray-200 bg-gray-50 rounded-t-lg">
                      <h4 className="font-semibold text-sm text-gray-700">Load Template</h4>
                    </div>
                    <div className="max-h-96 overflow-y-auto">
                      {getTemplateOptions().map((template) => (
                        <button
                          key={template.value}
                          onClick={() => loadTemplate(template.value)}
                          className="w-full text-left px-4 py-3 hover:bg-gray-50 border-b border-gray-100 transition-colors"
                        >
                          <div className="font-medium text-sm text-gray-800">{template.label}</div>
                          <div className="text-xs text-gray-500 mt-1">{template.description}</div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              <button
                onClick={addLayer}
                className="flex items-center space-x-2 bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition-all"
              >
                <Plus className="w-4 h-4" />
                <span>Add Layer</span>
              </button>
            </div>
          </div>
          <p className="text-sm text-purple-700 mt-2">
            Infinite canvas • Drag to pan • Scroll to zoom • Multi-select (Shift+Click) • Templates for quick start • Connect nodes by dragging handles
          </p>
        </div>

        {/* React Flow Canvas */}
        <div ref={reactFlowWrapper} style={{ height: '700px' }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onSelectionChange={onSelectionChange}
            nodeTypes={nodeTypes}
            snapToGrid={snapToGrid}
            snapGrid={[15, 15]}
            fitView
            attributionPosition="bottom-left"
            multiSelectionKeyCode="Shift"
            deleteKeyCode="Delete"
          >
            <Background variant="dots" gap={16} size={1} color="#ddd" />
            <Controls showInteractive={false} />
            <MiniMap
              nodeColor={(node) => {
                if (node.type === 'swimlane') return '#f3f4f6';
                return '#cbd5e1';
              }}
              maskColor="rgba(0, 0, 0, 0.1)"
              style={{ background: '#f8f9fa' }}
            />

            {/* Sidebar Shape Palette */}
            {showToolbar && (
              <Panel position="top-left" className="bg-white rounded-lg shadow-lg p-3 max-w-48">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-sm text-gray-700">Shapes</h4>
                  <button
                    onClick={() => setShowToolbar(false)}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    ×
                  </button>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {availableShapes.slice(0, 12).map((shape) => {
                    const Icon = shape.icon;
                    return (
                      <button
                        key={shape.type}
                        onClick={() => addNode(shape.type)}
                        className="flex flex-col items-center justify-center p-2 border border-gray-200 rounded hover:bg-gray-50 hover:border-blue-400 transition-all"
                        title={shape.label}
                      >
                        <Icon className="w-5 h-5 text-gray-600" />
                        <span className="text-xs mt-1 text-gray-600 truncate w-full text-center">
                          {shape.label}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </Panel>
            )}

            {!showToolbar && (
              <Panel position="top-left">
                <button
                  onClick={() => setShowToolbar(true)}
                  className="bg-white rounded-lg shadow-lg p-2 hover:bg-gray-50"
                  title="Show Toolbar"
                >
                  <Plus className="w-5 h-5 text-gray-600" />
                </button>
              </Panel>
            )}
          </ReactFlow>
        </div>

        {/* Footer */}
        <div className="w-full bg-gray-100 px-6 py-3 border-t">
          <p className="text-xs text-gray-600">
            💡 Tip: Use Ctrl+Z/Y for undo/redo • Shift+Click for multi-select • Drag handles to connect • Del to delete
          </p>
        </div>
      </div>

      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
};

const CDEDiagramBuilderV2 = (props) => (
  <ReactFlowProvider>
    <CDEDiagramBuilderInner {...props} />
  </ReactFlowProvider>
);

export default CDEDiagramBuilderV2;
