import React, { useEffect, useMemo, useCallback } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  ConnectionMode,
} from 'reactflow';
import 'reactflow/dist/style.css';

// Modular imports
import { SwimlaneBackground, SwimlaneHeader, SolutionNode } from './nodes';
import { LabeledStraightEdge } from './edges';
import { InputModal, DiagramToolbar, SettingsPanel } from './ui';
import { useDiagramState, useDiagramActions } from './hooks';
import { getInitialNodes, getInitialEdges } from './CDEFlowDiagram.constants';

/**
 * CDE Flow Diagram - Interactive workflow diagram for Common Data Environment
 * Based on ISO 19650 standards
 */
const CDEFlowDiagram = ({ field, value, onChange, error }) => {
  const { name, label, required } = field;

  // Suppress ResizeObserver errors (harmless console spam from ReactFlow)
  useEffect(() => {
    const errorHandler = (e) => {
      if (e.message?.includes('ResizeObserver loop')) {
        e.stopImmediatePropagation();
        e.preventDefault();
        return true;
      }
    };

    const unhandledRejectionHandler = (e) => {
      if (e.reason?.message?.includes('ResizeObserver loop')) {
        e.preventDefault();
      }
    };

    window.addEventListener('error', errorHandler);
    window.addEventListener('unhandledrejection', unhandledRejectionHandler);

    return () => {
      window.removeEventListener('error', errorHandler);
      window.removeEventListener('unhandledrejection', unhandledRejectionHandler);
    };
  }, []);

  // Use custom hook for state management
  const {
    nodes,
    setNodes,
    edges,
    setEdges,
    showSettings,
    setShowSettings,
    swimlaneCustom,
    setSwimlaneCustom,
    nodeStyle,
    setNodeStyle,
    modalState,
    setModalState,
    swimlaneMap,
    updateParent,
    handleNodesChange,
    handleEdgesChange,
    handleNodesDelete,
    handleEdgesDelete,
    onConnect,
    handleNodeDragStop,
  } = useDiagramState(value, onChange, name);

  // Handle node label change
  const handleNodeLabelChange = useCallback((nodeId, newLabel) => {
    setNodes((nds) => {
      const updatedNodes = nds.map((node) =>
        node.id === nodeId ? { ...node, data: { ...node.data, label: newLabel } } : node
      );
      updateParent(updatedNodes, edges);
      return updatedNodes;
    });
  }, [setNodes, edges, updateParent]);

  // Handle edge label change
  const handleEdgeLabelChange = useCallback((edgeId, newLabel) => {
    setEdges((eds) => {
      const updatedEdges = eds.map((edge) =>
        edge.id === edgeId ? { ...edge, data: { ...edge.data, label: newLabel } } : edge
      );
      updateParent(nodes, updatedEdges);
      return updatedEdges;
    });
  }, [setEdges, nodes, updateParent]);

  // Handle swimlane header label change
  const handleSwimlaneHeaderChange = useCallback((nodeId, property, newValue) => {
    setNodes((nds) => {
      const updatedNodes = nds.map((node) => {
        const swimlaneId = node.id.replace('header-', '').replace('bg-', '');
        const targetId = nodeId.replace('header-', '').replace('bg-', '');
        if (swimlaneId === targetId && (node.id.startsWith('header-') || node.id.startsWith('bg-'))) {
          return { ...node, data: { ...node.data, [property]: newValue } };
        }
        return node;
      });
      updateParent(updatedNodes, edges);
      return updatedNodes;
    });
  }, [setNodes, edges, updateParent]);

  // Use custom hook for actions
  const {
    openAddNodeModal,
    handleAddNode,
    openAddSwimlaneModal,
    handleAddSwimlane,
    removeSwimlane,
    clearDiagram
  } = useDiagramActions({
    nodes,
    edges,
    setNodes,
    setEdges,
    updateParent,
    setModalState,
    swimlaneMap,
    handleNodeLabelChange,
    handleEdgeLabelChange
  });

  // Reset to default
  const resetToDefault = useCallback(() => {
    if (window.confirm('Reset to default ISO 19650 workflow diagram?')) {
      const defaultNodes = getInitialNodes();
      const defaultEdges = getInitialEdges();
      setNodes(defaultNodes);
      setEdges(defaultEdges);
      updateParent(defaultNodes, defaultEdges);
    }
  }, [setNodes, setEdges, updateParent]);

  // Export diagram as JSON
  const exportDiagram = useCallback(() => {
    const diagramData = { nodes, edges };
    const dataStr = JSON.stringify(diagramData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `cde-diagram-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }, [nodes, edges]);

  // Import diagram from JSON
  const importDiagram = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const imported = JSON.parse(event.target?.result);
        if (imported.nodes && imported.edges) {
          setNodes(imported.nodes);
          setEdges(imported.edges);
          if (imported.swimlaneCustom) {
            setSwimlaneCustom(imported.swimlaneCustom);
          }
          if (imported.nodeStyle) {
            setNodeStyle(imported.nodeStyle);
          }
          updateParent(imported.nodes, imported.edges, imported.swimlaneCustom, imported.nodeStyle);
        }
      } catch (err) {
        alert('Invalid diagram file');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  }, [setNodes, setEdges, setSwimlaneCustom, setNodeStyle, updateParent]);

  // Update swimlane customization
  const updateSwimlaneCustom = useCallback((swimlaneId, property, newValue) => {
    setSwimlaneCustom(prev => {
      const updated = {
        ...prev,
        [swimlaneId]: {
          ...prev[swimlaneId],
          [property]: newValue
        }
      };

      setNodes(nds => {
        const updatedNodes = nds.map(node => {
          if ((node.id === `bg-${swimlaneId}` || node.id === `header-${swimlaneId}`) && updated[swimlaneId]) {
            return {
              ...node,
              data: {
                ...node.data,
                ...updated[swimlaneId]
              }
            };
          }
          return node;
        });
        updateParent(updatedNodes, edges, updated);
        return updatedNodes;
      });

      return updated;
    });
  }, [setNodes, edges, updateParent, setSwimlaneCustom]);

  // Update node style
  const updateNodeStyle = useCallback((property, newValue) => {
    setNodeStyle(prev => {
      const updated = { ...prev, [property]: newValue };
      updateParent(nodes, edges, swimlaneCustom, updated);
      return updated;
    });
  }, [nodes, edges, swimlaneCustom, updateParent, setNodeStyle]);

  // Memoize node types with onChange handlers
  const nodeTypes = useMemo(() => ({
    swimlaneBackground: (props) => <SwimlaneBackground {...props} data={{ ...props.data, onAddNode: openAddNodeModal }} />,
    swimlaneHeader: (props) => <SwimlaneHeader {...props} data={{ ...props.data, onAddSolution: openAddNodeModal, onLabelChange: handleSwimlaneHeaderChange }} />,
    solution: (props) => <SolutionNode {...props} data={{ ...props.data, onChange: handleNodeLabelChange, nodeStyle }} />,
  }), [handleNodeLabelChange, openAddNodeModal, handleSwimlaneHeaderChange, nodeStyle]);

  // Memoize edge types
  const edgeTypes = useMemo(() => ({
    labeledStraight: (props) => <LabeledStraightEdge {...props} data={{ ...props.data, onChange: handleEdgeLabelChange }} />,
  }), [handleEdgeLabelChange]);

  return (
    <div className="mb-8 w-full">
      <label className="block text-lg font-semibold mb-4 text-gray-800">
        {label} {required && <span className="text-red-500">*</span>}
      </label>

      <div style={{ height: '600px', border: '1px solid #e5e7eb', borderRadius: '8px', overflow: 'hidden' }}>
        <DiagramToolbar
          onReset={resetToDefault}
          onAddSwimlane={openAddSwimlaneModal}
          onRemoveSwimlane={removeSwimlane}
          onClear={clearDiagram}
          onExport={exportDiagram}
          onImport={importDiagram}
          onToggleSettings={() => setShowSettings(!showSettings)}
          showSettings={showSettings}
        />

        {showSettings && (
          <SettingsPanel
            swimlaneCustom={swimlaneCustom}
            nodeStyle={nodeStyle}
            onSwimlaneCustomChange={updateSwimlaneCustom}
            onNodeStyleChange={updateNodeStyle}
          />
        )}

        <div style={{ height: showSettings ? 'calc(100% - 400px)' : 'calc(100% - 60px)' }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={handleNodesChange}
            onEdgesChange={handleEdgesChange}
            onNodesDelete={handleNodesDelete}
            onEdgesDelete={handleEdgesDelete}
            onNodeDragStop={handleNodeDragStop}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            connectionMode={ConnectionMode.Loose}
            fitView
            deleteKeyCode="Delete"
            attributionPosition="bottom-left"
            minZoom={0.5}
            maxZoom={1.5}
          >
            <MiniMap nodeStrokeWidth={3} zoomable pannable />
            <Controls />
            <Background color="#e5e7eb" gap={16} />
          </ReactFlow>
        </div>
      </div>

      <div className="mt-2 text-sm text-gray-600">
        💡 <strong>Tips:</strong> Double-click nodes/edges to edit • Select and press Delete to remove • Drag horizontally to connect nodes across swimlanes
      </div>

      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}

      <InputModal
        isOpen={modalState.isOpen && modalState.type === 'addNode'}
        onClose={() => setModalState({ isOpen: false, type: null, swimlaneId: null })}
        onSubmit={(name) => handleAddNode(name, modalState)}
        title="Add Solution"
        placeholder="Enter solution name (e.g., SharePoint, BIM360)"
      />

      <InputModal
        isOpen={modalState.isOpen && modalState.type === 'addSwimlane'}
        onClose={() => setModalState({ isOpen: false, type: null, swimlaneId: null })}
        onSubmit={handleAddSwimlane}
        title="Add Swimlane"
        placeholder="Enter swimlane name (e.g., Review, Approved)"
      />
    </div>
  );
};

export default CDEFlowDiagram;
