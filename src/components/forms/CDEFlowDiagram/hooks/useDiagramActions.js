import { useCallback } from 'react';
import { SWIMLANES } from '../CDEFlowDiagram.constants';

/**
 * Custom hook for diagram actions (add/remove nodes, swimlanes, etc.)
 */
export const useDiagramActions = ({
  nodes,
  edges,
  setNodes,
  setEdges,
  updateParent,
  setModalState,
  swimlaneMap,
  handleNodeLabelChange,
  handleEdgeLabelChange
}) => {
  // Open modal for adding node to swimlane
  const openAddNodeModal = useCallback((swimlaneId) => {
    setModalState({ isOpen: true, type: 'addNode', swimlaneId });
  }, [setModalState]);

  // Handle adding node from modal
  const handleAddNode = useCallback((solutionName, modalState) => {
    const { swimlaneId } = modalState;
    const swimlane = swimlaneMap.get(swimlaneId);
    if (!swimlane) return;

    const existingNodesInLane = nodes.filter(n => n.data?.swimlane === swimlaneId && n.type === 'solution');
    const yPosition = 80 + (existingNodesInLane.length * 80);

    const newNodeId = `node-${Date.now()}`;
    const newNode = {
      id: newNodeId,
      type: 'solution',
      data: { label: solutionName, swimlane: swimlaneId, onChange: handleNodeLabelChange },
      position: { x: swimlane.x + 25, y: yPosition },
    };

    const newNodes = [...nodes, newNode];
    const currentIndex = SWIMLANES.findIndex(s => s.id === swimlaneId);
    const nextSwimlane = SWIMLANES[currentIndex + 1];

    let newEdges = [...edges];

    if (nextSwimlane) {
      const nextSwimlaneNodes = nodes.filter(n => n.data?.swimlane === nextSwimlane.id && n.type === 'solution');
      if (nextSwimlaneNodes.length > 0) {
        const targetNode = nextSwimlaneNodes[0];
        const newEdge = {
          id: `edge-${Date.now()}`,
          source: newNodeId,
          target: targetNode.id,
          type: 'labeledStraight',
          style: { stroke: nextSwimlane.borderColor, strokeWidth: 2 },
          data: { label: '', onChange: handleEdgeLabelChange }
        };
        newEdges = [...edges, newEdge];
      }
    }

    setNodes(newNodes);
    setEdges(newEdges);
    updateParent(newNodes, newEdges);
    setModalState({ isOpen: false, type: null, swimlaneId: null });
  }, [nodes, edges, setNodes, setEdges, updateParent, setModalState, swimlaneMap, handleNodeLabelChange, handleEdgeLabelChange]);

  // Open modal for adding swimlane
  const openAddSwimlaneModal = useCallback(() => {
    setModalState({ isOpen: true, type: 'addSwimlane', swimlaneId: null });
  }, [setModalState]);

  // Handle adding swimlane from modal
  const handleAddSwimlane = useCallback((name) => {
    const swimlaneHeaders = nodes.filter(n => n.id.startsWith('header-'));
    const newId = `lane-${Date.now()}`;
    const xPosition = swimlaneHeaders.length * 250;

    const newBg = {
      id: `bg-${newId}`,
      type: 'swimlaneBackground',
      data: {
        id: newId,
        label: name.toUpperCase(),
        fullLabel: name,
        color: '#e0e7ff',
        borderColor: '#6366f1',
        textColor: '#312e81',
        x: xPosition
      },
      position: { x: xPosition, y: 40 },
      draggable: false,
      selectable: false,
    };

    const newHeader = {
      id: `header-${newId}`,
      type: 'swimlaneHeader',
      data: {
        id: newId,
        label: name.toUpperCase(),
        fullLabel: name,
        color: '#e0e7ff',
        borderColor: '#6366f1',
        textColor: '#312e81',
      },
      position: { x: xPosition + 15, y: -35 },
      draggable: false,
    };

    const newNodes = [...nodes, newBg, newHeader];
    setNodes(newNodes);
    updateParent(newNodes, edges);
    setModalState({ isOpen: false, type: null, swimlaneId: null });
  }, [nodes, setNodes, edges, updateParent, setModalState]);

  // Remove last swimlane
  const removeSwimlane = useCallback(() => {
    const swimlaneHeaders = nodes.filter(n => n.id.startsWith('header-'));
    if (swimlaneHeaders.length <= 1) {
      alert('Cannot remove the last swimlane');
      return;
    }

    if (!window.confirm('Remove the last swimlane? This will delete all solutions in it.')) {
      return;
    }

    const lastHeader = swimlaneHeaders[swimlaneHeaders.length - 1];
    const swimlaneId = lastHeader.id.replace('header-', '');

    const filteredNodes = nodes.filter(n =>
      !n.id.includes(swimlaneId) && n.data?.swimlane !== swimlaneId
    );
    const filteredEdges = edges.filter(e => {
      const sourceNode = nodes.find(n => n.id === e.source);
      const targetNode = nodes.find(n => n.id === e.target);
      return sourceNode?.data?.swimlane !== swimlaneId && targetNode?.data?.swimlane !== swimlaneId;
    });

    setNodes(filteredNodes);
    setEdges(filteredEdges);
    updateParent(filteredNodes, filteredEdges);
  }, [nodes, edges, setNodes, setEdges, updateParent]);

  // Clear diagram
  const clearDiagram = useCallback(() => {
    if (window.confirm('Are you sure you want to clear all solutions?')) {
      const swimlaneNodesOnly = nodes.filter(n => n.id.startsWith('header-') || n.id.startsWith('bg-'));
      setNodes(swimlaneNodesOnly);
      setEdges([]);
      updateParent(swimlaneNodesOnly, []);
    }
  }, [nodes, setNodes, setEdges, updateParent]);

  return {
    openAddNodeModal,
    handleAddNode,
    openAddSwimlaneModal,
    handleAddSwimlane,
    removeSwimlane,
    clearDiagram
  };
};
