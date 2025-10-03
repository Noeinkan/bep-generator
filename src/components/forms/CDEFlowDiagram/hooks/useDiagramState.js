import { useState, useCallback, useMemo } from 'react';
import { useEdgesState, useNodesState, addEdge } from 'reactflow';
import { SWIMLANES, getInitialNodes, getInitialEdges } from '../CDEFlowDiagram.constants';

/**
 * Custom hook for managing diagram state
 */
export const useDiagramState = (initialValue, onChange, name) => {
  // Parse value or use defaults
  const parseValue = (val) => {
    if (typeof val === 'string') {
      try {
        return JSON.parse(val);
      } catch {
        return { nodes: getInitialNodes(), edges: getInitialEdges() };
      }
    }
    return val || { nodes: getInitialNodes(), edges: getInitialEdges() };
  };

  const parsedValue = parseValue(initialValue);

  const [nodes, setNodes, onNodesChange] = useNodesState(parsedValue.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(parsedValue.edges);
  const [showSettings, setShowSettings] = useState(false);
  const [swimlaneCustom, setSwimlaneCustom] = useState(parsedValue.swimlaneCustom || {});
  const [nodeStyle, setNodeStyle] = useState(parsedValue.nodeStyle || {
    background: '#ffffff',
    borderColor: '#6b7280',
    textColor: '#000000'
  });
  const [modalState, setModalState] = useState({ isOpen: false, type: null, swimlaneId: null });

  // Memoized swimlane lookup map for performance
  const swimlaneMap = useMemo(() => {
    const map = new Map();
    SWIMLANES.forEach(lane => map.set(lane.id, lane));
    return map;
  }, []);

  // Update parent component when diagram changes
  const updateParent = useCallback((newNodes, newEdges, customSettings = swimlaneCustom, customNodeStyle = nodeStyle) => {
    if (onChange) {
      onChange(name, JSON.stringify({
        nodes: newNodes,
        edges: newEdges,
        swimlaneCustom: customSettings,
        nodeStyle: customNodeStyle
      }, null, 2));
    }
  }, [onChange, name, swimlaneCustom, nodeStyle]);

  // Handle node changes
  const handleNodesChange = useCallback((changes) => {
    onNodesChange(changes);
  }, [onNodesChange]);

  // Handle edge changes
  const handleEdgesChange = useCallback((changes) => {
    onEdgesChange(changes);
  }, [onEdgesChange]);

  // Handle node deletion (protect headers and backgrounds)
  const handleNodesDelete = useCallback((deleted) => {
    const deletedIds = deleted.filter(n => !n.id.startsWith('header-') && !n.id.startsWith('bg-')).map(n => n.id);
    setNodes((nds) => {
      const remaining = nds.filter(n => !deletedIds.includes(n.id));
      updateParent(remaining, edges);
      return remaining;
    });
  }, [setNodes, edges, updateParent]);

  // Handle edge deletion
  const handleEdgesDelete = useCallback((deleted) => {
    const deletedIds = deleted.map(e => e.id);
    setEdges((eds) => {
      const remaining = eds.filter(e => !deletedIds.includes(e.id));
      updateParent(nodes, remaining);
      return remaining;
    });
  }, [setEdges, nodes, updateParent]);

  // Handle connection
  const onConnect = useCallback((params) => {
    const newEdge = {
      ...params,
      type: 'labeledStraight',
      data: { label: '' }
    };
    const newEdges = addEdge(newEdge, edges);
    setEdges(newEdges);
    updateParent(nodes, newEdges);
  }, [edges, setEdges, nodes, updateParent]);

  // Apply constraints after dragging stops
  const handleNodeDragStop = useCallback((event, node) => {
    if (node.id.startsWith('bg-') || node.id.startsWith('header-')) {
      const originalNode = nodes.find(n => n.id === node.id);
      if (originalNode) {
        setNodes((nds) =>
          nds.map((n) =>
            n.id === node.id ? { ...n, position: originalNode.position } : n
          )
        );
      }
      return;
    }

    if (node.type === 'solution' && node.data?.swimlane) {
      const swimlane = swimlaneMap.get(node.data.swimlane);
      if (swimlane) {
        setNodes((nds) => {
          const updatedNodes = nds.map((n) =>
            n.id === node.id ? { ...n, position: { x: swimlane.x + 25, y: node.position.y } } : n
          );
          updateParent(updatedNodes, edges);
          return updatedNodes;
        });
      }
    } else {
      updateParent(nodes, edges);
    }
  }, [nodes, edges, setNodes, updateParent, swimlaneMap]);

  return {
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
    onNodesChange,
    onEdgesChange
  };
};
