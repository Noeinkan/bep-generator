import React, { useCallback, useState } from 'react';
import ReactFlow, {
  addEdge,
  Background,
  Controls,
  MiniMap,
  useEdgesState,
  useNodesState,
  Handle,
  Position,
} from 'reactflow';
import 'reactflow/dist/style.css';

// Custom node component for CDE workflow states
const WorkflowStateNode = ({ data }) => {
  return (
    <div style={{
      padding: '12px 16px',
      border: '2px solid #3b82f6',
      borderRadius: '8px',
      background: '#dbeafe',
      minWidth: '180px',
      textAlign: 'center',
      fontWeight: '600',
      color: '#1e40af',
    }}>
      <Handle type="target" position={Position.Top} style={{ background: '#3b82f6' }} />
      <div>{data.label}</div>
      <Handle type="source" position={Position.Bottom} style={{ background: '#3b82f6' }} />
    </div>
  );
};

// Custom node component for solutions/platforms
const SolutionNode = ({ data }) => {
  return (
    <div style={{
      padding: '10px 14px',
      border: '1px solid #6b7280',
      borderRadius: '6px',
      background: '#ffffff',
      minWidth: '160px',
      textAlign: 'center',
      fontSize: '14px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    }}>
      <Handle type="target" position={Position.Top} style={{ background: '#6b7280' }} />
      <div>{data.label}</div>
      <Handle type="source" position={Position.Bottom} style={{ background: '#6b7280' }} />
    </div>
  );
};

const nodeTypes = {
  workflowState: WorkflowStateNode,
  solution: SolutionNode,
};

// Initial nodes based on ISO 19650 CDE workflow
const initialNodes = [
  // Workflow States (top row)
  {
    id: 'wip',
    type: 'workflowState',
    data: { label: 'WIP (Work in Progress)' },
    position: { x: 50, y: 50 },
  },
  {
    id: 'shared',
    type: 'workflowState',
    data: { label: 'SHARED (Coordination)' },
    position: { x: 300, y: 50 },
  },
  {
    id: 'published',
    type: 'workflowState',
    data: { label: 'PUBLISHED (Approved)' },
    position: { x: 550, y: 50 },
  },
  {
    id: 'archived',
    type: 'workflowState',
    data: { label: 'ARCHIVED (Reference)' },
    position: { x: 800, y: 50 },
  },
  // Example solutions (lower rows)
  {
    id: 'sol1',
    type: 'solution',
    data: { label: 'SharePoint' },
    position: { x: 50, y: 180 },
  },
  {
    id: 'sol2',
    type: 'solution',
    data: { label: 'Autodesk Docs' },
    position: { x: 300, y: 180 },
  },
  {
    id: 'sol3',
    type: 'solution',
    data: { label: 'BIMcollab' },
    position: { x: 300, y: 280 },
  },
  {
    id: 'sol4',
    type: 'solution',
    data: { label: 'Aconex' },
    position: { x: 550, y: 180 },
  },
];

// Initial edges showing workflow
const initialEdges = [
  // Workflow progression
  { id: 'e-wip-shared', source: 'wip', target: 'shared', animated: true, style: { stroke: '#3b82f6', strokeWidth: 2 } },
  { id: 'e-shared-published', source: 'shared', target: 'published', animated: true, style: { stroke: '#3b82f6', strokeWidth: 2 } },
  { id: 'e-published-archived', source: 'published', target: 'archived', animated: true, style: { stroke: '#3b82f6', strokeWidth: 2 } },
  // Solution connections
  { id: 'e-wip-sol1', source: 'wip', target: 'sol1', style: { stroke: '#9ca3af' } },
  { id: 'e-sol1-sol2', source: 'sol1', target: 'sol2', style: { stroke: '#9ca3af' } },
  { id: 'e-shared-sol2', source: 'shared', target: 'sol2', style: { stroke: '#9ca3af' } },
  { id: 'e-shared-sol3', source: 'shared', target: 'sol3', style: { stroke: '#9ca3af' } },
  { id: 'e-sol2-sol4', source: 'sol2', target: 'sol4', style: { stroke: '#9ca3af' } },
  { id: 'e-published-sol4', source: 'published', target: 'sol4', style: { stroke: '#9ca3af' } },
];

const CDEFlowDiagram = ({ field, value, onChange, error }) => {
  const { name, label, required } = field;

  // Parse value or use defaults
  const parseValue = (val) => {
    if (typeof val === 'string') {
      try {
        return JSON.parse(val);
      } catch {
        return { nodes: initialNodes, edges: initialEdges };
      }
    }
    return val || { nodes: initialNodes, edges: initialEdges };
  };

  const parsedValue = parseValue(value);
  const [nodes, setNodes, onNodesChange] = useNodesState(parsedValue.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(parsedValue.edges);
  const [nodeName, setNodeName] = useState('');
  const [nodeType, setNodeType] = useState('solution');

  // Update parent component when diagram changes
  const updateParent = useCallback((newNodes, newEdges) => {
    if (onChange) {
      onChange(name, JSON.stringify({ nodes: newNodes, edges: newEdges }, null, 2));
    }
  }, [onChange, name]);

  // Handle node changes and update parent
  const handleNodesChange = useCallback((changes) => {
    onNodesChange(changes);
    setNodes((nds) => {
      updateParent(nds, edges);
      return nds;
    });
  }, [onNodesChange, edges, updateParent, setNodes]);

  // Handle edge changes and update parent
  const handleEdgesChange = useCallback((changes) => {
    onEdgesChange(changes);
    setEdges((eds) => {
      updateParent(nodes, eds);
      return eds;
    });
  }, [onEdgesChange, nodes, updateParent, setEdges]);

  const onConnect = useCallback(
    (params) => {
      const newEdges = addEdge(params, edges);
      setEdges(newEdges);
      updateParent(nodes, newEdges);
    },
    [edges, setEdges, nodes, updateParent]
  );

  const addNode = useCallback(() => {
    if (!nodeName.trim()) return;

    const newNode = {
      id: `node-${Date.now()}`,
      type: nodeType,
      data: { label: nodeName },
      position: { x: Math.random() * 600, y: 200 + Math.random() * 200 },
    };

    const newNodes = [...nodes, newNode];
    setNodes(newNodes);
    updateParent(newNodes, edges);
    setNodeName('');
  }, [nodes, setNodes, nodeName, nodeType, edges, updateParent]);

  const clearDiagram = useCallback(() => {
    if (window.confirm('Are you sure you want to clear the entire diagram?')) {
      setNodes([]);
      setEdges([]);
      updateParent([], []);
    }
  }, [setNodes, setEdges, updateParent]);

  const resetToDefault = useCallback(() => {
    if (window.confirm('Reset to default ISO 19650 workflow diagram?')) {
      setNodes(initialNodes);
      setEdges(initialEdges);
      updateParent(initialNodes, initialEdges);
    }
  }, [setNodes, setEdges, updateParent]);

  return (
    <div className="mb-8 w-full">
      <label className="block text-lg font-semibold mb-4 text-gray-800">
        {label} {required && <span className="text-red-500">*</span>}
      </label>

      <div style={{ height: '600px', border: '1px solid #e5e7eb', borderRadius: '8px', overflow: 'hidden' }}>
        <div style={{
          padding: '12px 16px',
          background: '#f9fafb',
          borderBottom: '1px solid #e5e7eb',
          display: 'flex',
          gap: '12px',
          alignItems: 'center',
          flexWrap: 'wrap'
        }}>
          <input
            type="text"
            value={nodeName}
            onChange={(e) => setNodeName(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && addNode()}
            placeholder="Enter solution/platform name..."
            style={{
              flex: '1',
              minWidth: '200px',
              padding: '8px 12px',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              fontSize: '14px'
            }}
          />
          <select
            value={nodeType}
            onChange={(e) => setNodeType(e.target.value)}
            style={{
              padding: '8px 12px',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              fontSize: '14px',
              background: 'white'
            }}
          >
            <option value="solution">Solution/Platform</option>
            <option value="workflowState">Workflow State</option>
          </select>
          <button
            onClick={addNode}
            style={{
              padding: '8px 16px',
              background: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              fontSize: '14px',
              fontWeight: '500',
              cursor: 'pointer'
            }}
          >
            Add Node
          </button>
          <button
            onClick={resetToDefault}
            style={{
              padding: '8px 16px',
              background: '#6b7280',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              fontSize: '14px',
              fontWeight: '500',
              cursor: 'pointer'
            }}
          >
            Reset
          </button>
          <button
            onClick={clearDiagram}
            style={{
              padding: '8px 16px',
              background: '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              fontSize: '14px',
              fontWeight: '500',
              cursor: 'pointer'
            }}
          >
            Clear
          </button>
        </div>
        <div style={{ height: 'calc(100% - 60px)' }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={handleNodesChange}
            onEdgesChange={handleEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            fitView
            attributionPosition="bottom-left"
          >
            <MiniMap
              nodeStrokeWidth={3}
              zoomable
              pannable
            />
            <Controls />
            <Background color="#e5e7eb" gap={16} />
          </ReactFlow>
        </div>
      </div>

      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
};

export default CDEFlowDiagram;
