import React, { useState } from 'react';
import { Tree, TreeNode } from 'react-organizational-chart';

// Example data structure for the org chart
const orgData = {
  name: 'Project Manager',
  role: 'Delivery Lead',
  children: [
    {
      name: 'Alice Smith',
      role: 'Design Lead',
      children: [
        { name: 'Bob Brown', role: 'Designer' },
        { name: 'Carol White', role: 'Designer' },
      ],
    },
    {
      name: 'David Green',
      role: 'Engineering Lead',
      children: [
        { name: 'Eve Black', role: 'Engineer' },
        { name: 'Frank Blue', role: 'Engineer' },
      ],
    },
    {
      name: 'Grace Red',
      role: 'QA Lead',
      children: [
        { name: 'Heidi Yellow', role: 'QA Tester' },
      ],
    },
  ],
};


function EditableNode({ node, parent, onNodeChange, onDelete, editable }) {
  const [isEditing, setIsEditing] = useState(false);
  const [name, setName] = useState(node.name);
  const [role, setRole] = useState(node.role);

  const handleSave = () => {
    setIsEditing(false);
    onNodeChange({ ...node, name, role });
  };

  const handleAddChild = () => {
    const newChild = { name: 'New Member', role: 'Role', children: [] };
    const updated = {
      ...node,
      children: node.children ? [...node.children, newChild] : [newChild],
    };
    onNodeChange(updated);
  };

  const handleChildChange = (childIdx, updatedChild) => {
    const updatedChildren = node.children.map((c, i) => (i === childIdx ? updatedChild : c));
    onNodeChange({ ...node, children: updatedChildren });
  };

  const handleDeleteChild = (childIdx) => {
    const updatedChildren = node.children.filter((_, i) => i !== childIdx);
    onNodeChange({ ...node, children: updatedChildren });
  };

  return (
    <TreeNode
      label={
        <div style={{ minWidth: 160, background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8, padding: 8, boxShadow: '0 1px 4px #0001' }}>
          {isEditing ? (
            <div>
              <input
                className="border rounded px-1 py-0.5 text-sm mb-1 w-full"
                value={name}
                onChange={e => setName(e.target.value)}
                placeholder="Name"
                autoFocus
              />
              <input
                className="border rounded px-1 py-0.5 text-xs w-full"
                value={role}
                onChange={e => setRole(e.target.value)}
                placeholder="Role"
              />
              <div className="flex gap-1 mt-1">
                <button className="text-xs px-2 py-0.5 bg-green-500 text-white rounded" onClick={handleSave}>Save</button>
                <button className="text-xs px-2 py-0.5 bg-gray-300 rounded" onClick={() => setIsEditing(false)}>Cancel</button>
              </div>
            </div>
          ) : (
            <div>
              <strong>{node.name}</strong><br />
              <span style={{ fontSize: '0.85em', color: '#555' }}>{node.role}</span>
              {editable && (
                <div className="flex gap-1 mt-1">
                  <button className="text-xs px-2 py-0.5 bg-blue-500 text-white rounded" onClick={() => setIsEditing(true)}>Edit</button>
                  <button className="text-xs px-2 py-0.5 bg-red-500 text-white rounded" onClick={onDelete} disabled={!parent}>Delete</button>
                  <button className="text-xs px-2 py-0.5 bg-yellow-500 text-white rounded" onClick={handleAddChild}>Add</button>
                </div>
              )}
            </div>
          )}
        </div>
      }
    >
      {node.children && node.children.map((child, idx) => (
        <EditableNode
          key={idx}
          node={child}
          parent={node}
          onNodeChange={updatedChild => handleChildChange(idx, updatedChild)}
          onDelete={() => handleDeleteChild(idx)}
          editable={editable}
        />
      ))}
    </TreeNode>
  );
}


const OrgStructureChart = ({ data = orgData, onChange, editable = false }) => {
  // Only allow editing if onChange is provided
  const [tree, setTree] = useState(data);

  React.useEffect(() => {
    setTree(data);
  }, [data]);

  const handleRootChange = updatedRoot => {
    setTree(updatedRoot);
    if (onChange) onChange(updatedRoot);
  };

  return (
    <div
      className="w-full"
      style={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        width: '100%',
        minWidth: 0,
        maxWidth: '100%',
        overflow: 'auto',
        padding: 0,
      }}
    >
      <Tree
        lineWidth={'2px'}
        lineColor={'#1976d2'}
        lineBorderRadius={'8px'}
        label={null}
      >
        <EditableNode
          node={tree}
          parent={null}
          onNodeChange={handleRootChange}
          onDelete={() => {}}
          editable={editable && !!onChange}
        />
      </Tree>
    </div>
  );
};

export default OrgStructureChart;
