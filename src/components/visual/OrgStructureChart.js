import React, { useState } from 'react';
import { Tree, TreeNode } from 'react-organizational-chart';

// Helper to build org chart data from real project data
function buildOrgChartData(data) {
  if (!data) return null;
  const appointingParty = data.appointingParty || 'Appointing Party';

  // Support multiple lead appointed parties and mapping appointed parties to each
  // Accepts:
  // - data.leadAppointedParty: string or array of strings
  // - data.finalizedParties: array of objects, each with a 'Lead' or 'Lead Appointed Party' or 'Parent Lead' field

  // Normalize leads
  let leads = [];
  if (Array.isArray(data.leadAppointedParty)) {
    leads = data.leadAppointedParty;
  } else if (typeof data.leadAppointedParty === 'string') {
    leads = [data.leadAppointedParty];
  }

  // If finalizedParties is an object with mapping, use it; else, try to group by 'Parent Lead' or similar
  let appointedMap = {};
  if (Array.isArray(data.finalizedParties) && data.finalizedParties.length > 0) {
    // If any party has a 'Parent Lead' or 'Lead Appointed Party' field, group by it
    if (data.finalizedParties.some(p => p['Parent Lead'] || p['Lead Appointed Party'] || p['Lead'])) {
      data.finalizedParties.forEach(party => {
        const parent = party['Parent Lead'] || party['Lead Appointed Party'] || party['Lead'] || leads[0] || 'Lead Appointed Party';
        if (!appointedMap[parent]) appointedMap[parent] = [];
        appointedMap[parent].push(party);
      });
    } else if (leads.length > 1) {
      // If no mapping, but multiple leads, split finalizedParties evenly
      const perLead = Math.ceil(data.finalizedParties.length / leads.length);
      leads.forEach((lead, i) => {
        appointedMap[lead] = data.finalizedParties.slice(i * perLead, (i + 1) * perLead);
      });
    } else {
      // Single lead, all parties under it
      appointedMap[leads[0] || 'Lead Appointed Party'] = data.finalizedParties;
    }
  } else {
    // No finalizedParties, just show leads
    leads.forEach(lead => {
      appointedMap[lead] = [];
    });
  }

  return {
    name: appointingParty,
    role: 'Appointing Party',
    children: leads.map(lead => ({
      name: lead,
      role: 'Lead Appointed Party',
      children: (appointedMap[lead] || []).map(party => ({
        name: party['Company Name'] || party['Role/Service'] || 'Appointed Party',
        role: party['Role/Service'] || 'Appointed Party',
        contact: party['Lead Contact'] || '',
      }))
    }))
  };
}


function EditableNode({ node, parent, onNodeChange, onDelete, editable, onAddSibling }) {
  const [isEditing, setIsEditing] = useState(false);
  const [name, setName] = useState(node.name);
  const [role, setRole] = useState(node.role);

  const handleSave = () => {
    setIsEditing(false);
    const updatedNode = { ...node, name, role };
    // preserve contact where applicable
    if (node.contact !== undefined) updatedNode.contact = node.contact;
    onNodeChange(updatedNode);
  };

  const handleAddChild = () => {
    // Default new child role depends on current node's role
    let newChild = { name: 'New Member', role: 'Role', children: [] };
    if (node.role && node.role.toLowerCase().includes('appointing')) {
      newChild = { name: 'New Lead', role: 'Lead Appointed Party', children: [] };
    } else if (node.role && node.role.toLowerCase().includes('lead')) {
      newChild = { name: 'New Appointed Party', role: 'Appointed Party', contact: '', children: [] };
    }
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
              {/* Contact for appointed parties */}
              {(node.contact !== undefined || (role && role.toLowerCase().includes('appoint'))) && (
                <input
                  className="border rounded px-1 py-0.5 text-xs w-full mt-1"
                  value={node.contact || ''}
                  onChange={e => onNodeChange({ ...node, contact: e.target.value, name, role })}
                  placeholder="Contact"
                />
              )}
              <div className="flex gap-1 mt-1">
                <button className="text-xs px-2 py-0.5 bg-green-500 text-white rounded" onClick={handleSave}>Save</button>
                <button className="text-xs px-2 py-0.5 bg-gray-300 rounded" onClick={() => setIsEditing(false)}>Cancel</button>
              </div>
            </div>
          ) : (
            <div>
              <strong>{node.name}</strong><br />
              <span style={{ fontSize: '0.85em', color: '#555' }}>{node.role}</span>
              {node.contact && <div style={{ fontSize: '0.8em', color: '#666' }}>{node.contact}</div>}
              {editable && (
                <div className="flex gap-1 mt-1">
                  <button className="text-xs px-2 py-0.5 bg-blue-500 text-white rounded" onClick={() => setIsEditing(true)}>Edit</button>
                  <button className="text-xs px-2 py-0.5 bg-red-500 text-white rounded" onClick={onDelete} disabled={!parent}>Delete</button>
                  {/* If this is a Lead node, offer Add Appointed Party (child) and Add Lead (sibling under parent) */}
                  {node.role && node.role.toLowerCase().includes('lead') ? (
                    <>
                      <button className="text-xs px-2 py-0.5 bg-yellow-500 text-white rounded" onClick={handleAddChild}>Add Appointed</button>
                      <button
                        className="text-xs px-2 py-0.5 bg-indigo-500 text-white rounded"
                        onClick={() => {
                          // If parent exists, add a sibling by calling parent's onChange via onAddSibling
                          if (onAddSibling) onAddSibling({ name: 'New Lead', role: 'Lead Appointed Party', children: [] });
                        }}
                      >
                        Add Lead
                      </button>
                    </>
                  ) : (
                    // Default add behaviour: add child under this node
                    <button className="text-xs px-2 py-0.5 bg-yellow-500 text-white rounded" onClick={handleAddChild}>Add</button>
                  )}
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
          onAddSibling={(newSibling) => {
            // Insert the new sibling after this child
            const updatedChildren = [...(node.children || [])];
            updatedChildren.splice(idx + 1, 0, newSibling);
            onNodeChange({ ...node, children: updatedChildren });
          }}
        />
      ))}
    </TreeNode>
  );
}



const OrgStructureChart = ({ data, onChange, editable = false }) => {
  // If data is the full project object, build the org chart structure from it
  const treeData = data && data.appointingParty && data.leadAppointedParty && data.finalizedParties
    ? buildOrgChartData(data)
    : data;

  const [tree, setTree] = useState(treeData);

  React.useEffect(() => {
    setTree(treeData);
  }, [data, treeData]);

  const handleRootChange = updatedRoot => {
    setTree(updatedRoot);
    if (onChange) {
      // Build leadAppointedParty array and finalizedParties array from tree
      const leads = (updatedRoot.children || []).map(c => c.name);
      const finalized = [];
      (updatedRoot.children || []).forEach(leadNode => {
        (leadNode.children || []).forEach(p => {
          finalized.push({
            'Role/Service': p.role || 'Appointed Party',
            'Company Name': p.name,
            'Lead Contact': p.contact || '',
            'Parent Lead': leadNode.name,
          });
        });
      });
      onChange({ tree: updatedRoot, leadAppointedParty: leads, finalizedParties: finalized });
    }
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
