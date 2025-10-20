import { NODE_TYPES } from './nodeTypes';

export const getDefaultStructure = () => ({
  id: 'root',
  name: 'Volume Strategy',
  type: NODE_TYPES.ROOT,
  x: 500,
  y: 350,
  children: [
    {
      id: 'architecture',
      name: 'Architecture (ARC)',
      type: NODE_TYPES.DISCIPLINE,
      x: 200,
      y: 120,
      children: [
        { id: 'arc-l00', name: 'L00 - Ground Floor', type: NODE_TYPES.LEVEL, x: 80, y: 80, children: [] },
        { id: 'arc-l01', name: 'L01 - First Floor', type: NODE_TYPES.LEVEL, x: 80, y: 120, children: [] },
        { id: 'arc-l02', name: 'L02 - Second Floor', type: NODE_TYPES.LEVEL, x: 80, y: 160, children: [] }
      ]
    },
    {
      id: 'structural',
      name: 'Structural (STR)',
      type: NODE_TYPES.DISCIPLINE,
      x: 800,
      y: 120,
      children: [
        { id: 'str-sz01', name: 'SZ01 - Core Structure', type: NODE_TYPES.ZONE, x: 920, y: 80, children: [] },
        { id: 'str-sz02', name: 'SZ02 - Frame Structure', type: NODE_TYPES.ZONE, x: 920, y: 120, children: [] }
      ]
    },
    {
      id: 'mep',
      name: 'MEP Services',
      type: NODE_TYPES.DISCIPLINE,
      x: 500,
      y: 380,
      children: [
        { id: 'mep-mz01', name: 'MZ01 - HVAC Zone', type: NODE_TYPES.SYSTEM, x: 350, y: 440, children: [] },
        { id: 'mep-mz02', name: 'MZ02 - Electrical Zone', type: NODE_TYPES.SYSTEM, x: 500, y: 440, children: [] },
        { id: 'mep-mz03', name: 'MZ03 - Plumbing Zone', type: NODE_TYPES.SYSTEM, x: 650, y: 440, children: [] }
      ]
    },
    {
      id: 'site',
      name: 'Site (SITE)',
      type: NODE_TYPES.DISCIPLINE,
      x: 200,
      y: 380,
      children: [
        { id: 'site-ext', name: 'External Works', type: NODE_TYPES.LOCATION, x: 80, y: 420, children: [] }
      ]
    }
  ]
});

export const convertToText = (node, level = 0) => {
  const indent = '  '.repeat(level);
  let text = `${indent}${node.name}\n`;

  if (node.children && node.children.length > 0) {
    node.children.forEach(child => {
      text += convertToText(child, level + 1);
    });
  }

  return text;
};

export const parseValue = (val) => {
  if (typeof val === 'string') {
    return getDefaultStructure();
  }
  return val || getDefaultStructure();
};

export const addNodeToTree = (mindmapData, selectedNode, nodeType = NODE_TYPES.DISCIPLINE) => {
  const findAndAddNode = (node) => {
    if (node.id === selectedNode) {
      if (!node.children) node.children = [];
      const newNode = {
        id: `node-${Date.now()}`,
        name: 'New Node',
        type: nodeType,
        x: node.x + (Math.random() - 0.5) * 200,
        y: node.y + 100 + (Math.random() - 0.5) * 50,
        children: []
      };
      node.children.push(newNode);
      return true;
    }
    if (node.children && node.children.length > 0) {
      return node.children.some(findAndAddNode);
    }
    return false;
  };

  const newData = JSON.parse(JSON.stringify(mindmapData));
  return findAndAddNode(newData) ? newData : null;
};

export const removeNodeFromTree = (mindmapData, selectedNode) => {
  if (!selectedNode || selectedNode === 'root') {
    console.log('removeNodeFromTree: Invalid node or trying to delete root');
    return null;
  }

  console.log('removeNodeFromTree: Looking for node with ID:', selectedNode);

  const removeNode = (node, path = '') => {
    console.log(`Checking node at path ${path}: ${node.id} (${node.name})`);

    if (!node.children || node.children.length === 0) {
      console.log(`No children found at ${path}`);
      return false;
    }

    // Check if the target node is a direct child
    const initialLength = node.children.length;
    console.log(`Node ${node.id} has ${initialLength} children:`, node.children.map(c => `${c.id}(${c.name})`));

    node.children = node.children.filter(child => {
      const shouldKeep = child.id !== selectedNode;
      if (!shouldKeep) {
        console.log(`Found and removing child: ${child.id} (${child.name})`);
      }
      return shouldKeep;
    });

    // If we removed a direct child, return success
    if (node.children.length < initialLength) {
      console.log(`Successfully removed direct child. Children count: ${initialLength} -> ${node.children.length}`);
      return true;
    }

    // Otherwise, recursively search in all children
    for (let i = 0; i < node.children.length; i++) {
      console.log(`Searching recursively in child ${i}: ${node.children[i].id}`);
      if (removeNode(node.children[i], `${path}->${node.children[i].id}`)) {
        return true;
      }
    }

    console.log(`Node ${selectedNode} not found in subtree of ${node.id}`);
    return false;
  };

  const newData = JSON.parse(JSON.stringify(mindmapData));
  console.log('Starting removal process from root:', newData.id);
  const success = removeNode(newData, 'root');
  console.log('Removal result:', success ? 'SUCCESS' : 'FAILED');
  return success ? newData : null;
};

export const updateNodeInTree = (mindmapData, editingNode, newName) => {
  const isDuplicate = (node, newName) => {
    if (node.children) {
      return node.children.some(child =>
        child.name === newName && child.id !== editingNode
      ) || node.children.some(child => isDuplicate(child, newName));
    }
    return false;
  };

  const updateNodeName = (node) => {
    if (node.id === editingNode) {
      if (!newName.trim()) return false;
      if (node.children && isDuplicate(node, newName.trim())) return false;
      node.name = newName.trim();
      return true;
    }
    if (node.children) {
      return node.children.some(updateNodeName);
    }
    return false;
  };

  const newData = JSON.parse(JSON.stringify(mindmapData));
  return updateNodeName(newData) ? newData : null;
};

// Search and filter utilities
export const searchNodes = (rootNode, searchTerm, typeFilters = []) => {
  const results = [];
  const searchLower = searchTerm.toLowerCase();

  const traverse = (node) => {
    const matchesSearch = !searchTerm || node.name.toLowerCase().includes(searchLower);
    const matchesType = typeFilters.length === 0 || typeFilters.includes(node.type);

    if (matchesSearch && matchesType) {
      results.push(node);
    }

    if (node.children) {
      node.children.forEach(traverse);
    }
  };

  traverse(rootNode);
  return results;
};

export const findNodeById = (rootNode, nodeId) => {
  if (rootNode.id === nodeId) return rootNode;

  if (rootNode.children) {
    for (const child of rootNode.children) {
      const found = findNodeById(child, nodeId);
      if (found) return found;
    }
  }

  return null;
};

export const duplicateNodeInTree = (mindmapData, nodeId) => {
  if (!nodeId || nodeId === 'root') return null;

  const newData = JSON.parse(JSON.stringify(mindmapData));

  const duplicateNode = (node) => {
    if (!node.children) return false;

    for (let i = 0; i < node.children.length; i++) {
      if (node.children[i].id === nodeId) {
        // Found the node to duplicate
        const originalNode = node.children[i];
        const duplicatedNode = JSON.parse(JSON.stringify(originalNode));

        // Generate new IDs for the duplicated node and all its descendants
        const generateNewIds = (n) => {
          n.id = `${n.id}-copy-${Date.now()}`;
          // Offset position slightly
          n.x += 50;
          n.y += 50;
          if (n.children) {
            n.children.forEach(generateNewIds);
          }
        };

        generateNewIds(duplicatedNode);

        // Insert the duplicated node right after the original
        node.children.splice(i + 1, 0, duplicatedNode);
        return true;
      }

      // Recursively search in children
      if (duplicateNode(node.children[i])) {
        return true;
      }
    }

    return false;
  };

  return duplicateNode(newData) ? newData : null;
};

export const changeNodeTypeInTree = (mindmapData, nodeId, newType) => {
  if (!nodeId || nodeId === 'root') return null;

  const newData = JSON.parse(JSON.stringify(mindmapData));

  const changeType = (node) => {
    if (node.id === nodeId) {
      node.type = newType;
      return true;
    }

    if (node.children) {
      for (const child of node.children) {
        if (changeType(child)) {
          return true;
        }
      }
    }

    return false;
  };

  return changeType(newData) ? newData : null;
};