// Auto-layout utilities for mindmap nodes

export const LAYOUT_TYPES = {
  RADIAL: 'radial',
  TREE: 'tree',
  FORCE: 'force',
  GRID: 'grid'
};

export const organizeNodesRadial = (mindmapData) => {
  const result = JSON.parse(JSON.stringify(mindmapData));

  // Position root at center
  result.x = 500;
  result.y = 350;

  if (result.children && result.children.length > 0) {
    const angleStep = (2 * Math.PI) / result.children.length;
    const radius = 200;

    result.children.forEach((child, index) => {
      const angle = index * angleStep;
      child.x = result.x + Math.cos(angle) * radius;
      child.y = result.y + Math.sin(angle) * radius;

      // Position grandchildren in larger circles extending outward from their parents
      if (child.children && child.children.length > 0) {
        // const childAngleStep = (2 * Math.PI) / child.children.length;
        const childRadius = 150; // Increased from 100 for more external positioning

        // Calculate base angle from root to parent to orient children outward
        const parentAngle = Math.atan2(child.y - result.y, child.x - result.x);

        child.children.forEach((grandchild, grandIndex) => {
          // Start from the parent's angle and spread children around it
          const childAngle = parentAngle + (grandIndex - (child.children.length - 1) / 2) * (Math.PI / 3);
          grandchild.x = child.x + Math.cos(childAngle) * childRadius;
          grandchild.y = child.y + Math.sin(childAngle) * childRadius;

          // For great-grandchildren, continue the radial pattern even further out
          if (grandchild.children && grandchild.children.length > 0) {
            const greatChildRadius = 120;
            // const greatChildAngleStep = (2 * Math.PI) / grandchild.children.length;
            const greatParentAngle = Math.atan2(grandchild.y - child.y, grandchild.x - child.x);

            grandchild.children.forEach((greatGrandchild, ggIndex) => {
              const greatChildAngle = greatParentAngle + (ggIndex - (grandchild.children.length - 1) / 2) * (Math.PI / 4);
              greatGrandchild.x = grandchild.x + Math.cos(greatChildAngle) * greatChildRadius;
              greatGrandchild.y = grandchild.y + Math.sin(greatChildAngle) * greatChildRadius;
            });
          }
        });
      }
    });
  }

  return result;
};

export const organizeNodesTree = (mindmapData) => {
  const result = JSON.parse(JSON.stringify(mindmapData));

  // Tree layout with root at top
  result.x = 500;
  result.y = 100;

  // Track branch zones to prevent interference between different branches
  const branchZones = new Map(); // key: branchId, value: { minX, maxX, maxY }
  const occupiedPositions = new Map(); // key: "x,y", value: node

  // Calculate branch zones for root children to create non-interference corridors
  const calculateBranchZones = (rootNode) => {
    if (!rootNode.children || rootNode.children.length === 0) return;

    const branchWidth = 300; // Width allocated to each main branch
    const branchSpacing = 50; // Additional spacing between branches
    const totalBranchWidth = branchWidth + branchSpacing;

    // Calculate total width needed for all branches
    const totalWidth = rootNode.children.length * totalBranchWidth - branchSpacing;
    const startX = rootNode.x - totalWidth / 2;

    rootNode.children.forEach((child, index) => {
      const branchCenterX = startX + index * totalBranchWidth + branchWidth / 2;
      const branchMinX = branchCenterX - branchWidth / 2;
      const branchMaxX = branchCenterX + branchWidth / 2;

      branchZones.set(child.id, {
        minX: branchMinX,
        maxX: branchMaxX,
        maxY: rootNode.y + 120, // Start tracking from first level below root
        centerX: branchCenterX
      });
    });
  };

  // Get the branch zone for a given node by traversing up to find the root child
  const getBranchZone = (node, allNodes) => {
    // Find which root child this node belongs to
    const findRootBranch = (currentNode) => {
      // Find parent of current node
      for (const n of allNodes) {
        if (n.children && n.children.some(child => child.id === currentNode.id)) {
          if (n.id === result.id) {
            // Parent is root, so currentNode is a root child
            return currentNode.id;
          } else {
            // Continue traversing up
            return findRootBranch(n);
          }
        }
      }
      return null;
    };

    const rootBranchId = findRootBranch(node);
    return branchZones.get(rootBranchId);
  };

  // Collect all nodes for branch detection
  const getAllNodes = (node, nodes = []) => {
    nodes.push(node);
    if (node.children) {
      node.children.forEach(child => getAllNodes(child, nodes));
    }
    return nodes;
  };

  const allNodes = getAllNodes(result);

  const layoutTree = (node, level = 0) => {
    if (!node.children || node.children.length === 0) return;

    // Calculate vertical position - children are positioned directly below their parent
    const verticalSpacing = 120; // Distance between parent and child vertically
    const baseChildY = node.y + verticalSpacing;

    // Get branch zone for this node's descendants
    const branchZone = getBranchZone(node, allNodes);

    node.children.forEach((child, index) => {
      let proposedX, proposedY;

      if (level === 0) {
        // For root children (main branches), get their specific branch zone
        const childBranchZone = branchZones.get(child.id);
        proposedX = childBranchZone ? childBranchZone.centerX : (node.x + (index - (node.children.length - 1) / 2) * 200);
        proposedY = baseChildY; // All main branches at same vertical level
      } else {
        proposedY = baseChildY; // For deeper levels, use normal vertical spacing
        // For deeper levels, distribute within the branch zone
        if (branchZone) {
          const branchWidth = branchZone.maxX - branchZone.minX;
          const childSpacing = node.children.length > 1 ? branchWidth / (node.children.length + 1) : 0;
          proposedX = branchZone.minX + (index + 1) * childSpacing;

          // Ensure we stay within branch boundaries
          proposedX = Math.max(branchZone.minX + 50, Math.min(branchZone.maxX - 50, proposedX));
        } else {
          // Fallback to centered distribution if no branch zone found
          const horizontalSpacing = 150;
          const totalWidth = (node.children.length - 1) * horizontalSpacing;
          const startX = node.x - totalWidth / 2;
          proposedX = startX + index * horizontalSpacing;
        }
      }

      child.x = proposedX;

      if (level === 0) {
        // For root children (main branches), they stay on the same horizontal line
        child.y = proposedY;
      } else {
        // For deeper levels, check for vertical collisions within the same branch zone
        const collisionRadius = 80;
        let yOffset = 0;
        let attempts = 0;
        const maxAttempts = 20;

        while (attempts < maxAttempts) {
          const currentY = proposedY + yOffset;
          let hasCollision = false;

          // Check for collisions only within the same branch zone or with nodes outside zones
          for (const [posKey, existingNode] of occupiedPositions) {
            const [existingX, existingY] = posKey.split(',').map(Number);

            // Skip collision check if nodes are in different branch zones
            if (branchZone) {
              const existingNodeBranch = getBranchZone(existingNode, allNodes);
              if (existingNodeBranch && existingNodeBranch !== branchZone) {
                continue; // Different branches, no collision check needed
              }
            }

            const distance = Math.sqrt(
              Math.pow(child.x - existingX, 2) + Math.pow(currentY - existingY, 2)
            );

            if (distance < collisionRadius && existingNode.id !== child.id) {
              hasCollision = true;
              break;
            }
          }

          if (!hasCollision) {
            child.y = currentY;
            break;
          }

          // Try offsetting down in smaller increments
          yOffset += 35;
          attempts++;
        }

        // If still colliding after max attempts, place it further down
        if (attempts >= maxAttempts) {
          child.y = proposedY + yOffset;
        }
      }

      // Update branch zone max Y to track the extent of this branch
      if (branchZone && child.y > branchZone.maxY) {
        branchZone.maxY = child.y;
      }

      // Record this node's position
      const positionKey = `${Math.round(child.x)},${Math.round(child.y)}`;
      occupiedPositions.set(positionKey, child);

      // Recursively layout children of this child
      layoutTree(child, level + 1);
    });
  };

  // Initialize branch zones for root children
  calculateBranchZones(result);

  // Start the recursive layout from the root
  occupiedPositions.set(`${result.x},${result.y}`, result);
  layoutTree(result);

  return result;
};

export const organizeNodesGrid = (mindmapData) => {
  const result = JSON.parse(JSON.stringify(mindmapData));

  // Collect all nodes
  const allNodes = [];
  const traverse = (node) => {
    allNodes.push(node);
    if (node.children) {
      node.children.forEach(traverse);
    }
  };
  traverse(result);

  // Arrange in grid
  const cols = Math.ceil(Math.sqrt(allNodes.length));
  const cellWidth = 180;
  const cellHeight = 120;
  const startX = 100;
  const startY = 100;

  allNodes.forEach((node, index) => {
    const row = Math.floor(index / cols);
    const col = index % cols;
    node.x = startX + col * cellWidth;
    node.y = startY + row * cellHeight;
  });

  return result;
};

export const organizeNodesForce = (mindmapData) => {
  const result = JSON.parse(JSON.stringify(mindmapData));

  // Simple force-directed layout simulation
  const allNodes = [];
  const traverse = (node) => {
    allNodes.push(node);
    if (node.children) {
      node.children.forEach(traverse);
    }
  };
  traverse(result);

  // Initialize with current positions or random if not set
  allNodes.forEach(node => {
    if (!node.x) node.x = 500 + (Math.random() - 0.5) * 400;
    if (!node.y) node.y = 350 + (Math.random() - 0.5) * 500;
    node.vx = 0;
    node.vy = 0;
  });

  // Simple spring force simulation (simplified)
  for (let i = 0; i < 100; i++) {
    // Repulsion between all nodes
    for (let j = 0; j < allNodes.length; j++) {
      for (let k = j + 1; k < allNodes.length; k++) {
        const node1 = allNodes[j];
        const node2 = allNodes[k];
        const dx = node2.x - node1.x;
        const dy = node2.y - node1.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance > 0) {
          const force = 2000 / (distance * distance);
          const fx = (dx / distance) * force;
          const fy = (dy / distance) * force;

          node1.vx -= fx;
          node1.vy -= fy;
          node2.vx += fx;
          node2.vy += fy;
        }
      }
    }

    // Apply velocity and damping
    allNodes.forEach(node => {
      node.x += node.vx * 0.1;
      node.y += node.vy * 0.1;
      node.vx *= 0.9;
      node.vy *= 0.9;

      // Keep within bounds
      node.x = Math.max(50, Math.min(950, node.x));
      node.y = Math.max(50, Math.min(650, node.y));
    });
  }

  // Clean up temporary properties
  allNodes.forEach(node => {
    delete node.vx;
    delete node.vy;
  });

  return result;
};

export const organizeNodes = (mindmapData, layoutType) => {
  switch (layoutType) {
    case LAYOUT_TYPES.RADIAL:
      return organizeNodesRadial(mindmapData);
    case LAYOUT_TYPES.TREE:
      return organizeNodesTree(mindmapData);
    case LAYOUT_TYPES.GRID:
      return organizeNodesGrid(mindmapData);
    case LAYOUT_TYPES.FORCE:
      return organizeNodesForce(mindmapData);
    default:
      return mindmapData;
  }
};

export const snapToGrid = (mindmapData, gridSize = 50) => {
  const result = JSON.parse(JSON.stringify(mindmapData));

  const snapValue = (value) => Math.round(value / gridSize) * gridSize;

  const traverse = (node) => {
    node.x = snapValue(node.x);
    node.y = snapValue(node.y);
    if (node.children) {
      node.children.forEach(traverse);
    }
  };

  traverse(result);
  return result;
};