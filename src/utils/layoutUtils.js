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
        const childAngleStep = (2 * Math.PI) / child.children.length;
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
            const greatChildAngleStep = (2 * Math.PI) / grandchild.children.length;
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

  // Track occupied positions to detect collisions
  const occupiedPositions = new Map(); // key: "x,y", value: node

  const layoutTree = (node, level = 0) => {
    if (!node.children || node.children.length === 0) return;

    // Calculate base vertical position - children are further down from their specific parent
    const baseChildY = node.y + 180;

    // Calculate horizontal spacing based on number of children
    const childSpacing = Math.max(180, 600 / node.children.length);
    const totalWidth = (node.children.length - 1) * childSpacing;
    const startX = node.x - totalWidth / 2;

    node.children.forEach((child, index) => {
      child.x = startX + index * childSpacing;
      child.y = baseChildY;

      // Check for collisions with existing nodes at similar positions
      const collisionRadius = 120; // Minimum distance between nodes
      let yOffset = 0;
      let attempts = 0;
      const maxAttempts = 10;

      while (attempts < maxAttempts) {
        const currentY = baseChildY + yOffset;
        let hasCollision = false;

        // Check all existing nodes for collision
        for (const [posKey, existingNode] of occupiedPositions) {
          const [existingX, existingY] = posKey.split(',').map(Number);
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

        // Try offsetting down in increments
        yOffset += 60;
        attempts++;
      }

      // Record this node's position
      const positionKey = `${Math.round(child.x)},${Math.round(child.y)}`;
      occupiedPositions.set(positionKey, child);

      // Recursively layout children of this child
      layoutTree(child, level + 1);
    });
  };

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