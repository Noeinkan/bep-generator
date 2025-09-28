import { useCallback, useEffect } from 'react';
import * as d3 from 'd3';
import { getNodeTypeConfig, NODE_TYPES } from '../utils/nodeTypes';

export const useMindmapD3 = (svgRef, mindmapData, selectedNode, setSelectedNode, setEditingNode, setEditingText, setZoom, updateValue, highlightedNodes = []) => {

  // Helper function to traverse tree in preorder
  const getPreorderNodes = useCallback((node, result = []) => {
    result.push(node);
    if (node.children) {
      node.children.forEach(child => getPreorderNodes(child, result));
    }
    return result;
  }, []);

  // Helper function to get all links
  const getAllLinks = useCallback((node, links = []) => {
    if (node.children) {
      node.children.forEach(child => {
        links.push({ parent: node, child });
        getAllLinks(child, links);
      });
    }
    return links;
  }, []);

  const renderText = useCallback((nodeGroup, d) => {
    const nodeTypeConfig = getNodeTypeConfig(d.type || NODE_TYPES.DISCIPLINE);
    const text = nodeGroup.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', 5)
      .attr('fill', nodeTypeConfig.textColor)
      .attr('font-size', d.type === NODE_TYPES.ROOT ? '12px' : '12px')
      .attr('font-weight', d.type === NODE_TYPES.ROOT ? 'bold' : 'normal')
      .attr('pointer-events', 'none')
      .attr('aria-label', `Node: ${d.name}`);

    // Handle text wrapping for root node to fit within circle
    if (d.type === NODE_TYPES.ROOT) {
      const words = d.name.split(' ');
      text.text('');

      if (words.length <= 2) {
        if (words.length === 1) {
          text.append('tspan')
            .attr('x', 0)
            .attr('dy', 0)
            .text(words[0]);
        } else {
          text.append('tspan')
            .attr('x', 0)
            .attr('dy', -6)
            .text(words[0]);
          text.append('tspan')
            .attr('x', 0)
            .attr('dy', 12)
            .text(words[1]);
        }
      } else {
        text.append('tspan')
          .attr('x', 0)
          .attr('dy', -6)
          .text(words.slice(0, 2).join(' '));
        text.append('tspan')
          .attr('x', 0)
          .attr('dy', 12)
          .text(words.slice(2).join(' '));
      }
    } else {
      text.text(d.name);
    }
  }, []);

  const drawMindmap = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const g = svg.append('g');

    // Custom zoom behavior that only responds to right mouse button and scroll
    const zoomBehavior = d3.zoom()
      .scaleExtent([0.5, 3])
      .filter((event) => {
        // Allow scroll wheel zoom and right mouse button drag
        return event.type === 'wheel' || event.button === 2;
      })
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
        setZoom(event.transform.k);
      });

    svg.call(zoomBehavior);

    // Prevent context menu on right click
    svg.on('contextmenu', (event) => {
      event.preventDefault();
    });

    // Add click handler to SVG to clear selection when clicking empty space
    svg.on('click', (event) => {
      if (event.target === svg.node()) {
        setSelectedNode(null);
      }
    });

    // Get all nodes and links
    const allNodes = getPreorderNodes(mindmapData);
    const allLinks = getAllLinks(mindmapData);

    // Drag behavior - only responds to left mouse button with proper click separation
    let isDragging = false;

    const drag = d3.drag()
      .filter((event) => {
        // Only allow left mouse button (button 0) for dragging nodes
        return event.button === 0;
      })
      .on('start', function(event, d) {
        isDragging = false; // Reset drag flag
        console.log('Drag start detected for:', d.id);
      })
      .on('drag', function(event, d) {
        if (!isDragging) {
          console.log('First drag movement for:', d.id);
          isDragging = true;
          d3.select(this).raise().select('circle, rect').attr('stroke', '#EF4444').attr('stroke-width', 3);
        }

        d.x = event.x;
        d.y = event.y;
        d3.select(this).attr('transform', `translate(${d.x}, ${d.y})`);

        // Update links in real-time
        g.selectAll('.link').each(function(linkData) {
          if (linkData.parent.id === d.id || linkData.child.id === d.id) {
            d3.select(this)
              .attr('x1', linkData.parent.x)
              .attr('y1', linkData.parent.y)
              .attr('x2', linkData.child.x)
              .attr('y2', linkData.child.y);
          }
        });
      })
      .on('end', function(event, d) {
        if (isDragging) {
          console.log('Drag ended for:', d.id);
          const isSelected = selectedNode === d.id;
          const isHighlighted = highlightedNodes.includes(d.id);
          d3.select(this).select('circle, rect')
            .attr('stroke', isSelected ? '#EF4444' : isHighlighted ? '#F59E0B' : '#374151')
            .attr('stroke-width', isSelected ? 3 : isHighlighted ? 2 : 1);
          updateValue({ ...mindmapData });
        }
        isDragging = false;
      });

    // Draw links
    g.selectAll('.link')
      .data(allLinks)
      .join('line')
      .attr('class', 'link')
      .attr('x1', d => d.parent.x)
      .attr('y1', d => d.parent.y)
      .attr('x2', d => d.child.x)
      .attr('y2', d => d.child.y)
      .attr('stroke', '#6B7280')
      .attr('stroke-width', 2)
      .attr('opacity', 0.6);

    // Draw nodes
    const nodes = g.selectAll('.node-group')
      .data(allNodes)
      .join('g')
      .attr('class', 'node-group')
      .attr('transform', d => `translate(${d.x}, ${d.y})`)
      .call(drag);

    // Handle root nodes (circles) and child nodes (rectangles) separately
    nodes.each(function(d) {
      const nodeGroup = d3.select(this);
      nodeGroup.selectAll('*').remove();

      const nodeTypeConfig = getNodeTypeConfig(d.type || NODE_TYPES.DISCIPLINE);
      const isSelected = selectedNode === d.id;
      const isHighlighted = highlightedNodes.includes(d.id);

      if (nodeTypeConfig.shape === 'circle') {
        // Create circle for root node
        nodeGroup.append('circle')
          .attr('r', 40)
          .attr('cx', 0)
          .attr('cy', 0)
          .attr('fill', nodeTypeConfig.bgColor)
          .attr('stroke', isSelected ? '#EF4444' : isHighlighted ? '#F59E0B' : '#374151')
          .attr('stroke-width', isSelected ? 3 : isHighlighted ? 2 : 1)
          .attr('cursor', 'pointer')
          .on('click', (event) => {
            event.stopPropagation();
            console.log('Node clicked:', d.id, d.name);
            console.log('Current selection:', selectedNode);
            const newSelection = selectedNode === d.id ? null : d.id;
            console.log('Setting selection to:', newSelection);
            setSelectedNode(newSelection);
          })
          .on('dblclick', (event) => {
            event.stopPropagation();
            setEditingNode(d.id);
            setEditingText(d.name);
          });
      } else {
        // Create rectangle for child nodes
        const textLength = d.name.length * 8;
        nodeGroup.append('rect')
          .attr('width', Math.max(120, textLength))
          .attr('height', 30)
          .attr('x', -Math.max(60, textLength / 2))
          .attr('y', -15)
          .attr('rx', 5)
          .attr('fill', nodeTypeConfig.bgColor)
          .attr('stroke', isSelected ? '#EF4444' : isHighlighted ? '#F59E0B' : '#374151')
          .attr('stroke-width', isSelected ? 3 : isHighlighted ? 2 : 1)
          .attr('cursor', 'pointer')
          .on('click', (event) => {
            event.stopPropagation();
            console.log('Node clicked:', d.id, d.name);
            console.log('Current selection:', selectedNode);
            const newSelection = selectedNode === d.id ? null : d.id;
            console.log('Setting selection to:', newSelection);
            setSelectedNode(newSelection);
          })
          .on('dblclick', (event) => {
            event.stopPropagation();
            setEditingNode(d.id);
            setEditingText(d.name);
          });
      }

      // Add text label
      renderText(nodeGroup, d);
    });

  }, [mindmapData, selectedNode, updateValue, getPreorderNodes, getAllLinks, setSelectedNode, setEditingNode, setEditingText, setZoom, renderText, highlightedNodes]);

  useEffect(() => {
    drawMindmap();
  }, [drawMindmap]);

  const resetView = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.transition().duration(750).call(
      d3.zoom().transform,
      d3.zoomIdentity
    );
    setZoom(1);
  }, [svgRef, setZoom]);

  return { resetView };
};