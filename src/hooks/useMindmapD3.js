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
            const dx = linkData.child.x - linkData.parent.x;
            const dy = linkData.child.y - linkData.parent.y;
            const dr = Math.sqrt(dx * dx + dy * dy) * 0.3;
            d3.select(this)
              .attr('d', `M${linkData.parent.x},${linkData.parent.y} Q${linkData.parent.x + dx / 2},${linkData.parent.y + dy / 2 - dr} ${linkData.child.x},${linkData.child.y}`);
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

    // Draw links with curves
    g.selectAll('.link')
      .data(allLinks)
      .join('path')
      .attr('class', 'link')
      .attr('d', d => {
        const dx = d.child.x - d.parent.x;
        const dy = d.child.y - d.parent.y;
        const dr = Math.sqrt(dx * dx + dy * dy) * 0.3; // Control point distance
        return `M${d.parent.x},${d.parent.y} Q${d.parent.x + dx / 2},${d.parent.y + dy / 2 - dr} ${d.child.x},${d.child.y}`;
      })
      .attr('stroke', '#9CA3AF')
      .attr('stroke-width', 2.5)
      .attr('fill', 'none')
      .attr('opacity', 0.4)
      .style('filter', 'drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.1))')
      .style('transition', 'all 0.3s ease');

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
        // Add glow effect for selected/highlighted nodes
        if (isSelected || isHighlighted) {
          nodeGroup.append('circle')
            .attr('r', 45)
            .attr('cx', 0)
            .attr('cy', 0)
            .attr('fill', isSelected ? '#EF4444' : '#F59E0B')
            .attr('opacity', 0.2)
            .attr('class', 'animate-pulse-ring');
        }

        // Create gradient
        const gradientId = `gradient-${d.id}`;
        const gradient = svg.append('defs')
          .append('linearGradient')
          .attr('id', gradientId)
          .attr('x1', '0%')
          .attr('y1', '0%')
          .attr('x2', '0%')
          .attr('y2', '100%');

        gradient.append('stop')
          .attr('offset', '0%')
          .attr('stop-color', nodeTypeConfig.bgColor)
          .attr('stop-opacity', 1);

        gradient.append('stop')
          .attr('offset', '100%')
          .attr('stop-color', d3.rgb(nodeTypeConfig.bgColor).darker(0.3))
          .attr('stop-opacity', 1);

        // Create circle for root node
        nodeGroup.append('circle')
          .attr('r', 40)
          .attr('cx', 0)
          .attr('cy', 0)
          .attr('fill', `url(#${gradientId})`)
          .attr('stroke', isSelected ? '#EF4444' : isHighlighted ? '#F59E0B' : '#374151')
          .attr('stroke-width', isSelected ? 3 : isHighlighted ? 2 : 1.5)
          .attr('cursor', 'pointer')
          .style('filter', 'drop-shadow(0px 4px 6px rgba(0, 0, 0, 0.15))')
          .style('transition', 'all 0.3s ease')
          .on('mouseenter', function() {
            if (!isDragging) {
              d3.select(this)
                .transition()
                .duration(200)
                .attr('r', 43)
                .style('filter', 'drop-shadow(0px 6px 12px rgba(0, 0, 0, 0.25))');
            }
          })
          .on('mouseleave', function() {
            if (!isDragging) {
              d3.select(this)
                .transition()
                .duration(200)
                .attr('r', 40)
                .style('filter', 'drop-shadow(0px 4px 6px rgba(0, 0, 0, 0.15))');
            }
          })
          .on('click', (event) => {
            // Only handle click if we weren't dragging
            if (!isDragging) {
              event.stopPropagation();
              console.log('Node clicked:', d.id, d.name);
              console.log('Current selection:', selectedNode);
              const newSelection = selectedNode === d.id ? null : d.id;
              console.log('Setting selection to:', newSelection);
              setSelectedNode(newSelection);
            } else {
              console.log('Click ignored - was dragging');
            }
          })
          .on('dblclick', (event) => {
            event.stopPropagation();
            setEditingNode(d.id);
            setEditingText(d.name);
          });
      } else {
        // Create rectangle for child nodes
        const textLength = d.name.length * 8;
        const width = Math.max(120, textLength);
        const height = 30;

        // Add glow effect for selected/highlighted nodes
        if (isSelected || isHighlighted) {
          nodeGroup.append('rect')
            .attr('width', width + 8)
            .attr('height', height + 8)
            .attr('x', -Math.max(60, textLength / 2) - 4)
            .attr('y', -19)
            .attr('rx', 7)
            .attr('fill', isSelected ? '#EF4444' : '#F59E0B')
            .attr('opacity', 0.2)
            .attr('class', 'animate-pulse-ring');
        }

        // Create gradient
        const gradientId = `gradient-${d.id}`;
        const gradient = svg.append('defs')
          .append('linearGradient')
          .attr('id', gradientId)
          .attr('x1', '0%')
          .attr('y1', '0%')
          .attr('x2', '0%')
          .attr('y2', '100%');

        gradient.append('stop')
          .attr('offset', '0%')
          .attr('stop-color', nodeTypeConfig.bgColor)
          .attr('stop-opacity', 1);

        gradient.append('stop')
          .attr('offset', '100%')
          .attr('stop-color', d3.rgb(nodeTypeConfig.bgColor).darker(0.2))
          .attr('stop-opacity', 1);

        nodeGroup.append('rect')
          .attr('width', width)
          .attr('height', height)
          .attr('x', -Math.max(60, textLength / 2))
          .attr('y', -15)
          .attr('rx', 6)
          .attr('fill', `url(#${gradientId})`)
          .attr('stroke', isSelected ? '#EF4444' : isHighlighted ? '#F59E0B' : '#374151')
          .attr('stroke-width', isSelected ? 3 : isHighlighted ? 2 : 1.5)
          .attr('cursor', 'pointer')
          .style('filter', 'drop-shadow(0px 2px 4px rgba(0, 0, 0, 0.12))')
          .style('transition', 'all 0.3s ease')
          .on('mouseenter', function() {
            if (!isDragging) {
              d3.select(this)
                .transition()
                .duration(200)
                .attr('y', -17)
                .attr('height', height + 4)
                .style('filter', 'drop-shadow(0px 4px 8px rgba(0, 0, 0, 0.2))');
            }
          })
          .on('mouseleave', function() {
            if (!isDragging) {
              d3.select(this)
                .transition()
                .duration(200)
                .attr('y', -15)
                .attr('height', height)
                .style('filter', 'drop-shadow(0px 2px 4px rgba(0, 0, 0, 0.12))');
            }
          })
          .on('click', (event) => {
            // Only handle click if we weren't dragging
            if (!isDragging) {
              event.stopPropagation();
              console.log('Node clicked:', d.id, d.name);
              console.log('Current selection:', selectedNode);
              const newSelection = selectedNode === d.id ? null : d.id;
              console.log('Setting selection to:', newSelection);
              setSelectedNode(newSelection);
            } else {
              console.log('Click ignored - was dragging');
            }
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

  }, [mindmapData, selectedNode, updateValue, getPreorderNodes, getAllLinks, setSelectedNode, setEditingNode, setEditingText, setZoom, renderText, highlightedNodes, svgRef]);

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