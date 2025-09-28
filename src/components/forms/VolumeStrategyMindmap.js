import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3';
import {
  Plus,
  Trash2,
  Maximize2,
  Brain
} from 'lucide-react';

const VolumeStrategyMindmap = ({ field, value, onChange, error }) => {
  const { name, label, required } = field;
  const svgRef = useRef(null);
  const [editingNode, setEditingNode] = useState(null);
  const [editingText, setEditingText] = useState('');
  const [zoom, setZoom] = useState(1);
  const [selectedNode, setSelectedNode] = useState(null);

  // Parse the value (could be string or structured data)
  const parseValue = (val) => {
    if (typeof val === 'string') {
      // Try to extract structure from text
      return getDefaultStructure();
    }
    return val || getDefaultStructure();
  };

  const getDefaultStructure = () => ({
    id: 'root',
    name: 'Volume Strategy',
    x: 500,
    y: 250,
    children: [
      {
        id: 'architecture',
        name: 'Architecture (ARC)',
        x: 200,
        y: 120,
        children: [
          { id: 'arc-l00', name: 'L00 - Ground Floor', x: 80, y: 80, children: [] },
          { id: 'arc-l01', name: 'L01 - First Floor', x: 80, y: 120, children: [] },
          { id: 'arc-l02', name: 'L02 - Second Floor', x: 80, y: 160, children: [] }
        ]
      },
      {
        id: 'structural',
        name: 'Structural (STR)',
        x: 800,
        y: 120,
        children: [
          { id: 'str-sz01', name: 'SZ01 - Core Structure', x: 920, y: 80, children: [] },
          { id: 'str-sz02', name: 'SZ02 - Frame Structure', x: 920, y: 120, children: [] }
        ]
      },
      {
        id: 'mep',
        name: 'MEP Services',
        x: 500,
        y: 380,
        children: [
          { id: 'mep-mz01', name: 'MZ01 - HVAC Zone', x: 350, y: 440, children: [] },
          { id: 'mep-mz02', name: 'MZ02 - Electrical Zone', x: 500, y: 440, children: [] },
          { id: 'mep-mz03', name: 'MZ03 - Plumbing Zone', x: 650, y: 440, children: [] }
        ]
      },
      {
        id: 'site',
        name: 'Site (SITE)',
        x: 200,
        y: 380,
        children: [
          { id: 'site-ext', name: 'External Works', x: 80, y: 420, children: [] }
        ]
      }
    ]
  });

  const [mindmapData, setMindmapData] = useState(() => parseValue(value));

  const convertToText = useCallback((node, level = 0) => {
    const indent = '  '.repeat(level);
    let text = `${indent}${node.name}\n`;

    if (node.children && node.children.length > 0) {
      node.children.forEach(child => {
        text += convertToText(child, level + 1);
      });
    }

    return text;
  }, []);

  const updateValue = useCallback((newData) => {
    setMindmapData(newData);
    // Convert to simplified text representation for storage
    const textRepresentation = convertToText(newData);
    onChange(name, textRepresentation);
  }, [name, onChange, convertToText]);

  const drawMindmap = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g');

    // Zoom behavior
    const zoomBehavior = d3.zoom()
      .scaleExtent([0.5, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
        setZoom(event.transform.k);
      });

    svg.call(zoomBehavior);

    // Draw links first (so they appear behind nodes)
    const drawLinks = (node) => {
      if (node.children) {
        node.children.forEach(child => {
          g.append('line')
            .attr('x1', node.x)
            .attr('y1', node.y)
            .attr('x2', child.x)
            .attr('y2', child.y)
            .attr('stroke', '#6B7280')
            .attr('stroke-width', 2)
            .attr('opacity', 0.6);

          drawLinks(child);
        });
      }
    };

    drawLinks(mindmapData);

    // Draw nodes
    const drawNodes = (node, isRoot = false) => {
      const nodeGroup = g.append('g')
        .attr('class', 'node-group')
        .attr('transform', `translate(${node.x}, ${node.y})`);

      // Node circle/rectangle
      const nodeElement = nodeGroup.append(isRoot ? 'circle' : 'rect')
        .attr('fill', isRoot ? '#3B82F6' : '#10B981')
        .attr('stroke', selectedNode === node.id ? '#EF4444' : '#374151')
        .attr('stroke-width', selectedNode === node.id ? 3 : 1)
        .attr('cursor', 'pointer')
        .on('click', () => {
          setSelectedNode(selectedNode === node.id ? null : node.id);
        })
        .on('dblclick', () => {
          setEditingNode(node.id);
          setEditingText(node.name);
        });

      if (isRoot) {
        nodeElement
          .attr('r', 40)
          .attr('cx', 0)
          .attr('cy', 0);
      } else {
        const textLength = node.name.length * 8;
        nodeElement
          .attr('width', Math.max(120, textLength))
          .attr('height', 30)
          .attr('x', -Math.max(60, textLength / 2))
          .attr('y', -15)
          .attr('rx', 5);
      }

      // Node text
      nodeGroup.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', isRoot ? 5 : 5)
        .attr('fill', 'white')
        .attr('font-size', isRoot ? '14px' : '12px')
        .attr('font-weight', isRoot ? 'bold' : 'normal')
        .text(node.name)
        .attr('pointer-events', 'none');

      // Draw children
      if (node.children) {
        node.children.forEach(child => drawNodes(child, false));
      }
    };

    drawNodes(mindmapData, true);

  }, [mindmapData, selectedNode]);

  useEffect(() => {
    drawMindmap();
  }, [drawMindmap]);

  const addNode = () => {
    if (!selectedNode) return;

    const findAndAddNode = (node) => {
      if (node.id === selectedNode) {
        if (!node.children) node.children = [];
        const newNode = {
          id: `node-${Date.now()}`,
          name: 'New Node',
          x: node.x + (Math.random() - 0.5) * 200,
          y: node.y + 100 + (Math.random() - 0.5) * 50,
          children: []
        };
        node.children.push(newNode);
        return true;
      }
      if (node.children) {
        return node.children.some(findAndAddNode);
      }
      return false;
    };

    const newData = { ...mindmapData };
    findAndAddNode(newData);
    updateValue(newData);
  };

  const deleteNode = () => {
    if (!selectedNode || selectedNode === 'root') return;

    const removeNode = (node) => {
      if (node.children) {
        node.children = node.children.filter(child => {
          if (child.id === selectedNode) {
            return false;
          }
          removeNode(child);
          return true;
        });
      }
    };

    const newData = { ...mindmapData };
    removeNode(newData);
    setSelectedNode(null);
    updateValue(newData);
  };

  const saveEdit = () => {
    if (!editingNode || !editingText.trim()) return;

    const updateNodeName = (node) => {
      if (node.id === editingNode) {
        node.name = editingText.trim();
        return true;
      }
      if (node.children) {
        return node.children.some(updateNodeName);
      }
      return false;
    };

    const newData = { ...mindmapData };
    updateNodeName(newData);
    updateValue(newData);
    setEditingNode(null);
    setEditingText('');
  };

  const cancelEdit = () => {
    setEditingNode(null);
    setEditingText('');
  };

  const resetView = () => {
    const svg = d3.select(svgRef.current);
    svg.transition().duration(750).call(
      d3.zoom().transform,
      d3.zoomIdentity
    );
    setZoom(1);
  };

  return (
    <div className="mb-8 w-full">
      <label className="block text-lg font-semibold mb-4 text-gray-800">
        {label} {required && <span className="text-red-500">*</span>}
      </label>

      <div className="w-full border rounded-xl overflow-hidden shadow-sm bg-white">
        {/* Header Controls */}
        <div className="bg-gradient-to-r from-green-50 to-green-100 px-6 py-4 border-b border-green-200">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <Brain className="w-5 h-5 text-green-600" />
              <span className="text-base font-semibold text-green-800">Volume Strategy Mindmap</span>
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={addNode}
                disabled={!selectedNode}
                className={`p-2 rounded ${
                  selectedNode
                    ? 'text-green-600 hover:bg-green-100'
                    : 'text-gray-400 cursor-not-allowed'
                }`}
                title="Add Child Node"
              >
                <Plus className="w-4 h-4" />
              </button>

              <button
                onClick={deleteNode}
                disabled={!selectedNode || selectedNode === 'root'}
                className={`p-2 rounded ${
                  selectedNode && selectedNode !== 'root'
                    ? 'text-red-600 hover:bg-red-100'
                    : 'text-gray-400 cursor-not-allowed'
                }`}
                title="Delete Node"
              >
                <Trash2 className="w-4 h-4" />
              </button>

              <div className="w-px h-6 bg-gray-300" />

              <button
                onClick={resetView}
                className="p-2 text-gray-600 hover:bg-gray-100 rounded"
                title="Reset View"
              >
                <Maximize2 className="w-4 h-4" />
              </button>

              <span className="text-sm text-gray-600 min-w-12 text-center">
                {Math.round(zoom * 100)}%
              </span>
            </div>
          </div>

          <p className="text-sm text-green-700 mt-2">
            Click nodes to select • Double-click to edit • Drag to pan • Scroll to zoom
          </p>
        </div>

        {/* Mindmap Canvas */}
        <div className="relative bg-gray-50 w-full min-h-[500px]">
          <svg
            ref={svgRef}
            width="100%"
            height="500"
            viewBox="0 0 1000 500"
            preserveAspectRatio="none"
            className="border-none w-full h-full"
            style={{ background: 'linear-gradient(45deg, #f8fafc 25%, transparent 25%), linear-gradient(-45deg, #f8fafc 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #f8fafc 75%), linear-gradient(-45deg, transparent 75%, #f8fafc 75%)', backgroundSize: '20px 20px', backgroundPosition: '0 0, 0 10px, 10px -10px, -10px 0px' }}
          />

          {/* Editing Modal */}
          {editingNode && (
            <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-10">
              <div className="bg-white p-6 rounded-lg shadow-xl min-w-80">
                <h3 className="text-lg font-semibold mb-4">Edit Node</h3>
                <input
                  type="text"
                  value={editingText}
                  onChange={(e) => setEditingText(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 mb-4"
                  placeholder="Node name"
                  autoFocus
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') saveEdit();
                    if (e.key === 'Escape') cancelEdit();
                  }}
                />
                <div className="flex justify-end space-x-2">
                  <button
                    onClick={cancelEdit}
                    className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={saveEdit}
                    className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                  >
                    Save
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="w-full bg-gray-100 px-6 py-3 border-t">
          <p className="text-xs text-gray-600">
            💡 Tip: Select a node and click + to add children. Double-click nodes to rename them.
            Use mouse wheel to zoom and drag to pan around the mindmap.
          </p>
        </div>
      </div>

      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
};

export default VolumeStrategyMindmap;