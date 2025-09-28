import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Plus,
  Minus,
  ChevronUp,
  ChevronDown,
  Edit3,
  Trash2,
  Database,
  FileText,
  Box,
  Move,
  Save,
  Upload,
  Download,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Layers,
  Link,
  X
} from 'lucide-react';

const CDEDiagramBuilder = ({ field, value, onChange, error }) => {
  const { name, label, required } = field;
  const canvasRef = useRef(null);
  const [dragState, setDragState] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [selectedModel, setSelectedModel] = useState(null);
  const [editingText, setEditingText] = useState(null);
  const [connectionMode, setConnectionMode] = useState(false);
  const [tempConnection, setTempConnection] = useState(null);

  // Initialize with default CDE structure or parse existing value
  const getDefaultStructure = () => ({
    layers: [
      {
        id: 'layer-1',
        name: 'WIP (Work in Progress)',
        y: 50,
        models: [
          { id: 'model-1', type: 'document', name: 'SharePoint', x: 100, y: 0 },
          { id: 'model-2', type: 'document', name: 'Local Files', x: 250, y: 0 }
        ]
      },
      {
        id: 'layer-2',
        name: 'SHARED (Coordination)',
        y: 150,
        models: [
          { id: 'model-3', type: 'database', name: 'BIM 360', x: 150, y: 0 },
          { id: 'model-4', type: 'database', name: 'ProjectWise', x: 300, y: 0 }
        ]
      },
      {
        id: 'layer-3',
        name: 'PUBLISHED (Approved)',
        y: 250,
        models: [
          { id: 'model-5', type: 'cylinder', name: 'Aconex', x: 200, y: 0 },
          { id: 'model-6', type: 'cylinder', name: 'Client Portal', x: 350, y: 0 }
        ]
      }
    ],
    connections: [
      { id: 'conn-1', from: 'model-1', to: 'model-3' },
      { id: 'conn-2', from: 'model-2', to: 'model-4' },
      { id: 'conn-3', from: 'model-3', to: 'model-5' },
      { id: 'conn-4', from: 'model-4', to: 'model-6' }
    ]
  });

  const parseValue = (val) => {
    if (typeof val === 'string') {
      try {
        return JSON.parse(val);
      } catch {
        return getDefaultStructure();
      }
    }
    return val || getDefaultStructure();
  };

  const [diagramData, setDiagramData] = useState(() => parseValue(value));

  const updateValue = useCallback((newData) => {
    setDiagramData(newData);
    onChange(name, JSON.stringify(newData, null, 2));
  }, [name, onChange]);

  // Layer Management Functions
  const addLayer = () => {
    const newLayer = {
      id: `layer-${Date.now()}`,
      name: 'New Layer',
      y: Math.max(...diagramData.layers.map(l => l.y)) + 100,
      models: []
    };

    const newData = {
      ...diagramData,
      layers: [...diagramData.layers, newLayer]
    };
    updateValue(newData);
  };

  const removeLayer = (layerId) => {
    const newData = {
      ...diagramData,
      layers: diagramData.layers.filter(l => l.id !== layerId),
      connections: diagramData.connections.filter(conn => {
        const fromModel = findModelById(conn.from);
        const toModel = findModelById(conn.to);
        return fromModel?.layerId !== layerId && toModel?.layerId !== layerId;
      })
    };
    updateValue(newData);
  };

  const moveLayer = (layerId, direction) => {
    const layers = [...diagramData.layers];
    const layerIndex = layers.findIndex(l => l.id === layerId);

    if (layerIndex === -1) return;

    const newIndex = direction === 'up' ? layerIndex - 1 : layerIndex + 1;
    if (newIndex < 0 || newIndex >= layers.length) return;

    // Swap Y positions
    const tempY = layers[layerIndex].y;
    layers[layerIndex].y = layers[newIndex].y;
    layers[newIndex].y = tempY;

    updateValue({ ...diagramData, layers });
  };

  const updateLayerName = (layerId, newName) => {
    const newData = {
      ...diagramData,
      layers: diagramData.layers.map(layer =>
        layer.id === layerId ? { ...layer, name: newName } : layer
      )
    };
    updateValue(newData);
  };

  // Model Management Functions
  const addModel = (layerId, modelType) => {
    const layer = diagramData.layers.find(l => l.id === layerId);
    if (!layer) return;

    const newModel = {
      id: `model-${Date.now()}`,
      type: modelType,
      name: `New ${modelType}`,
      x: layer.models.length * 120 + 50,
      y: 0
    };

    const newData = {
      ...diagramData,
      layers: diagramData.layers.map(l =>
        l.id === layerId
          ? { ...l, models: [...l.models, newModel] }
          : l
      )
    };
    updateValue(newData);
  };

  const removeModel = (layerId, modelId) => {
    const newData = {
      ...diagramData,
      layers: diagramData.layers.map(layer =>
        layer.id === layerId
          ? { ...layer, models: layer.models.filter(m => m.id !== modelId) }
          : layer
      ),
      connections: diagramData.connections.filter(conn =>
        conn.from !== modelId && conn.to !== modelId
      )
    };
    updateValue(newData);
  };

  const updateModelName = (layerId, modelId, newName) => {
    const newData = {
      ...diagramData,
      layers: diagramData.layers.map(layer =>
        layer.id === layerId
          ? {
              ...layer,
              models: layer.models.map(model =>
                model.id === modelId ? { ...model, name: newName } : model
              )
            }
          : layer
      )
    };
    updateValue(newData);
  };

  const updateModelPosition = (layerId, modelId, x, y) => {
    const newData = {
      ...diagramData,
      layers: diagramData.layers.map(layer =>
        layer.id === layerId
          ? {
              ...layer,
              models: layer.models.map(model =>
                model.id === modelId ? { ...model, x, y } : model
              )
            }
          : layer
      )
    };
    updateValue(newData);
  };

  // Helper Functions
  const findModelById = (modelId) => {
    for (const layer of diagramData.layers) {
      const model = layer.models.find(m => m.id === modelId);
      if (model) return { ...model, layerId: layer.id, layerY: layer.y };
    }
    return null;
  };

  const getModelIcon = (type) => {
    switch (type) {
      case 'document': return FileText;
      case 'database': return Database;
      case 'cylinder': return Box;
      default: return Box;
    }
  };

  const getModelColor = (type) => {
    switch (type) {
      case 'document': return 'bg-blue-100 border-blue-300 text-blue-700';
      case 'database': return 'bg-green-100 border-green-300 text-green-700';
      case 'cylinder': return 'bg-purple-100 border-purple-300 text-purple-700';
      default: return 'bg-gray-100 border-gray-300 text-gray-700';
    }
  };

  // Connection Functions
  const addConnection = (fromId, toId) => {
    if (fromId === toId) return;

    const existingConnection = diagramData.connections.find(conn =>
      (conn.from === fromId && conn.to === toId) ||
      (conn.from === toId && conn.to === fromId)
    );

    if (existingConnection) return;

    const newConnection = {
      id: `conn-${Date.now()}`,
      from: fromId,
      to: toId
    };

    const newData = {
      ...diagramData,
      connections: [...diagramData.connections, newConnection]
    };
    updateValue(newData);
  };

  const removeConnection = (connectionId) => {
    const newData = {
      ...diagramData,
      connections: diagramData.connections.filter(conn => conn.id !== connectionId)
    };
    updateValue(newData);
  };

  // Rendering Functions
  const renderModel = (model, layer) => {
    const Icon = getModelIcon(model.type);
    const colorClass = getModelColor(model.type);
    const isSelected = selectedModel === model.id;
    const isEditing = editingText === model.id;

    return (
      <div
        key={model.id}
        className={`absolute cursor-pointer group transition-all duration-200 ${colorClass} ${
          isSelected ? 'ring-2 ring-blue-500 shadow-lg' : 'hover:shadow-md'
        }`}
        style={{
          left: model.x * zoom,
          top: model.y * zoom,
          transform: `scale(${zoom})`,
          transformOrigin: 'top left'
        }}
        onClick={() => setSelectedModel(model.id)}
        onDoubleClick={() => setEditingText(model.id)}
      >
        <div className="flex items-center space-x-2 px-3 py-2 border-2 rounded-lg min-w-24">
          <Icon className="w-4 h-4" />
          {isEditing ? (
            <input
              type="text"
              defaultValue={model.name}
              className="text-sm bg-transparent border-none outline-none min-w-16"
              autoFocus
              onBlur={(e) => {
                updateModelName(layer.id, model.id, e.target.value);
                setEditingText(null);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  updateModelName(layer.id, model.id, e.target.value);
                  setEditingText(null);
                }
                if (e.key === 'Escape') {
                  setEditingText(null);
                }
              }}
            />
          ) : (
            <span className="text-sm font-medium">{model.name}</span>
          )}

          {/* Delete button on hover */}
          <button
            className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-red-100 rounded"
            onClick={(e) => {
              e.stopPropagation();
              removeModel(layer.id, model.id);
            }}
          >
            <X className="w-3 h-3 text-red-600" />
          </button>
        </div>
      </div>
    );
  };

  const renderConnection = (connection) => {
    const fromModel = findModelById(connection.from);
    const toModel = findModelById(connection.to);

    if (!fromModel || !toModel) return null;

    const startX = (fromModel.x + 60) * zoom;
    const startY = (fromModel.layerY + fromModel.y + 20) * zoom;
    const endX = (toModel.x + 60) * zoom;
    const endY = (toModel.layerY + toModel.y + 20) * zoom;

    const midY = (startY + endY) / 2;

    return (
      <g key={connection.id}>
        <path
          d={`M ${startX} ${startY} Q ${startX} ${midY} ${(startX + endX) / 2} ${midY} Q ${endX} ${midY} ${endX} ${endY}`}
          fill="none"
          stroke="#6B7280"
          strokeWidth="2"
          className="hover:stroke-blue-500 cursor-pointer"
          onClick={() => removeConnection(connection.id)}
        />
        <circle
          cx={(startX + endX) / 2}
          cy={midY}
          r="3"
          fill="#6B7280"
          className="hover:fill-blue-500 cursor-pointer"
          onClick={() => removeConnection(connection.id)}
        />
      </g>
    );
  };

  const renderLayer = (layer) => {
    const isEditing = editingText === layer.id;

    return (
      <div
        key={layer.id}
        className="relative mb-6 border-2 border-dashed border-gray-300 rounded-lg bg-gray-50 p-4 min-h-32"
        style={{ marginTop: layer.y * 0.5 }}
      >
        {/* Layer Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <Layers className="w-4 h-4 text-gray-600" />
            {isEditing ? (
              <input
                type="text"
                defaultValue={layer.name}
                className="text-lg font-semibold bg-transparent border-b border-gray-400 outline-none"
                autoFocus
                onBlur={(e) => {
                  updateLayerName(layer.id, e.target.value);
                  setEditingText(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    updateLayerName(layer.id, e.target.value);
                    setEditingText(null);
                  }
                  if (e.key === 'Escape') {
                    setEditingText(null);
                  }
                }}
              />
            ) : (
              <h3
                className="text-lg font-semibold text-gray-800 cursor-pointer hover:text-blue-600"
                onDoubleClick={() => setEditingText(layer.id)}
              >
                {layer.name}
              </h3>
            )}
          </div>

          {/* Layer Controls */}
          <div className="flex items-center space-x-1">
            <button
              onClick={() => addModel(layer.id, 'document')}
              className="p-1 text-blue-600 hover:bg-blue-100 rounded"
              title="Add Document"
            >
              <FileText className="w-4 h-4" />
            </button>
            <button
              onClick={() => addModel(layer.id, 'database')}
              className="p-1 text-green-600 hover:bg-green-100 rounded"
              title="Add Database"
            >
              <Database className="w-4 h-4" />
            </button>
            <button
              onClick={() => addModel(layer.id, 'cylinder')}
              className="p-1 text-purple-600 hover:bg-purple-100 rounded"
              title="Add System"
            >
              <Box className="w-4 h-4" />
            </button>
            <div className="w-px h-4 bg-gray-300" />
            <button
              onClick={() => moveLayer(layer.id, 'up')}
              className="p-1 text-gray-600 hover:bg-gray-100 rounded"
              title="Move Up"
            >
              <ChevronUp className="w-4 h-4" />
            </button>
            <button
              onClick={() => moveLayer(layer.id, 'down')}
              className="p-1 text-gray-600 hover:bg-gray-100 rounded"
              title="Move Down"
            >
              <ChevronDown className="w-4 h-4" />
            </button>
            <button
              onClick={() => removeLayer(layer.id)}
              className="p-1 text-red-600 hover:bg-red-100 rounded"
              title="Delete Layer"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Models Container */}
        <div className="relative min-h-20">
          {layer.models.map(model => renderModel(model, layer))}

          {/* Add Model Hint */}
          {layer.models.length === 0 && (
            <div className="flex items-center justify-center h-16 text-gray-500 text-sm">
              Click the icons above to add models to this layer
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="mb-8 w-full">
      <label className="block text-lg font-semibold mb-4 text-gray-800">
        {label} {required && <span className="text-red-500">*</span>}
      </label>

      <div className="w-full border rounded-xl overflow-hidden shadow-sm bg-white">
        {/* Header Controls */}
        <div className="bg-gradient-to-r from-purple-50 to-purple-100 px-6 py-4 border-b border-purple-200">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <Layers className="w-5 h-5 text-purple-600" />
              <span className="text-base font-semibold text-purple-800">CDE Diagram Builder</span>
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={() => setZoom(Math.max(0.5, zoom - 0.1))}
                className="p-2 text-gray-600 hover:bg-gray-100 rounded"
                title="Zoom Out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <span className="text-sm text-gray-600 min-w-12 text-center">
                {Math.round(zoom * 100)}%
              </span>
              <button
                onClick={() => setZoom(Math.min(2, zoom + 0.1))}
                className="p-2 text-gray-600 hover:bg-gray-100 rounded"
                title="Zoom In"
              >
                <ZoomIn className="w-4 h-4" />
              </button>

              <div className="w-px h-6 bg-gray-300" />

              <button
                onClick={addLayer}
                className="flex items-center space-x-2 bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition-all"
              >
                <Plus className="w-4 h-4" />
                <span>Add Layer</span>
              </button>
            </div>
          </div>

          <p className="text-sm text-purple-700 mt-2">
            Double-click to rename • Use layer controls to add models • Click connections to remove them
          </p>
        </div>

        {/* Main Canvas Area */}
        <div className="relative p-6 bg-gray-50 min-h-[500px] max-h-[800px] overflow-auto">
          {/* SVG for connections */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 1 }}>
            {diagramData.connections.map(connection => renderConnection(connection))}
          </svg>

          {/* Layers */}
          <div style={{ zIndex: 2 }} className="relative">
            {diagramData.layers
              .sort((a, b) => a.y - b.y)
              .map(layer => renderLayer(layer))}
          </div>

          {/* Empty State */}
          {diagramData.layers.length === 0 && (
            <div className="flex flex-col items-center justify-center h-64 text-gray-500">
              <Layers className="w-16 h-16 mb-4 text-gray-300" />
              <p className="text-lg">No layers yet. Click "Add Layer" to get started.</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="w-full bg-gray-100 px-6 py-3 border-t">
          <p className="text-xs text-gray-600">
            💡 Tip: Double-click layer names and model names to edit them. Use the controls to add different model types and organize your CDE structure.
          </p>
        </div>
      </div>

      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
};

export default CDEDiagramBuilder;