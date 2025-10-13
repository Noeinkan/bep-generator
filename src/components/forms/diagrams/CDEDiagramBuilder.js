import React, { useState, useCallback, useReducer, useRef } from 'react';
import {
  Plus,
  ChevronUp,
  ChevronDown,
  Trash2,
  Database,
  FileText,
  Box,
  ZoomIn,
  ZoomOut,
  Layers,
  X,
  Link,
  RotateCcw
} from 'lucide-react';

const initialState = (value) => {
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

  if (typeof value === 'string') {
    try {
      return JSON.parse(value);
    } catch (e) {
      console.warn('Invalid diagram JSON, using default:', e);
      return getDefaultStructure();
    }
  }
  return value || getDefaultStructure();
};

const diagramReducer = (state, action) => {
  switch (action.type) {
    case 'SET_STATE':
      return action.state;
    case 'ADD_LAYER': {
      const newLayer = {
        id: `layer-${Date.now()}`,
        name: 'New Layer',
        y: Math.max(...state.layers.map(l => l.y)) + 100,
        models: []
      };
      return { ...state, layers: [...state.layers, newLayer] };
    }
    case 'REMOVE_LAYER': {
      const layerId = action.layerId;
      return {
        ...state,
        layers: state.layers.filter(l => l.id !== layerId),
        connections: state.connections.filter(conn => {
          const fromModel = state.layers.flatMap(l => l.models).find(m => m.id === conn.from);
          const toModel = state.layers.flatMap(l => l.models).find(m => m.id === conn.to);
          const fromLayer = state.layers.find(l => l.models.includes(fromModel));
          const toLayer = state.layers.find(l => l.models.includes(toModel));
          return fromLayer?.id !== layerId && toLayer?.id !== layerId;
        })
      };
    }
    case 'MOVE_LAYER': {
      const { layerId, direction } = action;
      const layers = [...state.layers];
      const index = layers.findIndex(l => l.id === layerId);
      if (index === -1) return state;
      const newIndex = direction === 'up' ? index - 1 : index + 1;
      if (newIndex < 0 || newIndex >= layers.length) return state;
      const tempY = layers[index].y;
      layers[index].y = layers[newIndex].y;
      layers[newIndex].y = tempY;
      return { ...state, layers };
    }
    case 'UPDATE_LAYER_NAME': {
      const { layerId, newName } = action;
      return {
        ...state,
        layers: state.layers.map(l => l.id === layerId ? { ...l, name: newName } : l)
      };
    }
    case 'ADD_MODEL': {
      const { layerId, modelType } = action;
      const layer = state.layers.find(l => l.id === layerId);
      if (!layer) return state;
      const newModel = {
        id: `model-${Date.now()}`,
        type: modelType,
        name: `New ${modelType.charAt(0).toUpperCase() + modelType.slice(1)}`,
        x: layer.models.length * 120 + 50,
        y: 0
      };
      return {
        ...state,
        layers: state.layers.map(l => l.id === layerId ? { ...l, models: [...l.models, newModel] } : l)
      };
    }
    case 'REMOVE_MODEL': {
      const { layerId, modelId } = action;
      return {
        ...state,
        layers: state.layers.map(l => l.id === layerId ? { ...l, models: l.models.filter(m => m.id !== modelId) } : l),
        connections: state.connections.filter(c => c.from !== modelId && c.to !== modelId)
      };
    }
    case 'UPDATE_MODEL_NAME': {
      const { layerId, modelId, newName } = action;
      return {
        ...state,
        layers: state.layers.map(l => l.id === layerId ? {
          ...l,
          models: l.models.map(m => m.id === modelId ? { ...m, name: newName } : m)
        } : l)
      };
    }
    case 'UPDATE_MODEL_POSITION': {
      const { layerId, modelId, x, y } = action;
      return {
        ...state,
        layers: state.layers.map(l => l.id === layerId ? {
          ...l,
          models: l.models.map(m => m.id === modelId ? { ...m, x, y } : m)
        } : l)
      };
    }
    case 'ADD_CONNECTION': {
      const { from, to } = action;
      if (from === to || state.connections.some(c => (c.from === from && c.to === to) || (c.from === to && c.to === from))) return state;
      const newConn = { id: `conn-${Date.now()}`, from, to };
      return { ...state, connections: [...state.connections, newConn] };
    }
    case 'REMOVE_CONNECTION': {
      const { connectionId } = action;
      return { ...state, connections: state.connections.filter(c => c.id !== connectionId) };
    }
    default:
      return state;
  }
};

const findModelById = (state, modelId) => {
  for (const layer of state.layers) {
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

const CDEDiagramBuilder = ({ field, value, onChange, error }) => {
  const { name, label, required } = field;
  const [diagramData, dispatch] = useReducer(diagramReducer, null, () => initialState(value));
  const [history, setHistory] = useState([]);
  const [zoom, setZoom] = useState(1);
  const [selectedModel, setSelectedModel] = useState(null);
  const [connectMode, setConnectMode] = useState(false);
  const [connectStart, setConnectStart] = useState(null);
  const [editingText, setEditingText] = useState(null);
  const [dragState, setDragState] = useState(null);
  const canvasRef = useRef(null);

  const updateValue = useCallback((newData) => {
    setHistory(prev => [...prev, JSON.parse(JSON.stringify(diagramData))]);
    onChange(name, JSON.stringify(newData, null, 2));
  }, [name, onChange, diagramData]);

  const dispatchAndUpdate = useCallback((action) => {
    const newState = diagramReducer(diagramData, action);
    dispatch(action);
    updateValue(newState);
  }, [diagramData, dispatch, updateValue]);

  const handleModelClick = (modelId) => {
    if (connectMode) {
      if (connectStart) {
        if (connectStart !== modelId) {
          dispatchAndUpdate({ type: 'ADD_CONNECTION', from: connectStart, to: modelId });
        }
        setConnectStart(null);
        setConnectMode(false);
      } else {
        setConnectStart(modelId);
      }
    } else {
      setSelectedModel(modelId);
    }
  };

  const handleMouseDown = (e, modelId, layerId, currentX, currentY) => {
    if (!connectMode && !editingText) {
      e.stopPropagation();
      setDragState({ modelId, layerId, startX: e.clientX, startY: e.clientY, currentX, currentY });
    }
  };

  const handleMouseMove = (e) => {
    if (dragState) {
      const dx = (e.clientX - dragState.startX) / zoom;
      const dy = (e.clientY - dragState.startY) / zoom;
      dispatch({ 
        type: 'UPDATE_MODEL_POSITION', 
        layerId: dragState.layerId, 
        modelId: dragState.modelId, 
        x: dragState.currentX + dx, 
        y: dragState.currentY + dy 
      });
      setDragState({ ...dragState, startX: e.clientX, startY: e.clientY });
    }
  };

  const handleMouseUp = () => {
    if (dragState) {
      setDragState(null);
    }
  };

  const undo = () => {
    if (history.length > 0) {
      const prevState = history.pop();
      setHistory([...history]);
      dispatch({ type: 'SET_STATE', state: prevState });
      onChange(name, JSON.stringify(prevState, null, 2));
    }
  };

  const renderConnection = (connection) => {
    const fromModel = findModelById(diagramData, connection.from);
    const toModel = findModelById(diagramData, connection.to);
    if (!fromModel || !toModel) return null;

    const startX = fromModel.x + 60;
    const startY = fromModel.layerY + fromModel.y + 20;
    const endX = toModel.x + 60;
    const endY = toModel.layerY + toModel.y + 20;
    const midY = (startY + endY) / 2;

    return (
      <g key={connection.id}>
        <path
          d={`M ${startX} ${startY} Q ${startX} ${midY} ${(startX + endX) / 2} ${midY} Q ${endX} ${midY} ${endX} ${endY}`}
          fill="none"
          stroke="#6B7280"
          strokeWidth="2"
          className="hover:stroke-blue-500 cursor-pointer"
          onClick={() => dispatchAndUpdate({ type: 'REMOVE_CONNECTION', connectionId: connection.id })}
        />
        <circle
          cx={(startX + endX) / 2}
          cy={midY}
          r="3"
          fill="#6B7280"
          className="hover:fill-blue-500 cursor-pointer"
          onClick={() => dispatchAndUpdate({ type: 'REMOVE_CONNECTION', connectionId: connection.id })}
        />
      </g>
    );
  };

  const renderModel = (model, layer) => {
    const Icon = getModelIcon(model.type);
    const colorClass = getModelColor(model.type);
    const isSelected = selectedModel === model.id || connectStart === model.id;
    const isEditing = editingText === model.id;

    return (
      <div
        key={model.id}
        className={`absolute cursor-move group transition-all duration-200 ${colorClass} ${
          isSelected ? 'ring-2 ring-blue-500 shadow-lg' : 'hover:shadow-md'
        } ${connectMode ? 'cursor-pointer' : ''}`}
        style={{
          left: model.x,
          top: model.y,
        }}
        onClick={() => handleModelClick(model.id)}
        onDoubleClick={() => setEditingText(model.id)}
        onMouseDown={(e) => handleMouseDown(e, model.id, layer.id, model.x, model.y)}
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
                dispatchAndUpdate({ type: 'UPDATE_MODEL_NAME', layerId: layer.id, modelId: model.id, newName: e.target.value });
                setEditingText(null);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  dispatchAndUpdate({ type: 'UPDATE_MODEL_NAME', layerId: layer.id, modelId: model.id, newName: e.target.value });
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
          <button
            className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-red-100 rounded"
            onClick={(e) => {
              e.stopPropagation();
              dispatchAndUpdate({ type: 'REMOVE_MODEL', layerId: layer.id, modelId: model.id });
            }}
          >
            <X className="w-3 h-3 text-red-600" />
          </button>
        </div>
      </div>
    );
  };

  const renderLayer = (layer) => {
    const isEditing = editingText === layer.id;
    const maxModelY = Math.max(0, ...layer.models.map(m => m.y + 40)); // Approximate height
    const layerHeight = maxModelY + 60;

    return (
      <div
        key={layer.id}
        className="absolute w-full border-2 border-dashed border-gray-300 rounded-lg bg-gray-50 p-4"
        style={{ top: layer.y, height: layerHeight }}
      >
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
                  dispatchAndUpdate({ type: 'UPDATE_LAYER_NAME', layerId: layer.id, newName: e.target.value });
                  setEditingText(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    dispatchAndUpdate({ type: 'UPDATE_LAYER_NAME', layerId: layer.id, newName: e.target.value });
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
          <div className="flex items-center space-x-1">
            <button
              onClick={() => dispatchAndUpdate({ type: 'ADD_MODEL', layerId: layer.id, modelType: 'document' })}
              className="p-1 text-blue-600 hover:bg-blue-100 rounded"
              title="Add Document"
            >
              <FileText className="w-4 h-4" />
            </button>
            <button
              onClick={() => dispatchAndUpdate({ type: 'ADD_MODEL', layerId: layer.id, modelType: 'database' })}
              className="p-1 text-green-600 hover:bg-green-100 rounded"
              title="Add Database"
            >
              <Database className="w-4 h-4" />
            </button>
            <button
              onClick={() => dispatchAndUpdate({ type: 'ADD_MODEL', layerId: layer.id, modelType: 'cylinder' })}
              className="p-1 text-purple-600 hover:bg-purple-100 rounded"
              title="Add System"
            >
              <Box className="w-4 h-4" />
            </button>
            <div className="w-px h-4 bg-gray-300" />
            <button
              onClick={() => dispatchAndUpdate({ type: 'MOVE_LAYER', layerId: layer.id, direction: 'up' })}
              className="p-1 text-gray-600 hover:bg-gray-100 rounded"
              title="Move Up"
            >
              <ChevronUp className="w-4 h-4" />
            </button>
            <button
              onClick={() => dispatchAndUpdate({ type: 'MOVE_LAYER', layerId: layer.id, direction: 'down' })}
              className="p-1 text-gray-600 hover:bg-gray-100 rounded"
              title="Move Down"
            >
              <ChevronDown className="w-4 h-4" />
            </button>
            <button
              onClick={() => dispatchAndUpdate({ type: 'REMOVE_LAYER', layerId: layer.id })}
              className="p-1 text-red-600 hover:bg-red-100 rounded"
              title="Delete Layer"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>
        <div className="relative" style={{ height: maxModelY + 20 }}>
          {layer.models.map(model => renderModel(model, layer))}
          {layer.models.length === 0 && (
            <div className="flex items-center justify-center h-16 text-gray-500 text-sm">
              Click the icons above to add models to this layer
            </div>
          )}
        </div>
      </div>
    );
  };

  const sortedLayers = [...diagramData.layers].sort((a, b) => a.y - b.y);
  const totalHeight = Math.max(500, sortedLayers[sortedLayers.length - 1]?.y + 200 || 500);

  return (
    <div className="mb-8 w-full" onMouseMove={handleMouseMove} onMouseUp={handleMouseUp}>
      <label className="block text-lg font-semibold mb-4 text-gray-800">
        {label} {required && <span className="text-red-500">*</span>}
      </label>
      <div className="w-full border rounded-xl overflow-hidden shadow-sm bg-white">
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
                onClick={undo}
                disabled={history.length === 0}
                className="p-2 text-gray-600 hover:bg-gray-100 rounded disabled:opacity-50"
                title="Undo"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
              <button
                onClick={() => setConnectMode(!connectMode)}
                className={`p-2 ${connectMode ? 'bg-blue-500 text-white' : 'text-gray-600 hover:bg-gray-100'} rounded`}
                title="Connect Mode"
              >
                <Link className="w-4 h-4" />
              </button>
              <button
                onClick={() => dispatchAndUpdate({ type: 'ADD_LAYER' })}
                className="flex items-center space-x-2 bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition-all"
              >
                <Plus className="w-4 h-4" />
                <span>Add Layer</span>
              </button>
            </div>
          </div>
          <p className="text-sm text-purple-700 mt-2">
            Double-click to rename • Drag models to reposition • {connectMode ? 'Click two models to connect' : 'Toggle connect mode to link models'} • Click connections to remove
          </p>
        </div>
        <div 
          ref={canvasRef}
          className="relative overflow-auto"
          style={{ height: 800, transform: `scale(${zoom})`, transformOrigin: 'top left', width: `${100 / zoom}%`, height: `${800 / zoom}px` }}
        >
          <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ height: totalHeight }}>
            {diagramData.connections.map(renderConnection)}
          </svg>
          <div className="relative" style={{ height: totalHeight }}>
            {sortedLayers.map(renderLayer)}
          </div>
          {diagramData.layers.length === 0 && (
            <div className="flex flex-col items-center justify-center h-64 text-gray-500">
              <Layers className="w-16 h-16 mb-4 text-gray-300" />
              <p className="text-lg">No layers yet. Click "Add Layer" to get started.</p>
            </div>
          )}
        </div>
        <div className="w-full bg-gray-100 px-6 py-3 border-t">
          <p className="text-xs text-gray-600">
            💡 Tip: Use undo for mistakes. Zoom for better view. Connections are automatic curves; drag models for better layout.
          </p>
        </div>
      </div>
      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
};

export default CDEDiagramBuilder;