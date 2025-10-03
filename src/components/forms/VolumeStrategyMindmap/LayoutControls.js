import React, { useState } from 'react';
import { Grid, TreePine, Target, MoreHorizontal } from 'lucide-react';
import { LAYOUT_TYPES } from '../../../utils/layoutUtils';

const LayoutControls = ({ onOrganizeNodes, onSnapToGrid }) => {
  const [showLayoutOptions, setShowLayoutOptions] = useState(false);

  const layoutOptions = [
    {
      type: LAYOUT_TYPES.RADIAL,
      label: 'Radial',
      icon: Target,
      description: 'Arrange nodes in circles around the center'
    },
    {
      type: LAYOUT_TYPES.TREE,
      label: 'Tree',
      icon: TreePine,
      description: 'Hierarchical tree layout'
    }
  ];

  const handleLayoutSelect = (layoutType) => {
    onOrganizeNodes(layoutType);
    setShowLayoutOptions(false);
  };

  return (
    <div className="relative">
      <button
        onClick={() => setShowLayoutOptions(!showLayoutOptions)}
        className="p-2 text-purple-600 hover:bg-purple-100 rounded"
        title="Layout Options"
        aria-label="Layout Options"
      >
        <MoreHorizontal className="w-4 h-4" />
      </button>

      {showLayoutOptions && (
        <div className="absolute top-full right-0 mt-1 bg-white border border-gray-300 rounded-lg shadow-lg z-20 min-w-56">
          <div className="p-2 border-b border-gray-200">
            <span className="text-sm font-medium text-gray-700">Auto Layout</span>
          </div>

          <div className="p-1">
            {layoutOptions.map((option) => {
              const Icon = option.icon;
              return (
                <button
                  key={option.type}
                  onClick={() => handleLayoutSelect(option.type)}
                  className="w-full flex items-start space-x-3 px-3 py-2 text-left hover:bg-gray-50 rounded"
                >
                  <Icon className="w-4 h-4 text-purple-600 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium text-gray-900">{option.label}</div>
                    <div className="text-xs text-gray-500">{option.description}</div>
                  </div>
                </button>
              );
            })}
          </div>

          <div className="p-2 border-t border-gray-200">
            <button
              onClick={() => {
                onSnapToGrid();
                setShowLayoutOptions(false);
              }}
              className="w-full flex items-center space-x-2 px-3 py-2 text-left text-sm text-gray-600 hover:bg-gray-50 rounded"
            >
              <Grid className="w-4 h-4" />
              <span>Snap to Grid</span>
            </button>
          </div>

          <div className="p-2 border-t border-gray-200">
            <button
              onClick={() => setShowLayoutOptions(false)}
              className="w-full text-sm text-gray-500 hover:text-gray-700"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default LayoutControls;