import React from 'react';
import { Plus } from 'lucide-react';

/**
 * Swimlane background component (clickable to add solutions)
 */
const SwimlaneBackground = ({ data }) => {
  const [showAddButton, setShowAddButton] = React.useState(false);

  const handleClick = () => {
    if (data.onAddNode) {
      data.onAddNode(data.id);
    }
  };

  return (
    <div
      style={{
        width: '230px',
        height: '500px',
        background: data.color,
        border: `2px solid ${data.borderColor}`,
        borderRadius: '8px',
        opacity: showAddButton ? 0.5 : 0.3,
        cursor: 'pointer',
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'opacity 0.2s',
      }}
      onMouseEnter={() => setShowAddButton(true)}
      onMouseLeave={() => setShowAddButton(false)}
      onClick={handleClick}
      className="nodrag"
    >
      {showAddButton && (
        <div
          style={{
            background: data.borderColor,
            color: 'white',
            borderRadius: '50%',
            width: '48px',
            height: '48px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 6px rgba(0,0,0,0.2)',
            pointerEvents: 'none',
          }}
        >
          <Plus size={28} strokeWidth={3} />
        </div>
      )}
    </div>
  );
};

export default SwimlaneBackground;
