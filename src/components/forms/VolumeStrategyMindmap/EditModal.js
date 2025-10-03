import React from 'react';

const EditModal = ({
  editingNode,
  editingText,
  setEditingText,
  onSave,
  onCancel
}) => {
  if (!editingNode) return null;

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') onSave();
    if (e.key === 'Escape') onCancel();
  };

  return (
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
          aria-label="Edit node name"
          onKeyDown={handleKeyDown}
        />
        <div className="flex justify-end space-x-2">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
            aria-label="Cancel edit"
          >
            Cancel
          </button>
          <button
            onClick={onSave}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
            aria-label="Save edit"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
};

export default EditModal;