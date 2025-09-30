import React from 'react';

const MIDPForm = ({ midpForm, onMidpFormChange, onSubmit, onCancel }) => {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Create New MIDP</h2>
          <button
            type="button"
            onClick={onCancel}
            className="text-gray-500 hover:text-gray-700"
          >
            ✕
          </button>
        </div>
        <form
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            onSubmit();
          }}
        >
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Project Name</label>
            <input
              value={midpForm.projectName}
              onChange={(e) => onMidpFormChange({ ...midpForm, projectName: e.target.value })}
              type="text"
              className="w-full p-3 border border-gray-300 rounded-lg"
              placeholder="Office Complex Project"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Description</label>
            <input
              value={midpForm.description}
              onChange={(e) => onMidpFormChange({ ...midpForm, description: e.target.value })}
              type="text"
              className="w-full p-3 border border-gray-300 rounded-lg"
              placeholder="Brief description"
            />
          </div>
          <div className="flex space-x-4">
            <button
              type="submit"
              className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg"
            >
              Create MIDP
            </button>
            <button
              type="button"
              onClick={onCancel}
              className="bg-gray-300 hover:bg-gray-400 text-gray-700 px-6 py-2 rounded-lg"
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default MIDPForm;