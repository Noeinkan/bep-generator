import React from 'react';

const TIDPDetailsPanel = ({ tidp, detailsForm, onDetailsFormChange, onSave, onDelete, onClose }) => {
  const updateContainer = (ci, field, value) => {
    const updated = [...(detailsForm.containers || [])];
    updated[ci] = { ...updated[ci], [field]: value };
    onDetailsFormChange({ ...detailsForm, containers: updated });
  };

  const removeContainer = (ci) => {
    const updated = (detailsForm.containers || []).filter((_, idx) => idx !== ci);
    onDetailsFormChange({ ...detailsForm, containers: updated });
  };

  const addContainer = () => {
    const updated = [
      ...(detailsForm.containers || []),
      {
        id: `IC-${Date.now()}`,
        'Information Container ID': `IC-${Date.now()}`,
        'Information Container Name/Title': '',
        'Due Date': ''
      }
    ];
    onDetailsFormChange({ ...detailsForm, containers: updated });
  };

  return (
    <div className="fixed right-6 top-20 w-96 bg-white border rounded-lg shadow-lg p-4 z-50">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-semibold">TIDP Details</h4>
        <button type="button" onClick={onClose} className="text-sm text-gray-600">
          Close
        </button>
      </div>
      <div className="text-sm text-gray-700">
        <label className="block text-xs font-medium mb-1">Task Team</label>
        <input
          value={detailsForm.taskTeam}
          onChange={(e) => onDetailsFormChange({ ...detailsForm, taskTeam: e.target.value })}
          className="w-full p-2 border rounded mb-2 text-sm"
        />
        <label className="block text-xs font-medium mb-1">Description</label>
        <input
          value={detailsForm.description}
          onChange={(e) => onDetailsFormChange({ ...detailsForm, description: e.target.value })}
          className="w-full p-2 border rounded mb-3 text-sm"
        />

        <div className="mb-3">
          <label className="block text-xs font-medium mb-2">Deliverables</label>
          <div className="space-y-2 max-h-48 overflow-auto">
            {(detailsForm.containers || []).map((c, ci) => (
              <div key={c.id || ci} className="flex items-center space-x-2">
                <input
                  value={
                    c['Information Container Name/Title'] ||
                    c['Container Name'] ||
                    c.name ||
                    ''
                  }
                  onChange={(e) => updateContainer(ci, 'Information Container Name/Title', e.target.value)}
                  className="flex-1 p-2 border rounded text-sm"
                  placeholder="Container Name"
                />
                <input
                  value={c['Due Date'] || c.dueDate || ''}
                  onChange={(e) => updateContainer(ci, 'Due Date', e.target.value)}
                  type="date"
                  className="p-2 border rounded text-sm"
                />
                <button
                  type="button"
                  onClick={() => removeContainer(ci)}
                  className="text-red-600 text-sm"
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
          <div className="mt-2">
            <button
              type="button"
              onClick={addContainer}
              className="bg-gray-100 px-2 py-1 rounded text-sm"
            >
              Add deliverable
            </button>
          </div>
        </div>

        <div className="flex space-x-2">
          <button
            type="button"
            onClick={() => onSave(tidp.id, {
              taskTeam: detailsForm.taskTeam,
              description: detailsForm.description,
              containers: detailsForm.containers
            })}
            className="bg-blue-600 text-white px-3 py-1 rounded text-sm"
          >
            Save
          </button>
          <button
            type="button"
            onClick={() => onDelete(tidp.id)}
            className="bg-red-600 text-white px-3 py-1 rounded text-sm"
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  );
};

export default TIDPDetailsPanel;