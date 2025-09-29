import React from 'react';

const SaveDraftDialog = ({
  show,
  newDraftName,
  isNewDraftNameValid,
  newDraftNameValidation,
  onNewDraftNameChange,
  onSave,
  onCancel
}) => {
  if (!show) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-md">
        <h3 className="text-lg font-semibold mb-4">Save Draft</h3>
        <div className="space-y-2">
          <input
            type="text"
            value={newDraftName}
            onChange={(e) => onNewDraftNameChange(e.target.value)}
            placeholder="Draft name..."
            className={`w-full p-3 border rounded-lg focus:ring-2 focus:border-transparent transition-colors ${
              newDraftName && !isNewDraftNameValid
                ? 'border-red-300 focus:ring-red-500'
                : 'border-gray-300 focus:ring-blue-500'
            }`}
            autoFocus
            onKeyPress={(e) => {
              if (e.key === 'Enter' && isNewDraftNameValid) {
                onSave();
              }
            }}
          />
          {newDraftName && !isNewDraftNameValid && newDraftNameValidation.error && (
            <p className="text-sm text-red-600 flex items-center">
              <svg className="w-4 h-4 mr-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              {newDraftNameValidation.error}
            </p>
          )}
          {newDraftName && isNewDraftNameValid && newDraftNameValidation.sanitized !== newDraftName && (
            <p className="text-sm text-gray-600 flex items-center">
              <svg className="w-4 h-4 mr-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
              Will be saved as: "{newDraftNameValidation.sanitized}"
            </p>
          )}
        </div>
        <div className="flex space-x-3 mt-4">
          <button
            onClick={onSave}
            disabled={!isNewDraftNameValid}
            className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Save
          </button>
          <button
            onClick={onCancel}
            className="flex-1 bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

export default SaveDraftDialog;