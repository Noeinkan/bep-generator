import React from 'react';
import { FolderOpen, Trash2, Edit3, Download, Calendar } from 'lucide-react';

const DraftListItem = ({
  draft,
  editingId,
  editingName,
  isEditingNameValid,
  editingNameValidation,
  debouncedSearchQuery,
  onEditingNameChange,
  onStartEdit,
  onCancelEdit,
  onRenameDraft,
  onLoadDraft,
  onExportDraft,
  onDeleteDraft,
  formatDate,
  highlightSearchTerm
}) => {
  const isEditing = editingId === draft.id;

  return (
    <div className="p-6 hover:bg-gray-50 transition-colors">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          {isEditing ? (
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <input
                  type="text"
                  value={editingName}
                  onChange={(e) => onEditingNameChange(e.target.value)}
                  className={`flex-1 p-2 border rounded focus:ring-2 focus:border-transparent transition-colors ${
                    editingName && !isEditingNameValid
                      ? 'border-red-300 focus:ring-red-500'
                      : 'border-gray-300 focus:ring-blue-500'
                  }`}
                  autoFocus
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' && isEditingNameValid) {
                      onRenameDraft(draft.id, editingName);
                    }
                  }}
                />
                <button
                  onClick={() => onRenameDraft(draft.id, editingName)}
                  disabled={!isEditingNameValid}
                  className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-3 py-2 rounded transition-colors"
                >
                  Save
                </button>
                <button
                  onClick={onCancelEdit}
                  className="bg-gray-500 hover:bg-gray-600 text-white px-3 py-2 rounded transition-colors"
                >
                  Cancel
                </button>
              </div>
              {editingName && !isEditingNameValid && editingNameValidation.error && (
                <p className="text-sm text-red-600 flex items-center">
                  <svg className="w-4 h-4 mr-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  {editingNameValidation.error}
                </p>
              )}
              {editingName && isEditingNameValid && editingNameValidation.sanitized !== editingName && (
                <p className="text-sm text-gray-600 flex items-center">
                  <svg className="w-4 h-4 mr-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                  </svg>
                  Will be saved as: "{editingNameValidation.sanitized}"
                </p>
              )}
            </div>
          ) : (
            <div>
              <h3 className="text-lg font-medium text-gray-900">
                {highlightSearchTerm(draft.name, debouncedSearchQuery)}
              </h3>
              <div className="flex items-center space-x-4 mt-2 text-sm text-gray-500">
                <div className="flex items-center space-x-1">
                  <Calendar className="w-4 h-4" />
                  <span>{formatDate(draft.lastModified)}</span>
                </div>
                <span>•</span>
                <span>
                  {highlightSearchTerm(draft.projectName, debouncedSearchQuery)}
                </span>
                <span>•</span>
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                  {draft.bepType === 'pre-appointment' ? 'Pre-Appointment' : 'Post-Appointment'}
                </span>
              </div>
            </div>
          )}
        </div>

        {!isEditing && (
          <div className="flex items-center space-x-2 ml-4">
            <button
              onClick={() => onLoadDraft(draft)}
              className="flex items-center space-x-1 bg-green-600 hover:bg-green-700 text-white px-3 py-2 rounded transition-colors"
              title="Load draft"
            >
              <FolderOpen className="w-4 h-4" />
              <span>Load</span>
            </button>

            <button
              onClick={() => onExportDraft(draft)}
              className="flex items-center space-x-1 bg-purple-600 hover:bg-purple-700 text-white px-3 py-2 rounded transition-colors"
              title="Export draft"
            >
              <Download className="w-4 h-4" />
            </button>

            <button
              onClick={() => onStartEdit(draft.id, draft.name)}
              className="flex items-center space-x-1 bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded transition-colors"
              title="Rename"
            >
              <Edit3 className="w-4 h-4" />
            </button>

            <button
              onClick={() => onDeleteDraft(draft.id)}
              className="flex items-center space-x-1 bg-red-600 hover:bg-red-700 text-white px-3 py-2 rounded transition-colors"
              title="Delete"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default DraftListItem;