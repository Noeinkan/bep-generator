import React from 'react';
import { FileText, Package } from 'lucide-react';

/**
 * Matrix Selector Component
 * Toggles between IM Activities Matrix and Information Deliverables Matrix
 */
const MatrixSelector = ({ activeMatrix, onMatrixChange, imActivitiesCount, deliverablesCount }) => {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-1 inline-flex gap-1">
      <button
        onClick={() => onMatrixChange('im-activities')}
        className={`flex items-center gap-3 px-6 py-3 rounded-lg font-medium transition-all ${
          activeMatrix === 'im-activities'
            ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-md'
            : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
        }`}
      >
        <FileText size={20} />
        <div className="text-left">
          <div className="font-semibold">
            Matrix 1: IM Activities
          </div>
          <div className={`text-xs mt-0.5 ${
            activeMatrix === 'im-activities' ? 'text-blue-100' : 'text-gray-500'
          }`}>
            {imActivitiesCount || 0} activities
          </div>
        </div>
      </button>

      <button
        onClick={() => onMatrixChange('deliverables')}
        className={`flex items-center gap-3 px-6 py-3 rounded-lg font-medium transition-all ${
          activeMatrix === 'deliverables'
            ? 'bg-gradient-to-r from-indigo-600 to-indigo-700 text-white shadow-md'
            : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
        }`}
      >
        <Package size={20} />
        <div className="text-left">
          <div className="font-semibold">
            Matrix 2: Deliverables
          </div>
          <div className={`text-xs mt-0.5 ${
            activeMatrix === 'deliverables' ? 'text-indigo-100' : 'text-gray-500'
          }`}>
            {deliverablesCount || 0} deliverables
          </div>
        </div>
      </button>
    </div>
  );
};

export default MatrixSelector;
