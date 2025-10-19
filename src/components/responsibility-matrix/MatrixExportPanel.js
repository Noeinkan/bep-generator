import React, { useState } from 'react';
import { Download, FileSpreadsheet, FileText, X, CheckCircle } from 'lucide-react';

/**
 * Matrix Export Panel
 * Provides export options for both matrices
 */
const MatrixExportPanel = ({
  onExport,
  onClose,
  loading = false
}) => {
  const [selectedFormat, setSelectedFormat] = useState('excel');
  const [selectedMatrices, setSelectedMatrices] = useState({
    imActivities: true,
    deliverables: true
  });
  const [includeDetails, setIncludeDetails] = useState({
    descriptions: true,
    isoReferences: true,
    syncStatus: true,
    summary: true
  });

  const toggleMatrix = (matrix) => {
    setSelectedMatrices(prev => ({
      ...prev,
      [matrix]: !prev[matrix]
    }));
  };

  const toggleDetail = (detail) => {
    setIncludeDetails(prev => ({
      ...prev,
      [detail]: !prev[detail]
    }));
  };

  const handleExport = () => {
    onExport({
      format: selectedFormat,
      matrices: selectedMatrices,
      details: includeDetails
    });
  };

  const canExport = selectedMatrices.imActivities || selectedMatrices.deliverables;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full">
        {/* Header */}
        <div className="bg-gradient-to-r from-green-600 to-green-700 text-white px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Download size={24} />
            <div>
              <h2 className="text-xl font-semibold">Export Responsibility Matrices</h2>
              <p className="text-sm text-green-100 mt-1">
                Choose format and content to export
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-white hover:bg-green-800 p-1 rounded transition-colors"
          >
            <X size={24} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Export Format */}
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">Export Format</h3>
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => setSelectedFormat('excel')}
                className={`flex items-center gap-3 p-4 border-2 rounded-lg transition-all ${
                  selectedFormat === 'excel'
                    ? 'border-green-600 bg-green-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <FileSpreadsheet size={24} className={selectedFormat === 'excel' ? 'text-green-600' : 'text-gray-400'} />
                <div className="text-left">
                  <div className={`font-medium ${selectedFormat === 'excel' ? 'text-green-900' : 'text-gray-900'}`}>
                    Excel Workbook
                  </div>
                  <div className="text-xs text-gray-600 mt-0.5">
                    Multi-sheet XLSX file
                  </div>
                </div>
                {selectedFormat === 'excel' && (
                  <CheckCircle size={20} className="text-green-600 ml-auto" />
                )}
              </button>

              <button
                onClick={() => setSelectedFormat('pdf')}
                className={`flex items-center gap-3 p-4 border-2 rounded-lg transition-all ${
                  selectedFormat === 'pdf'
                    ? 'border-green-600 bg-green-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <FileText size={24} className={selectedFormat === 'pdf' ? 'text-green-600' : 'text-gray-400'} />
                <div className="text-left">
                  <div className={`font-medium ${selectedFormat === 'pdf' ? 'text-green-900' : 'text-gray-900'}`}>
                    PDF Document
                  </div>
                  <div className="text-xs text-gray-600 mt-0.5">
                    ISO 19650 format
                  </div>
                </div>
                {selectedFormat === 'pdf' && (
                  <CheckCircle size={20} className="text-green-600 ml-auto" />
                )}
              </button>
            </div>
          </div>

          {/* Select Matrices */}
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">Include Matrices</h3>
            <div className="space-y-2">
              <label className="flex items-center gap-3 p-3 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors">
                <input
                  type="checkbox"
                  checked={selectedMatrices.imActivities}
                  onChange={() => toggleMatrix('imActivities')}
                  className="w-5 h-5 rounded border-gray-300 text-green-600 focus:ring-green-500"
                />
                <div className="flex-1">
                  <div className="font-medium text-gray-900">Information Management Activities</div>
                  <div className="text-xs text-gray-600">ISO 19650-2 Annex A activities with RACI assignments</div>
                </div>
              </label>

              <label className="flex items-center gap-3 p-3 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors">
                <input
                  type="checkbox"
                  checked={selectedMatrices.deliverables}
                  onChange={() => toggleMatrix('deliverables')}
                  className="w-5 h-5 rounded border-gray-300 text-green-600 focus:ring-green-500"
                />
                <div className="flex-1">
                  <div className="font-medium text-gray-900">Information Deliverables</div>
                  <div className="text-xs text-gray-600">Deliverables from TIDPs with schedule and responsibilities</div>
                </div>
              </label>
            </div>
          </div>

          {/* Additional Details */}
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">Additional Details</h3>
            <div className="grid grid-cols-2 gap-2">
              <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeDetails.descriptions}
                  onChange={() => toggleDetail('descriptions')}
                  className="rounded border-gray-300 text-green-600 focus:ring-green-500"
                />
                Include descriptions
              </label>

              <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeDetails.isoReferences}
                  onChange={() => toggleDetail('isoReferences')}
                  className="rounded border-gray-300 text-green-600 focus:ring-green-500"
                />
                Include ISO references
              </label>

              <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeDetails.syncStatus}
                  onChange={() => toggleDetail('syncStatus')}
                  className="rounded border-gray-300 text-green-600 focus:ring-green-500"
                />
                Include sync status
              </label>

              <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeDetails.summary}
                  onChange={() => toggleDetail('summary')}
                  className="rounded border-gray-300 text-green-600 focus:ring-green-500"
                />
                Include summary sheet
              </label>
            </div>
          </div>

          {/* Export Preview */}
          {selectedFormat === 'excel' && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="font-medium text-blue-900 mb-2 text-sm">Excel Export Contents</h4>
              <ul className="space-y-1 text-sm text-blue-800">
                {includeDetails.summary && <li>• Summary & Overview Sheet</li>}
                {selectedMatrices.imActivities && <li>• Information Management Activities Matrix</li>}
                {selectedMatrices.deliverables && <li>• Information Deliverables Matrix</li>}
                {selectedMatrices.deliverables && includeDetails.syncStatus && <li>• TIDP Sync Status</li>}
              </ul>
            </div>
          )}

          {selectedFormat === 'pdf' && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="font-medium text-blue-900 mb-2 text-sm">PDF Export Contents</h4>
              <ul className="space-y-1 text-sm text-blue-800">
                <li>• Document header with project info</li>
                {selectedMatrices.imActivities && <li>• Section 5.1: Information Management Activities</li>}
                {selectedMatrices.deliverables && <li>• Section 5.2: Information Deliverables</li>}
                {includeDetails.summary && <li>• Executive summary</li>}
                <li>• Version control and approval signatures</li>
              </ul>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="bg-gray-50 px-6 py-4 border-t flex items-center justify-end gap-3">
          <button
            onClick={onClose}
            className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-100 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleExport}
            disabled={!canExport || loading}
            className="flex items-center gap-2 px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                Exporting...
              </>
            ) : (
              <>
                <Download size={18} />
                Export {selectedFormat.toUpperCase()}
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default MatrixExportPanel;
