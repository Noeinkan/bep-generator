import React from 'react';
import { Calendar, Download, Plus, TrendingUp } from 'lucide-react';

const MIDPList = ({
  midps,
  templates,
  selectedTemplate,
  onTemplateChange,
  exportLoading,
  onExportPdf,
  onExportExcel,
  onViewDetails,
  onShowMidpForm,
  onShowEvolutionDashboard
}) => {
  const midp = midps.length > 0 ? midps[0] : null;

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="mb-4 flex items-center space-x-3">
          <label className="text-sm">Export template:</label>
          <select
            value={selectedTemplate || ''}
            onChange={(e) => onTemplateChange(e.target.value || null)}
            className="border p-2 rounded"
          >
            <option value="">Default</option>
            {templates.map((t) => (
              <option key={t.id || t.name} value={t.id || t.name}>
                {t.name || t.id}
              </option>
            ))}
          </select>
        </div>
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Master Information Delivery Plan</h2>
          {!midp && (
            <button
              onClick={onShowMidpForm}
              className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span>Generate MIDP</span>
            </button>
          )}
        </div>

        {!midp ? (
          <div className="text-center py-12 text-gray-500">
            <Calendar className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <p className="text-lg">No MIDP generated yet</p>
            <p className="text-sm">Generate the Master Information Delivery Plan from your TIDPs</p>
          </div>
        ) : (
          <div className="border border-gray-200 rounded-lg p-6">
            <div className="mb-4">
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                {midp.projectName || 'Project MIDP'}
              </h3>
              <p className="text-gray-600 text-sm mb-1">
                {midp.description || 'Master information delivery plan aggregating all project TIDPs'}
              </p>
              <p className="text-xs text-gray-500">
                {midp.includedTIDPs?.length || 0} TIDPs included • Last updated: {midp.updatedAt ? new Date(midp.updatedAt).toLocaleDateString() : 'N/A'}
              </p>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => onViewDetails(midp)}
                className="flex-1 bg-green-50 text-green-700 py-2 px-4 rounded text-sm hover:bg-green-100 transition-colors"
              >
                View Details
              </button>
              <button
                onClick={() => onShowEvolutionDashboard(midp.id)}
                className="bg-blue-50 text-blue-700 py-2 px-4 rounded text-sm hover:bg-blue-100 transition-colors flex items-center space-x-2"
                title="Evolution Dashboard"
              >
                <TrendingUp className="w-4 h-4" />
                <span>Evolution</span>
              </button>
              <button
                disabled={!!exportLoading[midp.id]}
                onClick={() => onExportPdf(midp.id)}
                className="bg-gray-50 text-gray-700 py-2 px-4 rounded text-sm hover:bg-gray-100 transition-colors flex items-center space-x-2"
              >
                {exportLoading[midp.id] ? '...' : (
                  <>
                    <Download className="w-4 h-4" />
                    <span>PDF</span>
                  </>
                )}
              </button>
              <button
                disabled={!!exportLoading[midp.id]}
                onClick={() => onExportExcel(midp.id)}
                className="bg-gray-50 text-gray-700 py-2 px-4 rounded text-sm hover:bg-gray-100 transition-colors"
              >
                {exportLoading[midp.id] ? '...' : 'XLSX'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MIDPList;