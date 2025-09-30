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
          <h2 className="text-2xl font-bold text-gray-900">Master Information Delivery Plans</h2>
          <button
            onClick={onShowMidpForm}
            className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span>New MIDP</span>
          </button>
        </div>

        {midps.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <Calendar className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <p className="text-lg">No MIDPs created yet</p>
            <p className="text-sm">Create your first Master Information Delivery Plan</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {midps.map((midp, index) => (
              <div
                key={midp.id || index}
                className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
              >
                <h3 className="font-semibold text-gray-900 mb-2">
                  {midp.projectName || `MIDP ${index + 1}`}
                </h3>
                <p className="text-gray-600 text-sm mb-3">
                  {midp.description || 'Master information delivery plan'}
                </p>
                <div className="flex space-x-2">
                  <button
                    onClick={() => onViewDetails(midp)}
                    className="flex-1 bg-green-50 text-green-700 py-2 px-3 rounded text-sm hover:bg-green-100 transition-colors"
                  >
                    View
                  </button>
                  <button
                    onClick={() => onShowEvolutionDashboard(midp.id)}
                    className="bg-blue-50 text-blue-700 py-2 px-3 rounded text-sm hover:bg-blue-100 transition-colors flex items-center"
                    title="Evolution Dashboard"
                  >
                    <TrendingUp className="w-4 h-4" />
                  </button>
                  <button
                    disabled={!!exportLoading[midp.id]}
                    onClick={() => onExportPdf(midp.id)}
                    className="bg-gray-50 text-gray-700 py-2 px-3 rounded text-sm hover:bg-gray-100 transition-colors"
                  >
                    {exportLoading[midp.id] ? '...' : <Download className="w-4 h-4" />}
                  </button>
                  <button
                    disabled={!!exportLoading[midp.id]}
                    onClick={() => onExportExcel(midp.id)}
                    className="bg-gray-50 text-gray-700 py-2 px-3 rounded text-sm hover:bg-gray-100 transition-colors"
                  >
                    {exportLoading[midp.id] ? '...' : 'XLSX'}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default MIDPList;