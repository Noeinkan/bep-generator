import React, { useRef } from 'react';
import { Plus, Calendar, Users, Download, Upload } from 'lucide-react';
import { exportTidpCsvTemplate, importTidpFromCsv } from '../../utils/csvHelpers';

const TIDPDashboard = ({ tidps, midps, onShowTidpForm, onShowMidpForm, onImportCsv, onToast }) => {
  const csvInputRef = useRef(null);

  const handleCsvImport = (event) => {
    const file = event.target.files[0];
    importTidpFromCsv(
      file,
      (containers) => {
        onImportCsv(containers);
        onToast({ open: true, message: `Imported ${containers.length} deliverables from CSV. Please fill in TIDP details and save.`, type: 'success' });
      },
      (error) => {
        onToast({ open: true, message: 'Failed to import CSV: ' + error.message, type: 'error' });
      }
    );
    event.target.value = '';
  };

  const handleExportTemplate = () => {
    try {
      exportTidpCsvTemplate();
      onToast({ open: true, message: 'TIDP CSV template downloaded successfully!', type: 'success' });
    } catch (error) {
      console.error('Download failed:', error);
      onToast({ open: true, message: 'Failed to download CSV template', type: 'error' });
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">TIDP/MIDP Dashboard</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-blue-50 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-blue-900">TIDPs</h3>
                <p className="text-3xl font-bold text-blue-600">{tidps.length}</p>
              </div>
              <Users className="w-8 h-8 text-blue-600" />
            </div>
            <p className="text-blue-700 text-sm mt-2">Task Information Delivery Plans</p>
          </div>

          <div className="bg-green-50 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-green-900">MIDPs</h3>
                <p className="text-3xl font-bold text-green-600">{midps.length}</p>
              </div>
              <Calendar className="w-8 h-8 text-green-600" />
            </div>
            <p className="text-green-700 text-sm mt-2">Master Information Delivery Plans</p>
          </div>

          <div className="bg-purple-50 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-purple-900">Projects</h3>
                <p className="text-3xl font-bold text-purple-600">1</p>
              </div>
              <Calendar className="w-8 h-8 text-purple-600" />
            </div>
            <p className="text-purple-700 text-sm mt-2">Active Projects</p>
          </div>
        </div>

        <div className="flex flex-wrap gap-4">
          <button
            onClick={onShowTidpForm}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>Create New TIDP</span>
          </button>
          <button
            onClick={onShowMidpForm}
            className="bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>Create New MIDP</span>
          </button>
          <button
            onClick={handleExportTemplate}
            className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2"
          >
            <Download className="w-4 h-4" />
            <span>Download TIDP CSV Template</span>
          </button>
          <div className="relative">
            <input
              ref={csvInputRef}
              type="file"
              accept=".csv"
              onChange={handleCsvImport}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              id="csv-import"
            />
            <label
              htmlFor="csv-import"
              className="bg-orange-600 hover:bg-orange-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2 cursor-pointer"
            >
              <Upload className="w-4 h-4" />
              <span>Import TIDP from CSV</span>
            </label>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TIDPDashboard;