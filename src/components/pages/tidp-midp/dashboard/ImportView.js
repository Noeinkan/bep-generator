import React from 'react';
import { Upload } from 'lucide-react';

const ImportView = ({ onImport }) => {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-12">
        <div className="text-center mb-12">
          <div className="w-24 h-24 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-8">
            <Upload className="w-12 h-12 text-blue-600" />
          </div>
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Import TIDPs</h2>
          <p className="text-gray-600 text-lg max-w-2xl mx-auto leading-relaxed">
            Import TIDP data from Excel or CSV files created by external teams.
            This allows seamless integration of team plans from various sources.
          </p>
        </div>

        <div className="space-y-8">
          <button
            onClick={onImport}
            className="w-full flex items-center justify-center px-8 py-6 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 group"
          >
            <div className="text-center">
              <Upload className="w-8 h-8 mx-auto mb-4 text-gray-400 group-hover:text-blue-500 transition-colors" />
              <span className="text-xl font-semibold text-gray-600 group-hover:text-blue-700 transition-colors">Import from Excel/CSV</span>
              <p className="text-gray-500 mt-2">Click to select and upload your files</p>
            </div>
          </button>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="p-6 border border-gray-200 rounded-lg bg-gray-50">
              <h3 className="font-bold text-gray-900 text-lg mb-4">Supported Formats</h3>
              <ul className="text-gray-700 space-y-2">
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  Excel (.xlsx, .xls)
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  CSV (.csv)
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  UTF-8 encoding recommended
                </li>
              </ul>
            </div>

            <div className="p-6 border border-gray-200 rounded-lg bg-blue-50">
              <h3 className="font-bold text-gray-900 text-lg mb-4">What's Imported</h3>
              <ul className="text-gray-700 space-y-2">
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mr-3"></div>
                  Team information
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mr-3"></div>
                  Deliverable containers
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mr-3"></div>
                  Schedules and milestones
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mr-3"></div>
                  Dependencies
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImportView;
