import React from 'react';
import { Download, FileText, Eye, FileType, Printer } from 'lucide-react';

const PreviewExportPage = ({ generateBEPContent, exportFormat, setExportFormat, previewBEP, downloadBEP, isExporting }) => {
  const content = generateBEPContent();
  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold">Preview & Export</h3>
      <div className="flex items-center space-x-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <span className="text-sm font-medium text-blue-900">Export Format:</span>
        <div className="flex space-x-3">
          {[
            { value: 'html', icon: FileType, label: 'HTML' },
            { value: 'pdf', icon: Printer, label: 'PDF' },
            { value: 'word', icon: FileText, label: 'Word' }
          ].map(format => (
            <label key={format.value} className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                value={format.value}
                checked={exportFormat === format.value}
                onChange={(e) => setExportFormat(e.target.value)}
                className="text-blue-600"
              />
              <format.icon className="w-4 h-4 text-blue-600" />
              <span className="text-sm text-blue-900">{format.label}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="flex space-x-3">
        <button
          onClick={previewBEP}
          className="flex items-center space-x-2 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg transition-all shadow-lg"
        >
          <Eye className="w-5 h-5" />
          <span>Preview BEP</span>
        </button>

        <button
          onClick={downloadBEP}
          disabled={isExporting}
          className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-8 py-3 rounded-lg transition-all transform hover:scale-105 shadow-lg disabled:opacity-50"
        >
          <Download className="w-5 h-5" />
          <span>{isExporting ? 'Exporting...' : 'Download Professional BEP'}</span>
        </button>
      </div>

      <iframe
        srcDoc={content}
        title="BEP Preview"
        className="w-full border rounded-lg"
        style={{ height: '600px' }}
      />
    </div>
  );
};

export default PreviewExportPage;