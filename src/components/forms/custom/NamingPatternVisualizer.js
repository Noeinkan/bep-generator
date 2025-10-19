import React from 'react';
import { ArrowRight, FileText } from 'lucide-react';

/**
 * NamingPatternVisualizer
 * Dynamically visualizes the naming convention pattern based on defined fields
 */
const NamingPatternVisualizer = ({ namingFields = [] }) => {
  // Generate pattern string from fields
  const generatePattern = () => {
    if (!namingFields || namingFields.length === 0) {
      return '[No fields defined]';
    }
    return namingFields
      .map(field => field.fieldName || '[Field]')
      .join('-');
  };

  // Generate example string from fields
  const generateExample = () => {
    if (!namingFields || namingFields.length === 0) {
      return 'Define fields to see example';
    }
    return namingFields
      .map(field => field.exampleValue || 'XXX')
      .join('-');
  };

  const pattern = generatePattern();
  const example = generateExample();

  return (
    <div className="space-y-6">
      {/* Visual Pattern Builder */}
      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-200 rounded-xl p-6">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
            <FileText className="w-6 h-6 text-white" />
          </div>
          <div>
            <h4 className="font-semibold text-gray-900">Pattern Structure</h4>
            <p className="text-sm text-gray-600">Dynamic naming convention pattern</p>
          </div>
        </div>

        {/* Pattern Nodes */}
        <div className="bg-white rounded-lg p-4 border border-blue-200">
          {namingFields && namingFields.length > 0 ? (
            <div className="flex flex-wrap items-center gap-2">
              {namingFields.map((field, index) => (
                <React.Fragment key={index}>
                  {/* Field Node */}
                  <div className="relative group">
                    <div className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-4 py-2 rounded-lg shadow-md hover:shadow-lg transition-all cursor-pointer border-2 border-blue-600">
                      <div className="text-xs font-medium opacity-90">
                        {field.fieldName || `Field ${index + 1}`}
                      </div>
                    </div>
                    
                    {/* Tooltip on hover */}
                    <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-64 bg-gray-900 text-white text-xs rounded-lg p-3 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 shadow-xl">
                      <div className="font-semibold mb-1">{field.fieldName || 'Unnamed Field'}</div>
                      <div className="text-gray-300 mb-1">Example: <span className="text-blue-300 font-mono">{field.exampleValue || 'N/A'}</span></div>
                      <div className="text-gray-400 text-xs">{field.description || 'No description'}</div>
                      <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1">
                        <div className="border-4 border-transparent border-t-gray-900"></div>
                      </div>
                    </div>
                  </div>

                  {/* Separator */}
                  {index < namingFields.length - 1 && (
                    <div className="text-gray-400 font-bold text-xl">-</div>
                  )}
                </React.Fragment>
              ))}
              
              {/* File Extension */}
              <div className="text-gray-400 font-bold text-xl ml-1">.rvt</div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-400">
              <FileText className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p className="text-sm">Add naming fields above to build your pattern</p>
            </div>
          )}
        </div>
      </div>

      {/* Pattern Formula */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Pattern */}
        <div className="bg-white border-2 border-purple-200 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-8 h-8 bg-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">P</span>
            </div>
            <h5 className="font-semibold text-gray-900">Pattern</h5>
          </div>
          <div className="bg-purple-50 rounded-lg p-3 border border-purple-200">
            <code className="text-sm text-purple-900 font-mono break-all">
              {pattern}
            </code>
          </div>
        </div>

        {/* Example */}
        <div className="bg-white border-2 border-green-200 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-8 h-8 bg-green-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">E</span>
            </div>
            <h5 className="font-semibold text-gray-900">Example</h5>
          </div>
          <div className="bg-green-50 rounded-lg p-3 border border-green-200">
            <code className="text-sm text-green-900 font-mono break-all">
              {example}.rvt
            </code>
          </div>
        </div>
      </div>

      {/* Detailed Breakdown */}
      {namingFields && namingFields.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
          <div className="bg-gradient-to-r from-gray-50 to-gray-100 px-4 py-3 border-b border-gray-200">
            <h5 className="font-semibold text-gray-900">Field Breakdown</h5>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">#</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Field Name</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Example Value</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Description</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {namingFields.map((field, index) => (
                  <tr key={index} className="hover:bg-gray-50 transition-colors">
                    <td className="px-4 py-3 whitespace-nowrap">
                      <div className="w-6 h-6 bg-blue-100 text-blue-700 rounded-full flex items-center justify-center text-xs font-semibold">
                        {index + 1}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <code className="text-sm font-mono bg-purple-50 text-purple-700 px-2 py-1 rounded">
                        {field.fieldName || 'Unnamed'}
                      </code>
                    </td>
                    <td className="px-4 py-3">
                      <code className="text-sm font-mono bg-green-50 text-green-700 px-2 py-1 rounded">
                        {field.exampleValue || 'N/A'}
                      </code>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">
                      {field.description || 'No description provided'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Additional Examples with Different File Types */}
      {namingFields && namingFields.length > 0 && (
        <div className="bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 rounded-lg p-4">
          <h5 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
            <ArrowRight className="w-4 h-4 text-amber-600" />
            Example Filenames with Different Extensions
          </h5>
          <div className="space-y-2">
            {['rvt', 'dwg', 'pdf', 'ifc', 'nwc'].map((ext, idx) => (
              <div key={idx} className="bg-white rounded-lg p-3 border border-amber-200 flex items-center gap-3">
                <div className="bg-amber-100 text-amber-700 px-2 py-1 rounded text-xs font-semibold uppercase min-w-[50px] text-center">
                  .{ext}
                </div>
                <code className="text-sm font-mono text-gray-700 flex-1">
                  {example}.{ext}
                </code>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default NamingPatternVisualizer;
