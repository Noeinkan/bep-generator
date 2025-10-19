import React from 'react';
import { Tag, Database, Shield, Star, FileCheck, Info } from 'lucide-react';

/**
 * DeliverableAttributesVisualizer
 * Dynamically visualizes deliverable attributes and metadata
 */
const DeliverableAttributesVisualizer = ({ deliverableAttributes = [] }) => {
  // Icon mapping for common attribute types
  const getIconForAttribute = (attributeName) => {
    const name = attributeName.toLowerCase();
    if (name.includes('format') || name.includes('file')) return FileCheck;
    if (name.includes('classification') || name.includes('system')) return Database;
    if (name.includes('security') || name.includes('confidential')) return Shield;
    if (name.includes('level') || name.includes('lod') || name.includes('loi')) return Star;
    if (name.includes('suitability') || name.includes('status')) return Tag;
    return Info;
  };

  // Color scheme for different attribute types
  const getColorForAttribute = (attributeName) => {
    const name = attributeName.toLowerCase();
    if (name.includes('format') || name.includes('file')) return 'blue';
    if (name.includes('classification') || name.includes('system')) return 'purple';
    if (name.includes('security') || name.includes('confidential')) return 'red';
    if (name.includes('level') || name.includes('lod') || name.includes('loi')) return 'green';
    if (name.includes('suitability') || name.includes('status')) return 'amber';
    return 'gray';
  };

  const colorClasses = {
    blue: {
      bg: 'bg-blue-500',
      bgLight: 'bg-blue-50',
      border: 'border-blue-200',
      text: 'text-blue-700',
      bgGradient: 'from-blue-500 to-blue-600',
      icon: 'bg-blue-600'
    },
    purple: {
      bg: 'bg-purple-500',
      bgLight: 'bg-purple-50',
      border: 'border-purple-200',
      text: 'text-purple-700',
      bgGradient: 'from-purple-500 to-purple-600',
      icon: 'bg-purple-600'
    },
    red: {
      bg: 'bg-red-500',
      bgLight: 'bg-red-50',
      border: 'border-red-200',
      text: 'text-red-700',
      bgGradient: 'from-red-500 to-red-600',
      icon: 'bg-red-600'
    },
    green: {
      bg: 'bg-green-500',
      bgLight: 'bg-green-50',
      border: 'border-green-200',
      text: 'text-green-700',
      bgGradient: 'from-green-500 to-green-600',
      icon: 'bg-green-600'
    },
    amber: {
      bg: 'bg-amber-500',
      bgLight: 'bg-amber-50',
      border: 'border-amber-200',
      text: 'text-amber-700',
      bgGradient: 'from-amber-500 to-amber-600',
      icon: 'bg-amber-600'
    },
    gray: {
      bg: 'bg-gray-500',
      bgLight: 'bg-gray-50',
      border: 'border-gray-200',
      text: 'text-gray-700',
      bgGradient: 'from-gray-500 to-gray-600',
      icon: 'bg-gray-600'
    }
  };

  return (
    <div className="space-y-6">
      {/* Attributes Card Grid */}
      {deliverableAttributes && deliverableAttributes.length > 0 ? (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {deliverableAttributes.map((attr, index) => {
              const color = getColorForAttribute(attr.attributeName || '');
              const colors = colorClasses[color];
              const Icon = getIconForAttribute(attr.attributeName || '');

              return (
                <div
                  key={index}
                  className={`bg-white border-2 ${colors.border} rounded-xl overflow-hidden hover:shadow-lg transition-all duration-200 hover:-translate-y-1`}
                >
                  {/* Card Header */}
                  <div className={`bg-gradient-to-r ${colors.bgGradient} p-4`}>
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-white bg-opacity-20 rounded-lg flex items-center justify-center backdrop-blur-sm">
                        <Icon className="w-6 h-6 text-white" />
                      </div>
                      <div className="flex-1">
                        <h5 className="font-semibold text-white text-sm">
                          {attr.attributeName || 'Unnamed Attribute'}
                        </h5>
                        <div className="text-xs text-white opacity-75">
                          Attribute {index + 1}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Card Body */}
                  <div className="p-4">
                    {/* Example Value */}
                    <div className="mb-3">
                      <div className="text-xs font-medium text-gray-500 mb-1.5">
                        Example Value
                      </div>
                      <div className={`${colors.bgLight} ${colors.border} border rounded-lg p-2.5`}>
                        <code className={`text-sm ${colors.text} font-mono font-medium`}>
                          {attr.exampleValue || 'N/A'}
                        </code>
                      </div>
                    </div>

                    {/* Description */}
                    <div>
                      <div className="text-xs font-medium text-gray-500 mb-1.5">
                        Description
                      </div>
                      <p className="text-sm text-gray-600 leading-relaxed">
                        {attr.description || 'No description provided'}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Summary Table */}
          <div className="bg-gradient-to-r from-slate-50 to-gray-50 border-2 border-gray-200 rounded-xl overflow-hidden">
            <div className="bg-gradient-to-r from-gray-700 to-slate-600 px-6 py-4">
              <h5 className="font-semibold text-white flex items-center gap-2">
                <Database className="w-5 h-5" />
                Deliverable Metadata Summary
              </h5>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead>
                  <tr className="bg-gray-100 border-b border-gray-200">
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      Attribute
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      Value
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      Purpose
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {deliverableAttributes.map((attr, index) => {
                    const color = getColorForAttribute(attr.attributeName || '');
                    const colors = colorClasses[color];
                    const Icon = getIconForAttribute(attr.attributeName || '');

                    return (
                      <tr key={index} className="hover:bg-gray-50 transition-colors">
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <div className={`${colors.icon} w-8 h-8 rounded-lg flex items-center justify-center`}>
                              <Icon className="w-4 h-4 text-white" />
                            </div>
                            <span className="font-medium text-gray-900">
                              {attr.attributeName || 'Unnamed'}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <code className={`text-sm font-mono ${colors.bgLight} ${colors.text} px-3 py-1.5 rounded-md border ${colors.border}`}>
                            {attr.exampleValue || 'N/A'}
                          </code>
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-600 max-w-md">
                          {attr.description || 'No description'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Key Benefits Section */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-200 rounded-lg p-5">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center flex-shrink-0">
                  <FileCheck className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h6 className="font-semibold text-gray-900 mb-2">
                    Standardization Benefits
                  </h6>
                  <ul className="text-sm text-gray-600 space-y-1">
                    <li>• Consistent metadata across all deliverables</li>
                    <li>• Automated quality checking and validation</li>
                    <li>• Improved searchability and retrieval</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-emerald-50 border-2 border-green-200 rounded-lg p-5">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 bg-green-600 rounded-lg flex items-center justify-center flex-shrink-0">
                  <Shield className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h6 className="font-semibold text-gray-900 mb-2">
                    Compliance & Governance
                  </h6>
                  <ul className="text-sm text-gray-600 space-y-1">
                    <li>• ISO 19650 information management compliance</li>
                    <li>• Security classification enforcement</li>
                    <li>• Audit trail and version control</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="bg-gradient-to-br from-gray-50 to-slate-50 border-2 border-dashed border-gray-300 rounded-xl p-12">
          <div className="text-center">
            <Database className="w-16 h-16 mx-auto mb-4 text-gray-300" />
            <h5 className="text-lg font-semibold text-gray-600 mb-2">
              No Attributes Defined
            </h5>
            <p className="text-sm text-gray-500">
              Add deliverable attributes above to define metadata and properties for your information containers
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default DeliverableAttributesVisualizer;
