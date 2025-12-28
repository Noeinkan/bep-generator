import React, { useState } from 'react';
import { X, BookTemplate, Building2, Home, Route, CheckCircle, AlertCircle } from 'lucide-react';
import { getAvailableTemplates } from '../../../data/templateRegistry';

const TemplateGallery = ({ show, onSelectTemplate, onCancel }) => {
  const [selectedTemplate, setSelectedTemplate] = useState(null);

  if (!show) return null;

  // Load templates from registry
  const registryTemplates = getAvailableTemplates();

  // Map registry templates to UI format with icons and colors
  const templateIconMap = {
    'commercial-office': { icon: Building2, color: 'purple' },
    'residential-complex': { icon: Home, color: 'blue' },
    'hospital': { icon: Building2, color: 'green' },
    'infrastructure': { icon: Route, color: 'green' }
  };

  const templates = registryTemplates.map(template => ({
    ...template,
    icon: templateIconMap[template.id]?.icon || Building2,
    color: templateIconMap[template.id]?.color || 'blue',
    bepType: 'post', // Default to post-appointment
    complexity: 'Intermediate',
    comingSoon: false // All registered templates are available
  }));

  const colorClasses = {
    blue: {
      iconBg: 'bg-blue-100',
      iconText: 'text-blue-600',
      border: 'border-blue-300',
      bg: 'bg-blue-50',
      badge: 'bg-blue-100 text-blue-700'
    },
    purple: {
      iconBg: 'bg-purple-100',
      iconText: 'text-purple-600',
      border: 'border-purple-300',
      bg: 'bg-purple-50',
      badge: 'bg-purple-100 text-purple-700'
    },
    green: {
      iconBg: 'bg-green-100',
      iconText: 'text-green-600',
      border: 'border-green-300',
      bg: 'bg-green-50',
      badge: 'bg-green-100 text-green-700'
    }
  };

  const handleSelectTemplate = () => {
    if (selectedTemplate && !selectedTemplate.comingSoon) {
      onSelectTemplate(selectedTemplate);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-5xl w-full max-h-[90vh] overflow-y-auto p-6 sm:p-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-6 sticky top-0 bg-white pb-4 border-b border-slate-200">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
              <BookTemplate className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-slate-900">Template Gallery</h3>
              <p className="text-sm text-slate-600">Start with a pre-configured BEP template</p>
            </div>
          </div>
          <button
            onClick={onCancel}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-slate-600" />
          </button>
        </div>

        {/* Info Banner */}
        <div className="mb-6 p-4 bg-purple-50 border border-purple-200 rounded-lg">
          <div className="flex items-start space-x-3">
            <div className="w-5 h-5 bg-purple-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <span className="text-white text-xs font-bold">i</span>
            </div>
            <div className="text-sm text-purple-900">
              <p className="font-semibold mb-1">What are templates?</p>
              <p className="text-purple-800">
                Templates are pre-filled BEPs with example content for common project types.
                They help you get started quickly and provide guidance on industry best practices.
              </p>
            </div>
          </div>
        </div>

        {/* Template Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-6">
          {templates.map((template) => {
            const colors = colorClasses[template.color];
            const isSelected = selectedTemplate?.id === template.id;

            return (
              <div
                key={template.id}
                className={`relative border-2 rounded-xl p-6 cursor-pointer transition-all duration-200 ${
                  template.comingSoon
                    ? 'opacity-60 cursor-not-allowed border-slate-200 bg-slate-50'
                    : isSelected
                    ? `${colors.border} ${colors.bg} shadow-lg`
                    : 'border-slate-200 hover:border-slate-300 hover:shadow-md'
                }`}
                onClick={() => !template.comingSoon && setSelectedTemplate(template)}
              >
                {/* Template Icon */}
                <div className="flex items-start justify-between mb-4">
                  <div className={`p-3 rounded-lg ${colors.iconBg}`}>
                    <template.icon className={`w-8 h-8 ${colors.iconText}`} />
                  </div>
                  {template.comingSoon && (
                    <span className="px-3 py-1 bg-yellow-100 text-yellow-700 text-xs font-semibold rounded-full">
                      Coming Soon
                    </span>
                  )}
                  {isSelected && !template.comingSoon && (
                    <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                      <CheckCircle className="w-4 h-4 text-white" />
                    </div>
                  )}
                </div>

                {/* Template Info */}
                <h4 className="text-xl font-bold text-slate-900 mb-2">
                  {template.name}
                </h4>
                <p className="text-sm text-slate-600 mb-4 leading-relaxed">
                  {template.description}
                </p>

                {/* Template Metadata */}
                <div className="flex flex-wrap gap-2">
                  <span className={`px-2 py-1 rounded text-xs font-semibold ${colors.badge}`}>
                    {template.category}
                  </span>
                  <span className="px-2 py-1 bg-slate-100 text-slate-700 rounded text-xs font-semibold">
                    {template.complexity}
                  </span>
                  <span className="px-2 py-1 bg-slate-100 text-slate-700 rounded text-xs font-semibold">
                    {template.bepType === 'pre-appointment' ? 'Pre-App' : 'Post-App'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Coming Soon Notice */}
        <div className="mb-6 p-4 bg-amber-50 border border-amber-200 rounded-lg">
          <div className="flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-amber-900">
              <p className="font-semibold mb-1">Template System in Development</p>
              <p className="text-amber-800">
                The template gallery is currently under development. More templates will be added soon,
                and you'll be able to save your own BEPs as reusable templates.
              </p>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center justify-end space-x-3 pt-4 border-t border-slate-200">
          <button
            onClick={onCancel}
            className="px-6 py-3 border border-slate-300 text-slate-700 rounded-lg font-semibold hover:bg-slate-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSelectTemplate}
            disabled={!selectedTemplate || selectedTemplate.comingSoon}
            className="px-6 py-3 bg-purple-600 text-white rounded-lg font-semibold hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
          >
            <BookTemplate className="w-5 h-5" />
            <span>Use Template</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default TemplateGallery;
