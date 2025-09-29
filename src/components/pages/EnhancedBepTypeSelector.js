import React from 'react';
import { CheckCircle, Zap, Target, Eye } from 'lucide-react';
import CONFIG from '../../config/bepConfig';

const EnhancedBepTypeSelector = ({ bepType, setBepType, onProceed }) => (
  <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 flex items-center justify-center p-4 sm:p-6 lg:p-8">
    <div className="bg-white rounded-3xl shadow-2xl w-full max-w-6xl p-6 sm:p-8 lg:p-12 xl:p-8 2xl:p-6 border border-slate-200">
      {/* Header Section */}
      <div className="text-center mb-12">
        <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-3xl flex items-center justify-center mx-auto mb-8 shadow-xl">
          <Zap className="w-12 h-12 text-white" />
        </div>
        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-slate-900 mb-4 tracking-tight">
          Choose Your BEP Type
        </h1>
        <p className="text-xl text-slate-600 mb-12 max-w-3xl mx-auto leading-relaxed">
          Select the BIM Execution Plan that best fits your project needs
        </p>

        {/* Info Box */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-200 rounded-2xl p-8 text-left max-w-4xl mx-auto shadow-lg">
          <div className="flex items-start space-x-4">
            <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
              <span className="text-white text-lg font-bold">?</span>
            </div>
            <div>
              <h3 className="text-2xl font-bold text-blue-900 mb-3">What is a BEP?</h3>
              <p className="text-lg text-blue-800 leading-relaxed">
                A BIM Execution Plan outlines how information management will be handled by the delivery team.
                It establishes how information requirements are managed and delivered by all project parties.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Cards Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 xl:gap-12 mb-16">
        {Object.entries(CONFIG.bepTypeDefinitions).map(([key, definition]) => {
          const IconComponent = definition.icon;
          const isSelected = bepType === key;

          return (
            <div
              key={key}
              className={`group relative bg-white border-2 rounded-3xl cursor-pointer transition-all duration-300 transform hover:scale-[1.02] hover:-translate-y-1 shadow-lg hover:shadow-2xl break-words ${
                isSelected
                  ? `border-${definition.color}-500 bg-gradient-to-br from-${definition.color}-50 to-${definition.color}-100 shadow-2xl ring-4 ring-${definition.color}-200 ring-opacity-50`
                  : 'border-slate-200 hover:border-slate-300 hover:shadow-xl'
              }`}
              onClick={() => {
                setBepType(key);
                setTimeout(() => onProceed(key), 300);
              }}
              role="button"
              tabIndex={0}
              aria-pressed={isSelected}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setBepType(key);
                  setTimeout(() => onProceed(key), 300);
                }
              }}
            >
              {/* Card Header */}
              <div className="p-8 pb-6 border-b border-slate-100">
                <div className="flex items-start space-x-6">
                  <div className={`p-4 rounded-2xl flex-shrink-0 transition-all duration-300 ${
                    isSelected
                      ? `bg-${definition.color}-100 shadow-lg`
                      : 'bg-slate-100 group-hover:bg-slate-200'
                  }`}>
                    <IconComponent className={`w-12 h-12 transition-colors duration-300 ${
                      isSelected ? `text-${definition.color}-600` : 'text-slate-600'
                    }`} />
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between mb-4">
                      <h3 className={`text-3xl font-bold transition-colors duration-300 ${
                        isSelected ? `text-${definition.color}-900` : 'text-slate-900'
                      }`}>
                        {definition.title}
                      </h3>
                      <span className={`px-4 py-2 rounded-full text-sm font-semibold transition-all duration-300 ${
                        isSelected
                          ? `bg-${definition.color}-100 text-${definition.color}-700 shadow-md`
                          : 'bg-slate-100 text-slate-600'
                      }`}>
                        {definition.subtitle}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Card Content */}
              <div className="p-8 pt-6">
                <p className="text-slate-700 mb-8 leading-relaxed text-lg">
                  {definition.description}
                </p>

                {/* Key Information */}
                <div className="space-y-6">
                  <div className="flex items-start space-x-4">
                    <div className="w-10 h-10 bg-blue-100 rounded-xl flex items-center justify-center flex-shrink-0">
                      <Target className="w-5 h-5 text-blue-600" />
                    </div>
                    <div className="flex-1">
                      <h4 className="text-lg font-bold text-slate-900 mb-1">Purpose</h4>
                      <p className="text-slate-600 leading-relaxed">{definition.purpose}</p>
                    </div>
                  </div>

                  <div className="flex items-start space-x-4">
                    <div className="w-10 h-10 bg-green-100 rounded-xl flex items-center justify-center flex-shrink-0">
                      <Eye className="w-5 h-5 text-green-600" />
                    </div>
                    <div className="flex-1">
                      <h4 className="text-lg font-bold text-slate-900 mb-1">Focus</h4>
                      <p className="text-slate-600 leading-relaxed">{definition.focus}</p>
                    </div>
                  </div>
                </div>

                {/* Language Style Box */}
                <div className="mt-8 p-6 bg-gradient-to-r from-slate-50 to-slate-100 rounded-2xl border-2 border-slate-200">
                  <div className="flex items-center space-x-3 mb-3">
                    <div className="w-6 h-6 bg-slate-400 rounded-full flex items-center justify-center">
                      <span className="text-white text-xs font-bold">💬</span>
                    </div>
                    <h4 className="text-lg font-bold text-slate-900 uppercase tracking-wide">Language Style</h4>
                  </div>
                  <p className="text-slate-700 italic leading-relaxed text-base">{definition.language}</p>
                </div>
              </div>

              {/* Selection Indicator */}
              {isSelected && (
                <div className="absolute top-6 right-6">
                  <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center shadow-xl animate-pulse">
                    <CheckCircle className="w-7 h-7 text-white" />
                  </div>
                </div>
              )}

              {/* Hover CTA */}
              <div className={`absolute bottom-6 left-8 right-8 transition-all duration-300 ${
                isSelected ? 'opacity-0 translate-y-2' : 'opacity-0 group-hover:opacity-100 group-hover:translate-y-0'
              }`}>
                <div className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-6 py-3 rounded-xl text-center font-semibold shadow-lg">
                  Select this BEP Type →
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Selection Status */}
      {bepType && (
        <div className="text-center">
          <div className="inline-flex items-center space-x-4 bg-green-50 border-2 border-green-200 rounded-full px-8 py-4 shadow-lg">
            <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-green-800">
              Selected: {CONFIG.bepTypeDefinitions[bepType].title}
            </span>
          </div>
          <p className="mt-4 text-lg text-slate-500">Proceeding automatically...</p>
        </div>
      )}
    </div>
  </div>
);

export default EnhancedBepTypeSelector;