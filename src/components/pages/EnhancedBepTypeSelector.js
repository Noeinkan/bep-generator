import React from 'react';
import { ChevronRight, CheckCircle, Zap, Target, Eye } from 'lucide-react';
import CONFIG from '../../config/bepConfig';

const EnhancedBepTypeSelector = ({ bepType, setBepType, onProceed }) => (
  <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
    <div className="bg-white rounded-2xl shadow-xl w-full max-w-4xl p-8">
      <div className="text-center mb-8">
        <Zap className="w-16 h-16 text-blue-600 mx-auto mb-4" />
        <h1 className="text-3xl font-bold text-gray-900 mb-2">BIM Execution Plan Generator</h1>
        <p className="text-gray-600 mb-6">Choose your BEP type to begin the tailored workflow</p>
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-left">
          <h3 className="font-semibold text-yellow-800 mb-2">What is a BEP?</h3>
          <p className="text-sm text-yellow-700">
            A BIM Execution Plan (BEP) explains how the information management aspects of the appointment will be carried out by the delivery team.
            It sets out how information requirements are managed and delivered collectively by all parties involved in the project.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {Object.entries(CONFIG.bepTypeDefinitions).map(([key, definition]) => {
          const IconComponent = definition.icon;
          const isSelected = bepType === key;

          return (
            <div
              key={key}
              className={`relative p-6 border-2 rounded-xl cursor-pointer transition-all transform hover:scale-105 ${
                isSelected
                  ? `border-${definition.color}-500 ${definition.bgClass} shadow-lg`
                  : 'border-gray-200 bg-white hover:border-gray-300 shadow-md'
              }`}
              onClick={() => setBepType(key)}
            >
              <div className="flex items-start space-x-4">
                <div className={`p-3 rounded-lg ${
                  isSelected ? `bg-${definition.color}-100` : 'bg-gray-100'
                }`}>
                  <IconComponent className={`w-8 h-8 ${
                    isSelected ? `text-${definition.color}-600` : 'text-gray-600'
                  }`} />
                </div>

                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-2">
                    <h3 className={`text-xl font-bold ${
                      isSelected ? definition.textClass : 'text-gray-900'
                    }`}>
                      {definition.title}
                    </h3>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      isSelected
                        ? `bg-${definition.color}-100 text-${definition.color}-700`
                        : 'bg-gray-100 text-gray-600'
                    }`}>
                      {definition.subtitle}
                    </span>
                  </div>

                  <p className="text-sm text-gray-600 mb-4 leading-relaxed">
                    {definition.description}
                  </p>

                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <Target className="w-4 h-4 text-gray-500" />
                      <span className="text-sm font-medium text-gray-700">Purpose:</span>
                      <span className="text-sm text-gray-600">{definition.purpose}</span>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Eye className="w-4 h-4 text-gray-500" />
                      <span className="text-sm font-medium text-gray-700">Focus:</span>
                      <span className="text-sm text-gray-600">{definition.focus}</span>
                    </div>

                    <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                      <span className="text-xs font-medium text-gray-700 block mb-1">Language Style:</span>
                      <span className="text-xs text-gray-600 italic">{definition.language}</span>
                    </div>
                  </div>
                </div>
              </div>

              {isSelected && (
                <div className="absolute top-3 right-3">
                  <CheckCircle className={`w-6 h-6 text-${definition.color}-600`} />
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="text-center">
        <button
          onClick={onProceed}
          disabled={!bepType}
          className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium px-8 py-3 rounded-lg transition-all transform hover:scale-105 shadow-lg mx-auto disabled:transform-none disabled:cursor-not-allowed"
        >
          <span>Proceed with {bepType ? CONFIG.bepTypeDefinitions[bepType].title : 'Selected BEP Type'}</span>
          <ChevronRight className="w-5 h-5" />
        </button>

        {bepType && (
          <p className="mt-3 text-sm text-gray-600">
            You've selected: <span className="font-medium">{CONFIG.bepTypeDefinitions[bepType].title}</span>
          </p>
        )}
      </div>
    </div>
  </div>
);

export default EnhancedBepTypeSelector;