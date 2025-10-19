import React, { useState, useRef, useEffect } from 'react';
import { HelpCircle, X, BookOpen, Shield, Lightbulb, FileText, AlertTriangle } from 'lucide-react';

/**
 * FieldHelpTooltip - Optimized version with lazy content loading
 * 
 * This component displays contextual help for BEP form fields with multiple tabs:
 * - Info: Basic description and related fields
 * - ISO 19650: Standards compliance information
 * - Best Practices: Professional recommendations
 * - Examples: Real-world examples by project type
 * - Common Mistakes: What to avoid
 * 
 * Performance optimizations:
 * - Lazy loads help content only when tooltip is opened
 * - Uses React.memo to prevent unnecessary re-renders
 * - Efficient event listener cleanup
 */
const FieldHelpTooltip = React.memo(({ fieldName, helpContent, position = 'right' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('info');
  const popoverRef = useRef(null);
  const buttonRef = useRef(null);

  // Define tabs configuration
  const allTabs = [
    { id: 'info', label: 'Info', icon: BookOpen, color: 'blue', contentKey: 'description' },
    { id: 'iso19650', label: 'ISO 19650', icon: Shield, color: 'green', contentKey: 'iso19650' },
    { id: 'bestPractices', label: 'Best Practices', icon: Lightbulb, color: 'yellow', contentKey: 'bestPractices' },
    { id: 'examples', label: 'Examples', icon: FileText, color: 'purple', contentKey: 'examples' },
    { id: 'commonMistakes', label: 'Common Mistakes', icon: AlertTriangle, color: 'red', contentKey: 'commonMistakes' }
  ];

  const tabs = helpContent ? allTabs.filter(tab => helpContent[tab.contentKey]) : [];

  // Set activeTab to first available tab when dialog opens
  useEffect(() => {
    if (isOpen && tabs.length > 0 && !tabs.find(tab => tab.id === activeTab)) {
      setActiveTab(tabs[0].id);
    }
  }, [isOpen, tabs, activeTab]);

  // Close on click outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        popoverRef.current &&
        !popoverRef.current.contains(event.target) &&
        buttonRef.current &&
        !buttonRef.current.contains(event.target)
      ) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      // Prevent body scroll when modal is open on mobile
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  // Close on Escape key
  useEffect(() => {
    const handleEscape = (event) => {
      if (event.key === 'Escape' && isOpen) {
        setIsOpen(false);
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen]);

  if (!helpContent) return null;

  const renderTabContent = () => {
    const currentTab = tabs.find(tab => tab.id === activeTab);
    const content = currentTab ? helpContent[currentTab.contentKey] : null;

    if (!content) {
      return <p className="text-gray-500 italic">No content available for this section.</p>;
    }

    switch (activeTab) {
      case 'info':
        return (
          <div className="prose prose-sm max-w-none">
            <p className="text-gray-700 whitespace-pre-line leading-relaxed">{content}</p>
            {helpContent.relatedFields && helpContent.relatedFields.length > 0 && (
              <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-xs font-semibold text-blue-900 mb-1">Related Fields:</p>
                <div className="flex flex-wrap gap-2">
                  {helpContent.relatedFields.map(field => (
                    <span key={field} className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded">
                      {field}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case 'iso19650':
        return (
          <div className="space-y-3">
            <div className="flex items-start gap-2">
              <Shield className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-green-900 text-sm mb-2">Standard Reference:</p>
                <p className="text-gray-700 whitespace-pre-line leading-relaxed text-sm">{content}</p>
              </div>
            </div>
            <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-xs text-green-800">
                💡 <strong>Compliance Tip:</strong> Following ISO 19650 guidelines ensures your BEP meets international standards for information management.
              </p>
            </div>
          </div>
        );

      case 'bestPractices':
        return (
          <div className="space-y-3">
            <div className="flex items-start gap-2 mb-3">
              <Lightbulb className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
              <p className="font-semibold text-yellow-900 text-sm">Professional Recommendations:</p>
            </div>
            <ul className="space-y-2">
              {Array.isArray(content) ? content.map((practice, index) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="w-5 h-5 flex items-center justify-center bg-yellow-100 text-yellow-700 rounded-full text-xs font-semibold flex-shrink-0 mt-0.5">
                    {index + 1}
                  </span>
                  <span className="text-sm text-gray-700 leading-relaxed">{practice}</span>
                </li>
              )) : <li className="text-sm text-gray-700">{content}</li>}
            </ul>
          </div>
        );

      case 'examples':
        if (typeof content === 'object' && !Array.isArray(content)) {
          return (
            <div className="space-y-4">
              {Object.entries(content).map(([projectType, example]) => (
                <div key={projectType} className="border border-purple-200 rounded-lg overflow-hidden">
                  <div className="bg-purple-50 px-3 py-2 border-b border-purple-200">
                    <p className="font-semibold text-purple-900 text-sm">{projectType}</p>
                  </div>
                  <div className="p-3 bg-white">
                    <p className="text-sm text-gray-700 whitespace-pre-line leading-relaxed italic">
                      "{example}"
                    </p>
                  </div>
                </div>
              ))}
            </div>
          );
        }
        return <p className="text-sm text-gray-700 italic">"{content}"</p>;

      case 'commonMistakes':
        return (
          <div className="space-y-3">
            <div className="flex items-start gap-2 mb-3">
              <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <p className="font-semibold text-red-900 text-sm">What to Avoid:</p>
            </div>
            <ul className="space-y-2">
              {Array.isArray(content) ? content.map((mistake, index) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="w-5 h-5 flex items-center justify-center bg-red-100 text-red-700 rounded-full flex-shrink-0 mt-0.5">
                    <X className="w-3 h-3" />
                  </span>
                  <span className="text-sm text-gray-700 leading-relaxed">{mistake}</span>
                </li>
              )) : <li className="text-sm text-gray-700">{content}</li>}
            </ul>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="relative inline-block">
      {/* Help Button */}
      <button
        ref={buttonRef}
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="p-1 text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1"
        aria-label={`Help for ${fieldName}`}
        aria-expanded={isOpen}
        title="Show field guidance"
      >
        <HelpCircle className="w-4 h-4" />
      </button>

      {/* Popover/Modal - Only render when open for performance */}
      {isOpen && (
        <>
          {/* Mobile: Full-screen modal backdrop */}
          <div className="md:hidden fixed inset-0 bg-black bg-opacity-50 z-[9998]" />

          <div
            ref={popoverRef}
            className={`
              fixed md:absolute z-[9999]

              // Mobile: Full screen modal
              inset-0 md:inset-auto

              // Desktop: Popover positioned right
              md:left-full md:top-0 md:ml-2
              md:w-[700px] md:max-h-[800px]

              bg-white rounded-lg md:rounded-xl shadow-2xl
              overflow-hidden
              flex flex-col
            `}
            role="dialog"
            aria-labelledby="help-dialog-title"
            aria-modal="true"
          >
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white px-4 md:px-6 py-4 flex-shrink-0">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <HelpCircle className="w-5 h-5" />
                  <h3 id="help-dialog-title" className="font-semibold text-sm md:text-base">
                    Field Guidance: {fieldName}
                  </h3>
                </div>
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-1 hover:bg-white hover:bg-opacity-20 rounded-lg transition-colors"
                  aria-label="Close help"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Tabs */}
            <div className="border-b border-gray-200 bg-gray-50 px-2 md:px-4 flex-shrink-0 overflow-x-auto">
              <div className="flex gap-1 min-w-max">
                {tabs.map(tab => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;
                  return (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => setActiveTab(tab.id)}
                      className={`
                        flex items-center gap-1.5 px-3 py-2.5 text-xs md:text-sm font-medium
                        border-b-2 transition-all
                        ${isActive
                          ? `border-${tab.color}-500 text-${tab.color}-700 bg-white`
                          : 'border-transparent text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                        }
                      `}
                      role="tab"
                      aria-selected={isActive}
                      aria-controls={`${tab.id}-panel`}
                    >
                      <Icon className="w-4 h-4" />
                      <span className="whitespace-nowrap">{tab.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Content */}
            <div 
              className="flex-1 overflow-y-auto p-4 md:p-6"
              role="tabpanel"
              id={`${activeTab}-panel`}
            >
              {renderTabContent()}
            </div>

            {/* Footer hint */}
            <div className="border-t border-gray-200 bg-gray-50 px-4 py-3 flex-shrink-0">
              <p className="text-xs text-gray-600 flex items-center gap-1">
                <Lightbulb className="w-3 h-3" />
                <span>Use this guidance to create ISO 19650 compliant content</span>
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
});

FieldHelpTooltip.displayName = 'FieldHelpTooltip';

export default FieldHelpTooltip;
