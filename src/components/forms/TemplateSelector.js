import React from 'react';
import { FileText, X, Sparkles } from 'lucide-react';
import INITIAL_DATA from '../../data/initialData';

// Map di esempi per diversi field names
const FIELD_EXAMPLES = {
  'projectDescription': INITIAL_DATA.projectDescription,
  'projectContext': INITIAL_DATA.projectContext,
  'bimStrategy': INITIAL_DATA.bimStrategy,
  'keyCommitments': INITIAL_DATA.keyCommitments,
  'valueProposition': INITIAL_DATA.valueProposition,
  'teamCapabilities': INITIAL_DATA.teamCapabilities,
  'proposedBimGoals': INITIAL_DATA.proposedBimGoals,
  'proposedObjectives': INITIAL_DATA.proposedObjectives,
  'tenderApproach': INITIAL_DATA.tenderApproach,
  'deliveryApproach': INITIAL_DATA.deliveryApproach,
  'referencedMaterial': INITIAL_DATA.referencedMaterial,
  'informationManagementResponsibilities': INITIAL_DATA.informationManagementResponsibilities,
  'organizationalStructure': INITIAL_DATA.organizationalStructure,
  'confirmedBimGoals': INITIAL_DATA.confirmedBimGoals,
  'implementationObjectives': INITIAL_DATA.implementationObjectives,
  'projectInformationRequirements': INITIAL_DATA.projectInformationRequirements,
  'midpDescription': INITIAL_DATA.midpDescription,
  'deliverySchedule': INITIAL_DATA.deliverySchedule,
  'tidpRequirements': INITIAL_DATA.tidpRequirements,
  'mobilisationPlan': INITIAL_DATA.mobilisationPlan,
  'teamCapabilitySummary': INITIAL_DATA.teamCapabilitySummary,
  'taskTeamExchange': INITIAL_DATA.taskTeamExchange,
  'modelReferencing3d': INITIAL_DATA.modelReferencing3d,
  'informationBreakdownStrategy': INITIAL_DATA.informationBreakdownStrategy,
  'federationProcess': INITIAL_DATA.federationProcess,
  'bimGoals': INITIAL_DATA.bimGoals,
  'primaryObjectives': INITIAL_DATA.primaryObjectives,
  'collaborativeProductionGoals': INITIAL_DATA.collaborativeProductionGoals,
  'alignmentStrategy': INITIAL_DATA.alignmentStrategy,
  'cdeStrategy': INITIAL_DATA.cdeStrategy,
  'volumeStrategy': INITIAL_DATA.volumeStrategy,
  'bimValueApplications': INITIAL_DATA.bimValueApplications,
  'strategicAlignment': INITIAL_DATA.strategicAlignment,
};

const TemplateSelector = ({ editor, onClose, fieldName }) => {
  const handleLoadExample = () => {
    if (!editor) return;

    // Get example text for the specific field
    const exampleText = FIELD_EXAMPLES[fieldName] || INITIAL_DATA.projectDescription;

    // Convert to HTML paragraph
    const htmlContent = `<p>${exampleText}</p>`;

    // Set content WITHOUT clearing - just insert at cursor or replace selection
    const currentContent = editor.getHTML();

    // If editor is empty or just has empty paragraph, replace it
    if (currentContent === '<p></p>' || currentContent === '') {
      editor.commands.setContent(htmlContent);
    } else {
      // Otherwise, insert at cursor position
      editor.commands.insertContent(htmlContent);
    }

    onClose();
  };

  // Get the example text to show preview
  const exampleText = FIELD_EXAMPLES[fieldName] || INITIAL_DATA.projectDescription;
  const previewText = exampleText.length > 200 ? exampleText.substring(0, 200) + '...' : exampleText;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[80vh] overflow-y-auto">
        <div className="sticky top-0 bg-gradient-to-r from-blue-500 to-blue-600 text-white p-6 rounded-t-xl">
          <div className="flex justify-between items-start">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-white bg-opacity-20 rounded-lg">
                <Sparkles size={24} />
              </div>
              <div>
                <h3 className="text-xl font-bold">Load Example Text</h3>
                <p className="text-blue-100 text-sm mt-1">
                  Prepopulate this field with professional example content
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-1 hover:bg-white hover:bg-opacity-20 rounded-lg transition-colors"
              title="Close"
              type="button"
            >
              <X size={24} />
            </button>
          </div>
        </div>

        <div className="p-6">
          {/* Info Box */}
          <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-start gap-3">
              <FileText className="text-blue-600 flex-shrink-0 mt-1" size={20} />
              <div>
                <h4 className="font-semibold text-blue-900 mb-1">How it works</h4>
                <p className="text-sm text-blue-800">
                  This will add professional example text to your editor. The example text will be <strong>inserted without removing</strong> any existing content you've already written.
                </p>
              </div>
            </div>
          </div>

          {/* Preview */}
          <div className="mb-6">
            <h4 className="font-medium text-gray-800 mb-2 flex items-center gap-2">
              <span className="w-1 h-4 bg-blue-500 rounded"></span>
              Example Text Preview
            </h4>
            <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
              <p className="text-sm text-gray-700 leading-relaxed italic">
                "{previewText}"
              </p>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={handleLoadExample}
              className="flex-1 px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 transition-all shadow-md hover:shadow-lg font-medium flex items-center justify-center gap-2"
              type="button"
            >
              <Sparkles size={20} />
              Load Example Text
            </button>
            <button
              onClick={onClose}
              className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors font-medium"
              type="button"
            >
              Cancel
            </button>
          </div>

          {/* Additional Info */}
          <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p className="text-xs text-yellow-800">
              <strong>Tip:</strong> You can edit the example text after loading it, or use it as inspiration for your own content. The example is based on a real commercial office project.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TemplateSelector;
