import React, { useState, useCallback } from 'react';
import { ChevronRight, ChevronLeft, Eye, Zap, FolderOpen, Save, ExternalLink } from 'lucide-react';

// Import all the existing BEP components
import ProgressSidebar from '../forms/controls/ProgressSidebar';
import CONFIG from '../../config/bepConfig';
import INITIAL_DATA from '../../data/initialData';
import FormStep from '../steps/FormStep';
import PreviewExportPage from './PreviewExportPage';
import EnhancedBepTypeSelector from './bep/EnhancedBepTypeSelector';
import DraftManager from './drafts/DraftManager';
import SaveDraftDialog from './drafts/SaveDraftDialog';
import { generateBEPContent } from '../../services/bepFormatter';
import { generatePDF } from '../../services/pdfGenerator';
import { generateDocx } from '../../services/docxGenerator';
import { useTidpData } from '../../hooks/useTidpData';
import { useMidpData } from '../../hooks/useMidpData';
import { useDraftOperations } from '../../hooks/useDraftOperations';
import { useAuth } from '../../contexts/AuthContext';
import { usePage } from '../../contexts/PageContext';
import { validateDraftName } from '../../utils/validationUtils';

const BEPGeneratorWrapper = () => {
  // const [searchParams] = useSearchParams();
  const { user } = useAuth();
  const { navigateTo } = usePage();

  // State variables
  const [formData, setFormData] = useState(INITIAL_DATA);
  const [bepType, setBepType] = useState('');
  const [currentStep, setCurrentStep] = useState(0);
  const [validationErrors, setValidationErrors] = useState({});
  const [completedSections, setCompletedSections] = useState(new Set());
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [showDraftManager, setShowDraftManager] = useState(false);
  const [newDraftName, setNewDraftName] = useState('');
  const [showSaveDraftDialog, setShowSaveDraftDialog] = useState(false);
  const [showSuccessToast, setShowSuccessToast] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [exportFormat, setExportFormat] = useState('pdf');

  // Check if we should show TIDP creation form
  // const shouldShowTidpForm = searchParams.get('createTidp') === 'true';

  // TIDP and MIDP data
  const { tidps, loading: _tidpLoading } = useTidpData();
  const { midps, loading: _midpLoading } = useMidpData();

  // Draft operations
  const { saveDraft, isLoading: savingDraft, error: _draftError } = useDraftOperations(user, formData, bepType, (loadedData, loadedType) => {
    setFormData(loadedData);
    setBepType(loadedType);
  }, () => {});

  // Close draft manager if user logs out
  React.useEffect(() => {
    if (!user && showDraftManager) {
      setShowDraftManager(false);
    }
  }, [user, showDraftManager]);

  // Initialize default milestones for step 5 (Information Delivery Planning)
  React.useEffect(() => {
    if (currentStep === 5 && (!formData.keyMilestones || formData.keyMilestones.length === 0)) {
      setFormData(prev => ({
        ...prev,
        keyMilestones: [
          { stage: 'Stage 3', description: 'Spatial Coordination', deliverables: 'Federated Models', dueDate: '' },
          { stage: 'Stage 4', description: 'Technical Design', deliverables: 'Construction Models', dueDate: '' },
          { stage: 'Stage 5', description: 'Manufacturing & Construction', deliverables: 'As-Built Models', dueDate: '' },
          { stage: 'Stage 6', description: 'Handover', deliverables: 'COBie Data', dueDate: '' }
        ]
      }));
    }
  }, [currentStep, formData.keyMilestones]);

  // Draft name validation
  const newDraftNameValidation = React.useMemo(() => {
    if (!newDraftName) return { isValid: false, error: null, sanitized: '' };
    return validateDraftName(newDraftName);
  }, [newDraftName]);

  // Navigation functions
  const goToTidpManager = () => {
    navigateTo('tidp-midp');
  };

  const goHome = () => {
    navigateTo('home');
  };

  // All existing BEP Generator methods (copied from App.js)
  const validateStep = useCallback((stepIndex, data) => {
    if (!bepType || !data) return {};

    const stepConfig = CONFIG.getFormFields(bepType, stepIndex);
    if (!stepConfig || !stepConfig.fields) return {};

    const errors = {};

    stepConfig.fields.forEach(field => {
      const value = data[field.name];

      if (field.required) {
        if (!value || (Array.isArray(value) && value.length === 0) ||
            (typeof value === 'string' && value.trim() === '')) {
          errors[field.name] = `${field.label} is required`;
        }
      }

      if (field.validation && value) {
        if (field.validation.minLength && value.length < field.validation.minLength) {
          errors[field.name] = `${field.label} must be at least ${field.validation.minLength} characters`;
        }
        if (field.validation.pattern && !field.validation.pattern.test(value)) {
          errors[field.name] = field.validation.message || `${field.label} format is invalid`;
        }
      }
    });

    return errors;
  }, [bepType]);

  const handleNext = useCallback(() => {
    const errors = validateStep(currentStep, formData);

    if (Object.keys(errors).length > 0) {
      setValidationErrors(errors);
      return;
    }

    setValidationErrors({});

    // Mark current step as completed
    setCompletedSections(prev => new Set(prev).add(currentStep));

    const totalSteps = CONFIG.steps?.length || 0;
    if (currentStep < totalSteps - 1) {
      setIsTransitioning(true);
      setTimeout(() => {
        setCurrentStep(prev => prev + 1);
        setIsTransitioning(false);
      }, 150);
    } else if (currentStep === totalSteps - 1) {
      // Last step reached, go to preview
      setShowPreview(true);
    }
  }, [currentStep, formData, validateStep]);

  const handlePrevious = useCallback(() => {
    if (currentStep > 0) {
      setIsTransitioning(true);
      setTimeout(() => {
        setCurrentStep(prev => prev - 1);
        setValidationErrors({});
        setIsTransitioning(false);
      }, 150);
    }
  }, [currentStep]);

  const updateFormData = useCallback((field, value) => {
    setFormData(prev => {
      const newData = { ...prev, [field]: value };

      if (validationErrors[field]) {
        const newErrors = { ...validationErrors };
        delete newErrors[field];
        setValidationErrors(newErrors);
      }

      return newData;
    });
  }, [validationErrors]);

  const handleTypeSelect = useCallback((selectedType) => {
    // Set all state synchronously to avoid race conditions
    setBepType(selectedType);
    setFormData({...INITIAL_DATA}); // Create a new object to ensure re-render
    setCurrentStep(0);
    setValidationErrors({});
    setShowPreview(false);
    setCompletedSections(new Set());
  }, []);

  const handlePreview = useCallback(() => {
    console.log('handlePreview called, currentStep:', currentStep, 'formData keys:', Object.keys(formData));
    // Allow preview even with validation errors for appendices step
    if (currentStep === CONFIG.steps?.length - 1) {
      console.log('On appendices step, allowing preview');
      setShowPreview(true);
      return;
    }

    const errors = validateStep(currentStep, formData);
    console.log('Validation errors:', errors);
    if (Object.keys(errors).length > 0) {
      setValidationErrors(errors);
      alert('Please fix validation errors before previewing');
      return;
    }
    setShowPreview(true);
  }, [currentStep, formData, validateStep]);

  const handlePreviewBEP = useCallback(() => {
    const content = generateBEPContent(formData, bepType, { tidpData: tidps, midpData: midps });
    const newWindow = window.open('', '_blank');
    if (newWindow) {
      newWindow.document.write(content);
      newWindow.document.close();
    }
  }, [formData, bepType, tidps, midps]);

  const handleSaveDraft = useCallback(async () => {
    if (!user) {
      alert('Please log in to save drafts');
      return;
    }
    if (!bepType) {
      alert('Please select a BEP type first');
      return;
    }
    setNewDraftName('');
    setShowSaveDraftDialog(true);
  }, [user, bepType]);

  const handleSaveDraftConfirm = useCallback(async () => {
    if (!newDraftNameValidation.isValid) return;

    try {
      await saveDraft(newDraftNameValidation.sanitized, formData);
      setShowSaveDraftDialog(false);
      setNewDraftName('');
      setShowSuccessToast(true);
      setTimeout(() => setShowSuccessToast(false), 3000);
    } catch (error) {
      alert('Failed to save draft: ' + error.message);
    }
  }, [newDraftNameValidation, formData, saveDraft]);

  const handleSaveDraftCancel = useCallback(() => {
    setShowSaveDraftDialog(false);
    setNewDraftName('');
  }, []);

  const generateContent = useCallback(async () => {
    setIsGenerating(true);
    try {
      const content = await generateBEPContent(formData, bepType, { tidpData: tidps, midpData: midps });
      // For HTML export, we skip DOMPurify sanitization since content is generated by our code
      // and is safe. DOMPurify was removing CSS styles needed for proper formatting.
      return content;
    } catch (error) {
      console.error('Content generation failed:', error);
      throw error;
    } finally {
      setIsGenerating(false);
    }
  }, [formData, bepType, tidps, midps]);

  const handleExport = useCallback(async () => {
    try {
      setIsGenerating(true);

      if (exportFormat === 'pdf') {
        try {
          const result = await generatePDF(formData, bepType, { tidpData: tidps, midpData: midps });
          if (result.success) {
            console.log(`PDF generated successfully: ${result.filename} (${result.size} bytes)`);
          }
        } catch (error) {
          console.error('PDF generation failed:', error);
          alert('PDF generation failed: ' + error.message);
        }
      } else if (exportFormat === 'word') {
        const docxBlob = await generateDocx(formData, bepType, { tidpData: tidps, midpData: midps });
        const url = URL.createObjectURL(docxBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `BEP_${bepType}_${new Date().toISOString().split('T')[0]}.docx`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } else if (exportFormat === 'html') {
        const content = await generateContent();
        const blob = new Blob([content], { type: 'text/html;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `BEP_${bepType}_${new Date().toISOString().split('T')[0]}.html`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed: ' + error.message);
    } finally {
      setIsGenerating(false);
    }
  }, [exportFormat, formData, bepType, generateContent, tidps, midps]);

  // If no BEP type selected, show type selector
  if (!bepType) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50" data-page-uri="/bep-generator">
        {/* Header with navigation */}
        <div className="bg-white shadow-lg border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                <Zap className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">BEP Generator</h1>
                <p className="text-gray-600">Create professional BIM Execution Plans</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={goToTidpManager}
                className="inline-flex items-center px-4 py-2 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors duration-200"
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                TIDP/MIDP Manager
              </button>
              <button
                onClick={goHome}
                className="inline-flex items-center px-4 py-2 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors duration-200"
              >
                <Zap className="w-4 h-4 mr-2" />
                Home
              </button>
            </div>
          </div>
        </div>

        <div className="max-w-6xl mx-auto px-4 py-8">
          <div className="bg-transparent rounded-xl p-0">
            <EnhancedBepTypeSelector
              bepType={bepType}
              setBepType={setBepType}
              onProceed={(selectedType) => handleTypeSelect(selectedType)}
            />
          </div>
        </div>
      </div>
    );
  }

  // Show preview if requested
  if (showPreview) {
    return (
      <PreviewExportPage
        formData={formData}
        bepType={bepType}
        onBack={() => setShowPreview(false)}
        onExport={handleExport}
        isGenerating={isGenerating}
        exportFormat={exportFormat}
        setExportFormat={setExportFormat}
        previewBEP={handlePreviewBEP}
        downloadBEP={handleExport}
        isExporting={isGenerating}
        tidpData={tidps}
        midpData={midps}
      />
    );
  }

  // Show draft manager if requested and user is logged in
  if (showDraftManager && user) {
    return (
      <DraftManager
        user={user}
        currentFormData={formData}
        onLoadDraft={(loadedData, loadedType) => {
          setFormData(loadedData);
          setBepType(loadedType);
          setShowDraftManager(false);
        }}
        onClose={() => setShowDraftManager(false)}
        bepType={bepType}
      />
    );
  }

  // Main BEP Generator interface
  return (
    <div className="h-screen bg-gray-50 flex relative" data-page-uri="/bep-generator">
      {/* Sidebar */}
      <div className="w-80 bg-white shadow-xl border-r border-gray-200 flex flex-col">
        <div className="p-6 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-indigo-50">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-xl font-bold text-gray-900 flex items-center">
                <Zap className="w-5 h-5 text-blue-600 mr-2" />
                BEP Generator
              </h1>
              <p className="text-sm text-gray-600 mt-1">{CONFIG.bepTypeDefinitions[bepType]?.title}</p>
            </div>
            <div className="flex items-center space-x-1">
              <button
                onClick={goToTidpManager}
                className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors duration-200"
                title="TIDP/MIDP Manager"
              >
                <ExternalLink className="w-4 h-4" />
              </button>
              <button
                onClick={goHome}
                className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors duration-200"
                title="Home"
              >
                <Zap className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="flex space-x-2">
            <button
              onClick={() => setShowDraftManager(true)}
              disabled={!user}
              className="flex-1 inline-flex items-center justify-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 hover:border-blue-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
            >
              <FolderOpen className="w-4 h-4 mr-2" />
              Drafts
            </button>
            <button
              onClick={() => setBepType('')}
              className="flex-1 inline-flex items-center justify-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 hover:border-blue-300 transition-all duration-200"
            >
              Change Type
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          <ProgressSidebar
            steps={CONFIG.steps || []}
            currentStep={currentStep}
            completedSections={completedSections}
            onStepClick={(stepIndex) => {
              setIsTransitioning(true);
              setTimeout(() => {
                setCurrentStep(stepIndex);
                setValidationErrors({});
                setIsTransitioning(false);
              }, 150);
            }}
            validateStep={validateStep}
            tidpData={tidps}
            midpData={midps}
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
  <div className="bg-white shadow-sm border-b border-gray-200 px-6 py-4 bg-gradient-to-r from-white to-gray-50 sticky top-0 z-10">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">
                  {CONFIG.steps[currentStep]?.title}
                </h2>
                <p className="text-sm text-gray-600">
                  {currentStep >= (CONFIG.steps?.length || 0) - 1 ? 'Ready for preview' : `Step ${currentStep + 1} of ${CONFIG.steps?.length || 0}`}
                </p>
              </div>
              {/* Progress indicator */}
              <div className="hidden md:flex items-center space-x-2">
                <div className="w-32 bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${((currentStep + 1) / (CONFIG.steps?.length || 1)) * 100}%` }}
                  ></div>
                </div>
                <span className="text-xs text-gray-500 font-medium">
                  {Math.round(((currentStep + 1) / (CONFIG.steps?.length || 1)) * 100)}%
                </span>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              {/* Navigation arrows */}
              <button
                onClick={handlePrevious}
                disabled={currentStep === 0}
                className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 hover:border-blue-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
              >
                <ChevronLeft className="w-4 h-4 mr-1" />
                Previous
              </button>

              <button
                onClick={handleNext}
                disabled={currentStep >= (CONFIG.steps?.length || 0)}
                className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 hover:border-blue-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
              >
                {currentStep >= (CONFIG.steps?.length || 0) - 1 ? 'Preview' : 'Next'}
                <ChevronRight className="w-4 h-4 ml-1" />
              </button>

              {/* TIDP/MIDP Integration */}
              <button
                onClick={goToTidpManager}
                className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-blue-50 hover:border-blue-300 transition-all duration-200"
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                TIDP/MIDP
              </button>

              <button
                onClick={handleSaveDraft}
                disabled={savingDraft || !user}
                className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-green-50 hover:border-green-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
              >
                <Save className="w-4 h-4 mr-2" />
                {savingDraft ? 'Saving...' : 'Save Draft'}
              </button>

              <button
                onClick={() => {
                  console.log('Preview button clicked, currentStep:', currentStep);
                  handlePreview();
                }}
                className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-lg text-white bg-blue-600 hover:bg-blue-700 hover:shadow-md transition-all duration-200 transform hover:scale-105"
              >
                <Eye className="w-4 h-4 mr-2" />
                Preview
              </button>
            </div>
          </div>
        </div>

        {/* Form Content */}
  <div className="flex-1 overflow-y-auto bg-gray-50">
          <div className={`mx-auto px-6 py-8 ${currentStep === CONFIG.steps?.length - 1 ? 'max-w-[297mm]' : 'max-w-[210mm]'}`}>
            <div className={`bg-white rounded-xl shadow-sm border border-gray-200 p-8 transition-all duration-300 ease-in-out ${isTransitioning ? 'opacity-50 transform scale-95' : 'opacity-100 transform scale-100'}`}>
              {formData && bepType ? (
                <FormStep
                  stepIndex={currentStep}
                  formData={formData}
                  updateFormData={updateFormData}
                  errors={validationErrors}
                  bepType={bepType}
                />
              ) : (
                <div className="flex items-center justify-center py-12">
                  <div className="text-center">
                    <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <div className="w-6 h-6 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                    </div>
                    <p className="text-gray-600">Loading form data...</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer Navigation */}
  <div className="bg-white border-t border-gray-200 px-6 py-4 shadow-lg flex-shrink-0">
          <div className="flex items-center justify-between">
            <button
              onClick={handlePrevious}
              disabled={currentStep === 0}
              className="inline-flex items-center px-6 py-3 border border-gray-300 shadow-sm text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 hover:border-blue-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
            >
              <ChevronLeft className="w-4 h-4 mr-2" />
              Previous
            </button>

            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-500 font-medium">
                {currentStep >= (CONFIG.steps?.length || 0) - 1 ? 'Ready for preview' : `Step ${currentStep + 1} of ${CONFIG.steps?.length || 0}`}
              </span>

              <button
                onClick={handleNext}
                disabled={currentStep >= (CONFIG.steps?.length || 0)}
                className="inline-flex items-center px-6 py-3 border border-transparent shadow-sm text-sm font-medium rounded-lg text-white bg-blue-600 hover:bg-blue-700 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105"
              >
                {currentStep >= (CONFIG.steps?.length || 0) - 1 ? 'Preview' : 'Next'}
                <ChevronRight className="w-4 h-4 ml-2" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Success Toast */}
      {showSuccessToast && (
        <div className="fixed top-4 right-4 z-50">
          <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded-lg shadow-lg flex items-center">
            <div className="w-5 h-5 bg-green-500 rounded-full flex items-center justify-center mr-3">
              <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            </div>
            <span className="font-medium">Draft saved successfully!</span>
          </div>
        </div>
      )}

      <SaveDraftDialog
        show={showSaveDraftDialog}
        newDraftName={newDraftName}
        isNewDraftNameValid={newDraftNameValidation.isValid}
        newDraftNameValidation={newDraftNameValidation}
        onNewDraftNameChange={setNewDraftName}
        onSave={handleSaveDraftConfirm}
        onCancel={handleSaveDraftCancel}
      />
    </div>
  );
};

export default BEPGeneratorWrapper;