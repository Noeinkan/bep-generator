import React, { useState, useCallback } from 'react';
import { useNavigate, useLocation, useSearchParams } from 'react-router-dom';
import { ChevronRight, ChevronLeft, Eye, Zap, FolderOpen, Save, ExternalLink } from 'lucide-react';
import DOMPurify from 'dompurify';

// Import all the existing BEP components
import ProgressSidebar from '../ui/ProgressSidebar';
import CONFIG from '../../config/bepConfig';
import INITIAL_DATA from '../../data/initialData';
import FormStep from '../steps/FormStep';
import PreviewExportPage from './PreviewExportPage';
import EnhancedBepTypeSelector from './EnhancedBepTypeSelector';
import DraftManager from './DraftManager';
import SaveDraftDialog from './SaveDraftDialog';
import { generateBEPContent } from '../../services/bepFormatter';
import { generatePDF } from '../../services/pdfGenerator';
import { useDraftOperations } from '../../hooks/useDraftOperations';
import { useAuth } from '../../contexts/AuthContext';
import { validateDraftName } from '../../utils/validationUtils';

const BEPGeneratorWrapper = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const { user } = useAuth();

  // Check if we should show TIDP creation form
  const shouldShowTidpForm = searchParams.get('createTidp') === 'true';

  // All the existing BEP Generator state (copied from App.js)
  const [currentStep, setCurrentStep] = useState(0);
  const [bepType, setBepType] = useState('');
  const [formData, setFormData] = useState(INITIAL_DATA);
  const [validationErrors, setValidationErrors] = useState({});
  const [showPreview, setShowPreview] = useState(false);
  const [showDraftManager, setShowDraftManager] = useState(false);
  const [showSaveDraftDialog, setShowSaveDraftDialog] = useState(false);
  const [newDraftName, setNewDraftName] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [exportFormat, setExportFormat] = useState('pdf');
  const [completedSections, setCompletedSections] = useState(new Set());

  // Draft operations
  const { saveDraft, isLoading: savingDraft, error: draftError } = useDraftOperations(user, formData, bepType, (loadedData, loadedType) => {
    setFormData(loadedData);
    setBepType(loadedType);
  }, () => {});

  // Close draft manager if user logs out
  React.useEffect(() => {
    if (!user && showDraftManager) {
      setShowDraftManager(false);
    }
  }, [user, showDraftManager]);

  // Draft name validation
  const newDraftNameValidation = React.useMemo(() => {
    if (!newDraftName) return { isValid: false, error: null, sanitized: '' };
    return validateDraftName(newDraftName);
  }, [newDraftName]);

  // Navigation functions
  const goToTidpManager = () => {
    navigate('/tidp-midp');
  };

  const goHome = () => {
    navigate('/');
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
      setCurrentStep(prev => prev + 1);
    }
  }, [currentStep, formData, validateStep, bepType]);

  const handlePrevious = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
      setValidationErrors({});
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
    const errors = validateStep(currentStep, formData);
    if (Object.keys(errors).length > 0) {
      setValidationErrors(errors);
      return;
    }
    setShowPreview(true);
  }, [currentStep, formData, validateStep]);

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
      alert('Draft saved successfully!');
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
      const content = await generateBEPContent(formData, bepType);
      return DOMPurify.sanitize(content);
    } catch (error) {
      console.error('Content generation failed:', error);
      throw error;
    } finally {
      setIsGenerating(false);
    }
  }, [formData, bepType]);

  const handleExport = useCallback(async () => {
    try {
      setIsGenerating(true);

      if (exportFormat === 'pdf') {
        await generatePDF(formData, bepType);
      } else {
        const content = await generateContent();
        // Handle other export formats...
      }
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsGenerating(false);
    }
  }, [exportFormat, formData, bepType, generateContent]);

  // If no BEP type selected, show type selector
  if (!bepType) {
    return (
      <div className="min-h-screen bg-gray-50">
        {/* Header with navigation */}
        <div className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">BEP Generator</h1>
              <p className="text-gray-600">Create professional BIM Execution Plans</p>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={goToTidpManager}
                className="inline-flex items-center text-gray-600 hover:text-gray-900"
              >
                <ExternalLink className="w-4 h-4 mr-1" />
                TIDP/MIDP Manager
              </button>
              <button
                onClick={goHome}
                className="text-gray-600 hover:text-gray-900"
              >
                Home
              </button>
            </div>
          </div>
        </div>

        <div className="max-w-4xl mx-auto px-4 py-8">
          <EnhancedBepTypeSelector
            bepType={bepType}
            setBepType={setBepType}
            onProceed={() => handleTypeSelect(bepType)}
          />
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
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <div className="w-80 bg-white shadow-lg">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-xl font-bold text-gray-900">BEP Generator</h1>
              <p className="text-sm text-gray-600">{CONFIG.bepTypeDefinitions[bepType]?.title}</p>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={goToTidpManager}
                className="p-2 text-gray-400 hover:text-gray-600"
                title="TIDP/MIDP Manager"
              >
                <ExternalLink className="w-4 h-4" />
              </button>
              <button
                onClick={goHome}
                className="p-2 text-gray-400 hover:text-gray-600"
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
              className="flex-1 inline-flex items-center justify-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <FolderOpen className="w-4 h-4 mr-2" />
              Drafts
            </button>
            <button
              onClick={() => setBepType('')}
              className="flex-1 inline-flex items-center justify-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
            >
              Change Type
            </button>
          </div>
        </div>

        <ProgressSidebar
          steps={CONFIG.steps || []}
          currentStep={currentStep}
          completedSections={completedSections}
          onStepClick={(stepIndex) => setCurrentStep(stepIndex)}
          validateStep={validateStep}
        />
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white shadow-sm border-b px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-gray-900">
                {CONFIG.steps[currentStep]?.title}
              </h2>
              <p className="text-sm text-gray-600">
                Step {currentStep + 1} of {CONFIG.steps?.length || 0}
              </p>
            </div>

            <div className="flex items-center space-x-3">
              {/* Navigation arrows */}
              <button
                onClick={handlePrevious}
                disabled={currentStep === 0}
                className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-4 h-4 mr-1" />
                Previous
              </button>

              <button
                onClick={handleNext}
                disabled={currentStep >= (CONFIG.steps?.length || 0) - 1}
                className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
                <ChevronRight className="w-4 h-4 ml-1" />
              </button>

              {/* TIDP/MIDP Integration */}
              <button
                onClick={goToTidpManager}
                className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                TIDP/MIDP Manager
              </button>

              <button
                onClick={handleSaveDraft}
                disabled={savingDraft || !user}
                className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Save className="w-4 h-4 mr-2" />
                {savingDraft ? 'Saving...' : 'Save Draft'}
              </button>

              <button
                onClick={handlePreview}
                className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
              >
                <Eye className="w-4 h-4 mr-2" />
                Preview
              </button>
            </div>
          </div>
        </div>

        {/* Form Content */}
        <div className="flex-1 overflow-auto">
          <div className="max-w-4xl mx-auto px-6 py-8">
            {formData && bepType ? (
              <FormStep
                stepIndex={currentStep}
                formData={formData}
                updateFormData={updateFormData}
                errors={validationErrors}
                bepType={bepType}
              />
            ) : (
              <div>Loading form data...</div>
            )}
          </div>
        </div>

        {/* Footer Navigation */}
        <div className="bg-white border-t px-6 py-4">
          <div className="flex items-center justify-between">
            <button
              onClick={handlePrevious}
              disabled={currentStep === 0}
              className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeft className="w-4 h-4 mr-2" />
              Previous
            </button>

            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-500">
                {currentStep + 1} of {CONFIG.steps?.length || 0}
              </span>

              <button
                onClick={handleNext}
                disabled={currentStep >= (CONFIG.steps?.length || 0) - 1}
                className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
                <ChevronRight className="w-4 h-4 ml-2" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <>
      <SaveDraftDialog
        show={showSaveDraftDialog}
        newDraftName={newDraftName}
        isNewDraftNameValid={newDraftNameValidation.isValid}
        newDraftNameValidation={newDraftNameValidation}
        onNewDraftNameChange={setNewDraftName}
        onSave={handleSaveDraftConfirm}
        onCancel={handleSaveDraftCancel}
      />
    </>
  );
};

export default BEPGeneratorWrapper;