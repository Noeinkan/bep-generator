import React, { useCallback, useEffect, useState } from 'react';
import { useNavigate, useParams, useLocation } from 'react-router-dom';
import { ChevronRight, ChevronLeft, Eye, Zap, FolderOpen, Save, ExternalLink } from 'lucide-react';
import { useBepForm } from '../../../contexts/BepFormContext';
import ProgressSidebar from '../../forms/controls/ProgressSidebar';
import FormStepRHF from '../../steps/FormStepRHF';
import CONFIG from '../../../config/bepConfig';
import { useTidpData } from '../../../hooks/useTidpData';
import { useMidpData } from '../../../hooks/useMidpData';
import { useDraftOperations } from '../../../hooks/useDraftOperations';
import { useAuth } from '../../../contexts/AuthContext';
import SaveDraftDialog from '../drafts/SaveDraftDialog';
import { validateDraftName } from '../../../utils/validationUtils';
import HiddenComponentsRenderer from '../../export/HiddenComponentsRenderer';

/**
 * Main form view component for BEP Generator
 * Handles form step navigation and displays the current step
 */
const BepFormView = () => {
  const { slug, step: stepParam } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();

  // Get form context
  const {
    bepType,
    completedSections,
    markStepCompleted,
    validateStep,
    currentDraft,
    setCurrentDraft,
    getFormData,
  } = useBepForm();

  // TIDP and MIDP data
  const { tidps } = useTidpData();
  const { midps } = useMidpData();

  // Local state
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [newDraftName, setNewDraftName] = useState('');
  const [showSaveDraftDialog, setShowSaveDraftDialog] = useState(false);
  const [showSuccessToast, setShowSuccessToast] = useState(false);
  const [existingDraftToOverwrite, setExistingDraftToOverwrite] = useState(null);

  const currentStep = parseInt(stepParam, 10) || 0;

  // Get current form data for draft operations
  const formData = getFormData();

  // Draft operations
  const { saveDraft, isLoading: savingDraft } = useDraftOperations(
    user,
    formData,
    bepType,
    () => {}, // onLoadDraft - handled in DraftManager
    () => {}  // onClose - not needed here
  );

  // Helper function to create URL-safe document names
  const createDocumentSlug = useCallback((name) => {
    return encodeURIComponent(
      name
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .substring(0, 50) || 'untitled'
    );
  }, []);

  // Helper function to get current document slug
  const getDocumentSlug = useCallback(() => {
    if (currentDraft && currentDraft.name) {
      return createDocumentSlug(currentDraft.name);
    }
    return slug || 'new-document';
  }, [currentDraft, createDocumentSlug, slug]);

  // Scroll to top when step changes
  useEffect(() => {
    // Find the scrollable content container and scroll it to top
    const contentContainer = document.querySelector('.flex-1.overflow-y-auto.bg-gray-50');
    if (contentContainer) {
      contentContainer.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, [currentStep]);

  // Initialize default milestones for step 5 (Information Delivery Planning)
  useEffect(() => {
    if (currentStep === 5 && (!formData.keyMilestones || formData.keyMilestones.length === 0)) {
      // This would be handled by React Hook Form's setValue
      // We'll update this when integrating with FormStep components
    }
  }, [currentStep, formData.keyMilestones]);

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
    navigate('/home');
  };

  const handleNext = () => {
    const totalSteps = CONFIG.steps?.length || 0;
    const isLastStep = currentStep === totalSteps - 1;

    // Validate current step
    if (!isLastStep) {
      const stepErrors = validateStep(currentStep);
      if (Object.keys(stepErrors).length > 0) {
        // Errors will be displayed by React Hook Form
        return;
      }
    }

    // Mark current step as completed
    markStepCompleted(currentStep);

    if (isLastStep) {
      // Last step reached - go to preview (screenshots already captured by useEffect)
      navigate(`/bep-generator/${getDocumentSlug()}/preview`);
    } else {
      // Move to next step
      const nextStep = currentStep + 1;
      const docSlug = getDocumentSlug();
      setIsTransitioning(true);

      requestAnimationFrame(() => {
        navigate(`/bep-generator/${docSlug}/step/${nextStep}`);
        setTimeout(() => setIsTransitioning(false), 150);
      });
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      const prevStep = currentStep - 1;
      const docSlug = getDocumentSlug();
      setIsTransitioning(true);

      requestAnimationFrame(() => {
        navigate(`/bep-generator/${docSlug}/step/${prevStep}`);
        setTimeout(() => setIsTransitioning(false), 150);
      });
    }
  };

  const handlePreview = useCallback(() => {
    const docSlug = getDocumentSlug();
    navigate(`/bep-generator/${docSlug}/preview`);
  }, [navigate, getDocumentSlug]);

  const handleSaveDraft = useCallback(async () => {
    if (!user) {
      alert('Please log in to save drafts');
      return;
    }
    if (!bepType) {
      alert('Please select a BEP type first');
      return;
    }

    // Microsoft Word behavior: if we're working on an existing draft, save it directly
    if (currentDraft && currentDraft.id) {
      try {
        const result = await saveDraft(currentDraft.name, formData, true);
        if (result.success) {
          setShowSuccessToast(true);
          setTimeout(() => setShowSuccessToast(false), 3000);
        } else {
          alert('Failed to save draft');
        }
      } catch (error) {
        alert('Failed to save draft: ' + error.message);
      }
    } else {
      // New BEP, ask for name
      setNewDraftName('');
      setShowSaveDraftDialog(true);
    }
  }, [user, bepType, currentDraft, formData, saveDraft]);

  const handleSaveDraftConfirm = useCallback(async () => {
    if (!newDraftNameValidation.isValid) return;

    try {
      const result = await saveDraft(newDraftNameValidation.sanitized, formData, false);

      if (result.success) {
        const draftInfo = { id: result.draftId, name: newDraftNameValidation.sanitized };
        setCurrentDraft(draftInfo);
        setShowSaveDraftDialog(false);
        setNewDraftName('');
        setExistingDraftToOverwrite(null);
        setShowSuccessToast(true);
        setTimeout(() => setShowSuccessToast(false), 3000);

        const newSlug = createDocumentSlug(newDraftNameValidation.sanitized);
        navigate(`/bep-generator/${newSlug}/step/${currentStep}`, { replace: true });
      } else if (result.existingDraft) {
        setExistingDraftToOverwrite(result.existingDraft);
      }
    } catch (error) {
      alert('Failed to save draft: ' + error.message);
    }
  }, [newDraftNameValidation, formData, saveDraft, createDocumentSlug, navigate, currentStep, setCurrentDraft]);

  const handleOverwriteDraft = useCallback(async () => {
    if (!newDraftNameValidation.isValid) return;

    try {
      const result = await saveDraft(newDraftNameValidation.sanitized, formData, true);

      if (result.success) {
        const draftInfo = { id: result.draftId, name: newDraftNameValidation.sanitized };
        setCurrentDraft(draftInfo);
        setShowSaveDraftDialog(false);
        setNewDraftName('');
        setExistingDraftToOverwrite(null);
        setShowSuccessToast(true);
        setTimeout(() => setShowSuccessToast(false), 3000);

        const newSlug = createDocumentSlug(newDraftNameValidation.sanitized);
        navigate(`/bep-generator/${newSlug}/step/${currentStep}`, { replace: true });
      } else {
        alert('Failed to overwrite draft');
      }
    } catch (error) {
      alert('Failed to overwrite draft: ' + error.message);
    }
  }, [newDraftNameValidation, formData, saveDraft, createDocumentSlug, navigate, currentStep, setCurrentDraft]);

  const handleSaveAsNewDraft = useCallback(() => {
    setExistingDraftToOverwrite(null);
    setNewDraftName('');
  }, []);

  const handleSaveDraftCancel = useCallback(() => {
    setShowSaveDraftDialog(false);
    setNewDraftName('');
    setExistingDraftToOverwrite(null);
  }, []);

  if (!bepType) {
    // Redirect to start menu if no BEP type
    navigate('/bep-generator');
    return null;
  }

  return (
    <div className="h-screen bg-gray-50 flex relative" data-page-uri={location.pathname}>
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
              {currentDraft && (
                <p className="text-xs text-blue-600 mt-1 font-medium flex items-center">
                  <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9 2a2 2 0 00-2 2v8a2 2 0 002 2h6a2 2 0 002-2V6.414A2 2 0 0016.414 5L14 2.586A2 2 0 0012.586 2H9z" />
                    <path d="M3 8a2 2 0 012-2v10h8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
                  </svg>
                  {currentDraft.name}
                </p>
              )}
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

          <button
            onClick={() => navigate('/bep-generator/drafts')}
            disabled={!user}
            className="w-full inline-flex items-center justify-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 hover:border-blue-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          >
            <FolderOpen className="w-4 h-4 mr-2" />
            Drafts
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          <ProgressSidebar
            steps={CONFIG.steps || []}
            currentStep={currentStep}
            completedSections={completedSections}
            onStepClick={(stepIndex) => {
              const docSlug = getDocumentSlug();
              setIsTransitioning(true);
              setTimeout(() => {
                navigate(`/bep-generator/${docSlug}/step/${stepIndex}`);
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

            <div className="flex items-center space-x-2">
              {/* Navigation arrows */}
              <button
                onClick={handlePrevious}
                disabled={currentStep === 0}
                className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 hover:border-blue-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
              >
                <ChevronLeft className="w-4 h-4 mr-1" />
                <span className="hidden xl:inline">Previous</span>
              </button>

              <button
                onClick={handleNext}
                className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 hover:border-blue-300 transition-all duration-200"
              >
                <span className="hidden xl:inline">{currentStep >= (CONFIG.steps?.length || 0) - 1 ? 'Preview' : 'Next'}</span>
                <ChevronRight className="w-4 h-4 xl:ml-1" />
              </button>

              {/* Separator */}
              <div className="hidden lg:block w-px h-8 bg-gray-300 mx-1"></div>

              <button
                onClick={handleSaveDraft}
                disabled={savingDraft || !user}
                className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-green-50 hover:border-green-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
                title="Save Draft"
              >
                <Save className="w-4 h-4" />
                <span className="hidden lg:inline ml-2">{savingDraft ? 'Saving...' : 'Save'}</span>
              </button>

              <button
                onClick={handlePreview}
                className="inline-flex items-center px-3 py-2 border border-transparent shadow-sm text-sm font-medium rounded-lg text-white bg-blue-600 hover:bg-blue-700 hover:shadow-md transition-all duration-200"
                title="Preview BEP"
              >
                <Eye className="w-4 h-4" />
                <span className="hidden lg:inline ml-2">Preview</span>
              </button>
            </div>
          </div>
        </div>

        {/* Form Content */}
        <div className="flex-1 overflow-y-auto bg-gray-50">
          <div className={`mx-auto px-6 py-8 ${currentStep === CONFIG.steps?.length - 1 ? 'max-w-[297mm]' : 'max-w-[210mm]'}`}>
            <div className={`bg-white rounded-xl shadow-sm border border-gray-200 p-8 transition-all duration-300 ease-in-out ${isTransitioning ? 'opacity-50 transform scale-95' : 'opacity-100 transform scale-100'}`}>
              <FormStepRHF
                stepIndex={currentStep}
                bepType={bepType}
              />
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
                className="inline-flex items-center px-6 py-3 border border-transparent shadow-sm text-sm font-medium rounded-lg text-white bg-blue-600 hover:bg-blue-700 hover:shadow-md transition-all duration-200 transform hover:scale-105"
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
        existingDraft={existingDraftToOverwrite}
        onOverwrite={handleOverwriteDraft}
        onSaveAsNew={handleSaveAsNewDraft}
      />

      {/* Hidden components for PDF screenshot capture */}
      <HiddenComponentsRenderer formData={formData} bepType={bepType} />
    </div>
  );
};

export default BepFormView;
