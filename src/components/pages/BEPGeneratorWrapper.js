import React, { useState, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { ChevronRight, ChevronLeft, Eye, Zap, FolderOpen, Save, ExternalLink } from 'lucide-react';

// Import all the existing BEP components
import ProgressSidebar from '../forms/controls/ProgressSidebar';
import CONFIG from '../../config/bepConfig';
import { getEmptyBepData, getTemplateById } from '../../data/templateRegistry';
import FormStep from '../steps/FormStep';
import PreviewExportPage from './PreviewExportPage';
import BepTypeSelector from './bep/BepTypeSelector';
import BepStartMenu from './bep/BepStartMenu';
import ImportBepDialog from './bep/ImportBepDialog';
import TemplateGallery from './bep/TemplateGallery';
import DraftManager from './drafts/DraftManager';
import SaveDraftDialog from './drafts/SaveDraftDialog';
import { generateBEPContent } from '../../services/bepFormatter';
import { generatePDF } from '../../services/pdfGenerator';
import { generateDocx } from '../../services/docxGenerator';
import { useTidpData } from '../../hooks/useTidpData';
import { useMidpData } from '../../hooks/useMidpData';
import { useDraftOperations } from '../../hooks/useDraftOperations';
import { useAuth } from '../../contexts/AuthContext';
import { validateDraftName } from '../../utils/validationUtils';

const BEPGeneratorWrapper = () => {
  const { user, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // Extract URL parameters
  const pathParts = location.pathname.split('/').filter(Boolean);
  const documentId = pathParts[1] && !['select-type', 'templates', 'drafts', 'import'].includes(pathParts[1]) ? decodeURIComponent(pathParts[1]) : null;
  const stepFromUrl = pathParts[3] ? parseInt(pathParts[3], 10) : null;

  // Determine current view from URL path
  const currentPath = location.pathname;
  const isSelectTypePage = currentPath.includes('/select-type');
  const isFormPage = documentId && stepFromUrl !== null;
  const isTemplatesPage = currentPath.includes('/templates');
  const isDraftsPage = currentPath.includes('/drafts');
  const isImportPage = currentPath.includes('/import');
  const isStartMenu = !isSelectTypePage && !isFormPage && !isTemplatesPage && !isDraftsPage && !isImportPage;

  // Derive currentStep from URL (single source of truth)
  const currentStep = isFormPage ? stepFromUrl : 0;

  // State variables
  const [formData, setFormData] = useState(getEmptyBepData());
  const [bepType, setBepType] = useState('');
  const [validationErrors, setValidationErrors] = useState({});
  const [completedSections, setCompletedSections] = useState(new Set());
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [newDraftName, setNewDraftName] = useState('');
  const [showSaveDraftDialog, setShowSaveDraftDialog] = useState(false);
  const [showSuccessToast, setShowSuccessToast] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [exportFormat, setExportFormat] = useState('pdf');
  const [existingDraftToOverwrite, setExistingDraftToOverwrite] = useState(null);
  // Track current draft being edited (like Word tracks the current document)
  const [currentDraft, setCurrentDraft] = useState(null); // { id, name }

  // Check if we should show TIDP creation form
  // const shouldShowTidpForm = searchParams.get('createTidp') === 'true';

  // TIDP and MIDP data
  const { tidps } = useTidpData();
  const { midps } = useMidpData();

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
    return 'new-document';
  }, [currentDraft, createDocumentSlug]);

  // Draft operations
  const { saveDraft, isLoading: savingDraft, importBepFromJson } = useDraftOperations(user, formData, bepType, (loadedData, loadedType, draftInfo) => {
    setFormData(loadedData);
    setBepType(loadedType);
    // Set current draft when loading from draft manager
    if (draftInfo) {
      setCurrentDraft({ id: draftInfo.id, name: draftInfo.name });
      const slug = createDocumentSlug(draftInfo.name);
      navigate(`/bep-generator/${slug}/step/0`);
    } else {
      navigate('/bep-generator/new-document/step/0');
    }
  }, () => {
    // Import dialog close callback (not used with route-based navigation)
  });

  // Sync documentId from URL on mount or when URL changes
  React.useEffect(() => {
    if (documentId && documentId !== 'new-document') {
      // Try to find matching draft by slug
      const decodedName = decodeURIComponent(documentId).replace(/-/g, ' ');
      console.log('Document ID from URL:', documentId, 'decoded:', decodedName);
      // If we have a matching draft in state, ensure it's set
      if (currentDraft && createDocumentSlug(currentDraft.name) !== documentId) {
        // URL doesn't match current draft, might need to reload
        console.log('URL document ID mismatch with current draft');
      }
    }
  }, [documentId, currentDraft, createDocumentSlug, bepType]);

  // Save BEP state to sessionStorage whenever it changes
  React.useEffect(() => {
    if (formData && bepType) {
      try {
        sessionStorage.setItem('bep-temp-state', JSON.stringify({
          formData,
          bepType,
          completedSections: Array.from(completedSections),
          currentDraft,
          timestamp: Date.now()
        }));
      } catch (error) {
        console.error('Failed to save BEP state:', error);
      }
    }
  }, [formData, bepType, completedSections, currentDraft]);

  // Restore BEP state from sessionStorage on mount
  React.useEffect(() => {
    try {
      const savedState = sessionStorage.getItem('bep-temp-state');
      if (savedState) {
        const {
          formData: savedFormData,
          bepType: savedBepType,
          completedSections: savedCompleted,
          currentDraft: savedDraft,
          timestamp
        } = JSON.parse(savedState);

        // Only restore if saved within last hour and we don't already have data
        const oneHour = 60 * 60 * 1000;
        if (timestamp && (Date.now() - timestamp < oneHour) && !bepType && savedBepType) {
          // Check if we're on a specific document/step URL
          const isOnSpecificStep = location.pathname.match(/\/bep-generator\/[^/]+\/step\/\d+/);

          // Only auto-restore and navigate if we're on a specific step URL
          // If we're on the base route, let the user see the start menu
          if (isOnSpecificStep) {
            setFormData(savedFormData);
            setBepType(savedBepType);
            setCompletedSections(new Set(savedCompleted || []));
            if (savedDraft) {
              setCurrentDraft(savedDraft);
            }
          }
          // If on base /bep-generator route, don't auto-restore - let user choose
        }
      }
    } catch (error) {
      console.error('Failed to restore BEP state:', error);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync URL whenever bepType is present but URL doesn't match expected format
  React.useEffect(() => {
    // IMPORTANT: Only sync if we're showing the form interface
    // Don't sync if user is intentionally on start menu or other pages
    if (bepType && !isSelectTypePage && !isTemplatesPage && !isDraftsPage && !isImportPage) {
      // If we're on the start menu with a bepType, user intentionally clicked to go back
      // Don't auto-redirect them
      if (isStartMenu) {
        return;
      }

      const slug = getDocumentSlug();
      const expectedUrl = `/bep-generator/${slug}/step/${currentStep}`;

      // Only navigate if the current URL doesn't match the expected format
      if (location.pathname !== expectedUrl) {
        console.log('URL sync: current =', location.pathname, 'expected =', expectedUrl);
        navigate(expectedUrl, { replace: true });
      }
    }
  }, [bepType, currentStep, location.pathname, isSelectTypePage, isTemplatesPage, isDraftsPage, isImportPage, isStartMenu, getDocumentSlug, navigate]);

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
    navigate('/tidp-midp');
  };

  const goHome = () => {
    navigate('/home');
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

  const handleNext = () => {
    const totalSteps = CONFIG.steps?.length || 0;
    const isLastStep = currentStep === totalSteps - 1;

    // For the last step (Appendices), allow preview even with validation errors
    if (!isLastStep) {
      const errors = validateStep(currentStep, formData);
      if (Object.keys(errors).length > 0) {
        setValidationErrors(errors);
        return;
      }
    }

    setValidationErrors({});

    // Mark current step as completed
    setCompletedSections(prev => new Set(prev).add(currentStep));

    if (isLastStep) {
      // Last step reached, go to preview
      setShowPreview(true);
    } else {
      // Move to next step
      const nextStep = currentStep + 1;
      const slug = getDocumentSlug();
      setIsTransitioning(true);

      // Use requestAnimationFrame to ensure smooth transition
      requestAnimationFrame(() => {
        navigate(`/bep-generator/${slug}/step/${nextStep}`);
        setTimeout(() => setIsTransitioning(false), 150);
      });
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      const prevStep = currentStep - 1;
      const slug = getDocumentSlug();
      setIsTransitioning(true);
      setValidationErrors({});

      // Use requestAnimationFrame to ensure smooth transition
      requestAnimationFrame(() => {
        navigate(`/bep-generator/${slug}/step/${prevStep}`);
        setTimeout(() => setIsTransitioning(false), 150);
      });
    }
    // At step 0, button will be disabled - use "Change Type" button to go back
  };

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
    // Use empty data for new BEPs (not pre-filled template data)
    setBepType(selectedType);
    setFormData(getEmptyBepData()); // Get fresh empty data
    setValidationErrors({});
    setShowPreview(false);
    setCompletedSections(new Set());
    setCurrentDraft(null); // Reset draft tracking for new BEP
    // Navigate to form page with step 0 (URL is the source of truth for currentStep)
    navigate('/bep-generator/new-document/step/0');
  }, [navigate]);

  // Start Menu Handlers
  const handleNewBep = useCallback(() => {
    navigate('/bep-generator/select-type');
  }, [navigate]);

  const handleLoadTemplate = useCallback(() => {
    navigate('/bep-generator/templates');
  }, [navigate]);

  const handleContinueDraft = useCallback(() => {
    navigate('/bep-generator/drafts');
  }, [navigate]);

  const handleImportBep = useCallback(() => {
    navigate('/bep-generator/import');
  }, [navigate]);

  const handleSelectTemplate = useCallback((template) => {
    console.log('Loading template:', template);

    // Load template data from registry
    const templateData = getTemplateById(template.id);

    if (templateData) {
      setFormData(templateData);
      setBepType(template.bepType);
      setValidationErrors({});
      setCompletedSections(new Set());
      setCurrentDraft(null); // Reset draft tracking for template
      // Create slug from template name
      const slug = createDocumentSlug(template.name || 'template');
      // Navigate to form with template loaded (URL is the source of truth for currentStep)
      navigate(`/bep-generator/${slug}/step/0`);
    } else {
      console.error('Template not found:', template.id);
      alert('Failed to load template. Please try again.');
    }
  }, [navigate, createDocumentSlug]);

  const handleCloseTemplateGallery = useCallback(() => {
    navigate('/bep-generator');
  }, [navigate]);

  const handleCloseDraftManager = useCallback(() => {
    navigate('/bep-generator');
  }, [navigate]);

  const handleCloseImportDialog = useCallback(() => {
    navigate('/bep-generator');
  }, [navigate]);

  const handleImportFile = useCallback(async (file) => {
    try {
      await importBepFromJson(file);
      // Navigate to form after import (importBepFromJson callback will handle the navigation)
    } catch (error) {
      console.error('Import failed:', error);
    }
  }, [importBepFromJson]);

  const handlePreview = useCallback(() => {
    const totalSteps = CONFIG.steps?.length || 0;
    const isLastStep = currentStep === totalSteps - 1;

    // Allow preview even with validation errors for the last step (Appendices)
    if (!isLastStep) {
      const errors = validateStep(currentStep, formData);
      if (Object.keys(errors).length > 0) {
        setValidationErrors(errors);
        alert('Please fix validation errors before previewing');
        return;
      }
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
        // Set as current draft after first save
        const draftInfo = { id: result.draftId, name: newDraftNameValidation.sanitized };
        setCurrentDraft(draftInfo);
        setShowSaveDraftDialog(false);
        setNewDraftName('');
        setExistingDraftToOverwrite(null);
        setShowSuccessToast(true);
        setTimeout(() => setShowSuccessToast(false), 3000);

        // Update URL to reflect the saved draft name
        const slug = createDocumentSlug(newDraftNameValidation.sanitized);
        navigate(`/bep-generator/${slug}/step/${currentStep}`, { replace: true });
      } else if (result.existingDraft) {
        // Draft already exists, show overwrite confirmation
        setExistingDraftToOverwrite(result.existingDraft);
      }
    } catch (error) {
      alert('Failed to save draft: ' + error.message);
    }
  }, [newDraftNameValidation, formData, saveDraft, createDocumentSlug, navigate, currentStep]);

  const handleOverwriteDraft = useCallback(async () => {
    if (!newDraftNameValidation.isValid) return;

    try {
      const result = await saveDraft(newDraftNameValidation.sanitized, formData, true);

      if (result.success) {
        // Set as current draft after overwrite
        const draftInfo = { id: result.draftId, name: newDraftNameValidation.sanitized };
        setCurrentDraft(draftInfo);
        setShowSaveDraftDialog(false);
        setNewDraftName('');
        setExistingDraftToOverwrite(null);
        setShowSuccessToast(true);
        setTimeout(() => setShowSuccessToast(false), 3000);

        // Update URL to reflect the saved draft name
        const slug = createDocumentSlug(newDraftNameValidation.sanitized);
        navigate(`/bep-generator/${slug}/step/${currentStep}`, { replace: true });
      } else {
        alert('Failed to overwrite draft');
      }
    } catch (error) {
      alert('Failed to overwrite draft: ' + error.message);
    }
  }, [newDraftNameValidation, formData, saveDraft, createDocumentSlug, navigate, currentStep]);

  const handleSaveAsNewDraft = useCallback(() => {
    // Reset to allow user to enter a new name
    setExistingDraftToOverwrite(null);
    setNewDraftName('');
  }, []);

  const handleSaveDraftCancel = useCallback(() => {
    setShowSaveDraftDialog(false);
    setNewDraftName('');
    setExistingDraftToOverwrite(null);
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

  // Show draft manager if on drafts route and user is logged in
  if (isDraftsPage && user) {
    return (
      <DraftManager
        user={user}
        currentFormData={formData}
        onLoadDraft={(loadedData, loadedType, draftInfo) => {
          setFormData(loadedData);
          setBepType(loadedType);
          setValidationErrors({});
          setCompletedSections(new Set());
          // Set current draft info when loading from draft manager
          if (draftInfo) {
            setCurrentDraft({ id: draftInfo.id, name: draftInfo.name });
            const slug = createDocumentSlug(draftInfo.name);
            navigate(`/bep-generator/${slug}/step/0`);
          } else {
            navigate('/bep-generator/new-document/step/0');
          }
        }}
        onClose={handleCloseDraftManager}
        bepType={bepType}
      />
    );
  }

  // Redirect to start menu if trying to access drafts without being logged in
  if (isDraftsPage && !user && !authLoading) {
    navigate('/bep-generator');
    return null;
  }

  // If no BEP type selected, show start menu or type selector
  if (!bepType) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50" data-page-uri={location.pathname}>
        {/* Header with navigation */}
        <div className="bg-white shadow-lg border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 py-2.5 lg:py-3 flex items-center justify-between">
            <div className="flex items-center space-x-2.5 lg:space-x-3">
              <div className="w-9 h-9 lg:w-10 lg:h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                <Zap className="w-4 h-4 lg:w-5 lg:h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl lg:text-2xl font-bold text-gray-900">BEP Generator</h1>
                <p className="text-sm lg:text-base text-gray-600">Create professional BIM Execution Plans</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 lg:space-x-3">
              <button
                onClick={goToTidpManager}
                className="inline-flex items-center px-2.5 lg:px-3 py-1.5 lg:py-2 text-sm lg:text-base text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors duration-200"
              >
                <ExternalLink className="w-3.5 h-3.5 lg:w-4 lg:h-4 mr-1.5 lg:mr-2" />
                TIDP/MIDP Manager
              </button>
              <button
                onClick={goHome}
                className="inline-flex items-center px-2.5 lg:px-3 py-1.5 lg:py-2 text-sm lg:text-base text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors duration-200"
              >
                <Zap className="w-3.5 h-3.5 lg:w-4 lg:h-4 mr-1.5 lg:mr-2" />
                Home
              </button>
            </div>
          </div>
        </div>

        {/* Show Start Menu, Type Selector, or Form based on current route */}
        {isStartMenu ? (
          authLoading ? (
            <div className="flex items-center justify-center min-h-screen">
              <div className="text-center">
                <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-gray-600">Loading...</p>
              </div>
            </div>
          ) : (
            <BepStartMenu
              onNewBep={handleNewBep}
              onLoadTemplate={handleLoadTemplate}
              onContinueDraft={handleContinueDraft}
              onImportBep={handleImportBep}
              user={user}
            />
          )
        ) : isSelectTypePage ? (
          <div className="max-w-6xl mx-auto px-4 py-4 lg:py-6">
            <div className="bg-transparent rounded-xl p-0">
              <BepTypeSelector
                bepType={bepType}
                setBepType={setBepType}
                onProceed={(selectedType) => handleTypeSelect(selectedType)}
              />
            </div>
          </div>
        ) : isTemplatesPage ? (
          <div className="max-w-6xl mx-auto px-4 py-4 lg:py-6">
            <TemplateGallery
              show={true}
              onSelectTemplate={handleSelectTemplate}
              onCancel={handleCloseTemplateGallery}
            />
          </div>
        ) : isImportPage ? (
          <div className="max-w-6xl mx-auto px-4 py-4 lg:py-6">
            <ImportBepDialog
              show={true}
              onImport={handleImportFile}
              onCancel={handleCloseImportDialog}
              isLoading={savingDraft}
            />
          </div>
        ) : null}
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

  // Main BEP Generator interface
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

          <div className="flex space-x-2">
            <button
              onClick={() => navigate('/bep-generator/drafts')}
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
              const slug = getDocumentSlug();
              setIsTransitioning(true);
              setTimeout(() => {
                navigate(`/bep-generator/${slug}/step/${stepIndex}`);
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
                className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 hover:border-blue-300 transition-all duration-200"
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
    </div>
  );
};

export default BEPGeneratorWrapper;