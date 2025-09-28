import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { ChevronRight, ChevronLeft, Eye, Zap, FolderOpen, Save } from 'lucide-react';
import DOMPurify from 'dompurify';

// Import separated components
import { AuthProvider, useAuth } from './contexts/AuthContext';
// Login/Register are intentionally not imported during development to avoid
// unused auth UI while authentication is temporarily bypassed.
import ProgressSidebar from './components/ui/ProgressSidebar';
import CONFIG from './config/bepConfig';
import INITIAL_DATA from './data/initialData';
import FormStep from './components/steps/FormStep';
import PreviewExportPage from './components/pages/PreviewExportPage';
import EnhancedBepTypeSelector from './components/pages/EnhancedBepTypeSelector';
import DraftManager from './components/pages/DraftManager';
import { generateBEPContent } from './services/bepFormatter';
import { generatePDF } from './services/pdfGenerator';
import { generateDocx } from './services/docxGenerator';



// Componenti riutilizzabili

const AppContent = () => {
  const { user, loading } = useAuth();
  // Authentication UI currently bypassed in development/demo mode

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Zap className="w-12 h-12 text-blue-600 animate-pulse mx-auto mb-4" />
          <p className="text-gray-600">Loading BEP Generator...</p>
        </div>
      </div>
    );
  }

  // Temporarily bypass authentication - re-enable the block below to restore
  // the login/register flow.

  // Use a mock user when authentication is disabled
  const mockUser = user || { id: 'demo-user', name: 'Demo User', email: 'demo@example.com' };
  return <ProfessionalBEPGenerator user={mockUser} />;
};

const ProfessionalBEPGenerator = ({ user }) => {
  const { logout } = useAuth();
  const [currentStep, setCurrentStep] = useState(0);
  const [bepType, setBepType] = useState('');
  const [formData, setFormData] = useState(INITIAL_DATA);
  const [completedSections, setCompletedSections] = useState(new Set());
  const [exportFormat, setExportFormat] = useState('html');
  const [errors, setErrors] = useState({});
  const [isExporting, setIsExporting] = useState(false);
  const [showBepTypeSelector, setShowBepTypeSelector] = useState(true);
  const [showDraftManager, setShowDraftManager] = useState(false);
  const [showSaveDraftDialog, setShowSaveDraftDialog] = useState(false);
  const [saveDraftName, setSaveDraftName] = useState('');

  useEffect(() => {
    const savedData = localStorage.getItem(`bepData_${user.id}`);
    if (savedData) {
      try {
        const parsedData = JSON.parse(savedData);
        // Merge saved data with INITIAL_DATA to ensure new fields have example values
        // Don't overwrite initial data arrays if saved data has empty arrays
        const mergedData = { ...INITIAL_DATA };
        Object.keys(parsedData).forEach(key => {
          const savedValue = parsedData[key];
          const initialValue = INITIAL_DATA[key];

          // If both are arrays and saved is empty but initial has data, keep initial
          if (Array.isArray(savedValue) && Array.isArray(initialValue) &&
              savedValue.length === 0 && initialValue.length > 0) {
            mergedData[key] = initialValue;
          } else {
            mergedData[key] = savedValue;
          }
        });
        setFormData(mergedData);
      } catch (error) {
        console.error('Error parsing saved data:', error);
        // If there's an error, use INITIAL_DATA
        setFormData(INITIAL_DATA);
      }
    }
  }, [user.id]);

  const debounce = (func, delay) => {
    let timeoutId;
    return (...args) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => func(...args), delay);
    };
  };

  useEffect(() => {
    const debouncedSave = debounce(() => {
      localStorage.setItem(`bepData_${user.id}`, JSON.stringify(formData));
    }, 500);
    debouncedSave();
  }, [formData, user.id]);

  const updateFormData = useCallback((field, value) => {
    const sanitizedValue = typeof value === 'string' ? DOMPurify.sanitize(value) : value;
    setFormData(prev => ({ ...prev, [field]: sanitizedValue }));
    const stepConfig = CONFIG.getFormFields(bepType, currentStep);
    const fieldConfig = stepConfig?.fields.find(f => f.name === field);
    if (fieldConfig) {
      const error = validateField(field, sanitizedValue, fieldConfig.required);
      setErrors(prev => ({ ...prev, [field]: error }));
    }
  }, [currentStep, bepType]);

  const validateField = (name, value, required) => {
    if (required && (!value || (Array.isArray(value) && value.length === 0) || (typeof value === 'string' && value.trim() === ''))) {
      return `${name.replace(/([A-Z])/g, ' $1').trim()} is required`;
    }
    return null;
  };

  const validateStep = useCallback((stepIndex) => {
    const stepConfig = CONFIG.getFormFields(bepType, stepIndex);
    if (!stepConfig) return true;

    return stepConfig.fields.every(field => {
      const value = formData[field.name];
      return !field.required || (value && (Array.isArray(value) ? value.length > 0 : value.trim() !== ''));
    });
  }, [formData, bepType]);

  const validatedSteps = useMemo(() => {
    return CONFIG.steps.map((_, index) => validateStep(index));
  }, [validateStep]);

  const validateCurrentStep = () => {
    const stepConfig = CONFIG.getFormFields(bepType, currentStep);
    if (!stepConfig) return true;

    const newErrors = {};
    let isValid = true;

    stepConfig.fields.forEach(field => {
      const error = validateField(field.name, formData[field.name], field.required);
      if (error) {
        newErrors[field.name] = error;
        isValid = false;
      }
    });

    setErrors(newErrors);
    return isValid;
  };

  const nextStep = () => {
    if (validateCurrentStep()) {
      setCompletedSections(prev => new Set([...prev, currentStep]));
      if (currentStep < CONFIG.steps.length - 1) {
        setCurrentStep(currentStep + 1);
      }
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const onStepClick = useCallback((index) => setCurrentStep(index), []);

  const goToPreview = () => {
    if (validateCurrentStep()) {
      setCompletedSections(prev => new Set([...prev, currentStep]));
      setCurrentStep(CONFIG.steps.length);
    }
  };



  const downloadBEP = async () => {
    setIsExporting(true);
  const content = generateBEPContent(formData, bepType);
    const currentDate = new Date().toISOString().split('T')[0];
    const fileName = `Professional_BEP_${formData.projectName || 'Project'}_${currentDate}`;

    try {
      if (exportFormat === 'html') {
        const blob = new Blob([content], { type: 'text/html;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${fileName}.html`;
        a.click();
        URL.revokeObjectURL(url);
      } else if (exportFormat === 'pdf') {
        const pdf = generatePDF(formData, bepType);
        pdf.save(`${fileName}.pdf`);
      } else if (exportFormat === 'word') {
        // Lazily load the docx generator and Packer so the heavy `docx`
        // dependency isn't included in the initial bundle.
        const { generateDocx } = await import('./services/docxGenerator');
        const { Packer } = await import('docx');
        const doc = await generateDocx(formData, bepType);
        const blob = await Packer.toBlob(doc);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${fileName}.docx`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Export error:', error);
    } finally {
      setIsExporting(false);
    }
  };

  const previewBEP = () => {
    const content = generateBEPContent(formData, bepType);
    const previewWindow = window.open('', '_blank', 'width=1200,height=800');
    previewWindow.document.write(content);
    previewWindow.document.close();
  };

  const handleBepTypeProceed = () => {
    setShowBepTypeSelector(false);
    setCurrentStep(0);
  };

  const handleLoadDraft = (draftData, draftBepType) => {
    setFormData(draftData);
    setBepType(draftBepType);
    setShowDraftManager(false);
    setShowBepTypeSelector(false);
    setCurrentStep(0);
    setCompletedSections(new Set());
  };

  const handleSaveDraft = () => {
    if (!bepType) {
      alert('Seleziona prima il tipo di BEP prima di salvare un draft.');
      return;
    }
    setShowSaveDraftDialog(true);
  };

  const saveDraft = (name) => {
    if (!name.trim()) return;

    const draftsKey = `bepDrafts_${user.id}`;
    const existingDrafts = JSON.parse(localStorage.getItem(draftsKey) || '{}');

    const draftId = Date.now().toString();
    const draft = {
      id: draftId,
      name: name.trim(),
      data: formData,
      bepType: bepType,
      lastModified: new Date().toISOString(),
      projectName: formData.projectName || 'Progetto senza nome'
    };

    existingDrafts[draftId] = draft;
    localStorage.setItem(draftsKey, JSON.stringify(existingDrafts));

    setSaveDraftName('');
    setShowSaveDraftDialog(false);

    // Show success message
    alert(`Draft "${name}" salvato con successo!`);
  };

  // Wrapper functions for the services
  const generateBEPContentWrapper = () => generateBEPContent(formData, bepType);


  // Show Draft Manager if requested
  if (showDraftManager) {
    return (
      <DraftManager
        user={user}
        currentFormData={formData}
        onLoadDraft={handleLoadDraft}
        onClose={() => setShowDraftManager(false)}
        bepType={bepType}
      />
    );
  }

  // Show BEP type selector if no type is selected
  if (showBepTypeSelector || !bepType) {
    return (
      <EnhancedBepTypeSelector
        bepType={bepType}
        setBepType={setBepType}
        onProceed={handleBepTypeProceed}
      />
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2">
                <Zap className="w-8 h-8 text-blue-600" />
                <h1 className="text-2xl font-bold text-gray-900">Professional BEP Generator</h1>
              </div>
              <span className="text-sm text-gray-500">ISO 19650-2 Compliant</span>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-600">
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                  {bepType === 'pre-appointment' ? 'Pre-Appointment BEP' : 'Post-Appointment BEP'}
                </span>
              </div>
              <div className="flex items-center space-x-3">
                <button
                  onClick={handleSaveDraft}
                  className="flex items-center space-x-2 bg-green-600 hover:bg-green-700 text-white px-3 py-2 rounded-lg transition-colors"
                  title="Salva Draft"
                >
                  <Save className="w-4 h-4" />
                  <span>Save Draft</span>
                </button>
                <button
                  onClick={() => setShowDraftManager(true)}
                  className="flex items-center space-x-2 bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-2 rounded-lg transition-colors"
                  title="Gestisci Draft"
                >
                  <FolderOpen className="w-4 h-4" />
                  <span>Draft</span>
                </button>
                <span className="text-sm text-gray-600">
                  Welcome, {user.name}
                </span>
                <button
                  onClick={logout}
                  className="text-sm text-gray-500 hover:text-gray-700 underline"
                >
                  Logout
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          <div className="lg:col-span-1">
            <ProgressSidebar
              steps={CONFIG.steps}
              currentStep={currentStep}
              completedSections={completedSections}
              onStepClick={onStepClick}
              validateStep={(index) => validatedSteps[index]}
            />
          </div>

          <div className="lg:col-span-3">
            <div className="bg-white rounded-lg shadow-sm p-8">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">{currentStep < CONFIG.steps.length ? CONFIG.steps[currentStep].title : 'Preview & Export'}</h2>
                  <p className="text-gray-600 mt-1">{currentStep < CONFIG.steps.length ? CONFIG.steps[currentStep].description : 'Preview and export the generated BEP'}</p>
                </div>
                {currentStep < CONFIG.steps.length && (
                  <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                    CONFIG.categories[CONFIG.steps[currentStep].category].bg
                  }`}>
                    {CONFIG.steps[currentStep].category} Aspects
                  </span>
                )}
              </div>

              {/* Top Navigation Bar */}
              <div className="flex justify-between items-center mb-6 pb-4 border-b bg-gray-50 rounded-lg p-4">
                <button
                  onClick={prevStep}
                  disabled={currentStep === 0}
                  className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeft className="w-4 h-4" />
                  <span>Previous</span>
                </button>

                <div className="text-sm text-gray-600 font-medium">
                  Step {currentStep + 1} of {CONFIG.steps.length + (currentStep >= CONFIG.steps.length ? 1 : 0)}
                </div>

                <div className="flex space-x-3">
                  {currentStep < CONFIG.steps.length - 1 ? (
                    <button
                      onClick={nextStep}
                      className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors"
                    >
                      <span>Next</span>
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  ) : currentStep === CONFIG.steps.length - 1 ? (
                    <button
                      onClick={goToPreview}
                      className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors"
                    >
                      <span>Preview & Export</span>
                      <Eye className="w-4 h-4" />
                    </button>
                  ) : null}
                </div>
              </div>

              {currentStep < CONFIG.steps.length ? (
                <FormStep
                  stepIndex={currentStep}
                  formData={formData}
                  updateFormData={updateFormData}
                  errors={errors}
                  bepType={bepType}
                />
              ) : (
                <PreviewExportPage
                  generateBEPContent={generateBEPContentWrapper}
                  exportFormat={exportFormat}
                  setExportFormat={setExportFormat}
                  previewBEP={previewBEP}
                  downloadBEP={downloadBEP}
                  isExporting={isExporting}
                />
              )}

              <div className="flex justify-between items-center mt-8 pt-6 border-t">
                <button
                  onClick={prevStep}
                  disabled={currentStep === 0}
                  className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <ChevronLeft className="w-4 h-4" />
                  <span>Previous</span>
                </button>

                <div className="text-sm text-gray-500">
                  Step {currentStep + 1} of {CONFIG.steps.length + (currentStep >= CONFIG.steps.length ? 1 : 0)}
                </div>

                <div className="flex space-x-3">
                  {currentStep < CONFIG.steps.length - 1 ? (
                    <button
                      onClick={nextStep}
                      className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors"
                    >
                      <span>Next</span>
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  ) : currentStep === CONFIG.steps.length - 1 ? (
                    <button
                      onClick={goToPreview}
                      className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors"
                    >
                      <span>Preview & Export</span>
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  ) : null}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Save Draft Dialog */}
      {showSaveDraftDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">Salva Draft</h3>
            <p className="text-gray-600 mb-4">Dai un nome al tuo draft per poterlo recuperare in seguito.</p>
            <input
              type="text"
              value={saveDraftName}
              onChange={(e) => setSaveDraftName(e.target.value)}
              placeholder="Nome del draft..."
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent mb-4"
              autoFocus
              onKeyPress={(e) => {
                if (e.key === 'Enter' && saveDraftName.trim()) {
                  saveDraft(saveDraftName);
                }
              }}
            />
            <div className="flex space-x-3">
              <button
                onClick={() => saveDraft(saveDraftName)}
                disabled={!saveDraftName.trim()}
                className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg transition-colors flex items-center justify-center space-x-2"
              >
                <Save className="w-4 h-4" />
                <span>Salva</span>
              </button>
              <button
                onClick={() => {
                  setShowSaveDraftDialog(false);
                  setSaveDraftName('');
                }}
                className="flex-1 bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors"
              >
                Annulla
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const App = () => {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
};

export default App;