import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePage } from '../../contexts/PageContext';
import { ArrowLeft } from 'lucide-react';
import { useTidpData } from '../../hooks/useTidpData';
import { useMidpData } from '../../hooks/useMidpData';
import { useExport } from '../../hooks/useExport';
import { getDefaultContainer } from '../../utils/csvHelpers';
import Toast from '../common/Toast';
import TIDPImportDialog from '../TIDPImportDialog';
import MIDPEvolutionDashboard from '../MIDPEvolutionDashboard';
import TIDPList from '../tidp/TIDPList';
import TIDPForm from '../tidp/TIDPForm';
import TIDPDetailsPanel from '../tidp/TIDPDetailsPanel';
import TIDPDashboard from '../tidp/TIDPDashboard';
import MIDPList from '../midp/MIDPList';
import MIDPForm from '../midp/MIDPForm';

const TidpMidpManager = ({ onClose, initialShowTidpForm = false, initialShowMidpForm = false, initialTidpId = null }) => {
  const navigate = useNavigate();
  const { navigateTo } = usePage();
  const [activeTab, setActiveTab] = useState('dashboard');
  const [showTidpForm, setShowTidpForm] = useState(initialShowTidpForm);
  const [showMidpForm, setShowMidpForm] = useState(initialShowMidpForm);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [showEvolutionDashboard, setShowEvolutionDashboard] = useState(null);
  const [detailsItem, setDetailsItem] = useState(null);
  const [detailsForm, setDetailsForm] = useState({ taskTeam: '', description: '', containers: [] });
  const [toast, setToast] = useState({ open: false, message: '', type: 'info' });

  // Custom hooks
  const { tidps, loading: tidpLoading, loadTidps, createTidp, updateTidp, deleteTidp } = useTidpData();
  const { midps, loading: midpLoading, loadMidps, createMidp } = useMidpData();
  const {
    templates,
    selectedTemplate,
    setSelectedTemplate,
    exportLoading,
    bulkExportRunning,
    bulkProgress,
    exportTidpExcel,
    exportTidpPdf,
    exportMidpExcel,
    exportMidpPdf,
    exportAllTidpPdfs,
    exportConsolidatedProject
  } = useExport();

  // Form states
  const [tidpForm, setTidpForm] = useState({
    taskTeam: '',
    discipline: '',
    teamLeader: '',
    description: '',
    containers: [
      {
        ...getDefaultContainer(),
        'Information Container Name/Title': 'Initial deliverable',
        'Due Date': new Date().toISOString().slice(0, 10)
      }
    ]
  });

  const [midpForm, setMidpForm] = useState({ projectName: '', description: '' });

  useEffect(() => {
    const loadData = async () => {
      try {
        await Promise.all([loadTidps(), loadMidps()]);
        // If an initial TIDP id was provided (e.g. via /tidp-editor/:id), load it into the form
        if (initialTidpId) {
          try {
            const t = await (async () => {
              const ApiService = require('../../services/apiService').default || require('../../services/apiService');
              const resp = await ApiService.getTIDP(initialTidpId);
              return resp.data || resp;
            })();

            if (t) {
              // Populate tidpForm and detailsForm, and show the tidp form for editing
              setTidpForm((prev) => ({
                ...prev,
                taskTeam: t.teamName || t.taskTeam || prev.taskTeam,
                discipline: t.discipline || prev.discipline,
                teamLeader: t.leader || t.teamLeader || prev.teamLeader,
                description: t.responsibilities || t.description || prev.description,
                containers: t.containers && t.containers.length > 0 ? t.containers : prev.containers
              }));

              setDetailsItem({ type: 'tidp', data: t });
              setDetailsForm({ taskTeam: t.teamName || t.taskTeam, description: t.description || t.responsibilities, containers: t.containers || [] });
              setShowTidpForm(true);
            }
          } catch (err) {
            console.warn('Failed to load initial TIDP by id:', initialTidpId, err);
          }
        }
      } catch (error) {
        console.error('Failed to load initial data:', error);
        setToast({
          open: true,
          message: 'Failed to load data. Please ensure the backend server is running.',
          type: 'error'
        });
      }
    };
    loadData();
  }, [loadTidps, loadMidps, initialTidpId]);

  // TIDP handlers
  const handleCreateTidp = async () => {
    if (!tidpForm.taskTeam || tidpForm.taskTeam.trim().length < 2) {
      setToast({ open: true, message: 'Task team is required', type: 'error' });
      return;
    }
    if (!tidpForm.discipline) {
      setToast({ open: true, message: 'Discipline is required', type: 'error' });
      return;
    }
    if (!tidpForm.teamLeader || tidpForm.teamLeader.trim().length < 2) {
      setToast({ open: true, message: 'Team leader is required', type: 'error' });
      return;
    }

    try {
      const created = await createTidp(tidpForm);
      setToast({ open: true, message: 'TIDP created', type: 'success' });
      setShowTidpForm(false);
      resetTidpForm();

      if (created) {
        const t = created.data ? created.data : created;
        // Switch to the TIDPs tab and reload the list so the newly created item is visible
        try {
          setActiveTab('tidps');
          await loadTidps();
        } catch (loadErr) {
          console.warn('Failed to reload TIDPs after create:', loadErr);
        }

        // Show details for the newly created TIDP
        setDetailsItem({ type: 'tidp', data: t });
        setDetailsForm({
          taskTeam: t.taskTeam || '',
          description: t.description || '',
          containers: t.containers || []
        });
      }
    } catch (err) {
      console.error('Create TIDP failed', err);
      setToast({ open: true, message: err.message || 'Failed to create TIDP', type: 'error' });
    }
  };

  const handleUpdateTidp = async (id, update) => {
    try {
      await updateTidp(id, update);
      setToast({ open: true, message: 'TIDP updated', type: 'success' });
      setDetailsItem(null);
    } catch (err) {
      console.error('Update TIDP failed', err);
      setToast({ open: true, message: err.message || 'Failed to update TIDP', type: 'error' });
    }
  };

  const handleDeleteTidp = async (id) => {
    if (!window.confirm('Delete this TIDP?')) return;
    try {
      await deleteTidp(id);
      setToast({ open: true, message: 'TIDP deleted', type: 'success' });
      setDetailsItem(null);
    } catch (err) {
      console.error('Delete TIDP failed', err);
      setToast({ open: true, message: err.message || 'Failed to delete TIDP', type: 'error' });
    }
  };

  // MIDP handlers
  const handleCreateMidp = async () => {
    if (!midpForm.projectName || midpForm.projectName.trim().length < 2) {
      setToast({ open: true, message: 'Project name is required', type: 'error' });
      return;
    }
    try {
      const result = await createMidp(midpForm);
      const message = result.autoGenerated
        ? 'MIDP auto-generated from existing TIDPs'
        : 'MIDP created (add TIDPs to populate)';
      setToast({ open: true, message, type: 'success' });
      setShowMidpForm(false);
      setMidpForm({ projectName: '', description: '' });
    } catch (err) {
      console.error('Create MIDP failed', err);
      setToast({ open: true, message: err.message || 'Failed to create MIDP', type: 'error' });
    }
  };

  const handleImportComplete = async (importResults) => {
    setToast({
      open: true,
      message: `Imported ${importResults.successful.length} TIDPs successfully`,
      type: 'success'
    });
    setShowImportDialog(false);
    await Promise.all([loadTidps(), loadMidps()]);
  };

  const handleImportCsv = (containers) => {
    setTidpForm((prev) => ({
      ...prev,
      containers: containers
    }));
    setShowTidpForm(true);
  };

  // Export handlers
  const handleExportTidpPdf = async (id) => {
    try {
      await exportTidpPdf(id);
      setToast({ open: true, message: 'TIDP PDF export downloaded', type: 'success' });
    } catch (err) {
      setToast({ open: true, message: 'Failed to export TIDP to PDF: ' + (err.message || err), type: 'error' });
    }
  };

  const handleExportTidpExcel = async (id) => {
    try {
      await exportTidpExcel(id);
      setToast({ open: true, message: 'TIDP Excel export downloaded', type: 'success' });
    } catch (err) {
      setToast({ open: true, message: 'Failed to export TIDP to Excel: ' + (err.message || err), type: 'error' });
    }
  };

  const handleExportMidpPdf = async (id) => {
    try {
      await exportMidpPdf(id);
      setToast({ open: true, message: 'MIDP PDF export downloaded', type: 'success' });
    } catch (err) {
      setToast({ open: true, message: 'Failed to export MIDP to PDF: ' + (err.message || err), type: 'error' });
    }
  };

  const handleExportMidpExcel = async (id) => {
    try {
      await exportMidpExcel(id);
      setToast({ open: true, message: 'MIDP Excel export downloaded', type: 'success' });
    } catch (err) {
      setToast({ open: true, message: 'Failed to export MIDP to Excel: ' + (err.message || err), type: 'error' });
    }
  };

  const handleExportAllPdfs = async () => {
    try {
      const result = await exportAllTidpPdfs(tidps, 3);
      if (result && result.errors && result.errors.length > 0) {
        setToast({ open: true, message: `Completed with ${result.errors.length} failures`, type: 'error' });
      } else {
        setToast({ open: true, message: `All TIDP exports completed (${tidps.length})`, type: 'success' });
      }
    } catch (err) {
      setToast({ open: true, message: 'Some TIDP exports failed', type: 'error' });
    }
  };

  const resetTidpForm = () => {
    setTidpForm({
      taskTeam: '',
      discipline: '',
      teamLeader: '',
      description: '',
      containers: [
        {
          ...getDefaultContainer(),
          'Information Container Name/Title': 'Initial deliverable',
          'Due Date': new Date().toISOString().slice(0, 10)
        }
      ]
    });
  };

  const loading = tidpLoading || midpLoading;

  const handleCloseTidpForm = () => {
    // Always hide the form and reset local state
    setShowTidpForm(false);
    resetTidpForm();

    // If this manager was mounted via the /tidp-editor URL (direct open),
    // navigating back to a known page is preferable to leaving the app at a
    // stray URL like /tidp-editor (without id) or other unexpected path.
    const path = window.location.pathname || '';

    // If an external onClose handler was provided (parent embedded context), call it
    if (typeof onClose === 'function') {
      try { onClose(); } catch (e) { /* noop */ }
      return;
    }

    if (path.startsWith('/tidp-editor')) {
      try { navigateTo('tidp-midp'); } catch (e) { /* noop */ }
      try { navigate('/tidp-midp'); } catch (e) { /* noop */ }
    }
  };

  if (showTidpForm) {
    return (
      <TIDPForm
        tidpForm={tidpForm}
        onTidpFormChange={setTidpForm}
        onSubmit={handleCreateTidp}
        onCancel={handleCloseTidpForm}
      />
    );
  }

  if (showMidpForm) {
    return (
      <MIDPForm
        midpForm={midpForm}
        onMidpFormChange={setMidpForm}
        onSubmit={handleCreateMidp}
        onCancel={() => {
          setShowMidpForm(false);
          setMidpForm({ projectName: '', description: '' });
        }}
      />
    );
  }

  return (
    <>
      <div className="min-h-full">
        <div className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
            <div>
              {onClose ? (
                <>
                  <h1 className="text-2xl font-bold text-gray-900">TIDP/MIDP Manager</h1>
                  <p className="text-gray-600">Manage Task and Master Information Delivery Plans</p>
                </>
              ) : (
                <div className="flex items-center space-x-4">
                  <button
                    onClick={() => {
                      if (window.history.length > 1) {
                        window.history.back();
                        setTimeout(() => {
                          try { navigateTo('tidp-midp'); } catch (e) { /* noop */ }
                        }, 200);
                      } else {
                        // Fallback to explicit navigation
                        try { navigateTo('tidp-midp'); } catch (e) { /* noop */ }
                        try { navigate('/tidp-midp-dashboard'); } catch (e) { /* noop */ }
                      }
                    }}
                    className="flex items-center text-gray-600 hover:text-gray-900 transition-colors"
                  >
                    <ArrowLeft className="w-5 h-5 mr-2" />
                    Back to Dashboard
                  </button>
                  <div>
                    <h1 className="text-2xl font-bold text-gray-900">TIDP Editor</h1>
                    <p className="text-gray-600">Create and edit Task Information Delivery Plans</p>
                  </div>
                </div>
              )}
            </div>
            {onClose && (
              <div className="flex items-center space-x-3">
                <button onClick={onClose} className="text-sm text-gray-600 hover:text-gray-800 underline">
                  Back to BEP
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="mb-6">
            <nav className="flex space-x-8">
              {[
                { id: 'dashboard', label: 'Dashboard' },
                { id: 'tidps', label: 'TIDPs' },
                { id: 'midps', label: 'MIDP' }
              ].map(({ id, label }) => (
                <button
                  key={id}
                  onClick={() => setActiveTab(id)}
                  className={`py-2 px-1 border-b-2 font-medium text-sm ${
                    activeTab === id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {label}
                </button>
              ))}
            </nav>
          </div>

          {loading ? (
            <div className="text-center py-12">
              <p className="text-gray-500">Loading...</p>
            </div>
          ) : (
            <>
              {activeTab === 'dashboard' && (
                <TIDPDashboard
                  tidps={tidps}
                  midps={midps}
                  onShowTidpForm={() => setShowTidpForm(true)}
                  onShowMidpForm={() => setShowMidpForm(true)}
                  onImportCsv={handleImportCsv}
                  onToast={setToast}
                />
              )}
              {activeTab === 'tidps' && (
                <TIDPList
                  tidps={tidps}
                  templates={templates}
                  selectedTemplate={selectedTemplate}
                  onTemplateChange={setSelectedTemplate}
                  exportLoading={exportLoading}
                  onExportPdf={handleExportTidpPdf}
                  onExportExcel={handleExportTidpExcel}
                  onViewDetails={(tidp) => {
                    setDetailsItem({ type: 'tidp', data: tidp });
                    setDetailsForm({
                      taskTeam: tidp.taskTeam || '',
                      description: tidp.description || '',
                      containers: tidp.containers || []
                    });
                  }}
                  onShowTidpForm={() => setShowTidpForm(true)}
                  onShowImportDialog={() => setShowImportDialog(true)}
                  onImportCsv={handleImportCsv}
                  onExportAllPdfs={handleExportAllPdfs}
                  onExportConsolidated={exportConsolidatedProject}
                  bulkExportRunning={bulkExportRunning}
                  bulkProgress={bulkProgress}
                  midps={midps}
                  onToast={setToast}
                />
              )}
              {activeTab === 'midps' && (
                <MIDPList
                  midps={midps}
                  templates={templates}
                  selectedTemplate={selectedTemplate}
                  onTemplateChange={setSelectedTemplate}
                  exportLoading={exportLoading}
                  onExportPdf={handleExportMidpPdf}
                  onExportExcel={handleExportMidpExcel}
                  onViewDetails={(midp) => setDetailsItem({ type: 'midp', data: midp })}
                  onShowMidpForm={() => setShowMidpForm(true)}
                  onShowEvolutionDashboard={setShowEvolutionDashboard}
                />
              )}
            </>
          )}
        </div>

        {/* Details panel */}
        {detailsItem && detailsItem.type === 'tidp' && (
          <TIDPDetailsPanel
            tidp={detailsItem.data}
            detailsForm={detailsForm}
            onDetailsFormChange={setDetailsForm}
            onSave={handleUpdateTidp}
            onDelete={handleDeleteTidp}
            onClose={() => setDetailsItem(null)}
          />
        )}

        {/* MIDP Details panel (simple) */}
        {detailsItem && detailsItem.type === 'midp' && (
          <div className="fixed right-6 top-20 w-96 bg-white border rounded-lg shadow-lg p-4 z-50">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-semibold">MIDP Details</h4>
              <button type="button" onClick={() => setDetailsItem(null)} className="text-sm text-gray-600">
                Close
              </button>
            </div>
            <div className="text-sm text-gray-700">
              <h5 className="font-medium mb-2">{detailsItem.data.projectName}</h5>
              <p className="text-xs text-gray-600 mb-3">{detailsItem.data.description}</p>
              <div className="flex space-x-2">
                <button
                  type="button"
                  onClick={() => setDetailsItem(null)}
                  className="bg-gray-200 px-3 py-1 rounded text-sm"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Toast Notification */}
      <Toast
        open={toast.open}
        message={toast.message}
        type={toast.type}
        onClose={() => setToast((t) => ({ ...t, open: false }))}
      />

      {/* Import Dialog */}
      <TIDPImportDialog
        open={showImportDialog}
        onClose={() => setShowImportDialog(false)}
        onImportComplete={handleImportComplete}
      />

      {/* Evolution Dashboard */}
      {showEvolutionDashboard && (
        <MIDPEvolutionDashboard
          midpId={showEvolutionDashboard}
          onClose={() => setShowEvolutionDashboard(null)}
        />
      )}
    </>
  );
};

export default TidpMidpManager;