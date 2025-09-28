import React, { useState, useEffect, useRef } from 'react';
import { Plus, Calendar, Users, Download } from 'lucide-react';
import ApiService from '../../services/apiService';
import Toast from '../common/Toast';

const TidpMidpManager = ({ onClose }) => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [tidps, setTidps] = useState([]);
  const [midps, setMidps] = useState([]);
  const [loading, setLoading] = useState(false);
  const [exportLoading, setExportLoading] = useState({});
  const [detailsItem, setDetailsItem] = useState(null);
  const [detailsForm, setDetailsForm] = useState({ taskTeam: '', description: '', containers: [] });
  const [templates, setTemplates] = useState([]);
  const [selectedTemplate, setSelectedTemplate] = useState(null);
  const [showTidpForm, setShowTidpForm] = useState(false);
  const [showMidpForm, setShowMidpForm] = useState(false);
  // Toast state
  const [toast, setToast] = useState({ open: false, message: '', type: 'info' });

  // Forms state
  const [tidpForm, setTidpForm] = useState({ taskTeam: '', description: '' });
  const [midpForm, setMidpForm] = useState({ projectName: '', description: '' });

  // Bulk export state
  const [bulkExportRunning, setBulkExportRunning] = useState(false);
  const [bulkProgress, setBulkProgress] = useState({ done: 0, total: 0 });

  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    loadData();
    return () => { mountedRef.current = false; };
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [tidpData, midpData] = await Promise.all([
        ApiService.getAllTIDPs(),
        ApiService.getAllMIDPs()
      ]);
      if (!mountedRef.current) return;
      setTidps(tidpData.tidps || []);
      setMidps(midpData.midps || []);
    } catch (error) {
      if (mountedRef.current) console.error('Failed to load TIDP/MIDP data:', error);
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  };

  const loadTemplates = async () => {
    try {
      const resp = await ApiService.getExportTemplates();
      // Server currently returns { success: true, data: { tidp: {...}, midp: {...} } }
      // Normalize into an array of template descriptors: [{ id, name, ... }]
      if (resp && Array.isArray(resp)) {
        setTemplates(resp);
      } else if (resp && resp.templates && Array.isArray(resp.templates)) {
        setTemplates(resp.templates);
      } else if (resp && resp.data && typeof resp.data === 'object') {
        const normalized = Object.keys(resp.data).map(key => ({ id: key, name: key.toUpperCase(), ...resp.data[key] }));
        setTemplates(normalized);
      } else if (resp) {
        // Fallback: try to coerce to array if possible
        try {
          const coerced = Array.isArray(resp) ? resp : [resp];
          setTemplates(coerced);
        } catch (e) {
          setTemplates([]);
        }
      } else {
        setTemplates([]);
      }
    } catch (err) {
      console.warn('Failed to load export templates', err);
    }
  };

  useEffect(() => {
    loadTemplates();
  }, []);

  // Export / Preview handlers
  const exportTidpExcel = async (id) => {
    setExportLoading(prev => ({ ...prev, [id]: true }));
    try {
      await ApiService.exportTIDPToExcel(id, selectedTemplate);
      setToast({ open: true, message: 'TIDP Excel export downloaded', type: 'success' });
    } catch (err) {
      console.error(err);
      setToast({ open: true, message: 'Failed to export TIDP to Excel: ' + (err.message || err), type: 'error' });
    } finally {
      setExportLoading(prev => ({ ...prev, [id]: false }));
    }
  };

  const exportTidpPdf = async (id) => {
    return exportTidpPdfInternal(id, { silent: false });
  };

  const exportTidpPdfInternal = async (id, { silent = false } = {}) => {
    setExportLoading(prev => ({ ...prev, [id]: true }));
    try {
      await ApiService.exportTIDPToPDF(id, selectedTemplate);
      if (!silent) setToast({ open: true, message: 'TIDP PDF export downloaded', type: 'success' });
      return { success: true };
    } catch (err) {
      console.error(err);
      if (!silent) setToast({ open: true, message: 'Failed to export TIDP to PDF: ' + (err.message || err), type: 'error' });
      return { success: false, error: err };
    } finally {
      setExportLoading(prev => ({ ...prev, [id]: false }));
    }
  };

  

  const exportMidpExcel = async (id) => {
    setExportLoading(prev => ({ ...prev, [id]: true }));
    try {
      await ApiService.exportMIDPToExcel(id, selectedTemplate);
      setToast({ open: true, message: 'MIDP Excel export downloaded', type: 'success' });
    } catch (err) {
      console.error(err);
      setToast({ open: true, message: 'Failed to export MIDP to Excel: ' + (err.message || err), type: 'error' });
    } finally {
      setExportLoading(prev => ({ ...prev, [id]: false }));
    }
  };

  const exportMidpPdf = async (id) => {
    setExportLoading(prev => ({ ...prev, [id]: true }));
    try {
      await ApiService.exportMIDPToPDF(id, selectedTemplate);
      setToast({ open: true, message: 'MIDP PDF export downloaded', type: 'success' });
    } catch (err) {
      console.error(err);
      setToast({ open: true, message: 'Failed to export MIDP to PDF: ' + (err.message || err), type: 'error' });
    } finally {
      setExportLoading(prev => ({ ...prev, [id]: false }));
    }
  };

  // Bulk export helpers with limited concurrency
  const runConcurrent = async (items, worker, concurrency = 3, progressCb) => {
    let i = 0;
    let done = 0;
    const errors = [];
    const runners = Array.from({ length: Math.min(concurrency, items.length) }).map(async () => {
      while (i < items.length) {
        const idx = i++;
        try {
          await worker(items[idx]);
        } catch (e) {
          console.error('Worker error', e);
          errors.push({ item: items[idx], error: e });
        }
        done++;
        if (progressCb) progressCb(done, items.length);
      }
    });
    await Promise.all(runners);
    return { errors };
  };

  const exportAllTidpPdfs = async (concurrency = 3) => {
    if (bulkExportRunning) return;
    if (tidps.length === 0) {
      setToast({ open: true, message: 'No TIDPs to export', type: 'info' });
      return;
    }
    setBulkExportRunning(true);
    setBulkProgress({ done: 0, total: tidps.length });
    try {
      const result = await runConcurrent(tidps, async (t) => {
        // silent to avoid per-file toasts
        await exportTidpPdfInternal(t.id, { silent: true });
      }, concurrency, (done, total) => setBulkProgress({ done, total }));
      if (result && result.errors && result.errors.length > 0) {
        console.error('Some exports failed', result.errors);
        setToast({ open: true, message: `Completed with ${result.errors.length} failures`, type: 'error' });
      } else {
        setToast({ open: true, message: `All TIDP exports completed (${tidps.length})`, type: 'success' });
      }
    } catch (err) {
      console.error(err);
      setToast({ open: true, message: 'Some TIDP exports failed', type: 'error' });
    } finally {
      if (mountedRef.current) {
        setBulkExportRunning(false);
        setBulkProgress({ done: 0, total: 0 });
      }
    }
  };

  

  const TIDPList = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        {/* Template selector for exports (per-list level) */}
        <div className="mb-4 flex items-center space-x-3">
          <label className="text-sm">Export template:</label>
          <select value={selectedTemplate || ''} onChange={(e) => setSelectedTemplate(e.target.value || null)} className="border p-2 rounded">
            <option value="">Default</option>
            {templates.map((t) => (
              <option key={t.id || t.name} value={t.id || t.name}>{t.name || t.id}</option>
            ))}
          </select>
        </div>
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Task Information Delivery Plans</h2>
          {activeTab === 'dashboard' && (
            <div className="space-y-4">
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold mb-3">Bulk exports</h3>
                <div className="flex items-center space-x-3">
                  <button onClick={async () => { await exportAllTidpPdfs(3); }} disabled={bulkExportRunning} className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-60">
                    {bulkExportRunning ? `${bulkProgress.done}/${bulkProgress.total} exporting...` : 'Export all TIDP PDFs'}
                  </button>

                  <button onClick={async () => {
                    if (midps.length === 0) { setToast({ open: true, message: 'No MIDP available to consolidate', type: 'info' }); return; }
                    const projectId = 'project-1';
                    try {
                      await ApiService.exportConsolidatedProject(projectId, midps[0].id);
                      setToast({ open: true, message: 'Consolidated project export downloaded', type: 'success' });
                    } catch (err) {
                      console.error(err);
                      setToast({ open: true, message: 'Failed consolidated export: ' + (err.message || err), type: 'error' });
                    }
                  }} className="bg-purple-600 text-white px-4 py-2 rounded">Export consolidated project</button>
                </div>
              </div>
            </div>
          )}
          <button
            onClick={() => setShowTidpForm(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span>New TIDP</span>
          </button>
          {/* Details panel - moved outside of the create button so it renders correctly */}
          {detailsItem && (
            <div className="fixed right-6 top-20 w-96 bg-white border rounded-lg shadow-lg p-4 z-50">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold">{detailsItem.type === 'tidp' ? 'TIDP Details' : 'MIDP Details'}</h4>
                <button type="button" onClick={() => setDetailsItem(null)} className="text-sm text-gray-600">Close</button>
              </div>
              <div className="text-sm text-gray-700">
                {detailsItem.type === 'tidp' ? (
                  <div>
                    <label className="block text-xs font-medium mb-1">Task Team</label>
                    <input value={detailsForm.taskTeam} onChange={(e) => setDetailsForm((s) => ({ ...s, taskTeam: e.target.value }))} className="w-full p-2 border rounded mb-2 text-sm" />
                    <label className="block text-xs font-medium mb-1">Description</label>
                    <input value={detailsForm.description} onChange={(e) => setDetailsForm((s) => ({ ...s, description: e.target.value }))} className="w-full p-2 border rounded mb-3 text-sm" />

                    <div className="mb-3">
                      <label className="block text-xs font-medium mb-2">Deliverables</label>
                      <div className="space-y-2 max-h-48 overflow-auto">
                        {(detailsForm.containers || []).map((c, ci) => (
                          <div key={c.id || ci} className="flex items-center space-x-2">
                            <input value={c['Container Name'] || c.name || ''} onChange={(e) => {
                              const updated = [...(detailsForm.containers || [])];
                              updated[ci] = { ...updated[ci], 'Container Name': e.target.value };
                              setDetailsForm((s) => ({ ...s, containers: updated }));
                            }} className="flex-1 p-2 border rounded text-sm" placeholder="Container Name" />
                            <input value={c['Due Date'] || c.dueDate || ''} onChange={(e) => {
                              const updated = [...(detailsForm.containers || [])];
                              updated[ci] = { ...updated[ci], 'Due Date': e.target.value };
                              setDetailsForm((s) => ({ ...s, containers: updated }));
                            }} type="date" className="p-2 border rounded text-sm" />
                            <button type="button" onClick={() => {
                              const updated = (detailsForm.containers || []).filter((_, idx) => idx !== ci);
                              setDetailsForm((s) => ({ ...s, containers: updated }));
                            }} className="text-red-600 text-sm">Remove</button>
                          </div>
                        ))}
                      </div>
                      <div className="mt-2">
                        <button type="button" onClick={() => {
                          const updated = [...(detailsForm.containers || []), { id: `c-${Date.now()}`, 'Container Name': '', 'Due Date': '' }];
                          setDetailsForm((s) => ({ ...s, containers: updated }));
                        }} className="bg-gray-100 px-2 py-1 rounded text-sm">Add deliverable</button>
                      </div>
                    </div>

                    <div className="flex space-x-2">
                      <button type="button" onClick={() => {
                        const updated = {
                          taskTeam: detailsForm.taskTeam,
                          description: detailsForm.description,
                          containers: detailsForm.containers
                        };
                        handleUpdateTidp(detailsItem.data.id, updated);
                      }} className="bg-blue-600 text-white px-3 py-1 rounded text-sm">Save</button>
                      <button type="button" onClick={() => handleDeleteTidp(detailsItem.data.id)} className="bg-red-600 text-white px-3 py-1 rounded text-sm">Delete</button>
                    </div>
                  </div>
                ) : (
                  <div>
                    <h5 className="font-medium mb-2">{detailsItem.data.projectName}</h5>
                    <p className="text-xs text-gray-600 mb-3">{detailsItem.data.description}</p>
                    <div className="flex space-x-2">
                      <button type="button" onClick={() => setDetailsItem(null)} className="bg-gray-200 px-3 py-1 rounded text-sm">Close</button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {tidps.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <Users className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <p className="text-lg">No TIDPs created yet</p>
            <p className="text-sm">Create your first Task Information Delivery Plan</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {tidps.map((tidp, index) => (
              <div key={tidp.id || index} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <h3 className="font-semibold text-gray-900 mb-2">{tidp.taskTeam || `TIDP ${index + 1}`}</h3>
                <p className="text-gray-600 text-sm mb-3">{tidp.description || tidp.discipline || 'Task information delivery plan'}</p>
                <div className="flex space-x-2">
                  <button onClick={() => { setDetailsItem({ type: 'tidp', data: tidp }); setDetailsForm({ taskTeam: tidp.taskTeam || '', description: tidp.description || '', containers: tidp.containers || [] }); }} className="flex-1 bg-blue-50 text-blue-700 py-2 px-3 rounded text-sm hover:bg-blue-100 transition-colors">View</button>
                  <button disabled={!!exportLoading[tidp.id]} onClick={() => exportTidpPdf(tidp.id)} className="bg-gray-50 text-gray-700 py-2 px-3 rounded text-sm hover:bg-gray-100 transition-colors">
                    {exportLoading[tidp.id] ? '...' : <Download className="w-4 h-4" />}
                  </button>
                  <button disabled={!!exportLoading[tidp.id]} onClick={() => exportTidpExcel(tidp.id)} className="bg-gray-50 text-gray-700 py-2 px-3 rounded text-sm hover:bg-gray-100 transition-colors">{exportLoading[tidp.id] ? '...' : 'XLSX'}</button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  const MIDPList = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="mb-4 flex items-center space-x-3">
          <label className="text-sm">Export template:</label>
          <select value={selectedTemplate || ''} onChange={(e) => setSelectedTemplate(e.target.value || null)} className="border p-2 rounded">
            <option value="">Default</option>
            {templates.map((t) => (
              <option key={t.id || t.name} value={t.id || t.name}>{t.name || t.id}</option>
            ))}
          </select>
        </div>
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Master Information Delivery Plans</h2>
          <button
            onClick={() => setShowMidpForm(true)}
            className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span>New MIDP</span>
          </button>
        </div>

        {midps.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <Calendar className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <p className="text-lg">No MIDPs created yet</p>
            <p className="text-sm">Create your first Master Information Delivery Plan</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {midps.map((midp, index) => (
              <div key={midp.id || index} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <h3 className="font-semibold text-gray-900 mb-2">{midp.projectName || `MIDP ${index + 1}`}</h3>
                <p className="text-gray-600 text-sm mb-3">{midp.description || 'Master information delivery plan'}</p>
                <div className="flex space-x-2">
                  <button onClick={() => setDetailsItem({ type: 'midp', data: midp })} className="flex-1 bg-green-50 text-green-700 py-2 px-3 rounded text-sm hover:bg-green-100 transition-colors">View</button>
                  <button disabled={!!exportLoading[midp.id]} onClick={() => exportMidpPdf(midp.id)} className="bg-gray-50 text-gray-700 py-2 px-3 rounded text-sm hover:bg-gray-100 transition-colors">
                    {exportLoading[midp.id] ? '...' : <Download className="w-4 h-4" />}
                  </button>
                  <button disabled={!!exportLoading[midp.id]} onClick={() => exportMidpExcel(midp.id)} className="bg-gray-50 text-gray-700 py-2 px-3 rounded text-sm hover:bg-gray-100 transition-colors">{exportLoading[midp.id] ? '...' : 'XLSX'}</button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  const TIDPForm = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Create New TIDP</h2>
          <button type="button" onClick={() => setShowTidpForm(false)} className="text-gray-500 hover:text-gray-700">✕</button>
        </div>
        <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); handleCreateTidp(); }}>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Task Team</label>
            <input value={tidpForm.taskTeam} onChange={(e) => setTidpForm((s) => ({ ...s, taskTeam: e.target.value }))} type="text" className="w-full p-3 border border-gray-300 rounded-lg" placeholder="Architecture Team" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Description</label>
            <input value={tidpForm.description} onChange={(e) => setTidpForm((s) => ({ ...s, description: e.target.value }))} type="text" className="w-full p-3 border border-gray-300 rounded-lg" placeholder="Brief description" />
          </div>
          <div className="flex space-x-4">
            <button type="submit" className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg">Create TIDP</button>
            <button type="button" onClick={() => { setShowTidpForm(false); setTidpForm({ taskTeam: '', description: '' }); }} className="bg-gray-300 hover:bg-gray-400 text-gray-700 px-6 py-2 rounded-lg">Cancel</button>
          </div>
        </form>
      </div>
    </div>
  );

  const MIDPForm = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Create New MIDP</h2>
          <button type="button" onClick={() => setShowMidpForm(false)} className="text-gray-500 hover:text-gray-700">✕</button>
        </div>
        <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); handleCreateMidp(); }}>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Project Name</label>
            <input value={midpForm.projectName} onChange={(e) => setMidpForm((s) => ({ ...s, projectName: e.target.value }))} type="text" className="w-full p-3 border border-gray-300 rounded-lg" placeholder="Office Complex Project" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Description</label>
            <input value={midpForm.description} onChange={(e) => setMidpForm((s) => ({ ...s, description: e.target.value }))} type="text" className="w-full p-3 border border-gray-300 rounded-lg" placeholder="Brief description" />
          </div>
          <div className="flex space-x-4">
            <button type="submit" className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg">Create MIDP</button>
            <button type="button" onClick={() => { setShowMidpForm(false); setMidpForm({ projectName: '', description: '' }); }} className="bg-gray-300 hover:bg-gray-400 text-gray-700 px-6 py-2 rounded-lg">Cancel</button>
          </div>
        </form>
      </div>
    </div>
  );

  // Create handlers
  const handleCreateTidp = async () => {
    // simple validation
    if (!tidpForm.taskTeam || tidpForm.taskTeam.trim().length < 2) {
      setToast({ open: true, message: 'Task team is required', type: 'error' });
      return;
    }
    try {
      // The server validator expects specific fields: teamName, discipline, leader, company, responsibilities, containers
      // Provide minimal defaults so the TIDP can be created and the user can edit details in the UI.
      const payload = {
        teamName: tidpForm.taskTeam,
        discipline: 'Architecture',
        leader: 'TBD',
        company: 'TBD',
        responsibilities: tidpForm.description || 'TBD',
        description: tidpForm.description,
        containers: [
          {
            id: `c-${Date.now()}`,
            'Container Name': 'Initial deliverable',
            'Type': 'Model',
            'Format': 'IFC',
            'LOI Level': 'LOD 300',
            'Author': 'TBD',
            'Dependencies': [],
            'Est. Time': '1 day',
            'Milestone': 'Initial',
            'Due Date': new Date().toISOString().slice(0,10),
            'Status': 'Planned'
          }
        ]
      };
      const created = await ApiService.createTIDP(payload);
      setToast({ open: true, message: 'TIDP created', type: 'success' });
      setShowTidpForm(false);
      setTidpForm({ taskTeam: '', description: '' });
      // reload list and open details editor for the new TIDP (server returns { success: true, data: tidp })
      await loadData();
      const createdTidp = (created && (created.data || created.tidp)) || created;
      if (createdTidp) {
        // if the wrapper exists (data), unwrap
        const t = createdTidp.data ? createdTidp.data : createdTidp;
        setDetailsItem({ type: 'tidp', data: t });
        setDetailsForm({ taskTeam: t.taskTeam || '', description: t.description || '', containers: t.containers || [] });
      }
    } catch (err) {
      console.error('Create TIDP failed', err);
      setToast({ open: true, message: err.message || 'Failed to create TIDP', type: 'error' });
    }
  };

  const handleCreateMidp = async () => {
    if (!midpForm.projectName || midpForm.projectName.trim().length < 2) {
      setToast({ open: true, message: 'Project name is required', type: 'error' });
      return;
    }
    try {
  // There is no direct createMIDP; we'll use createMIDPFromTIDPs with empty tidpIds
  await ApiService.createMIDPFromTIDPs({ projectName: midpForm.projectName, description: midpForm.description }, []);
      setToast({ open: true, message: 'MIDP created', type: 'success' });
      setShowMidpForm(false);
      setMidpForm({ projectName: '', description: '' });
      await loadData();
    } catch (err) {
      console.error('Create MIDP failed', err);
      setToast({ open: true, message: err.message || 'Failed to create MIDP', type: 'error' });
    }
  };

  // Details edit/update/delete
  const handleUpdateTidp = async (id, update) => {
    try {
      await ApiService.updateTIDP(id, update);
      setToast({ open: true, message: 'TIDP updated', type: 'success' });
      setDetailsItem(null);
      await loadData();
    } catch (err) {
      console.error('Update TIDP failed', err);
      setToast({ open: true, message: err.message || 'Failed to update TIDP', type: 'error' });
    }
  };

  const handleDeleteTidp = async (id) => {
    if (!window.confirm('Delete this TIDP?')) return;
    try {
      await ApiService.deleteTIDP(id);
      setToast({ open: true, message: 'TIDP deleted', type: 'success' });
      setDetailsItem(null);
      await loadData();
    } catch (err) {
      console.error('Delete TIDP failed', err);
      setToast({ open: true, message: err.message || 'Failed to delete TIDP', type: 'error' });
    }
  };

  if (showTidpForm) return <TIDPForm />;
  if (showMidpForm) return <MIDPForm />;

  return (
    <>
    <div className="min-h-full">
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
      <h1 className="text-2xl font-bold text-gray-900">TIDP/MIDP Manager</h1>
      <p className="text-gray-600">Manage Task and Master Information Delivery Plans</p>

  
          </div>
          <div className="flex items-center space-x-3">
            <button onClick={onClose} className="text-sm text-gray-600 hover:text-gray-800 underline">Back to BEP</button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="mb-6">
          <nav className="flex space-x-8">
            {[
              { id: 'dashboard', label: 'Dashboard' },
              { id: 'tidps', label: 'TIDPs' },
              { id: 'midps', label: 'MIDPs' }
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
              <div className="space-y-6">
                <div className="bg-white rounded-lg shadow p-6">
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">TIDP/MIDP Dashboard</h2>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    <div className="bg-blue-50 rounded-lg p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-lg font-semibold text-blue-900">TIDPs</h3>
                          <p className="text-3xl font-bold text-blue-600">{tidps.length}</p>
                        </div>
                        <Users className="w-8 h-8 text-blue-600" />
                      </div>
                      <p className="text-blue-700 text-sm mt-2">Task Information Delivery Plans</p>
                    </div>

                    <div className="bg-green-50 rounded-lg p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-lg font-semibold text-green-900">MIDPs</h3>
                          <p className="text-3xl font-bold text-green-600">{midps.length}</p>
                        </div>
                        <Calendar className="w-8 h-8 text-green-600" />
                      </div>
                      <p className="text-green-700 text-sm mt-2">Master Information Delivery Plans</p>
                    </div>

                    <div className="bg-purple-50 rounded-lg p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-lg font-semibold text-purple-900">Projects</h3>
                          <p className="text-3xl font-bold text-purple-600">1</p>
                        </div>
                        <Calendar className="w-8 h-8 text-purple-600" />
                      </div>
                      <p className="text-purple-700 text-sm mt-2">Active Projects</p>
                    </div>
                  </div>

                  <div className="flex space-x-4">
                    <button onClick={() => setShowTidpForm(true)} className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2">
                      <Plus className="w-4 h-4" />
                      <span>Create New TIDP</span>
                    </button>
                    <button onClick={() => setShowMidpForm(true)} className="bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2">
                      <Plus className="w-4 h-4" />
                      <span>Create New MIDP</span>
                    </button>
                  </div>
                </div>
              </div>
            )}
            {activeTab === 'tidps' && <TIDPList />}
            {activeTab === 'midps' && <MIDPList />}
          </>
        )}
      </div>
    </div>
    {/* Toast Notification */}
    <Toast open={toast.open} message={toast.message} type={toast.type} onClose={() => setToast((t) => ({ ...t, open: false }))} />
    </>
  );
};

export default TidpMidpManager;
