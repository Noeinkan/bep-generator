import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Calendar, Users, Download, Upload, TrendingUp, ArrowLeft } from 'lucide-react';
import Papa from 'papaparse';
import ApiService from '../../services/apiService';
import Toast from '../common/Toast';
import TIDPImportDialog from '../TIDPImportDialog';
import MIDPEvolutionDashboard from '../MIDPEvolutionDashboard';

const TidpMidpManager = ({ onClose, initialShowTidpForm = false, initialShowMidpForm = false }) => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('dashboard');
  const [tidps, setTidps] = useState([]);
  const [midps, setMidps] = useState([]);
  const [loading, setLoading] = useState(false);
  const [exportLoading, setExportLoading] = useState({});
  const [detailsItem, setDetailsItem] = useState(null);
  const [detailsForm, setDetailsForm] = useState({ taskTeam: '', description: '', containers: [] });
  const [templates, setTemplates] = useState([]);
  const [selectedTemplate, setSelectedTemplate] = useState(null);
  const [showTidpForm, setShowTidpForm] = useState(initialShowTidpForm);
  const [showMidpForm, setShowMidpForm] = useState(initialShowMidpForm);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [showEvolutionDashboard, setShowEvolutionDashboard] = useState(null);
  // Toast state
  const [toast, setToast] = useState({ open: false, message: '', type: 'info' });

  const [tidpForm, setTidpForm] = useState({
    taskTeam: '',
    discipline: '',
    teamLeader: '',
    description: '',
    containers: [
      {
        id: `c-${Date.now()}`,
        'Container Name': 'Initial deliverable',
        'Type': 'Model',
        'Format': 'IFC',
        'LOI Level': 'LOD 300',
        'Author': '',
        'Dependencies': [],
        'Est. Time': '1 day',
        'Milestone': 'Initial',
        'Due Date': new Date().toISOString().slice(0,10),
        'Status': 'Planned'
      }
    ]
  });
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

  // CSV Template Export/Import functions
  const exportTidpCsvTemplate = () => {
    try {
      const csvData = [
        {
          'Information Container ID': 'IC-ARCH-001',
          'Information Container Name/Title': 'Federated Architectural Model',
          'Description': 'Complete architectural model including all building elements',
          'Task Name': 'Architectural Modeling',
          'Responsible Task Team/Party': 'Architecture Team',
          'Author': 'John Smith',
          'Dependencies/Predecessors': 'Site Survey, Structural Grid',
          'Level of Information Need (LOIN)': 'LOD 300',
          'Classification': 'Pr_20_30_60 - Building model',
          'Estimated Production Time': '3 days',
          'Delivery Milestone': 'Stage 3 - Developed Design',
          'Due Date': '2025-12-31',
          'Format/Type': 'IFC 4.0',
          'Purpose': 'Coordination and visualization',
          'Acceptance Criteria': 'Model validation passed, no clashes',
          'Review and Authorization Process': 'S4 - Issue for approval',
          'Status': 'Planned'
        },
        {
          'Information Container ID': 'IC-STRUC-001',
          'Information Container Name/Title': 'Structural Model',
          'Description': 'Complete structural model with foundations, columns, beams, and slabs',
          'Task Name': 'Structural Modeling',
          'Responsible Task Team/Party': 'Structural Engineering Team',
          'Author': 'Jane Doe',
          'Dependencies/Predecessors': 'Architectural Model',
          'Level of Information Need (LOIN)': 'LOD 350',
          'Classification': 'Pr_20_30_60 - Building model',
          'Estimated Production Time': '5 days',
          'Delivery Milestone': 'Stage 4 - Technical Design',
          'Due Date': '2026-01-15',
          'Format/Type': 'IFC 4.0',
          'Purpose': 'Structural analysis and coordination',
          'Acceptance Criteria': 'Structural analysis completed, coordination resolved',
          'Review and Authorization Process': 'S4 - Issue for approval',
          'Status': 'Planned'
        }
      ];

      const csv = Papa.unparse(csvData);
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      link.href = url;
      link.download = 'tidp-deliverables-template.csv';
      link.style.display = 'none';

      // Add to DOM, click, and remove
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Clean up the URL object
      URL.revokeObjectURL(url);

      setToast({ open: true, message: 'TIDP CSV template downloaded successfully!', type: 'success' });
    } catch (error) {
      console.error('Download failed:', error);
      setToast({ open: true, message: 'Failed to download CSV template', type: 'error' });
    }
  };

  const importTidpFromCsv = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        try {
          const containers = results.data.map((row, index) => ({
            id: `IC-${Date.now()}-${index}`,
            'Information Container ID': row['Information Container ID'] || `IC-${Date.now()}-${index}`,
            'Information Container Name/Title': row['Information Container Name/Title'] || '',
            'Description': row['Description'] || '',
            'Task Name': row['Task Name'] || '',
            'Responsible Task Team/Party': row['Responsible Task Team/Party'] || '',
            'Author': row['Author'] || '',
            'Dependencies/Predecessors': row['Dependencies/Predecessors'] || '',
            'Level of Information Need (LOIN)': row['Level of Information Need (LOIN)'] || 'LOD 200',
            'Classification': row['Classification'] || '',
            'Estimated Production Time': row['Estimated Production Time'] || '1 day',
            'Delivery Milestone': row['Delivery Milestone'] || '',
            'Due Date': row['Due Date'] || '',
            'Format/Type': row['Format/Type'] || 'IFC 4.0',
            'Purpose': row['Purpose'] || '',
            'Acceptance Criteria': row['Acceptance Criteria'] || '',
            'Review and Authorization Process': row['Review and Authorization Process'] || 'S1 - Work in progress',
            'Status': row['Status'] || 'Planned'
          })).filter(container => container['Information Container Name/Title'].trim() !== '');

          if (containers.length === 0) {
            setToast({ open: true, message: 'No valid deliverables found in CSV', type: 'error' });
            return;
          }

          // Create TIDP with imported containers
          setTidpForm(prev => ({
            ...prev,
            containers: containers
          }));
          
          setShowTidpForm(true);
          setToast({ 
            open: true, 
            message: `Imported ${containers.length} deliverables from CSV. Please fill in TIDP details and save.`, 
            type: 'success' 
          });
        } catch (error) {
          console.error('CSV import error:', error);
          setToast({ open: true, message: 'Failed to import CSV: ' + error.message, type: 'error' });
        }
      },
      error: (error) => {
        console.error('CSV parsing error:', error);
        setToast({ open: true, message: 'Failed to parse CSV file', type: 'error' });
      }
    });

    // Reset file input
    event.target.value = '';
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
          <div className="flex flex-wrap gap-3">
            <button
              onClick={() => setShowImportDialog(true)}
              className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
            >
              <Upload className="w-4 h-4" />
              <span>Import TIDPs</span>
            </button>
            <button
              onClick={() => setShowTidpForm(true)}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span>New TIDP</span>
            </button>
            <button
              onClick={exportTidpCsvTemplate}
              className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2 transition-colors"
              title="Download a CSV template with sample TIDP deliverables to fill and import"
            >
              <Download className="w-4 h-4" />
              <span>Download CSV Template</span>
            </button>
            <div className="relative">
              <input
                type="file"
                accept=".csv"
                onChange={importTidpFromCsv}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                id="csv-import-tidp"
              />
              <label htmlFor="csv-import-tidp" className="bg-orange-600 hover:bg-orange-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 cursor-pointer transition-colors">
                <Upload className="w-4 h-4" />
                <span>Import CSV</span>
              </label>
            </div>
          </div>
        </div>

        {/* Bulk Export Section */}
        <div className="bg-gray-50 rounded-lg p-4 mb-6">
          <h3 className="text-lg font-semibold mb-3">Bulk Operations</h3>
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

        {/* Details panel */}
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
                          <input value={c['Information Container Name/Title'] || c['Container Name'] || c.name || ''} onChange={(e) => {
                            const updated = [...(detailsForm.containers || [])];
                            updated[ci] = { ...updated[ci], 'Information Container Name/Title': e.target.value };
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
                        const updated = [...(detailsForm.containers || []), { 
                          id: `IC-${Date.now()}`, 
                          'Information Container ID': `IC-${Date.now()}`,
                          'Information Container Name/Title': '', 
                          'Due Date': '' 
                        }];
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
                  <button
                    onClick={() => setShowEvolutionDashboard(midp.id)}
                    className="bg-blue-50 text-blue-700 py-2 px-3 rounded text-sm hover:bg-blue-100 transition-colors flex items-center"
                    title="Evolution Dashboard"
                  >
                    <TrendingUp className="w-4 h-4" />
                  </button>
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

  const TIDPForm = () => {
    const addContainer = () => {
      const newContainer = {
        id: `IC-${Date.now()}`,
        'Information Container ID': `IC-${Date.now()}`,
        'Information Container Name/Title': '',
        'Description': '',
        'Task Name': '',
        'Responsible Task Team/Party': '',
        'Author': '',
        'Dependencies/Predecessors': '',
        'Level of Information Need (LOIN)': 'LOD 200',
        'Classification': '',
        'Estimated Production Time': '1 day',
        'Delivery Milestone': '',
        'Due Date': '',
        'Format/Type': 'IFC 4.0',
        'Purpose': '',
        'Acceptance Criteria': '',
        'Review and Authorization Process': 'S1 - Work in progress',
        'Status': 'Planned'
      };
      setTidpForm(prev => ({
        ...prev,
        containers: [...prev.containers, newContainer]
      }));
    };

    const updateContainer = (index, field, value) => {
      setTidpForm(prev => ({
        ...prev,
        containers: prev.containers.map((container, i) =>
          i === index ? { ...container, [field]: value } : container
        )
      }));
    };

    const removeContainer = (index) => {
      if (tidpForm.containers.length > 1) {
        setTidpForm(prev => ({
          ...prev,
          containers: prev.containers.filter((_, i) => i !== index)
        }));
      }
    };

    return (
      <div className="space-y-4">
        <div className="bg-white rounded-lg shadow px-4 py-3">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold text-gray-900">Create New TIDP</h2>
            <button type="button" onClick={() => setShowTidpForm(false)} className="text-gray-500 hover:text-gray-700 text-lg">✕</button>
          </div>
          <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); handleCreateTidp(); }}>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">Task Team</label>
                <input
                  value={tidpForm.taskTeam}
                  onChange={(e) => setTidpForm((s) => ({ ...s, taskTeam: e.target.value }))}
                  type="text"
                  className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:ring-1 focus:ring-blue-500"
                  placeholder="Architecture Team"
                  required
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">Discipline</label>
                <select
                  value={tidpForm.discipline}
                  onChange={(e) => setTidpForm((s) => ({ ...s, discipline: e.target.value }))}
                  className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:ring-1 focus:ring-blue-500"
                  required
                >
                  <option value="">Select Discipline</option>
                  <option value="architecture">Architecture</option>
                  <option value="structural">Structural Engineering</option>
                  <option value="mep">MEP Engineering</option>
                  <option value="civil">Civil Engineering</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">Team Leader</label>
                <input
                  value={tidpForm.teamLeader}
                  onChange={(e) => setTidpForm((s) => ({ ...s, teamLeader: e.target.value }))}
                  type="text"
                  className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:ring-1 focus:ring-blue-500"
                  placeholder="John Smith"
                  required
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">Description</label>
                <input
                  value={tidpForm.description}
                  onChange={(e) => setTidpForm((s) => ({ ...s, description: e.target.value }))}
                  type="text"
                  className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:ring-1 focus:ring-blue-500"
                  placeholder="TIDP description"
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-base font-semibold text-gray-900">Information Containers</h3>
                <button
                  type="button"
                  onClick={addContainer}
                  className="bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded text-xs flex items-center space-x-1"
                >
                  <Plus className="w-3 h-3" />
                  <span>Add Container</span>
                </button>
              </div>

              <div className="max-h-96 overflow-auto border border-gray-300 rounded-lg">
                <table className="w-full border-collapse">
                  <thead className="bg-gray-100 border-b border-gray-300 sticky top-0 z-10">
                    <tr>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[120px]">Information Container ID</th>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[140px]">Information Container Name/Title</th>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[140px]">Description</th>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[120px]">Task Name</th>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[140px]">Responsible Task Team/Party</th>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[100px]">Author</th>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[140px]">Dependencies/Predecessors</th>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[140px]">Level of Information Need (LOIN)</th>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[120px]">Classification</th>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[120px]">Estimated Production Time</th>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[120px]">Delivery Milestone</th>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[100px]">Due Date</th>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[100px]">Format/Type</th>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[120px]">Purpose</th>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[140px]">Acceptance Criteria</th>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[140px]">Review and Authorization Process</th>
                      <th className="px-1 py-1 text-left text-xs font-semibold text-gray-700 border-r border-gray-300 min-w-[80px]">Status</th>
                      <th className="px-1 py-1 text-center text-xs font-semibold text-gray-700 min-w-[80px]">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tidpForm.containers.map((container, index) => (
                      <tr key={container.id} className="border-b border-gray-200 hover:bg-gray-50">
                        <td className="px-1 py-1 border-r border-gray-300">
                          <input
                            type="text"
                            value={container['Information Container ID']}
                            onChange={(e) => updateContainer(index, 'Information Container ID', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                            placeholder="IC-ARCH-001"
                          />
                        </td>
                        <td className="px-1 py-1 border-r border-gray-300">
                          <input
                            type="text"
                            value={container['Information Container Name/Title']}
                            onChange={(e) => updateContainer(index, 'Information Container Name/Title', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                            placeholder="Federated Architectural Model"
                          />
                        </td>
                        <td className="px-1 py-1 border-r border-gray-300">
                          <input
                            type="text"
                            value={container['Description']}
                            onChange={(e) => updateContainer(index, 'Description', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                            placeholder="Complete architectural model including all building elements"
                          />
                        </td>
                        <td className="px-1 py-1 border-r border-gray-300">
                          <input
                            type="text"
                            value={container['Task Name']}
                            onChange={(e) => updateContainer(index, 'Task Name', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                            placeholder="Architectural Modeling"
                          />
                        </td>
                        <td className="px-1 py-1 border-r border-gray-300">
                          <input
                            type="text"
                            value={container['Responsible Task Team/Party']}
                            onChange={(e) => updateContainer(index, 'Responsible Task Team/Party', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                            placeholder="Architecture Team"
                          />
                        </td>
                        <td className="px-1 py-1 border-r border-gray-300">
                          <input
                            type="text"
                            value={container['Author']}
                            onChange={(e) => updateContainer(index, 'Author', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                            placeholder="John Smith"
                          />
                        </td>
                        <td className="px-1 py-1 border-r border-gray-300">
                          <input
                            type="text"
                            value={container['Dependencies/Predecessors']}
                            onChange={(e) => updateContainer(index, 'Dependencies/Predecessors', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                            placeholder="Site Survey, Structural Grid"
                          />
                        </td>
                        <td className="px-1 py-1 border-r border-gray-300">
                          <select
                            value={container['Level of Information Need (LOIN)']}
                            onChange={(e) => updateContainer(index, 'Level of Information Need (LOIN)', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                          >
                            <option value="LOD 100">LOD 100</option>
                            <option value="LOD 200">LOD 200</option>
                            <option value="LOD 300">LOD 300</option>
                            <option value="LOD 350">LOD 350</option>
                            <option value="LOD 400">LOD 400</option>
                            <option value="LOD 500">LOD 500</option>
                            <option value="As-Built">As-Built</option>
                          </select>
                        </td>
                        <td className="px-1 py-1 border-r border-gray-300">
                          <select
                            value={container['Classification']}
                            onChange={(e) => updateContainer(index, 'Classification', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                          >
                            <option value="">Select Classification</option>
                            <option value="Pr_20_30_60 - Building model">Pr_20_30_60 - Building model</option>
                            <option value="Pr_20_30_70 - Space model">Pr_20_30_70 - Space model</option>
                            <option value="Pr_20_30_80 - Zone model">Pr_20_30_80 - Zone model</option>
                            <option value="Pr_30_10 - Element">Pr_30_10 - Element</option>
                            <option value="Pr_30_20 - Component">Pr_30_20 - Component</option>
                            <option value="Pr_30_30 - Assembly">Pr_30_30 - Assembly</option>
                          </select>
                        </td>
                        <td className="px-1 py-1 border-r border-gray-300">
                          <input
                            type="text"
                            value={container['Estimated Production Time']}
                            onChange={(e) => updateContainer(index, 'Estimated Production Time', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                            placeholder="3 days"
                          />
                        </td>
                        <td className="px-1 py-1 border-r border-gray-300">
                          <select
                            value={container['Delivery Milestone']}
                            onChange={(e) => updateContainer(index, 'Delivery Milestone', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                          >
                            <option value="">Select Milestone</option>
                            <option value="Stage 1 - Preparation">Stage 1 - Preparation</option>
                            <option value="Stage 2 - Concept Design">Stage 2 - Concept Design</option>
                            <option value="Stage 3 - Developed Design">Stage 3 - Developed Design</option>
                            <option value="Stage 4 - Technical Design">Stage 4 - Technical Design</option>
                            <option value="Stage 5 - Manufacturing & Construction">Stage 5 - Manufacturing & Construction</option>
                            <option value="Stage 6 - Handover & Close Out">Stage 6 - Handover & Close Out</option>
                            <option value="Stage 7 - In Use">Stage 7 - In Use</option>
                          </select>
                        </td>
                        <td className="px-1 py-1 border-r border-gray-300">
                          <input
                            type="date"
                            value={container['Due Date']}
                            onChange={(e) => updateContainer(index, 'Due Date', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                          />
                        </td>
                        <td className="px-1 py-1 border-r border-gray-300">
                          <select
                            value={container['Format/Type']}
                            onChange={(e) => updateContainer(index, 'Format/Type', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                          >
                            <option value="IFC 2x3">IFC 2x3</option>
                            <option value="IFC 4.0">IFC 4.0</option>
                            <option value="DWG">DWG</option>
                            <option value="PDF">PDF</option>
                            <option value="XLSX">XLSX</option>
                            <option value="DOCX">DOCX</option>
                            <option value="RVT">RVT</option>
                            <option value="NWD">NWD</option>
                          </select>
                        </td>
                        <td className="px-1 py-1 border-r border-gray-300">
                          <input
                            type="text"
                            value={container['Purpose']}
                            onChange={(e) => updateContainer(index, 'Purpose', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                            placeholder="Coordination and visualization"
                          />
                        </td>
                        <td className="px-1 py-1 border-r border-gray-300">
                          <input
                            type="text"
                            value={container['Acceptance Criteria']}
                            onChange={(e) => updateContainer(index, 'Acceptance Criteria', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                            placeholder="Model validation passed, no clashes"
                          />
                        </td>
                        <td className="px-1 py-1 border-r border-gray-300">
                          <select
                            value={container['Review and Authorization Process']}
                            onChange={(e) => updateContainer(index, 'Review and Authorization Process', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                          >
                            <option value="S1 - Work in progress">S1 - Work in progress</option>
                            <option value="S2 - Shared for coordination">S2 - Shared for coordination</option>
                            <option value="S3 - Issue for comment">S3 - Issue for comment</option>
                            <option value="S4 - Issue for approval">S4 - Issue for approval</option>
                            <option value="S5 - Issue for construction">S5 - Issue for construction</option>
                          </select>
                        </td>
                        <td className="px-1 py-1 border-r border-gray-300">
                          <select
                            value={container['Status']}
                            onChange={(e) => updateContainer(index, 'Status', e.target.value)}
                            className="w-full px-1 py-0.5 text-xs border border-gray-300 rounded"
                          >
                            <option value="Planned">Planned</option>
                            <option value="In Progress">In Progress</option>
                            <option value="Under Review">Under Review</option>
                            <option value="Approved">Approved</option>
                            <option value="Completed">Completed</option>
                            <option value="Delayed">Delayed</option>
                          </select>
                        </td>
                        <td className="px-1 py-1 text-center">
                          <button
                            type="button"
                            onClick={() => removeContainer(index)}
                            className="text-red-600 hover:text-red-800 text-xs"
                            disabled={tidpForm.containers.length === 1}
                          >
                            Remove
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="flex space-x-3 pt-2">
              <button type="submit" className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-1 rounded text-sm">Create TIDP</button>
              <button type="button" onClick={() => { setShowTidpForm(false); setTidpForm({
                taskTeam: '',
                discipline: '',
                teamLeader: '',
                description: '',
                containers: [
                  {
                    id: `IC-${Date.now()}`,
                    'Information Container ID': `IC-${Date.now()}`,
                    'Information Container Name/Title': 'Initial deliverable',
                    'Description': '',
                    'Task Name': '',
                    'Responsible Task Team/Party': '',
                    'Author': '',
                    'Dependencies/Predecessors': '',
                    'Level of Information Need (LOIN)': 'LOD 200',
                    'Classification': '',
                    'Estimated Production Time': '1 day',
                    'Delivery Milestone': '',
                    'Due Date': new Date().toISOString().slice(0,10),
                    'Format/Type': 'IFC 4.0',
                    'Purpose': '',
                    'Acceptance Criteria': '',
                    'Review and Authorization Process': 'S1 - Work in progress',
                    'Status': 'Planned'
                  }
                ]
              }); }} className="bg-gray-300 hover:bg-gray-400 text-gray-700 px-4 py-1 rounded text-sm">Cancel</button>
            </div>
          </form>
        </div>
      </div>
    );
  };

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
    if (!tidpForm.discipline) {
      setToast({ open: true, message: 'Discipline is required', type: 'error' });
      return;
    }
    if (!tidpForm.teamLeader || tidpForm.teamLeader.trim().length < 2) {
      setToast({ open: true, message: 'Team leader is required', type: 'error' });
      return;
    }

    try {
      // The server validator expects specific fields: teamName, discipline, leader, company, responsibilities, containers
      // Provide minimal defaults so the TIDP can be created and the user can edit details in the UI.
      const payload = {
        teamName: tidpForm.taskTeam,
        discipline: tidpForm.discipline,
        leader: tidpForm.teamLeader,
        company: 'TBD',
        responsibilities: tidpForm.description || 'TBD',
        description: tidpForm.description,
        containers: tidpForm.containers
      };
      const created = await ApiService.createTIDP(payload);
      setToast({ open: true, message: 'TIDP created', type: 'success' });
      setShowTidpForm(false);
      setTidpForm({
        taskTeam: '',
        discipline: '',
        teamLeader: '',
        description: '',
        containers: [
          {
            id: `c-${Date.now()}`,
            'Container Name': 'Initial deliverable',
            'Type': 'Model',
            'Format': 'IFC',
            'LOI Level': 'LOD 300',
            'Author': '',
            'Dependencies': [],
            'Est. Time': '1 day',
            'Milestone': 'Initial',
            'Due Date': new Date().toISOString().slice(0,10),
            'Status': 'Planned'
          }
        ]
      });
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
      // Try to auto-generate MIDP from existing TIDPs first
      const projectId = 'project-1'; // In a real app, this would be dynamic
      try {
        await ApiService.autoGenerateMIDP(projectId, {
          projectName: midpForm.projectName,
          description: midpForm.description
        });
        setToast({ open: true, message: 'MIDP auto-generated from existing TIDPs', type: 'success' });
      } catch (autoError) {
        // If auto-generation fails (no TIDPs), create empty MIDP
        await ApiService.createMIDPFromTIDPs({
          projectName: midpForm.projectName,
          description: midpForm.description
        }, []);
        setToast({ open: true, message: 'MIDP created (add TIDPs to populate)', type: 'success' });
      }

      setShowMidpForm(false);
      setMidpForm({ projectName: '', description: '' });
      await loadData();
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
    await loadData();
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
            {onClose ? (
              <>
                <h1 className="text-2xl font-bold text-gray-900">TIDP/MIDP Manager</h1>
                <p className="text-gray-600">Manage Task and Master Information Delivery Plans</p>
              </>
            ) : (
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => navigate('/tidp-midp-dashboard')}
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
              <button onClick={onClose} className="text-sm text-gray-600 hover:text-gray-800 underline">Back to BEP</button>
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

                  <div className="flex flex-wrap gap-4">
                    <button onClick={() => setShowTidpForm(true)} className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2">
                      <Plus className="w-4 h-4" />
                      <span>Create New TIDP</span>
                    </button>
                    <button onClick={() => setShowMidpForm(true)} className="bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2">
                      <Plus className="w-4 h-4" />
                      <span>Create New MIDP</span>
                    </button>
                    <button onClick={exportTidpCsvTemplate} className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2">
                      <Download className="w-4 h-4" />
                      <span>Download TIDP CSV Template</span>
                    </button>
                    <div className="relative">
                      <input
                        type="file"
                        accept=".csv"
                        onChange={importTidpFromCsv}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                        id="csv-import"
                      />
                      <label htmlFor="csv-import" className="bg-orange-600 hover:bg-orange-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2 cursor-pointer">
                        <Upload className="w-4 h-4" />
                        <span>Import TIDP from CSV</span>
                      </label>
                    </div>
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
