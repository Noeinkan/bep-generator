import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Plus,
  Calendar,
  Users,
  Download,
  Upload,
  TrendingUp,
  BarChart3,
  ArrowLeft,
  FileText,
  Settings,
  Filter,
  Search
} from 'lucide-react';
import ApiService from '../../services/apiService';
import Toast from '../common/Toast';
import TIDPImportDialog from '../TIDPImportDialog';
import MIDPEvolutionDashboard from '../MIDPEvolutionDashboard';

const TIDPMIDPDashboard = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // Parse current view from URL
  const getCurrentView = () => {
    const path = location.pathname;
    if (path.includes('/tidps')) return 'tidps';
    if (path.includes('/midps')) return 'midps';
    if (path.includes('/import')) return 'import';
    return 'dashboard';
  };

  const [activeView, setActiveView] = useState(getCurrentView());
  const [tidps, setTidps] = useState([]);
  const [midps, setMidps] = useState([]);
  const [loading, setLoading] = useState(false);
  const [exportLoading, setExportLoading] = useState({});
  const [templates, setTemplates] = useState([]);
  const [selectedTemplate, setSelectedTemplate] = useState(null);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [showEvolutionDashboard, setShowEvolutionDashboard] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterDiscipline, setFilterDiscipline] = useState('all');

  // Toast state
  const [toast, setToast] = useState({ open: false, message: '', type: 'info' });

  // Statistics
  const [stats, setStats] = useState({
    totalTidps: 0,
    totalMidps: 0,
    totalDeliverables: 0,
    activeMilestones: 0
  });

  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    loadData();
    loadTemplates();
    return () => { mountedRef.current = false; };
  }, []);

  useEffect(() => {
    // Update URL when view changes
    const newPath = activeView === 'dashboard' ? '/tidp-midp' : `/tidp-midp/${activeView}`;
    if (location.pathname !== newPath) {
      navigate(newPath, { replace: true });
    }
  }, [activeView, navigate, location.pathname]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [tidpData, midpData] = await Promise.all([
        ApiService.getAllTIDPs(),
        ApiService.getAllMIDPs()
      ]);

      if (!mountedRef.current) return;

      const tidpList = tidpData.data || [];
      const midpList = midpData.data || [];

      setTidps(tidpList);
      setMidps(midpList);

      // Calculate statistics
      const totalDeliverables = tidpList.reduce((sum, tidp) =>
        sum + (tidp.containers?.length || 0), 0
      );

      const allMilestones = new Set();
      tidpList.forEach(tidp => {
        tidp.containers?.forEach(container => {
          const milestone = container.Milestone || container.deliveryMilestone;
          if (milestone) allMilestones.add(milestone);
        });
      });

      setStats({
        totalTidps: tidpList.length,
        totalMidps: midpList.length,
        totalDeliverables,
        activeMilestones: allMilestones.size
      });

    } catch (error) {
      if (mountedRef.current) {
        console.error('Failed to load TIDP/MIDP data:', error);
        setToast({ open: true, message: 'Failed to load data', type: 'error' });
      }
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  };

  const loadTemplates = async () => {
    try {
      const resp = await ApiService.getExportTemplates();
      if (resp && Array.isArray(resp)) {
        setTemplates(resp);
      } else if (resp && resp.templates && Array.isArray(resp.templates)) {
        setTemplates(resp.templates);
      } else if (resp && resp.data && typeof resp.data === 'object') {
        const normalized = Object.keys(resp.data).map(key => ({
          id: key,
          name: key.toUpperCase(),
          ...resp.data[key]
        }));
        setTemplates(normalized);
      }
    } catch (err) {
      console.warn('Failed to load export templates', err);
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

  const autoGenerateMIDP = async () => {
    if (tidps.length === 0) {
      setToast({ open: true, message: 'No TIDPs available to generate MIDP', type: 'info' });
      return;
    }

    try {
      const projectId = 'project-1'; // In a real app, this would be dynamic
      await ApiService.autoGenerateMIDP(projectId, {
        projectName: `Auto-generated MIDP ${new Date().toLocaleDateString()}`,
        description: `MIDP generated from ${tidps.length} TIDPs`
      });
      setToast({ open: true, message: 'MIDP auto-generated successfully', type: 'success' });
      await loadData();
    } catch (err) {
      console.error('Auto-generate MIDP failed', err);
      setToast({ open: true, message: err.message || 'Failed to auto-generate MIDP', type: 'error' });
    }
  };

  // Filter functions
  const getFilteredTidps = () => {
    return tidps.filter(tidp => {
      const matchesSearch = !searchTerm ||
        tidp.teamName?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        tidp.description?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        tidp.discipline?.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesDiscipline = filterDiscipline === 'all' || tidp.discipline === filterDiscipline;

      return matchesSearch && matchesDiscipline;
    });
  };

  const getFilteredMidps = () => {
    return midps.filter(midp => {
      return !searchTerm ||
        midp.projectName?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        midp.description?.toLowerCase().includes(searchTerm.toLowerCase());
    });
  };

  const getDisciplineOptions = () => {
    const disciplines = new Set(tidps.map(tidp => tidp.discipline).filter(Boolean));
    return Array.from(disciplines);
  };

  // Navigation items
  const navigationItems = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'tidps', label: 'TIDPs', icon: Users },
    { id: 'midps', label: 'MIDPs', icon: Calendar },
    { id: 'import', label: 'Import', icon: Upload }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => navigate('/')}
                className="inline-flex items-center text-gray-500 hover:text-gray-700"
              >
                <ArrowLeft className="w-4 h-4 mr-1" />
                Back to Home
              </button>
              <div className="h-6 border-l border-gray-300"></div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">TIDP/MIDP Manager</h1>
                <p className="text-gray-600">Information Delivery Planning & Management</p>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              {/* Quick Actions */}
              <button
                onClick={() => setShowImportDialog(true)}
                className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                <Upload className="w-4 h-4 mr-2" />
                Import
              </button>

              <button
                onClick={autoGenerateMIDP}
                className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700"
              >
                <TrendingUp className="w-4 h-4 mr-2" />
                Auto-Generate MIDP
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Navigation Tabs */}
        <div className="mb-8">
          <nav className="flex space-x-8">
            {navigationItems.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveView(id)}
                className={`py-2 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                  activeView === id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Dashboard View */}
        {activeView === 'dashboard' && (
          <div className="space-y-8">
            {/* Statistics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center">
                  <Users className="w-8 h-8 text-blue-600" />
                  <div className="ml-4">
                    <p className="text-2xl font-bold text-gray-900">{stats.totalTidps}</p>
                    <p className="text-gray-600">TIDPs</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center">
                  <Calendar className="w-8 h-8 text-green-600" />
                  <div className="ml-4">
                    <p className="text-2xl font-bold text-gray-900">{stats.totalMidps}</p>
                    <p className="text-gray-600">MIDPs</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center">
                  <FileText className="w-8 h-8 text-purple-600" />
                  <div className="ml-4">
                    <p className="text-2xl font-bold text-gray-900">{stats.totalDeliverables}</p>
                    <p className="text-gray-600">Deliverables</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center">
                  <TrendingUp className="w-8 h-8 text-orange-600" />
                  <div className="ml-4">
                    <p className="text-2xl font-bold text-gray-900">{stats.activeMilestones}</p>
                    <p className="text-gray-600">Milestones</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <button
                  onClick={() => setActiveView('tidps')}
                  className="flex items-center p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <Users className="w-6 h-6 text-blue-600 mr-3" />
                  <div className="text-left">
                    <p className="font-medium text-gray-900">Manage TIDPs</p>
                    <p className="text-sm text-gray-600">Create and edit team plans</p>
                  </div>
                </button>

                <button
                  onClick={() => setActiveView('midps')}
                  className="flex items-center p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <Calendar className="w-6 h-6 text-green-600 mr-3" />
                  <div className="text-left">
                    <p className="font-medium text-gray-900">View MIDPs</p>
                    <p className="text-sm text-gray-600">Monitor master plans</p>
                  </div>
                </button>

                <button
                  onClick={() => setShowImportDialog(true)}
                  className="flex items-center p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <Upload className="w-6 h-6 text-purple-600 mr-3" />
                  <div className="text-left">
                    <p className="font-medium text-gray-900">Import Data</p>
                    <p className="text-sm text-gray-600">Import from Excel/CSV</p>
                  </div>
                </button>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent TIDPs</h2>
              {tidps.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <Users className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                  <p>No TIDPs created yet</p>
                  <button
                    onClick={() => setActiveView('tidps')}
                    className="mt-2 text-blue-600 hover:text-blue-800"
                  >
                    Create your first TIDP
                  </button>
                </div>
              ) : (
                <div className="space-y-4">
                  {tidps.slice(0, 5).map((tidp, index) => (
                    <div key={tidp.id || index} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                      <div>
                        <h3 className="font-medium text-gray-900">{tidp.teamName || `TIDP ${index + 1}`}</h3>
                        <p className="text-sm text-gray-600">{tidp.discipline} • {tidp.containers?.length || 0} deliverables</p>
                      </div>
                      <div className="text-sm text-gray-500">
                        {new Date(tidp.updatedAt).toLocaleDateString()}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* TIDPs View */}
        {activeView === 'tidps' && (
          <div className="space-y-6">
            {/* Filters and Search */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex flex-col sm:flex-row gap-4">
                <div className="flex-1">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                    <input
                      type="text"
                      placeholder="Search TIDPs..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>

                <div className="flex items-center space-x-4">
                  <select
                    value={filterDiscipline}
                    onChange={(e) => setFilterDiscipline(e.target.value)}
                    className="border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="all">All Disciplines</option>
                    {getDisciplineOptions().map(discipline => (
                      <option key={discipline} value={discipline}>{discipline}</option>
                    ))}
                  </select>

                  <button
                    onClick={() => navigate('/bep-generator?createTidp=true')}
                    className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700"
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    New TIDP
                  </button>
                </div>
              </div>
            </div>

            {/* TIDPs Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {getFilteredTidps().map((tidp, index) => (
                <div key={tidp.id || index} className="bg-white rounded-lg shadow hover:shadow-md transition-shadow">
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900">{tidp.teamName || `TIDP ${index + 1}`}</h3>
                        <p className="text-sm text-gray-600">{tidp.discipline}</p>
                      </div>
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        {tidp.status || 'Draft'}
                      </span>
                    </div>

                    <p className="text-gray-600 text-sm mb-4 line-clamp-3">
                      {tidp.description || tidp.responsibilities || 'Task information delivery plan'}
                    </p>

                    <div className="space-y-2 mb-4">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-500">Deliverables:</span>
                        <span className="font-medium">{tidp.containers?.length || 0}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-500">Leader:</span>
                        <span className="font-medium">{tidp.leader || 'TBD'}</span>
                      </div>
                    </div>

                    <div className="flex space-x-2">
                      <button className="flex-1 bg-blue-50 text-blue-700 py-2 px-3 rounded text-sm hover:bg-blue-100 transition-colors">
                        View Details
                      </button>
                      <button className="bg-gray-50 text-gray-700 py-2 px-3 rounded text-sm hover:bg-gray-100 transition-colors">
                        <Download className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {getFilteredTidps().length === 0 && (
              <div className="text-center py-12 text-gray-500">
                <Users className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p className="text-lg">No TIDPs found</p>
                <p className="text-sm">Try adjusting your search or filters</p>
              </div>
            )}
          </div>
        )}

        {/* MIDPs View */}
        {activeView === 'midps' && (
          <div className="space-y-6">
            {/* MIDPs Header */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex justify-between items-center">
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">Master Information Delivery Plans</h2>
                  <p className="text-gray-600">Aggregated project delivery schedules</p>
                </div>
                <button
                  onClick={autoGenerateMIDP}
                  className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Generate MIDP
                </button>
              </div>
            </div>

            {/* MIDPs Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {getFilteredMidps().map((midp, index) => (
                <div key={midp.id || index} className="bg-white rounded-lg shadow hover:shadow-md transition-shadow">
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900">{midp.projectName || `MIDP ${index + 1}`}</h3>
                        <p className="text-sm text-gray-600">{midp.includedTIDPs?.length || 0} TIDPs included</p>
                      </div>
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                        {midp.status || 'Active'}
                      </span>
                    </div>

                    <p className="text-gray-600 text-sm mb-4">
                      {midp.description || 'Master information delivery plan'}
                    </p>

                    <div className="space-y-2 mb-4">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-500">Total Deliverables:</span>
                        <span className="font-medium">{midp.aggregatedData?.totalContainers || 0}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-500">Estimated Hours:</span>
                        <span className="font-medium">{midp.aggregatedData?.totalEstimatedHours || 0}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-500">Last Updated:</span>
                        <span className="font-medium">{new Date(midp.updatedAt).toLocaleDateString()}</span>
                      </div>
                    </div>

                    <div className="flex space-x-2">
                      <button className="flex-1 bg-green-50 text-green-700 py-2 px-3 rounded text-sm hover:bg-green-100 transition-colors">
                        View Details
                      </button>
                      <button
                        onClick={() => setShowEvolutionDashboard(midp.id)}
                        className="bg-blue-50 text-blue-700 py-2 px-3 rounded text-sm hover:bg-blue-100 transition-colors"
                        title="Evolution Dashboard"
                      >
                        <TrendingUp className="w-4 h-4" />
                      </button>
                      <button className="bg-gray-50 text-gray-700 py-2 px-3 rounded text-sm hover:bg-gray-100 transition-colors">
                        <Download className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {getFilteredMidps().length === 0 && (
              <div className="text-center py-12 text-gray-500">
                <Calendar className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p className="text-lg">No MIDPs created yet</p>
                <button
                  onClick={autoGenerateMIDP}
                  className="mt-2 text-green-600 hover:text-green-800"
                >
                  Generate your first MIDP
                </button>
              </div>
            )}
          </div>
        )}

        {/* Import View */}
        {activeView === 'import' && (
          <div className="max-w-2xl mx-auto">
            <div className="bg-white rounded-lg shadow p-8">
              <div className="text-center mb-8">
                <Upload className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                <h2 className="text-2xl font-semibold text-gray-900 mb-2">Import TIDPs</h2>
                <p className="text-gray-600">
                  Import TIDP data from Excel or CSV files created by external teams
                </p>
              </div>

              <div className="space-y-6">
                <button
                  onClick={() => setShowImportDialog(true)}
                  className="w-full flex items-center justify-center px-6 py-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-gray-400 transition-colors"
                >
                  <Upload className="w-6 h-6 mr-3 text-gray-400" />
                  <span className="text-lg font-medium text-gray-600">Import from Excel/CSV</span>
                </button>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 border border-gray-200 rounded-lg">
                    <h3 className="font-medium text-gray-900 mb-2">Supported Formats</h3>
                    <ul className="text-sm text-gray-600 space-y-1">
                      <li>• Excel (.xlsx, .xls)</li>
                      <li>• CSV (.csv)</li>
                      <li>• UTF-8 encoding recommended</li>
                    </ul>
                  </div>

                  <div className="p-4 border border-gray-200 rounded-lg">
                    <h3 className="font-medium text-gray-900 mb-2">What's Imported</h3>
                    <ul className="text-sm text-gray-600 space-y-1">
                      <li>• Team information</li>
                      <li>• Deliverable containers</li>
                      <li>• Schedules and milestones</li>
                      <li>• Dependencies</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

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

      {/* Toast Notification */}
      <Toast
        open={toast.open}
        message={toast.message}
        type={toast.type}
        onClose={() => setToast((t) => ({ ...t, open: false }))}
      />
    </div>
  );
};

export default TIDPMIDPDashboard;