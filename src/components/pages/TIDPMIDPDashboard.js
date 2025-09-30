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
  Search,
  CheckCircle
} from 'lucide-react';
import Papa from 'papaparse';
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
  const [showHelp, setShowHelp] = useState(false);
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

  // CSV Template Export function
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

  const handleComplianceCheck = async () => {
    try {
      // Check if MIDPs have required LOD/LOI information
      const compliant = midps.every(midp => {
        return midp.aggregatedData?.containers?.every(container => 
          container.loiLevel && container.format
        );
      });

      setToast({
        open: true,
        message: compliant ? 'All MIDPs are compliant with ISO 19650 standards' : 'Some MIDPs may not be fully compliant - check LOD/LOI details',
        type: compliant ? 'success' : 'warning'
      });
    } catch (error) {
      setToast({
        open: true,
        message: 'Failed to perform compliance check',
        type: 'error'
      });
    }
  };

  const generateReport = async () => {
    try {
      // Import jsPDF dynamically
      const { default: jsPDF } = await import('jspdf');

      const doc = new jsPDF();
      doc.setFontSize(16);
      doc.text('MIDP Compliance Report', 20, 20);
      doc.setFontSize(12);
      doc.text(`Generated on: ${new Date().toLocaleDateString()}`, 20, 35);

      // Add relationship summary
      doc.setFontSize(14);
      doc.text('TIDP-MIDP Relationship Summary', 20, 55);
      doc.setFontSize(10);
      const relationshipText = [
        'In the context of ISO 19650, TIDPs and MIDPs are key elements for BIM project planning.',
        'TIDPs provide detailed team-specific deliverables, while MIDPs integrate them into a unified plan.',
        'This hierarchical relationship ensures synchronized delivery and proactive collaboration.'
      ];
      
      relationshipText.forEach((line, index) => {
        doc.text(line, 20, 70 + (index * 5));
      });

      // Add compliance status
      doc.setFontSize(12);
      doc.text('Compliance Status:', 20, 100);
      const compliant = midps.every(midp => 
        midp.aggregatedData?.containers?.every(container => container.loiLevel)
      );
      doc.setFontSize(10);
      doc.text(compliant ? '✓ Compliant with ISO 19650' : '⚠ Review required', 20, 110);

      doc.save('MIDP_Compliance_Report.pdf');

      setToast({
        open: true,
        message: 'Report generated successfully',
        type: 'success'
      });
    } catch (error) {
      setToast({
        open: true,
        message: 'Failed to generate report',
        type: 'error'
      });
    }
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
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <button
                onClick={() => navigate('/')}
                className="inline-flex items-center text-gray-500 hover:text-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 rounded-md p-2 transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
              <div className="h-8 border-l border-gray-300"></div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">TIDP/MIDP Manager</h1>
                <p className="text-gray-600 text-lg mt-1">Information Delivery Planning & Management</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Quick Actions */}
              <button
                onClick={() => setShowImportDialog(true)}
                className="inline-flex items-center px-6 py-3 border border-gray-300 rounded-lg shadow-sm text-base font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200"
              >
                <Upload className="w-5 h-5 mr-3" />
                Import
              </button>

              <button
                onClick={autoGenerateMIDP}
                className="inline-flex items-center px-6 py-3 border border-transparent rounded-lg shadow-sm text-base font-semibold text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-all duration-200 hover:shadow-md"
              >
                <TrendingUp className="w-5 h-5 mr-3" />
                Auto-Generate MIDP
              </button>

              <button
                onClick={() => setShowHelp(true)}
                className="inline-flex items-center px-6 py-3 border border-gray-300 rounded-lg shadow-sm text-base font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200"
              >
                <FileText className="w-5 h-5 mr-3" />
                Help
              </button>

              <button
                onClick={handleComplianceCheck}
                className="inline-flex items-center px-6 py-3 border border-gray-300 rounded-lg shadow-sm text-base font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-all duration-200"
              >
                <CheckCircle className="w-5 h-5 mr-3" />
                Compliance Check
              </button>

              <button
                onClick={generateReport}
                className="inline-flex items-center px-6 py-3 border border-gray-300 rounded-lg shadow-sm text-base font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 transition-all duration-200"
              >
                <Download className="w-5 h-5 mr-3" />
                Generate Report
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Navigation Tabs */}
        <div className="mb-10">
          <nav className="flex space-x-1 bg-gray-100 p-2 rounded-lg">
            {navigationItems.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveView(id)}
                className={`flex-1 py-3 px-4 rounded-md font-semibold text-base flex items-center justify-center space-x-3 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                  activeView === id
                    ? 'bg-white text-blue-700 shadow-sm border border-gray-200'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span>{label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Dashboard View */}
        {activeView === 'dashboard' && (
          <div className="space-y-8">
            {/* Statistics Cards */}
            {loading ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {[...Array(4)].map((_, i) => (
                  <div key={i} className="bg-white rounded-lg border border-gray-200 shadow-sm p-6 animate-pulse">
                    <div className="flex items-center">
                      <div className="w-12 h-12 bg-gray-200 rounded-lg"></div>
                      <div className="ml-4 flex-1">
                        <div className="h-8 bg-gray-200 rounded w-16 mb-2"></div>
                        <div className="h-4 bg-gray-200 rounded w-20"></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-all duration-200 p-6 hover:border-blue-300">
                  <div className="flex items-center">
                    <div className="p-3 bg-blue-50 rounded-lg">
                      <Users className="w-8 h-8 text-blue-600" />
                    </div>
                    <div className="ml-4">
                      <p className="text-3xl font-bold text-gray-900">{stats.totalTidps}</p>
                      <p className="text-gray-600 font-medium">TIDPs</p>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-all duration-200 p-6 hover:border-green-300">
                  <div className="flex items-center">
                    <div className="p-3 bg-green-50 rounded-lg">
                      <Calendar className="w-8 h-8 text-green-600" />
                    </div>
                    <div className="ml-4">
                      <p className="text-3xl font-bold text-gray-900">{stats.totalMidps}</p>
                      <p className="text-gray-600 font-medium">MIDPs</p>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-all duration-200 p-6 hover:border-purple-300">
                  <div className="flex items-center">
                    <div className="p-3 bg-purple-50 rounded-lg">
                      <FileText className="w-8 h-8 text-purple-600" />
                    </div>
                    <div className="ml-4">
                      <p className="text-3xl font-bold text-gray-900">{stats.totalDeliverables}</p>
                      <p className="text-gray-600 font-medium">Deliverables</p>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-all duration-200 p-6 hover:border-orange-300">
                  <div className="flex items-center">
                    <div className="p-3 bg-orange-50 rounded-lg">
                      <TrendingUp className="w-8 h-8 text-orange-600" />
                    </div>
                    <div className="ml-4">
                      <p className="text-3xl font-bold text-gray-900">{stats.activeMilestones}</p>
                      <p className="text-gray-600 font-medium">Milestones</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Quick Actions */}
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Quick Actions</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <button
                  onClick={() => setActiveView('tidps')}
                  className="group flex items-center p-6 border-2 border-gray-200 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <div className="p-3 bg-blue-100 rounded-lg group-hover:bg-blue-200 transition-colors">
                    <Users className="w-6 h-6 text-blue-600" />
                  </div>
                  <div className="ml-4 text-left">
                    <p className="font-bold text-gray-900 text-lg">Manage TIDPs</p>
                    <p className="text-gray-600 mt-1">Create and edit team plans</p>
                  </div>
                </button>

                <button
                  onClick={() => setActiveView('midps')}
                  className="group flex items-center p-6 border-2 border-gray-200 rounded-lg hover:border-green-400 hover:bg-green-50 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                >
                  <div className="p-3 bg-green-100 rounded-lg group-hover:bg-green-200 transition-colors">
                    <Calendar className="w-6 h-6 text-green-600" />
                  </div>
                  <div className="ml-4 text-left">
                    <p className="font-bold text-gray-900 text-lg">View MIDP</p>
                    <p className="text-gray-600 mt-1">Monitor master plan</p>
                  </div>
                </button>

                <button
                  onClick={() => setShowImportDialog(true)}
                  className="group flex items-center p-6 border-2 border-gray-200 rounded-lg hover:border-purple-400 hover:bg-purple-50 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                >
                  <div className="p-3 bg-purple-100 rounded-lg group-hover:bg-purple-200 transition-colors">
                    <Upload className="w-6 h-6 text-purple-600" />
                  </div>
                  <div className="ml-4 text-left">
                    <p className="font-bold text-gray-900 text-lg">Import Data</p>
                    <p className="text-gray-600 mt-1">Import from Excel/CSV</p>
                  </div>
                </button>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Recent TIDPs</h2>
              {tidps.length === 0 ? (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
                    <Users className="w-8 h-8 text-gray-400" />
                  </div>
                  <p className="text-xl font-semibold text-gray-900 mb-2">No TIDPs created yet</p>
                  <p className="text-gray-600 mb-6">Get started by creating your first Team Information Delivery Plan</p>
                  <button
                    onClick={() => setActiveView('tidps')}
                    className="inline-flex items-center px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200"
                  >
                    <Plus className="w-5 h-5 mr-2" />
                    Create your first TIDP
                  </button>
                </div>
              ) : (
                <div className="space-y-4">
                  {tidps.slice(0, 5).map((tidp, index) => (
                    <div key={tidp.id || index} className="flex items-center justify-between p-6 border border-gray-200 rounded-lg hover:border-blue-300 hover:shadow-sm transition-all duration-200">
                      <div className="flex items-center space-x-4">
                        <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                          <Users className="w-6 h-6 text-blue-600" />
                        </div>
                        <div>
                          <h3 className="font-bold text-gray-900 text-lg">{tidp.teamName || `TIDP ${index + 1}`}</h3>
                          <p className="text-gray-600">{tidp.discipline} • {tidp.containers?.length || 0} deliverables</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium text-gray-900">
                          {new Date(tidp.updatedAt).toLocaleDateString()}
                        </div>
                        <div className="text-sm text-gray-500">Last updated</div>
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
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-8 mb-8">
              <div className="flex flex-col lg:flex-row gap-6">
                <div className="flex-1">
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Search TIDPs</label>
                  <div className="relative">
                    <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                    <input
                      type="text"
                      placeholder="Search by team name, description, or discipline..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-base"
                    />
                  </div>
                </div>

                <div className="flex items-end space-x-4">
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Filter by Discipline</label>
                    <select
                      value={filterDiscipline}
                      onChange={(e) => setFilterDiscipline(e.target.value)}
                      className="border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent text-base min-w-48"
                    >
                      <option value="all">All Disciplines</option>
                      {getDisciplineOptions().map(discipline => (
                        <option key={discipline} value={discipline}>{discipline}</option>
                      ))}
                    </select>
                  </div>

                  <button
                    onClick={() => navigate('/tidp-editor')}
                    className="inline-flex items-center px-8 py-3 border border-transparent rounded-lg shadow-sm text-base font-semibold text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 hover:shadow-md"
                  >
                    <Plus className="w-5 h-5 mr-3" />
                    New TIDP
                  </button>

                  <button
                    onClick={exportTidpCsvTemplate}
                    className="inline-flex items-center px-8 py-3 border border-purple-600 rounded-lg shadow-sm text-base font-semibold text-purple-600 bg-white hover:bg-purple-50 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 transition-all duration-200 hover:shadow-md"
                    title="Download a CSV template with sample TIDP deliverables to fill and import"
                  >
                    <Download className="w-5 h-5 mr-3" />
                    Download CSV Template
                  </button>
                </div>
              </div>
            </div>

            {/* TIDPs Grid */}
            {loading ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {[...Array(6)].map((_, i) => (
                  <div key={i} className="bg-white rounded-lg border border-gray-200 shadow-sm p-8 animate-pulse">
                    <div className="flex items-start justify-between mb-6">
                      <div className="flex-1">
                        <div className="h-6 bg-gray-200 rounded w-3/4 mb-2"></div>
                        <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                      </div>
                      <div className="h-6 bg-gray-200 rounded w-16"></div>
                    </div>
                    <div className="space-y-2 mb-6">
                      <div className="h-4 bg-gray-200 rounded"></div>
                      <div className="h-4 bg-gray-200 rounded w-5/6"></div>
                    </div>
                    <div className="space-y-3 mb-8">
                      <div className="flex justify-between">
                        <div className="h-4 bg-gray-200 rounded w-20"></div>
                        <div className="h-4 bg-gray-200 rounded w-8"></div>
                      </div>
                      <div className="flex justify-between">
                        <div className="h-4 bg-gray-200 rounded w-16"></div>
                        <div className="h-4 bg-gray-200 rounded w-12"></div>
                      </div>
                    </div>
                    <div className="flex space-x-3">
                      <div className="flex-1 h-10 bg-gray-200 rounded-lg"></div>
                      <div className="h-10 w-10 bg-gray-200 rounded-lg"></div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {getFilteredTidps().map((tidp, index) => (
                  <div key={tidp.id || index} className="bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-lg hover:border-blue-300 transition-all duration-200 group">
                    <div className="p-8">
                      <div className="flex items-start justify-between mb-6">
                        <div className="flex-1">
                          <h3 className="text-xl font-bold text-gray-900 mb-2 group-hover:text-blue-700 transition-colors">{tidp.teamName || `TIDP ${index + 1}`}</h3>
                          <p className="text-gray-600 font-medium">{tidp.discipline}</p>
                        </div>
                        <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold bg-blue-100 text-blue-800">
                          {tidp.status || 'Draft'}
                        </span>
                      </div>

                      <p className="text-gray-700 text-base mb-6 line-clamp-3 leading-relaxed">
                        {tidp.description || tidp.responsibilities || 'Task information delivery plan for team coordination and deliverables management.'}
                      </p>

                      <div className="space-y-3 mb-8">
                        <div className="flex justify-between items-center">
                          <span className="text-gray-600 font-medium">Deliverables:</span>
                          <span className="font-bold text-gray-900">{tidp.containers?.length || 0}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-gray-600 font-medium">Team Leader:</span>
                          <span className="font-bold text-gray-900">{tidp.leader || 'TBD'}</span>
                        </div>
                      </div>

                      <div className="flex space-x-3">
                        <button className="flex-1 bg-blue-600 text-white font-semibold py-3 px-4 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 hover:shadow-md">
                          View Details
                        </button>
                        <button className="bg-gray-100 text-gray-700 p-3 rounded-lg hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-all duration-200">
                          <Download className="w-5 h-5" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {getFilteredTidps().length === 0 && (
              <div className="text-center py-16">
                <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-8">
                  <Users className="w-10 h-10 text-gray-400" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4">No TIDPs found</h3>
                <p className="text-gray-600 text-lg mb-8 max-w-md mx-auto">Try adjusting your search terms or filters, or create a new TIDP to get started.</p>
                <button
                  onClick={() => navigate('/tidp-editor')}
                  className="inline-flex items-center px-8 py-4 bg-blue-600 text-white font-semibold text-lg rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 hover:shadow-lg"
                >
                  <Plus className="w-6 h-6 mr-3" />
                  Create New TIDP
                </button>
              </div>
            )}
          </div>
        )}

        {/* MIDPs View */}
        {activeView === 'midps' && (
          <div className="space-y-6">
            {/* MIDPs Header */}
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-8 mb-8">
              <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">Master Information Delivery Plans</h2>
                  <p className="text-gray-600 text-lg">Aggregated project delivery schedules from multiple team plans</p>
                </div>
                <button
                  onClick={autoGenerateMIDP}
                  className="inline-flex items-center px-8 py-4 border border-transparent rounded-lg shadow-sm text-base font-semibold text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-all duration-200 hover:shadow-md lg:shrink-0"
                >
                  <Plus className="w-5 h-5 mr-3" />
                  Generate MIDP
                </button>
              </div>
            </div>

            {/* MIDPs Grid */}
            {loading ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {[...Array(4)].map((_, i) => (
                  <div key={i} className="bg-white rounded-lg border border-gray-200 shadow-sm p-8 animate-pulse">
                    <div className="flex items-start justify-between mb-6">
                      <div className="flex-1">
                        <div className="h-6 bg-gray-200 rounded w-3/4 mb-2"></div>
                        <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                      </div>
                      <div className="h-6 bg-gray-200 rounded w-16"></div>
                    </div>
                    <div className="h-4 bg-gray-200 rounded mb-6"></div>
                    <div className="space-y-3 mb-8">
                      <div className="flex justify-between">
                        <div className="h-4 bg-gray-200 rounded w-24"></div>
                        <div className="h-4 bg-gray-200 rounded w-8"></div>
                      </div>
                      <div className="flex justify-between">
                        <div className="h-4 bg-gray-200 rounded w-28"></div>
                        <div className="h-4 bg-gray-200 rounded w-12"></div>
                      </div>
                      <div className="flex justify-between">
                        <div className="h-4 bg-gray-200 rounded w-20"></div>
                        <div className="h-4 bg-gray-200 rounded w-16"></div>
                      </div>
                    </div>
                    <div className="flex space-x-3">
                      <div className="flex-1 h-10 bg-gray-200 rounded-lg"></div>
                      <div className="h-10 w-10 bg-gray-200 rounded-lg"></div>
                      <div className="h-10 w-10 bg-gray-200 rounded-lg"></div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {getFilteredMidps().map((midp, index) => (
                  <div key={midp.id || index} className="bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-lg hover:border-green-300 transition-all duration-200 group">
                    <div className="p-8">
                      <div className="flex items-start justify-between mb-6">
                        <div className="flex-1">
                          <h3 className="text-xl font-bold text-gray-900 mb-2 group-hover:text-green-700 transition-colors">{midp.projectName || `MIDP ${index + 1}`}</h3>
                          <p className="text-gray-600 font-medium">{midp.includedTIDPs?.length || 0} TIDPs included</p>
                        </div>
                        <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold bg-green-100 text-green-800">
                          {midp.status || 'Active'}
                        </span>
                      </div>

                      <p className="text-gray-700 text-base mb-6 leading-relaxed">
                        {midp.description || 'Master information delivery plan aggregating multiple team plans for comprehensive project coordination.'}
                      </p>

                      <div className="space-y-3 mb-8">
                        <div className="flex justify-between items-center">
                          <span className="text-gray-600 font-medium">Total Deliverables:</span>
                          <span className="font-bold text-gray-900">{midp.aggregatedData?.totalContainers || 0}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-gray-600 font-medium">Estimated Hours:</span>
                          <span className="font-bold text-gray-900">{midp.aggregatedData?.totalEstimatedHours || 0}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-gray-600 font-medium">Last Updated:</span>
                          <span className="font-bold text-gray-900">{new Date(midp.updatedAt).toLocaleDateString()}</span>
                        </div>
                      </div>

                      <div className="flex space-x-3">
                        <button className="flex-1 bg-green-600 text-white font-semibold py-3 px-4 rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-all duration-200 hover:shadow-md">
                          View Details
                        </button>
                        <button
                          onClick={() => setShowEvolutionDashboard(midp.id)}
                          className="bg-blue-600 text-white p-3 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200"
                          title="Evolution Dashboard"
                        >
                          <TrendingUp className="w-5 h-5" />
                        </button>
                        <button className="bg-gray-100 text-gray-700 p-3 rounded-lg hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-all duration-200">
                          <Download className="w-5 h-5" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {getFilteredMidps().length === 0 && (
              <div className="text-center py-16">
                <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-8">
                  <Calendar className="w-10 h-10 text-gray-400" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4">No MIDPs created yet</h3>
                <p className="text-gray-600 text-lg mb-8 max-w-md mx-auto">Generate your first Master Information Delivery Plan by aggregating existing TIDPs.</p>
                <button
                  onClick={autoGenerateMIDP}
                  className="inline-flex items-center px-8 py-4 bg-green-600 text-white font-semibold text-lg rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-all duration-200 hover:shadow-lg"
                >
                  <Plus className="w-6 h-6 mr-3" />
                  Generate First MIDP
                </button>
              </div>
            )}
          </div>
        )}

        {/* Import View */}
        {activeView === 'import' && (
          <div className="max-w-4xl mx-auto">
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-12">
              <div className="text-center mb-12">
                <div className="w-24 h-24 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-8">
                  <Upload className="w-12 h-12 text-blue-600" />
                </div>
                <h2 className="text-3xl font-bold text-gray-900 mb-4">Import TIDPs</h2>
                <p className="text-gray-600 text-lg max-w-2xl mx-auto leading-relaxed">
                  Import TIDP data from Excel or CSV files created by external teams. 
                  This allows seamless integration of team plans from various sources.
                </p>
              </div>

              <div className="space-y-8">
                <button
                  onClick={() => setShowImportDialog(true)}
                  className="w-full flex items-center justify-center px-8 py-6 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 group"
                >
                  <div className="text-center">
                    <Upload className="w-8 h-8 mx-auto mb-4 text-gray-400 group-hover:text-blue-500 transition-colors" />
                    <span className="text-xl font-semibold text-gray-600 group-hover:text-blue-700 transition-colors">Import from Excel/CSV</span>
                    <p className="text-gray-500 mt-2">Click to select and upload your files</p>
                  </div>
                </button>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <div className="p-6 border border-gray-200 rounded-lg bg-gray-50">
                    <h3 className="font-bold text-gray-900 text-lg mb-4">Supported Formats</h3>
                    <ul className="text-gray-700 space-y-2">
                      <li className="flex items-center">
                        <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                        Excel (.xlsx, .xls)
                      </li>
                      <li className="flex items-center">
                        <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                        CSV (.csv)
                      </li>
                      <li className="flex items-center">
                        <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                        UTF-8 encoding recommended
                      </li>
                    </ul>
                  </div>

                  <div className="p-6 border border-gray-200 rounded-lg bg-blue-50">
                    <h3 className="font-bold text-gray-900 text-lg mb-4">What's Imported</h3>
                    <ul className="text-gray-700 space-y-2">
                      <li className="flex items-center">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mr-3"></div>
                        Team information
                      </li>
                      <li className="flex items-center">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mr-3"></div>
                        Deliverable containers
                      </li>
                      <li className="flex items-center">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mr-3"></div>
                        Schedules and milestones
                      </li>
                      <li className="flex items-center">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mr-3"></div>
                        Dependencies
                      </li>
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

      {/* Help Modal */}
      {showHelp && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-gray-900">Relationship between TIDP and MIDP</h2>
                <button
                  onClick={() => setShowHelp(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>
              <div className="prose max-w-none">
                <p>In the context of ISO 19650, TIDPs (Task Information Delivery Plans) and the MIDP (Master Information Delivery Plan) are key elements for planning information delivery in a BIM project.</p>
                
                <h3>TIDP (Task Information Delivery Plan)</h3>
                <p>These are detailed plans prepared by each team or task team involved in the project. Each TIDP describes the specific information that team must produce, including deliverables (models, documents, data), delivery milestones, responsibilities (who does what), formats, and levels of detail (LOD/LOI). It is focused on individual or subgroup activities, and derives from the detailed responsibility matrix (Detailed Responsibility Matrix).</p>
                
                <h3>MIDP (Master Information Delivery Plan)</h3>
                <p>It is the overall master plan of the project, which integrates and coordinates all TIDPs from the various teams. The MIDP acts as the "main calendar" that aligns delivery timelines, ensures consistency between team contributions, and includes details such as task dependencies, revisions, approvals, and integration into the CDE. It is produced by the lead appointee (for example, the project information manager) and evolves during the project.</p>
                
                <h3>Relationship</h3>
                <p>The MIDP is essentially a collation and harmonization of the individual TIDPs. TIDPs provide granular details for each team, while the MIDP unites them into a unified framework for the entire project, ensuring that deliveries are synchronized with project milestones (for example, design, construction, or handover phases). This relationship is hierarchical: TIDPs feed the MIDP, which in turn informs the BIM Execution Plan (BEP) and supports verification of compliance with client-required information. In practice, a delay in one TIDP can impact the entire MIDP, thus promoting proactive collaboration.</p>
              </div>
            </div>
          </div>
        </div>
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