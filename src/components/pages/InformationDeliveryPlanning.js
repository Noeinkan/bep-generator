import React, { useState, useEffect, useCallback } from 'react';
import { Calendar, Plus, FileText, Download, RefreshCw, AlertCircle, CheckCircle, Users, Target } from 'lucide-react';
import ApiService from '../../services/apiService';
import TidpMidpManager from './TidpMidpManager';
import ExcelTIDPEditor from '../ExcelTIDPEditor';
import TipTapEditor from '../forms/TipTapEditor';
import IntroTableField from '../forms/EditableTable/IntroTableField';
import FieldHelpTooltip from '../forms/FieldHelpTooltip';
import HELP_CONTENT from '../../data/helpContentData';

const InformationDeliveryPlanning = ({ formData, updateFormData, errors, bepType }) => {
  const [tidps, setTidps] = useState([]);
  const [midps, setMidps] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  // selection state is handled inside the inline manager; remove unused local state
  const [serverConnected, setServerConnected] = useState(false);
  const [showManager, setShowManager] = useState(false);

  // Check server connection on mount
  const loadTidpsAndMidps = useCallback(async () => {
    if (!serverConnected) return;

    setLoading(true);
    try {
      const [tidpData, midpData] = await Promise.all([
        ApiService.getAllTIDPs(formData.projectName || 'current'),
        ApiService.getAllMIDPs()
      ]);
      setTidps(tidpData.tidps || []);
      setMidps(midpData.midps || []);
    } catch (error) {
      console.error('Failed to load TIDP/MIDP data:', error);
    } finally {
      setLoading(false);
    }
  }, [serverConnected, formData.projectName]);

  const checkServerConnection = useCallback(async () => {
    try {
      await ApiService.healthCheck();
      setServerConnected(true);
      loadTidpsAndMidps();
    } catch (error) {
      console.log('TIDP/MIDP server not available - showing basic form interface');
      setServerConnected(false);
    }
  }, [loadTidpsAndMidps]);

  useEffect(() => {
    checkServerConnection();
  }, [checkServerConnection]);

  // ...existing code...

  // loadTidpsAndMidps is defined above using useCallback

  const createNewTidp = () => {
    setActiveTab('tidp-form');
    // Open inline manager for creation
    setShowManager(true);
  };

  const createNewMidp = () => {
    setActiveTab('midp-form');
    // Open inline manager for creation
    setShowManager(true);
  };

  const BasicFormInterface = () => (
    <div className="space-y-6">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <AlertCircle className="w-6 h-6 text-blue-600" />
            <div>
              <h4 className="font-medium text-blue-900">Information Delivery Planning</h4>
              <p className="text-blue-700 text-sm mt-1">
                Complete basic requirements or access advanced TIDP/MIDP management
              </p>
            </div>
          </div>
          <button
            onClick={() => setShowManager(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition-colors flex items-center space-x-2 font-medium"
          >
            <Calendar className="w-5 h-5" />
            <span>Open TIDP/MIDP Manager</span>
          </button>
        </div>
      </div>

      <div className="space-y-4">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <label className="block text-sm font-medium text-gray-700">
              Master Information Delivery Plan (MIDP) <span className="text-red-500">*</span>
            </label>
            {HELP_CONTENT.midpDescription && (
              <FieldHelpTooltip fieldName="midpDescription" helpContent={HELP_CONTENT.midpDescription} />
            )}
          </div>
          <TipTapEditor
            value={formData.midpDescription || ''}
            onChange={(html) => updateFormData('midpDescription', html)}
            placeholder="The MIDP establishes a structured schedule for information delivery aligned with RIBA Plan of Work 2020 stages. Key deliverables include: Stage 3 coordinated federated models by Month 8, Stage 4 construction-ready models with full MEP coordination by Month 14..."
            className={errors.midpDescription ? 'border-red-300' : ''}
            minHeight="150px"
            fieldName="midpDescription"
            autoSaveKey={`midp-description-${formData.projectName || 'default'}`}
          />
          {errors.midpDescription && (
            <p className="text-red-500 text-sm mt-1">{errors.midpDescription}</p>
          )}
        </div>

        <div>
          <div className="flex items-center gap-2 mb-2">
            <label className="block text-sm font-medium text-gray-700">
              Task Information Delivery Plans (TIDPs)
            </label>
            {HELP_CONTENT.tidpRequirements && (
              <FieldHelpTooltip fieldName="tidpRequirements" helpContent={HELP_CONTENT.tidpRequirements} />
            )}
          </div>
          <TipTapEditor
            value={formData.tidpRequirements || ''}
            onChange={(html) => updateFormData('tidpRequirements', html)}
            placeholder="TIDPs define discipline-specific delivery requirements: Architecture TIDP delivers spatial models and specification schedules biweekly, Structural TIDP provides analysis models and connection details monthly..."
            minHeight="120px"
            fieldName="tidpRequirements"
            autoSaveKey={`tidp-requirements-${formData.projectName || 'default'}`}
          />
        </div>

        <div>
          <div className="flex items-center gap-2 mb-2">
            <label className="block text-sm font-medium text-gray-700">
              Key Information Delivery Milestones <span className="text-red-500">*</span>
            </label>
            {HELP_CONTENT.keyMilestones && (
              <FieldHelpTooltip fieldName="keyMilestones" helpContent={HELP_CONTENT.keyMilestones} />
            )}
          </div>
          <IntroTableField
            field={{
              name: 'keyMilestones',
              label: '',
              required: true,
              type: 'introTable',
              introPlaceholder: 'The following milestones represent key information delivery points throughout the project:',
              tableColumns: ['Stage/Phase', 'Milestone Description', 'Deliverables', 'Due Date']
            }}
            value={formData.keyMilestones}
            onChange={(name, value) => updateFormData(name, value)}
            error={errors.keyMilestones}
          />
        </div>
      </div>
    </div>
  );

  const AdvancedInterface = () => (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="flex space-x-8">
          {[
            { id: 'overview', label: 'Overview', icon: Target },
            { id: 'tidps', label: 'TIDPs', icon: Users },
            { id: 'midps', label: 'MIDPs', icon: Calendar },
            { id: 'tidp-form', label: 'New TIDP', icon: Plus },
            { id: 'midp-form', label: 'New MIDP', icon: Plus }
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <Icon className="w-4 h-4 inline mr-2" />
              {label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="min-h-[400px]">
        {activeTab === 'overview' && <OverviewTab />}
        {activeTab === 'tidps' && <TidpsTab />}
        {activeTab === 'midps' && <MidpsTab />}
        {activeTab === 'tidp-form' && <TidpFormTab />}
        {activeTab === 'midp-form' && <MidpFormTab />}
      </div>
    </div>
  );

  const OverviewTab = () => (
    <div className="space-y-6">
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <div className="flex items-center space-x-2">
          <CheckCircle className="w-5 h-5 text-green-600" />
          <h4 className="font-medium text-green-900">TIDP/MIDP Server Connected</h4>
        </div>
        <p className="text-green-700 text-sm mt-2">
          Advanced TIDP/MIDP management is available. You can create, edit, and manage Task Information Delivery Plans and Master Information Delivery Plans.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">TIDPs</h3>
              <p className="text-3xl font-bold text-blue-600">{tidps.length}</p>
            </div>
            <Users className="w-8 h-8 text-blue-600" />
          </div>
          <p className="text-gray-600 text-sm mt-2">Task Information Delivery Plans</p>
          <button
            onClick={createNewTidp}
            className="mt-4 w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors"
            id="create-tidp-overview"
            data-testid="create-tidp-overview"
            aria-label="Create TIDP (overview)"
          >
            Create TIDP
          </button>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">MIDPs</h3>
              <p className="text-3xl font-bold text-green-600">{midps.length}</p>
            </div>
            <Calendar className="w-8 h-8 text-green-600" />
          </div>
          <p className="text-gray-600 text-sm mt-2">Master Information Delivery Plans</p>
          <button
            onClick={createNewMidp}
            className="mt-4 w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 transition-colors"
          >
            Create MIDP
          </button>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Project</h3>
              <p className="text-lg font-medium text-gray-900">{formData.projectName || 'No Project'}</p>
            </div>
            <Target className="w-8 h-8 text-purple-600" />
          </div>
          <p className="text-gray-600 text-sm mt-2">Current BEP Project</p>
          <button
            onClick={loadTidpsAndMidps}
            disabled={loading}
            className="mt-4 w-full bg-purple-600 text-white py-2 px-4 rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin inline mr-2" /> : <RefreshCw className="w-4 h-4 inline mr-2" />}
            Refresh
          </button>
        </div>
      </div>
    </div>
  );

  const TidpsTab = () => (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Task Information Delivery Plans</h3>
        <button
          onClick={createNewTidp}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
          id="create-tidp-tidps-tab"
          data-testid="create-tidp-tidps-tab"
          aria-label="Create TIDP (TIDPs tab)"
        >
          <Plus className="w-4 h-4" />
          <span>New TIDP</span>
        </button>
      </div>

      {tidps.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <Users className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <p>No TIDPs created yet</p>
          <p className="text-sm">Create your first Task Information Delivery Plan</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {tidps.map((tidp) => (
            <TidpCard key={tidp.id} tidp={tidp} onSelect={() => setShowManager(true)} />
          ))}
        </div>
      )}
    </div>
  );

  const MidpsTab = () => (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Master Information Delivery Plans</h3>
        <button
          onClick={createNewMidp}
          className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors flex items-center space-x-2"
        >
          <Plus className="w-4 h-4" />
          <span>New MIDP</span>
        </button>
      </div>

      {midps.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <Calendar className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <p>No MIDPs created yet</p>
          <p className="text-sm">Create your first Master Information Delivery Plan</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {midps.map((midp) => (
            <MidpCard key={midp.id} midp={midp} onSelect={() => setShowManager(true)} />
          ))}
        </div>
      )}
    </div>
  );

  const TidpFormTab = () => {
    const [showExcelEditor, setShowExcelEditor] = useState(false);

    if (showExcelEditor) {
      return <ExcelTIDPEditor onClose={() => setShowExcelEditor(false)} onSave={(tidpData) => {
        // Handle save logic here
        console.log('TIDP saved:', tidpData);
        setShowExcelEditor(false);
        // You could call an API to save the TIDP
      }} />;
    }

    return (
      <div className="max-w-2xl">
        <h3 className="text-lg font-semibold mb-4">Create New TIDP</h3>
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
          <p className="text-gray-600 mb-4">
            Open the Excel-style TIDP editor for intuitive creation and management of Task Information Delivery Plans.
          </p>
          <button
            onClick={() => setShowExcelEditor(true)}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
          >
            <FileText className="w-4 h-4" />
            <span>Open Excel TIDP Editor</span>
          </button>
        </div>
      </div>
    );
  };

  const MidpFormTab = () => (
    <div className="max-w-2xl">
      <h3 className="text-lg font-semibold mb-4">Create New MIDP</h3>
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
        <p className="text-gray-600 mb-4">
          Use the dedicated MIDP creation form in the advanced interface. This will open in a new window with full MIDP management capabilities.
        </p>
        <button
          onClick={() => window.open('/tidp-midp-manager', '_blank', 'width=1200,height=800')}
          className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition-colors flex items-center space-x-2"
        >
          <Calendar className="w-4 h-4" />
          <span>Open MIDP Manager</span>
        </button>
      </div>
    </div>
  );

  const TidpCard = ({ tidp, onSelect }) => (
    <div className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start mb-2">
        <h4 className="font-medium text-gray-900">{tidp.taskTeam}</h4>
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
          tidp.status === 'active' ? 'bg-green-100 text-green-800' :
          tidp.status === 'draft' ? 'bg-yellow-100 text-yellow-800' :
          'bg-gray-100 text-gray-800'
        }`}>
          {tidp.status}
        </span>
      </div>
      <p className="text-sm text-gray-600 mb-3">{tidp.discipline}</p>
      <div className="text-xs text-gray-500 mb-3">
        <p>Leader: {tidp.teamLeader}</p>
        <p>Due: {new Date(tidp.deliveryDates?.milestones?.[0]?.date || Date.now()).toLocaleDateString()}</p>
      </div>
      <div className="flex space-x-2">
        <button
          onClick={() => onSelect(tidp)}
          className="flex-1 bg-blue-50 text-blue-700 py-1 px-3 rounded text-sm hover:bg-blue-100 transition-colors"
        >
          View
        </button>
        <button className="bg-gray-50 text-gray-700 py-1 px-3 rounded text-sm hover:bg-gray-100 transition-colors">
          <Download className="w-4 h-4" />
        </button>
      </div>
    </div>
  );

  const MidpCard = ({ midp, onSelect }) => (
    <div className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start mb-2">
        <h4 className="font-medium text-gray-900">{midp.projectName}</h4>
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
          midp.status === 'active' ? 'bg-green-100 text-green-800' :
          midp.status === 'draft' ? 'bg-yellow-100 text-yellow-800' :
          'bg-gray-100 text-gray-800'
        }`}>
          {midp.status}
        </span>
      </div>
      <p className="text-sm text-gray-600 mb-3">Version: {midp.version}</p>
      <div className="text-xs text-gray-500 mb-3">
        <p>TIDPs: {midp.aggregatedTidps?.length || 0}</p>
        <p>Updated: {new Date(midp.lastUpdated).toLocaleDateString()}</p>
      </div>
      <div className="flex space-x-2">
        <button
          onClick={() => onSelect(midp)}
          className="flex-1 bg-green-50 text-green-700 py-1 px-3 rounded text-sm hover:bg-green-100 transition-colors"
        >
          View
        </button>
        <button className="bg-gray-50 text-gray-700 py-1 px-3 rounded text-sm hover:bg-gray-100 transition-colors">
          <Download className="w-4 h-4" />
        </button>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Mark page URI for debugging and testing */}
      <div data-page-uri="/information-delivery-planning" />
      <div className="flex items-center space-x-3 mb-6">
        <Calendar className="w-6 h-6 text-blue-600" />
        <h2 className="text-2xl font-bold text-gray-900">Information Delivery Planning</h2>
        {serverConnected && (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
            <div className="w-2 h-2 bg-green-600 rounded-full mr-1.5"></div>
            Server Connected
          </span>
        )}
      </div>

      {showManager ? (
        <TidpMidpManager 
          onClose={() => setShowManager(false)} 
          initialShowTidpForm={activeTab === 'tidp-form'} 
          initialShowMidpForm={activeTab === 'midp-form'} 
        />
      ) : (
        serverConnected ? <AdvancedInterface /> : <BasicFormInterface />
      )}
    </div>
  );
};

export default InformationDeliveryPlanning;