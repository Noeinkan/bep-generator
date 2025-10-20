import OrgStructureField from '../specialized/OrgStructureField';
import OrgStructureDataTable from '../specialized/OrgStructureDataTable';
import IMStrategyBuilder from '../custom/IMStrategyBuilder';
import NamingConventionBuilder from '../custom/NamingConventionBuilder';
// ...existing code...
import React, { useState } from 'react';
import CONFIG from '../../../config/bepConfig';
import EditableTable from './EditableTable';
import IntroTableField from './IntroTableField';
import FieldHeader from './FieldHeader';
import CheckboxGroup from './CheckboxGroup';
import FolderStructureDiagram from '../diagrams/FolderStructureDiagram';
import CDEDiagramBuilderV2 from '../diagrams/CDEDiagramBuilder';
import VolumeStrategyMindmap from '../diagrams/VolumeStrategyMindmap';
import TipTapEditor from '../editors/TipTapEditor';
import TimelineInput from '../specialized/TimelineInput';
import BudgetInput from '../specialized/BudgetInput';
import StandardsTable from '../tables/StandardsTable';
import { Calendar, Plus, Table2, FileText } from 'lucide-react';

const InputField = React.memo(({ field, value, onChange, error, formData = {} }) => {
  const { name, label, number, type, required, rows, placeholder, options: fieldOptions } = field;
  const optionsList = fieldOptions ? CONFIG.options[fieldOptions] : null;

  const baseClasses = "w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500";

  const handleCheckboxChange = (option) => {
    const current = Array.isArray(value) ? value : [];
    const updated = current.includes(option)
      ? current.filter(item => item !== option)
      : [...current, option];
    onChange(name, updated);
  };

  switch (type) {
    case 'timeline':
      return (
        <TimelineInput
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'budget':
      return (
        <BudgetInput
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'orgchart':
      return (
        <OrgStructureField
          field={field}
          value={value}
          onChange={(v) => {
            // If the org chart component emits an object with leadAppointedParty and finalizedParties,
            // persist those into separate form fields expected elsewhere in the app.
            if (v && typeof v === 'object') {
              if (v.leadAppointedParty !== undefined) {
                onChange('leadAppointedParty', v.leadAppointedParty);
              }
              if (v.finalizedParties !== undefined) {
                onChange('finalizedParties', v.finalizedParties);
              }
              // Also keep the organizationalStructure field (the org tree) for compatibility
              if (v.tree !== undefined) {
                onChange(name, v.tree);
              } else {
                onChange(name, v);
              }
            } else {
              onChange(name, v);
            }
          }}
          formData={formData}
        />
      );
    
    case 'orgstructure-data-table':
      return (
        <OrgStructureDataTable
          field={field}
          value={value}
          formData={formData}
        />
      );
    
    case 'table':
      return (
        <EditableTable
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'standardsTable':
      return (
        <StandardsTable
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'introTable':
      return (
        <IntroTableField
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'fileStructure':
      return (
        <FolderStructureDiagram
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'cdeDiagram':
      return (
        <CDEDiagramBuilderV2
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'mindmap':
      return (
        <VolumeStrategyMindmap
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'im-strategy-builder':
      return (
        <IMStrategyBuilder
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'naming-conventions':
      return (
        <NamingConventionBuilder
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'textarea':
      return (
        <div>
          <FieldHeader 
            fieldName={name}
            label={label}
            number={number}
            required={required}
            htmlFor={name}
          />
          <TipTapEditor
            id={name}
            aria-required={required}
            value={value || ''}
            onChange={(newValue) => onChange(name, newValue)}
            className=""
            placeholder={placeholder || `Enter ${label.toLowerCase()}...`}
            minHeight={`${(rows || 3) * 24}px`}
            autoSaveKey={`tiptap-${name}`}
            fieldName={name}
          />
          {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
        </div>
      );

    case 'select':
      return (
        <div>
          <FieldHeader 
            fieldName={name}
            label={label}
            number={number}
            required={required}
            htmlFor={name}
          />
          <select
            id={name}
            aria-required={required}
            value={value || ''}
            onChange={(e) => onChange(name, e.target.value)}
            className={baseClasses}
          >
            <option value="">Select {label.toLowerCase()}</option>
            {optionsList?.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
          {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
        </div>
      );

    case 'checkbox':
      return (
        <CheckboxGroup
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'section-header':
      return (
        <div className="mb-4">
          <h4 className="text-lg font-semibold text-gray-900 border-b-2 border-gray-300 pb-2">
            {number && <span className="text-blue-600">{number} </span>}
            {label}
          </h4>
        </div>
      );

    // Step 5 specialized field types
    case 'milestones-table':
      return <MilestonesTableField field={field} value={value} onChange={onChange} error={error} />;

    case 'tidp-reference':
      return <TidpReferenceField field={field} value={value} onChange={onChange} error={error} formData={formData} />;

    case 'tidp-section':
      return <TidpSectionField field={field} value={value} onChange={onChange} error={error} formData={formData} />;

    case 'deliverables-matrix':
      return <DeliverablesMatrixField field={field} value={value} onChange={onChange} error={error} formData={formData} />;

    case 'im-activities-matrix':
      return <ImActivitiesMatrixField field={field} value={value} onChange={onChange} error={error} formData={formData} />;

    default:
      return (
        <div>
          <FieldHeader 
            fieldName={name}
            label={label}
            number={number}
            required={required}
            htmlFor={name}
          />
          <input
            id={name}
            aria-required={required}
            type="text"
            value={value || ''}
            onChange={(e) => onChange(name, e.target.value)}
            className={baseClasses}
            placeholder={placeholder || `Enter ${label.toLowerCase()}`}
          />
          {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
        </div>
      );
  }
});

// Step 5 Specialized Components

const MilestonesTableField = ({ field, value, onChange, error }) => {
  const { name, label, number, required } = field;
  const milestones = value || [];

  const addMilestone = () => {
    onChange(name, [...milestones, { stage: '', description: '', deliverables: '', dueDate: '' }]);
  };

  const updateMilestone = (index, key, newValue) => {
    const updated = [...milestones];
    updated[index] = { ...updated[index], [key]: newValue };
    onChange(name, updated);
  };

  const removeMilestone = (index) => {
    onChange(name, milestones.filter((_, i) => i !== index));
  };

  return (
    <div>
      <FieldHeader 
        fieldName={name}
        label={label}
        number={number}
        required={required}
      />
      <div className="overflow-x-auto">
        <table className="min-w-full border border-gray-300 rounded-lg">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-2 text-left text-sm font-medium text-gray-700 border-b">Stage/Phase</th>
              <th className="px-4 py-2 text-left text-sm font-medium text-gray-700 border-b">Milestone Description</th>
              <th className="px-4 py-2 text-left text-sm font-medium text-gray-700 border-b">Deliverables</th>
              <th className="px-4 py-2 text-left text-sm font-medium text-gray-700 border-b">Due Date</th>
              <th className="px-4 py-2 text-left text-sm font-medium text-gray-700 border-b">Actions</th>
            </tr>
          </thead>
          <tbody>
            {milestones.map((milestone, index) => (
              <tr key={index} className="border-b">
                <td className="px-4 py-2">
                  <input
                    type="text"
                    value={milestone.stage || ''}
                    onChange={(e) => updateMilestone(index, 'stage', e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded"
                    placeholder="Stage 3"
                  />
                </td>
                <td className="px-4 py-2">
                  <input
                    type="text"
                    value={milestone.description || ''}
                    onChange={(e) => updateMilestone(index, 'description', e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded"
                    placeholder="Coordinated Federated Models"
                  />
                </td>
                <td className="px-4 py-2">
                  <input
                    type="text"
                    value={milestone.deliverables || ''}
                    onChange={(e) => updateMilestone(index, 'deliverables', e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded"
                    placeholder="Architecture, Structure, MEP Models"
                  />
                </td>
                <td className="px-4 py-2">
                  <input
                    type="date"
                    value={milestone.dueDate || ''}
                    onChange={(e) => updateMilestone(index, 'dueDate', e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded"
                  />
                </td>
                <td className="px-4 py-2">
                  <button
                    type="button"
                    onClick={() => removeMilestone(index)}
                    className="text-red-600 hover:text-red-800"
                  >
                    Remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <button
          type="button"
          onClick={addMilestone}
          className="mt-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
        >
          <Plus className="w-4 h-4" />
          <span>Add Milestone</span>
        </button>
      </div>
      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
};

const TidpReferenceField = ({ field, value, onChange, error, formData }) => {
  const { label, number } = field;
  const [showManager, setShowManager] = useState(false);
  const [tidps, setTidps] = useState([]);
  const [loading, setLoading] = useState(false);

  // Lazy load components and hooks
  const TidpMidpManager = React.lazy(() => import('../../pages/tidp-midp/TidpMidpManager'));

  // Load TIDPs function
  const loadTidps = React.useCallback(async () => {
    try {
      setLoading(true);
      // Dynamic import of API service
      const ApiService = (await import('../../../services/apiService')).default;
      const response = await ApiService.getAllTIDPs(formData.projectName || 'current');
      setTidps(response.tidps || []);
    } catch (error) {
      console.log('TIDPs not available:', error);
      setTidps([]);
    } finally {
      setLoading(false);
    }
  }, [formData.projectName]);

  // Load TIDPs on mount
  React.useEffect(() => {
    loadTidps();
  }, [loadTidps]);

  return (
    <div>
      <FieldHeader 
        fieldName={field.name}
        label={label}
        number={number}
        required={field.required}
      />

      {showManager ? (
        <React.Suspense fallback={<div className="p-4 text-center">Loading TIDP Manager...</div>}>
          <TidpMidpManager
            onClose={() => {
              setShowManager(false);
              // Reload TIDPs after closing manager
              loadTidps();
            }}
            initialShowTidpForm={false}
          />
        </React.Suspense>
      ) : (
        <div className="space-y-4">
          {/* TIDPs List */}
          <div className="bg-gradient-to-r from-indigo-50 to-blue-50 border-2 border-indigo-200 rounded-lg p-6">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-indigo-600 rounded-lg flex items-center justify-center">
                  <Calendar className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">Task Information Delivery Plans</h4>
                  <p className="text-sm text-gray-600">
                    {loading ? 'Loading TIDPs...' : `${tidps.length} TIDP${tidps.length !== 1 ? 's' : ''} created`}
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setShowManager(true)}
                className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center space-x-2 font-medium"
              >
                <Plus className="w-4 h-4" />
                <span>Create TIDP</span>
              </button>
            </div>

            {/* List of existing TIDPs */}
            {tidps.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mt-4">
                {tidps.map((tidp) => (
                  <div
                    key={tidp.id}
                    className="bg-white border border-indigo-200 rounded-lg p-3 hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex-1">
                        <h5 className="font-medium text-sm text-gray-900 truncate">{tidp.taskTeam}</h5>
                        <p className="text-xs text-gray-600">{tidp.discipline}</p>
                      </div>
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                        tidp.status === 'active' ? 'bg-green-100 text-green-800' :
                        tidp.status === 'draft' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {tidp.status}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500">
                      <p>Leader: {tidp.teamLeader}</p>
                      <p>Containers: {tidp.containers?.length || 0}</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              !loading && (
                <div className="text-center py-6 text-gray-500">
                  <p className="text-sm">No TIDPs created yet</p>
                  <p className="text-xs mt-1">Click "Create TIDP" to add your first Task Information Delivery Plan</p>
                </div>
              )
            )}
          </div>

          {/* Button to open full manager */}
          <button
            type="button"
            onClick={() => setShowManager(true)}
            className="w-full bg-white border-2 border-indigo-300 text-indigo-700 px-6 py-3 rounded-lg hover:bg-indigo-50 transition-colors flex items-center justify-center space-x-2 font-medium"
          >
            <Calendar className="w-5 h-5" />
            <span>Open TIDP/MIDP Manager Dashboard</span>
          </button>
        </div>
      )}
      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
};

const TidpSectionField = ({ field, value, onChange, error, formData }) => {
  const { name, label, number, placeholder } = field;
  const [showManager, setShowManager] = useState(false);

  // Lazy load the manager component
  const TidpMidpManager = React.lazy(() => import('../../pages/tidp-midp/TidpMidpManager'));

  return (
    <div>
      <FieldHeader 
        fieldName={name}
        label={label}
        number={number}
        required={field.required}
      />

      {showManager ? (
        <React.Suspense fallback={<div>Loading...</div>}>
          <TidpMidpManager
            onClose={() => setShowManager(false)}
            initialShowTidpForm={true}
          />
        </React.Suspense>
      ) : (
        <>
          <textarea
            value={value || ''}
            onChange={(e) => onChange(name, e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={3}
            placeholder={placeholder}
          />
          <button
            type="button"
            onClick={() => setShowManager(true)}
            className="mt-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition-colors flex items-center space-x-2 font-medium"
          >
            <Calendar className="w-5 h-5" />
            <span>Open TIDP/MIDP Manager</span>
          </button>
        </>
      )}
      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
};

const DeliverablesMatrixField = ({ field, value, onChange, error, formData }) => {
  const { label, number, placeholder } = field;
  const [showMatrixManager, setShowMatrixManager] = useState(false);

  // Lazy load the manager component
  const ResponsibilityMatrixManager = React.lazy(() => import('../../responsibility-matrix/ResponsibilityMatrixManager'));

  return (
    <div>
      <FieldHeader 
        fieldName={field.name}
        label={label}
        number={number}
        required={field.required}
      />

      {showMatrixManager ? (
        <React.Suspense fallback={<div>Loading...</div>}>
          <ResponsibilityMatrixManager
            projectId={formData.projectName || 'current'}
            onClose={() => setShowMatrixManager(false)}
          />
        </React.Suspense>
      ) : (
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-200 rounded-lg p-6">
          <div className="flex items-start space-x-4">
            <div className="flex-shrink-0">
              <div className="w-12 h-12 bg-green-600 rounded-lg flex items-center justify-center">
                <FileText className="w-6 h-6 text-white" />
              </div>
            </div>
            <div className="flex-1">
              <h4 className="font-semibold text-gray-900 mb-1">Deliverables Schedule with TIDP Sync</h4>
              <p className="text-sm text-gray-600 mb-4">
                {placeholder}
              </p>
              <button
                type="button"
                onClick={() => setShowMatrixManager(true)}
                className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg transition-colors flex items-center space-x-2 font-medium"
              >
                <FileText className="w-5 h-5" />
                <span>Open Deliverables Matrix</span>
              </button>
            </div>
          </div>
        </div>
      )}
      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
};

const ImActivitiesMatrixField = ({ field, value, onChange, error, formData }) => {
  const { label, number, placeholder } = field;
  const [showMatrixManager, setShowMatrixManager] = useState(false);

  // Lazy load the manager component
  const ResponsibilityMatrixManager = React.lazy(() => import('../../responsibility-matrix/ResponsibilityMatrixManager'));

  return (
    <div>
      <FieldHeader 
        fieldName={field.name}
        label={label}
        number={number}
        required={field.required}
      />

      {showMatrixManager ? (
        <React.Suspense fallback={<div>Loading...</div>}>
          <ResponsibilityMatrixManager
            projectId={formData.projectName || 'current'}
            onClose={() => setShowMatrixManager(false)}
          />
        </React.Suspense>
      ) : (
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-200 rounded-lg p-6">
          <div className="flex items-start space-x-4">
            <div className="flex-shrink-0">
              <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center">
                <Table2 className="w-6 h-6 text-white" />
              </div>
            </div>
            <div className="flex-1">
              <h4 className="font-semibold text-gray-900 mb-1">RACI Matrix for ISO 19650-2 Annex A Activities</h4>
              <p className="text-sm text-gray-600 mb-4">
                {placeholder}
              </p>
              <button
                type="button"
                onClick={() => setShowMatrixManager(true)}
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors flex items-center space-x-2 font-medium"
              >
                <Table2 className="w-5 h-5" />
                <span>Open Responsibility Matrix</span>
              </button>
            </div>
          </div>
        </div>
      )}
      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
};

export default InputField;