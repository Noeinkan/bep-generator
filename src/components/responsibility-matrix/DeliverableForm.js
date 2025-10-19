import React, { useState, useEffect } from 'react';
import { X, Save, AlertCircle } from 'lucide-react';

/**
 * Form for adding/editing Information Deliverables
 */
const DeliverableForm = ({
  deliverable = null,
  teams = [], // Available task teams from TIDPs
  onSave,
  onCancel,
  loading = false
}) => {
  const [formData, setFormData] = useState({
    deliverableName: '',
    description: '',
    responsibleTaskTeam: '',
    accountableParty: '',
    exchangeStage: '',
    dueDate: '',
    format: '',
    loinLod: '',
    dependencies: [],
    status: 'Planned'
  });

  const [errors, setErrors] = useState({});

  const STATUSES = ['Planned', 'In Progress', 'Delivered', 'Approved'];

  const EXCHANGE_STAGES = [
    'Stage 0 - Strategic Definition',
    'Stage 1 - Preparation and Brief',
    'Stage 2 - Concept Design',
    'Stage 3 - Spatial Coordination',
    'Stage 4 - Technical Design',
    'Stage 5 - Manufacturing and Construction',
    'Stage 6 - Handover',
    'Stage 7 - Use'
  ];

  const FORMATS = [
    'IFC',
    'DWG',
    'DXF',
    'PDF',
    'RVT',
    'NWD',
    'NWC',
    'COBie',
    'BCF',
    'XML',
    'JSON',
    'Excel',
    'Other'
  ];

  const LOD_LEVELS = [
    'LOD 100',
    'LOD 200',
    'LOD 300',
    'LOD 350',
    'LOD 400',
    'LOD 500',
    'LOIN A',
    'LOIN B',
    'LOIN C',
    'LOIN D',
    'LOIN E',
    'LOIN F'
  ];

  useEffect(() => {
    if (deliverable) {
      setFormData({
        deliverableName: deliverable.deliverable_name || '',
        description: deliverable.description || '',
        responsibleTaskTeam: deliverable.responsible_task_team || '',
        accountableParty: deliverable.accountable_party || '',
        exchangeStage: deliverable.exchange_stage || '',
        dueDate: deliverable.due_date ? deliverable.due_date.split('T')[0] : '',
        format: deliverable.format || '',
        loinLod: deliverable.loin_lod || '',
        dependencies: deliverable.dependencies || [],
        status: deliverable.status || 'Planned'
      });
    }
  }, [deliverable]);

  const handleChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    // Clear error for this field
    if (errors[field]) {
      setErrors(prev => ({
        ...prev,
        [field]: null
      }));
    }
  };

  const validate = () => {
    const newErrors = {};

    if (!formData.deliverableName.trim()) {
      newErrors.deliverableName = 'Deliverable name is required';
    }

    if (!formData.responsibleTaskTeam.trim()) {
      newErrors.responsibleTaskTeam = 'Responsible task team is required';
    }

    if (!formData.exchangeStage) {
      newErrors.exchangeStage = 'Exchange stage is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!validate()) {
      return;
    }

    // Format the data for submission
    const submitData = {
      ...formData,
      // Convert date to ISO string if provided
      dueDate: formData.dueDate ? new Date(formData.dueDate).toISOString() : null,
      // Ensure dependencies is an array
      dependencies: Array.isArray(formData.dependencies) ? formData.dependencies : []
    };

    onSave(submitData);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="bg-gradient-to-r from-indigo-600 to-indigo-700 text-white px-6 py-4 flex items-center justify-between sticky top-0">
          <h2 className="text-xl font-semibold">
            {deliverable ? 'Edit Deliverable' : 'Add Manual Deliverable'}
          </h2>
          <button
            onClick={onCancel}
            className="text-white hover:bg-indigo-800 p-1 rounded transition-colors"
          >
            <X size={24} />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Deliverable Name */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Deliverable Name <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              value={formData.deliverableName}
              onChange={(e) => handleChange('deliverableName', e.target.value)}
              className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 ${
                errors.deliverableName
                  ? 'border-red-500 focus:ring-red-500'
                  : 'border-gray-300 focus:ring-indigo-500'
              }`}
              placeholder="e.g., Architectural Model - Level 3"
            />
            {errors.deliverableName && (
              <p className="mt-1 text-sm text-red-600 flex items-center gap-1">
                <AlertCircle size={14} />
                {errors.deliverableName}
              </p>
            )}
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Description
            </label>
            <textarea
              value={formData.description}
              onChange={(e) => handleChange('description', e.target.value)}
              rows={3}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
              placeholder="Describe the deliverable content and purpose..."
            />
          </div>

          {/* Responsibility Section */}
          <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
            <h3 className="font-semibold text-gray-800 mb-4">Responsibility</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Responsible Task Team <span className="text-red-500">*</span>
                </label>
                {teams.length > 0 ? (
                  <select
                    value={formData.responsibleTaskTeam}
                    onChange={(e) => handleChange('responsibleTaskTeam', e.target.value)}
                    className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 ${
                      errors.responsibleTaskTeam
                        ? 'border-red-500 focus:ring-red-500'
                        : 'border-gray-300 focus:ring-indigo-500'
                    }`}
                  >
                    <option value="">Select team...</option>
                    {teams.map(team => (
                      <option key={team} value={team}>{team}</option>
                    ))}
                    <option value="custom">-- Custom Team --</option>
                  </select>
                ) : (
                  <input
                    type="text"
                    value={formData.responsibleTaskTeam}
                    onChange={(e) => handleChange('responsibleTaskTeam', e.target.value)}
                    className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 ${
                      errors.responsibleTaskTeam
                        ? 'border-red-500 focus:ring-red-500'
                        : 'border-gray-300 focus:ring-indigo-500'
                    }`}
                    placeholder="Enter team name"
                  />
                )}
                {errors.responsibleTaskTeam && (
                  <p className="mt-1 text-sm text-red-600 flex items-center gap-1">
                    <AlertCircle size={14} />
                    {errors.responsibleTaskTeam}
                  </p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Accountable Party
                </label>
                <input
                  type="text"
                  value={formData.accountableParty}
                  onChange={(e) => handleChange('accountableParty', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  placeholder="e.g., Lead Architect"
                />
              </div>
            </div>
          </div>

          {/* Delivery Details */}
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <h3 className="font-semibold text-gray-800 mb-4">Delivery Details</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Exchange Stage / Milestone <span className="text-red-500">*</span>
                </label>
                <select
                  value={formData.exchangeStage}
                  onChange={(e) => handleChange('exchangeStage', e.target.value)}
                  className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 ${
                    errors.exchangeStage
                      ? 'border-red-500 focus:ring-red-500'
                      : 'border-gray-300 focus:ring-indigo-500'
                  }`}
                >
                  <option value="">Select stage...</option>
                  {EXCHANGE_STAGES.map(stage => (
                    <option key={stage} value={stage}>{stage}</option>
                  ))}
                </select>
                {errors.exchangeStage && (
                  <p className="mt-1 text-sm text-red-600 flex items-center gap-1">
                    <AlertCircle size={14} />
                    {errors.exchangeStage}
                  </p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Due Date
                </label>
                <input
                  type="date"
                  value={formData.dueDate}
                  onChange={(e) => handleChange('dueDate', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Format / Type
                </label>
                <select
                  value={formData.format}
                  onChange={(e) => handleChange('format', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="">Select format...</option>
                  {FORMATS.map(format => (
                    <option key={format} value={format}>{format}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Level of Information Need (LOIN/LOD)
                </label>
                <select
                  value={formData.loinLod}
                  onChange={(e) => handleChange('loinLod', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="">Select level...</option>
                  {LOD_LEVELS.map(lod => (
                    <option key={lod} value={lod}>{lod}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Status
                </label>
                <select
                  value={formData.status}
                  onChange={(e) => handleChange('status', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  {STATUSES.map(status => (
                    <option key={status} value={status}>{status}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Info Notice */}
          {deliverable?.is_auto_populated === 1 && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 flex items-start gap-3">
              <AlertCircle size={20} className="text-yellow-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-yellow-800">
                <p className="font-medium">Auto-populated Deliverable</p>
                <p className="mt-1">
                  This deliverable was automatically populated from a TIDP. Changes will be overwritten
                  if you re-sync the associated TIDP unless you convert it to a manual deliverable.
                </p>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex items-center justify-end gap-3 pt-4 border-t">
            <button
              type="button"
              onClick={onCancel}
              className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex items-center gap-2 px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  Saving...
                </>
              ) : (
                <>
                  <Save size={18} />
                  {deliverable ? 'Update Deliverable' : 'Add Deliverable'}
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default DeliverableForm;
