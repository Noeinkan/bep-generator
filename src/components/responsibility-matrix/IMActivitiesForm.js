import React, { useState, useEffect } from 'react';
import { X, Save, AlertCircle } from 'lucide-react';
import { RACI_ROLES, RACI_ROLE_DESCRIPTIONS, ISO_ACTIVITY_PHASES } from '../../constants/iso19650ActivitiesTemplate';

/**
 * Form for adding/editing IM Activities
 */
const IMActivitiesForm = ({
  activity = null,
  onSave,
  onCancel,
  loading = false
}) => {
  const [formData, setFormData] = useState({
    activityName: '',
    activityDescription: '',
    activityPhase: ISO_ACTIVITY_PHASES.MOBILIZATION,
    isoReference: '',
    appointingPartyRole: 'N/A',
    leadAppointedPartyRole: 'N/A',
    appointedPartiesRole: 'N/A',
    thirdPartiesRole: 'N/A',
    notes: ''
  });

  const [errors, setErrors] = useState({});

  useEffect(() => {
    if (activity) {
      setFormData({
        activityName: activity.activity_name || '',
        activityDescription: activity.activity_description || '',
        activityPhase: activity.activity_phase || ISO_ACTIVITY_PHASES.MOBILIZATION,
        isoReference: activity.iso_reference || '',
        appointingPartyRole: activity.appointing_party_role || 'N/A',
        leadAppointedPartyRole: activity.lead_appointed_party_role || 'N/A',
        appointedPartiesRole: activity.appointed_parties_role || 'N/A',
        thirdPartiesRole: activity.third_parties_role || 'N/A',
        notes: activity.notes || ''
      });
    }
  }, [activity]);

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

    if (!formData.activityName.trim()) {
      newErrors.activityName = 'Activity name is required';
    }

    if (!formData.activityPhase) {
      newErrors.activityPhase = 'Activity phase is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!validate()) {
      return;
    }

    onSave(formData);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-4 flex items-center justify-between sticky top-0">
          <h2 className="text-xl font-semibold">
            {activity ? 'Edit IM Activity' : 'Add Custom IM Activity'}
          </h2>
          <button
            onClick={onCancel}
            className="text-white hover:bg-blue-800 p-1 rounded transition-colors"
          >
            <X size={24} />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Activity Name */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Activity Name <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              value={formData.activityName}
              onChange={(e) => handleChange('activityName', e.target.value)}
              className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 ${
                errors.activityName
                  ? 'border-red-500 focus:ring-red-500'
                  : 'border-gray-300 focus:ring-blue-500'
              }`}
              placeholder="e.g., Establish information standard"
            />
            {errors.activityName && (
              <p className="mt-1 text-sm text-red-600 flex items-center gap-1">
                <AlertCircle size={14} />
                {errors.activityName}
              </p>
            )}
          </div>

          {/* Activity Description */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Description
            </label>
            <textarea
              value={formData.activityDescription}
              onChange={(e) => handleChange('activityDescription', e.target.value)}
              rows={3}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Describe what this activity involves..."
            />
          </div>

          {/* Phase and ISO Reference */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Activity Phase <span className="text-red-500">*</span>
              </label>
              <select
                value={formData.activityPhase}
                onChange={(e) => handleChange('activityPhase', e.target.value)}
                className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 ${
                  errors.activityPhase
                    ? 'border-red-500 focus:ring-red-500'
                    : 'border-gray-300 focus:ring-blue-500'
                }`}
              >
                {Object.values(ISO_ACTIVITY_PHASES).map(phase => (
                  <option key={phase} value={phase}>
                    {phase}
                  </option>
                ))}
              </select>
              {errors.activityPhase && (
                <p className="mt-1 text-sm text-red-600 flex items-center gap-1">
                  <AlertCircle size={14} />
                  {errors.activityPhase}
                </p>
              )}
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                ISO 19650 Reference
              </label>
              <input
                type="text"
                value={formData.isoReference}
                onChange={(e) => handleChange('isoReference', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="e.g., ISO 19650-2:2018, 5.2.1"
              />
            </div>
          </div>

          {/* RACI Assignments Section */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-semibold text-gray-800 mb-4">RACI Role Assignments</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Appointing Party */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Appointing Party (Client)
                </label>
                <select
                  value={formData.appointingPartyRole}
                  onChange={(e) => handleChange('appointingPartyRole', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {Object.entries(RACI_ROLES).map(([key, value]) => (
                    <option key={value} value={value}>
                      {value} - {RACI_ROLE_DESCRIPTIONS[value]}
                    </option>
                  ))}
                </select>
              </div>

              {/* Lead Appointed Party */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Lead Appointed Party
                </label>
                <select
                  value={formData.leadAppointedPartyRole}
                  onChange={(e) => handleChange('leadAppointedPartyRole', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {Object.entries(RACI_ROLES).map(([key, value]) => (
                    <option key={value} value={value}>
                      {value} - {RACI_ROLE_DESCRIPTIONS[value]}
                    </option>
                  ))}
                </select>
              </div>

              {/* Appointed Parties */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Appointed Parties (Task Teams)
                </label>
                <select
                  value={formData.appointedPartiesRole}
                  onChange={(e) => handleChange('appointedPartiesRole', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {Object.entries(RACI_ROLES).map(([key, value]) => (
                    <option key={value} value={value}>
                      {value} - {RACI_ROLE_DESCRIPTIONS[value]}
                    </option>
                  ))}
                </select>
              </div>

              {/* Third Parties */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Third Parties
                </label>
                <select
                  value={formData.thirdPartiesRole}
                  onChange={(e) => handleChange('thirdPartiesRole', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {Object.entries(RACI_ROLES).map(([key, value]) => (
                    <option key={value} value={value}>
                      {value} - {RACI_ROLE_DESCRIPTIONS[value]}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Notes */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Notes / Comments
            </label>
            <textarea
              value={formData.notes}
              onChange={(e) => handleChange('notes', e.target.value)}
              rows={3}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Add any additional context, clarifications, or notes..."
            />
          </div>

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
              className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  Saving...
                </>
              ) : (
                <>
                  <Save size={18} />
                  {activity ? 'Update Activity' : 'Add Activity'}
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default IMActivitiesForm;
