import React, { useState, useMemo } from 'react';
import { Plus, Edit2, Trash2, Download, FileText, BookOpen } from 'lucide-react';
import { RACI_ROLES, ISO_ACTIVITY_PHASES } from '../../constants/iso19650ActivitiesTemplate';

/**
 * Information Management Activities Matrix (Matrix 1)
 * ISO 19650-2 Annex A compliant
 */
const IMActivitiesMatrix = ({
  activities = [],
  onEdit,
  onDelete,
  onAddCustom,
  onExport,
  loading = false
}) => {
  const [selectedPhase, setSelectedPhase] = useState('all');
  const [editingCell, setEditingCell] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  // Get unique phases from activities
  const phases = useMemo(() => {
    const uniquePhases = [...new Set(activities.map(a => a.activity_phase))];
    return ['all', ...uniquePhases];
  }, [activities]);

  // Filter activities by phase and search
  const filteredActivities = useMemo(() => {
    let filtered = activities;

    if (selectedPhase !== 'all') {
      filtered = filtered.filter(a => a.activity_phase === selectedPhase);
    }

    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(a =>
        a.activity_name?.toLowerCase().includes(term) ||
        a.activity_description?.toLowerCase().includes(term) ||
        a.iso_reference?.toLowerCase().includes(term)
      );
    }

    return filtered;
  }, [activities, selectedPhase, searchTerm]);

  // Group activities by phase for display
  const groupedActivities = useMemo(() => {
    const groups = {};
    filteredActivities.forEach(activity => {
      const phase = activity.activity_phase || 'Other';
      if (!groups[phase]) {
        groups[phase] = [];
      }
      groups[phase].push(activity);
    });
    return groups;
  }, [filteredActivities]);

  const handleCellClick = (activityId, column) => {
    setEditingCell({ activityId, column });
  };

  const handleCellChange = (activity, column, value) => {
    onEdit(activity.id, { [column]: value });
    setEditingCell(null);
  };

  const renderRACICell = (activity, column) => {
    const value = activity[column] || 'N/A';
    const isEditing = editingCell?.activityId === activity.id && editingCell?.column === column;

    if (isEditing) {
      return (
        <select
          autoFocus
          value={value}
          onChange={(e) => handleCellChange(activity, column, e.target.value)}
          onBlur={() => setEditingCell(null)}
          className="w-full px-2 py-1 text-sm border border-blue-500 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {Object.entries(RACI_ROLES).map(([key, val]) => (
            <option key={val} value={val}>{val}</option>
          ))}
        </select>
      );
    }

    return (
      <div
        onClick={() => handleCellClick(activity.id, column)}
        className={`px-3 py-2 text-center font-medium cursor-pointer hover:bg-gray-50 rounded transition-colors ${
          value === 'A' ? 'bg-blue-100 text-blue-800' :
          value === 'R' ? 'bg-green-100 text-green-800' :
          value === 'C' ? 'bg-yellow-100 text-yellow-800' :
          value === 'I' ? 'bg-purple-100 text-purple-800' :
          'bg-gray-100 text-gray-600'
        }`}
        title="Click to edit"
      >
        {value}
      </div>
    );
  };

  const renderPhaseSection = (phase, phaseActivities) => {
    return (
      <div key={phase} className="mb-8">
        <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-4 py-3 rounded-t-lg flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BookOpen size={20} />
            <h3 className="font-semibold text-lg">{phase}</h3>
            <span className="bg-blue-500 px-2 py-1 rounded text-xs">
              {phaseActivities.length} {phaseActivities.length === 1 ? 'activity' : 'activities'}
            </span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full border-collapse bg-white shadow-sm">
            <thead className="bg-gray-50 border-b-2 border-gray-200">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-r">
                  Activity / Function
                </th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider border-r">
                  Appointing Party<br/>(Client)
                </th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider border-r">
                  Lead Appointed<br/>Party
                </th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider border-r">
                  Appointed Parties<br/>(Task Teams)
                </th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider border-r">
                  Third Parties
                </th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {phaseActivities.map((activity, idx) => (
                <tr key={activity.id} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-4 py-3 border-r">
                    <div className="flex flex-col gap-1">
                      <div className="font-medium text-gray-900 text-sm">
                        {activity.activity_name}
                      </div>
                      {activity.activity_description && (
                        <div className="text-xs text-gray-600 leading-tight">
                          {activity.activity_description}
                        </div>
                      )}
                      {activity.iso_reference && (
                        <div className="flex items-center gap-1 mt-1">
                          <FileText size={12} className="text-gray-400" />
                          <span className="text-xs text-gray-500 italic">
                            {activity.iso_reference}
                          </span>
                        </div>
                      )}
                    </div>
                  </td>
                  <td className="px-2 py-3 border-r">
                    {renderRACICell(activity, 'appointing_party_role')}
                  </td>
                  <td className="px-2 py-3 border-r">
                    {renderRACICell(activity, 'lead_appointed_party_role')}
                  </td>
                  <td className="px-2 py-3 border-r">
                    {renderRACICell(activity, 'appointed_parties_role')}
                  </td>
                  <td className="px-2 py-3 border-r">
                    {renderRACICell(activity, 'third_parties_role')}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-center gap-2">
                      <button
                        onClick={() => onEdit(activity.id)}
                        className="p-1 text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded transition-colors"
                        title="Edit activity"
                      >
                        <Edit2 size={16} />
                      </button>
                      {activity.is_custom === 1 && (
                        <button
                          onClick={() => onDelete(activity.id)}
                          className="p-1 text-red-600 hover:text-red-800 hover:bg-red-50 rounded transition-colors"
                          title="Delete custom activity"
                        >
                          <Trash2 size={16} />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-3 text-gray-600">Loading activities...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header Controls */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div className="flex flex-wrap items-center gap-4">
          {/* Search */}
          <div className="flex-1 min-w-[200px]">
            <input
              type="text"
              placeholder="Search activities..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Phase Filter */}
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700">Phase:</label>
            <select
              value={selectedPhase}
              onChange={(e) => setSelectedPhase(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {phases.map(phase => (
                <option key={phase} value={phase}>
                  {phase === 'all' ? 'All Phases' : phase}
                </option>
              ))}
            </select>
          </div>

          {/* Action Buttons */}
          <button
            onClick={onAddCustom}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Plus size={18} />
            Add Custom Activity
          </button>

          <button
            onClick={onExport}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
          >
            <Download size={18} />
            Export Matrix
          </button>
        </div>
      </div>

      {/* RACI Legend */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="font-semibold text-gray-800 mb-3 text-sm">RACI Legend:</h4>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <div className="flex items-center gap-2">
            <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded font-medium text-sm">A</span>
            <span className="text-sm text-gray-700">Accountable</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="bg-green-100 text-green-800 px-2 py-1 rounded font-medium text-sm">R</span>
            <span className="text-sm text-gray-700">Responsible</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded font-medium text-sm">C</span>
            <span className="text-sm text-gray-700">Consulted</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="bg-purple-100 text-purple-800 px-2 py-1 rounded font-medium text-sm">I</span>
            <span className="text-sm text-gray-700">Informed</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="bg-gray-100 text-gray-600 px-2 py-1 rounded font-medium text-sm">N/A</span>
            <span className="text-sm text-gray-700">Not Applicable</span>
          </div>
        </div>
        <p className="text-xs text-gray-600 mt-2 italic">Click any role cell to edit assignments</p>
      </div>

      {/* Activities Table by Phase */}
      {filteredActivities.length === 0 ? (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
          <p className="text-gray-500">
            {searchTerm ? 'No activities match your search.' : 'No activities found. Add ISO 19650 standard activities to get started.'}
          </p>
        </div>
      ) : (
        <div>
          {Object.entries(groupedActivities).map(([phase, phaseActivities]) =>
            renderPhaseSection(phase, phaseActivities)
          )}
        </div>
      )}

      {/* Summary */}
      <div className="bg-gray-50 rounded-lg border border-gray-200 p-4">
        <div className="text-sm text-gray-600">
          Showing <span className="font-semibold text-gray-900">{filteredActivities.length}</span> of{' '}
          <span className="font-semibold text-gray-900">{activities.length}</span> activities
          {searchTerm && ` matching "${searchTerm}"`}
        </div>
      </div>
    </div>
  );
};

export default IMActivitiesMatrix;
