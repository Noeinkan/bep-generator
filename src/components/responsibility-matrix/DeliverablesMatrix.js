import React, { useState, useMemo } from 'react';
import { Plus, Edit2, Trash2, Download, Calendar, Package, RefreshCw, AlertCircle } from 'lucide-react';

/**
 * Information Deliverables Matrix (Matrix 2)
 * Shows deliverables from TIDPs with sync status
 */
const DeliverablesMatrix = ({
  deliverables = [],
  syncStatus = null,
  onEdit,
  onDelete,
  onAddManual,
  onSync,
  onExport,
  loading = false
}) => {
  const [viewMode, setViewMode] = useState('table'); // 'table' or 'timeline'
  const [filterStage, setFilterStage] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterTeam, setFilterTeam] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('dueDate');
  const [sortOrder, setSortOrder] = useState('asc');

  // Extract unique values for filters
  const stages = useMemo(() => {
    const uniqueStages = [...new Set(deliverables.map(d => d.exchange_stage).filter(Boolean))];
    return ['all', ...uniqueStages.sort()];
  }, [deliverables]);

  const statuses = ['all', 'Planned', 'In Progress', 'Delivered', 'Approved'];

  const teams = useMemo(() => {
    const uniqueTeams = [...new Set(deliverables.map(d => d.responsible_task_team).filter(Boolean))];
    return ['all', ...uniqueTeams.sort()];
  }, [deliverables]);

  // Filter and sort deliverables
  const filteredDeliverables = useMemo(() => {
    let filtered = deliverables;

    if (filterStage !== 'all') {
      filtered = filtered.filter(d => d.exchange_stage === filterStage);
    }

    if (filterStatus !== 'all') {
      filtered = filtered.filter(d => d.status === filterStatus);
    }

    if (filterTeam !== 'all') {
      filtered = filtered.filter(d => d.responsible_task_team === filterTeam);
    }

    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(d =>
        d.deliverable_name?.toLowerCase().includes(term) ||
        d.description?.toLowerCase().includes(term) ||
        d.responsible_task_team?.toLowerCase().includes(term) ||
        d.format?.toLowerCase().includes(term)
      );
    }

    // Sort
    filtered.sort((a, b) => {
      let aVal, bVal;

      switch (sortBy) {
        case 'dueDate':
          aVal = a.due_date || '';
          bVal = b.due_date || '';
          break;
        case 'name':
          aVal = a.deliverable_name || '';
          bVal = b.deliverable_name || '';
          break;
        case 'stage':
          aVal = a.exchange_stage || '';
          bVal = b.exchange_stage || '';
          break;
        case 'team':
          aVal = a.responsible_task_team || '';
          bVal = b.responsible_task_team || '';
          break;
        default:
          aVal = a.deliverable_name || '';
          bVal = b.deliverable_name || '';
      }

      if (sortOrder === 'asc') {
        return aVal > bVal ? 1 : -1;
      } else {
        return aVal < bVal ? 1 : -1;
      }
    });

    return filtered;
  }, [deliverables, filterStage, filterStatus, filterTeam, searchTerm, sortBy, sortOrder]);

  // Group by stage
  const groupedByStage = useMemo(() => {
    const groups = {};
    filteredDeliverables.forEach(deliverable => {
      const stage = deliverable.exchange_stage || 'Unassigned';
      if (!groups[stage]) {
        groups[stage] = [];
      }
      groups[stage].push(deliverable);
    });
    return groups;
  }, [filteredDeliverables]);

  const getStatusBadgeClass = (status) => {
    switch (status) {
      case 'Planned':
        return 'bg-gray-100 text-gray-800';
      case 'In Progress':
        return 'bg-blue-100 text-blue-800';
      case 'Delivered':
        return 'bg-green-100 text-green-800';
      case 'Approved':
        return 'bg-purple-100 text-purple-800';
      default:
        return 'bg-gray-100 text-gray-600';
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return '-';
    try {
      return new Date(dateString).toLocaleDateString('en-GB', {
        day: '2-digit',
        month: 'short',
        year: 'numeric'
      });
    } catch {
      return dateString;
    }
  };

  const isOverdue = (dueDate, status) => {
    if (!dueDate || status === 'Delivered' || status === 'Approved') return false;
    return new Date(dueDate) < new Date();
  };

  const toggleSort = (field) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('asc');
    }
  };

  const renderTableView = () => {
    return (
      <div className="overflow-x-auto">
        <table className="w-full border-collapse bg-white shadow-sm">
          <thead className="bg-gray-50 border-b-2 border-gray-200">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-r">
                <button
                  onClick={() => toggleSort('name')}
                  className="flex items-center gap-1 hover:text-blue-600"
                >
                  Deliverable
                  {sortBy === 'name' && (
                    <span className="text-blue-600">{sortOrder === 'asc' ? '↑' : '↓'}</span>
                  )}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-r">
                <button
                  onClick={() => toggleSort('team')}
                  className="flex items-center gap-1 hover:text-blue-600"
                >
                  Responsible Team
                  {sortBy === 'team' && (
                    <span className="text-blue-600">{sortOrder === 'asc' ? '↑' : '↓'}</span>
                  )}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-r">
                Accountable
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-r">
                <button
                  onClick={() => toggleSort('stage')}
                  className="flex items-center gap-1 hover:text-blue-600"
                >
                  Stage
                  {sortBy === 'stage' && (
                    <span className="text-blue-600">{sortOrder === 'asc' ? '↑' : '↓'}</span>
                  )}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-r">
                <button
                  onClick={() => toggleSort('dueDate')}
                  className="flex items-center gap-1 hover:text-blue-600"
                >
                  Due Date
                  {sortBy === 'dueDate' && (
                    <span className="text-blue-600">{sortOrder === 'asc' ? '↑' : '↓'}</span>
                  )}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-r">
                Format
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-r">
                LOD/LOIN
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-r">
                Status
              </th>
              <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {filteredDeliverables.map((deliverable, idx) => {
              const overdue = isOverdue(deliverable.due_date, deliverable.status);
              const isAutoPopulated = deliverable.is_auto_populated === 1;

              return (
                <tr
                  key={deliverable.id}
                  className={`${idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'} ${
                    overdue ? 'bg-red-50' : ''
                  }`}
                >
                  <td className="px-4 py-3 border-r">
                    <div className="flex flex-col gap-1">
                      <div className="font-medium text-gray-900 text-sm flex items-center gap-2">
                        {deliverable.deliverable_name}
                        {isAutoPopulated && (
                          <span
                            className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded"
                            title="Auto-populated from TIDP"
                          >
                            Auto
                          </span>
                        )}
                      </div>
                      {deliverable.description && (
                        <div className="text-xs text-gray-600 line-clamp-2">
                          {deliverable.description}
                        </div>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3 border-r text-sm text-gray-700">
                    {deliverable.responsible_task_team || '-'}
                  </td>
                  <td className="px-4 py-3 border-r text-sm text-gray-700">
                    {deliverable.accountable_party || '-'}
                  </td>
                  <td className="px-4 py-3 border-r text-sm text-gray-700">
                    {deliverable.exchange_stage || '-'}
                  </td>
                  <td className="px-4 py-3 border-r">
                    <div className="flex items-center gap-1">
                      {overdue && <AlertCircle size={14} className="text-red-600" />}
                      <span className={`text-sm ${overdue ? 'text-red-600 font-semibold' : 'text-gray-700'}`}>
                        {formatDate(deliverable.due_date)}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3 border-r text-sm text-gray-700">
                    {deliverable.format || '-'}
                  </td>
                  <td className="px-4 py-3 border-r text-sm text-gray-700">
                    {deliverable.loin_lod || '-'}
                  </td>
                  <td className="px-4 py-3 border-r">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusBadgeClass(deliverable.status)}`}>
                      {deliverable.status || 'Planned'}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-center gap-2">
                      <button
                        onClick={() => onEdit(deliverable)}
                        className="p-1 text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded transition-colors"
                        title="Edit deliverable"
                      >
                        <Edit2 size={16} />
                      </button>
                      {!isAutoPopulated && (
                        <button
                          onClick={() => onDelete(deliverable.id)}
                          className="p-1 text-red-600 hover:text-red-800 hover:bg-red-50 rounded transition-colors"
                          title="Delete deliverable"
                        >
                          <Trash2 size={16} />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  };

  const renderTimelineView = () => {
    return (
      <div className="space-y-6">
        {Object.entries(groupedByStage).map(([stage, stageDeliverables]) => (
          <div key={stage} className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
            <div className="bg-gradient-to-r from-indigo-600 to-indigo-700 text-white px-4 py-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Calendar size={20} />
                <h3 className="font-semibold text-lg">{stage}</h3>
                <span className="bg-indigo-500 px-2 py-1 rounded text-xs">
                  {stageDeliverables.length}
                </span>
              </div>
            </div>
            <div className="p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {stageDeliverables.map((deliverable) => {
                const overdue = isOverdue(deliverable.due_date, deliverable.status);
                const isAutoPopulated = deliverable.is_auto_populated === 1;

                return (
                  <div
                    key={deliverable.id}
                    className={`border rounded-lg p-4 hover:shadow-md transition-shadow ${
                      overdue ? 'border-red-300 bg-red-50' : 'border-gray-200'
                    }`}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex-1">
                        <div className="font-semibold text-gray-900 text-sm mb-1 flex items-center gap-2">
                          <Package size={14} className="text-gray-400" />
                          {deliverable.deliverable_name}
                        </div>
                        {isAutoPopulated && (
                          <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
                            Auto-synced
                          </span>
                        )}
                      </div>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusBadgeClass(deliverable.status)}`}>
                        {deliverable.status}
                      </span>
                    </div>

                    <div className="space-y-2 text-xs text-gray-600">
                      <div>
                        <span className="font-medium">Team:</span> {deliverable.responsible_task_team || '-'}
                      </div>
                      <div>
                        <span className="font-medium">Accountable:</span> {deliverable.accountable_party || '-'}
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="font-medium">Due:</span>
                        {overdue && <AlertCircle size={12} className="text-red-600" />}
                        <span className={overdue ? 'text-red-600 font-semibold' : ''}>
                          {formatDate(deliverable.due_date)}
                        </span>
                      </div>
                      <div>
                        <span className="font-medium">Format:</span> {deliverable.format || '-'}
                      </div>
                    </div>

                    <div className="mt-3 flex gap-2">
                      <button
                        onClick={() => onEdit(deliverable)}
                        className="flex-1 px-2 py-1 text-xs bg-blue-50 text-blue-600 rounded hover:bg-blue-100 transition-colors"
                      >
                        Edit
                      </button>
                      {!isAutoPopulated && (
                        <button
                          onClick={() => onDelete(deliverable.id)}
                          className="px-2 py-1 text-xs bg-red-50 text-red-600 rounded hover:bg-red-100 transition-colors"
                        >
                          Delete
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
        <span className="ml-3 text-gray-600">Loading deliverables...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Sync Status Banner */}
      {syncStatus && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-start justify-between">
            <div>
              <h4 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
                <RefreshCw size={18} className="text-blue-600" />
                TIDP Synchronization Status
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Total TIDPs:</span>{' '}
                  <span className="font-semibold">{syncStatus.totalTIDPs}</span>
                </div>
                <div>
                  <span className="text-gray-600">Synced:</span>{' '}
                  <span className="font-semibold text-green-600">{syncStatus.syncedTIDPs}</span>
                </div>
                <div>
                  <span className="text-gray-600">Unsynced:</span>{' '}
                  <span className="font-semibold text-orange-600">{syncStatus.unsyncedTIDPs}</span>
                </div>
              </div>
            </div>
            <button
              onClick={onSync}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <RefreshCw size={16} />
              Sync Now
            </button>
          </div>
        </div>
      )}

      {/* Header Controls */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div className="flex flex-wrap items-center gap-4 mb-4">
          {/* Search */}
          <div className="flex-1 min-w-[200px]">
            <input
              type="text"
              placeholder="Search deliverables..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          {/* View Mode Toggle */}
          <div className="flex items-center gap-2 bg-gray-100 p-1 rounded-lg">
            <button
              onClick={() => setViewMode('table')}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                viewMode === 'table'
                  ? 'bg-white text-indigo-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Table
            </button>
            <button
              onClick={() => setViewMode('timeline')}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                viewMode === 'timeline'
                  ? 'bg-white text-indigo-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Timeline
            </button>
          </div>

          {/* Action Buttons */}
          <button
            onClick={onAddManual}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
          >
            <Plus size={18} />
            Add Manual
          </button>

          <button
            onClick={onExport}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
          >
            <Download size={18} />
            Export
          </button>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700">Stage:</label>
            <select
              value={filterStage}
              onChange={(e) => setFilterStage(e.target.value)}
              className="px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {stages.map(stage => (
                <option key={stage} value={stage}>
                  {stage === 'all' ? 'All Stages' : stage}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700">Status:</label>
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {statuses.map(status => (
                <option key={status} value={status}>
                  {status === 'all' ? 'All Statuses' : status}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700">Team:</label>
            <select
              value={filterTeam}
              onChange={(e) => setFilterTeam(e.target.value)}
              className="px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {teams.map(team => (
                <option key={team} value={team}>
                  {team === 'all' ? 'All Teams' : team}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Deliverables Display */}
      {filteredDeliverables.length === 0 ? (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
          <Package size={48} className="mx-auto text-gray-400 mb-4" />
          <p className="text-gray-500 mb-2">
            {searchTerm ? 'No deliverables match your search.' : 'No deliverables found.'}
          </p>
          <p className="text-sm text-gray-400">
            Sync TIDPs or add manual deliverables to get started.
          </p>
        </div>
      ) : (
        <>
          {viewMode === 'table' ? renderTableView() : renderTimelineView()}

          {/* Summary */}
          <div className="bg-gray-50 rounded-lg border border-gray-200 p-4">
            <div className="text-sm text-gray-600">
              Showing <span className="font-semibold text-gray-900">{filteredDeliverables.length}</span> of{' '}
              <span className="font-semibold text-gray-900">{deliverables.length}</span> deliverables
              {searchTerm && ` matching "${searchTerm}"`}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default DeliverablesMatrix;
