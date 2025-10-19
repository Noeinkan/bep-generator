import React, { useState } from 'react';
import { RefreshCw, AlertCircle, CheckCircle, XCircle, ChevronDown, ChevronUp, Trash2 } from 'lucide-react';

/**
 * TIDP Synchronization Panel
 * Shows sync status and provides sync controls
 */
const TIDPSyncPanel = ({
  syncStatus,
  onSyncAll,
  onSyncSingle,
  onUnsync,
  loading = false,
  onClose
}) => {
  const [expandedTidps, setExpandedTidps] = useState(new Set());
  const [overwriteManual, setOverwriteManual] = useState(false);
  const [syncing, setSyncing] = useState(false);

  const toggleTidpExpanded = (tidpId) => {
    const newExpanded = new Set(expandedTidps);
    if (newExpanded.has(tidpId)) {
      newExpanded.delete(tidpId);
    } else {
      newExpanded.add(tidpId);
    }
    setExpandedTidps(newExpanded);
  };

  const handleSyncAll = async () => {
    setSyncing(true);
    try {
      await onSyncAll(overwriteManual);
    } finally {
      setSyncing(false);
    }
  };

  const handleSyncSingle = async (tidpId) => {
    setSyncing(true);
    try {
      await onSyncSingle(tidpId, overwriteManual);
    } finally {
      setSyncing(false);
    }
  };

  const handleUnsync = async (tidpId) => {
    if (window.confirm('Are you sure you want to remove all auto-populated deliverables from this TIDP?')) {
      await onUnsync(tidpId);
    }
  };

  if (!syncStatus) {
    return null;
  }

  const { totalTIDPs, syncedTIDPs, unsyncedTIDPs, tidpDetails = [] } = syncStatus;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-4 flex items-center justify-between sticky top-0">
          <div className="flex items-center gap-3">
            <RefreshCw size={24} />
            <div>
              <h2 className="text-xl font-semibold">TIDP Synchronization</h2>
              <p className="text-sm text-blue-100 mt-1">
                Manage automatic population of deliverables from TIDPs
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-white hover:bg-blue-800 p-1 rounded transition-colors"
          >
            <XCircle size={24} />
          </button>
        </div>

        {/* Sync Overview */}
        <div className="p-6 bg-gradient-to-b from-blue-50 to-white border-b">
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-white rounded-lg border border-gray-200 p-4 text-center">
              <div className="text-3xl font-bold text-gray-900">{totalTIDPs}</div>
              <div className="text-sm text-gray-600 mt-1">Total TIDPs</div>
            </div>
            <div className="bg-white rounded-lg border border-green-200 p-4 text-center">
              <div className="text-3xl font-bold text-green-600">{syncedTIDPs}</div>
              <div className="text-sm text-gray-600 mt-1">Synced</div>
            </div>
            <div className="bg-white rounded-lg border border-orange-200 p-4 text-center">
              <div className="text-3xl font-bold text-orange-600">{unsyncedTIDPs}</div>
              <div className="text-sm text-gray-600 mt-1">Needs Sync</div>
            </div>
          </div>

          {/* Sync All Controls */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <h3 className="font-semibold text-gray-900 mb-2">Sync All TIDPs</h3>
                <p className="text-sm text-gray-600 mb-3">
                  Automatically populate deliverables from all TIDP containers
                </p>
                <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={overwriteManual}
                    onChange={(e) => setOverwriteManual(e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span>Overwrite manually edited deliverables</span>
                </label>
              </div>
              <button
                onClick={handleSyncAll}
                disabled={syncing || loading}
                className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {syncing ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    Syncing...
                  </>
                ) : (
                  <>
                    <RefreshCw size={18} />
                    Sync All
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* TIDP Details List */}
        <div className="p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Individual TIDP Status</h3>
          <div className="space-y-3">
            {tidpDetails.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <AlertCircle size={48} className="mx-auto mb-3 text-gray-400" />
                <p>No TIDPs found for this project</p>
              </div>
            ) : (
              tidpDetails.map((tidp) => {
                const isExpanded = expandedTidps.has(tidp.tidpId);
                const needsAttention = tidp.needsSync;

                return (
                  <div
                    key={tidp.tidpId}
                    className={`border rounded-lg overflow-hidden transition-all ${
                      needsAttention ? 'border-orange-300 bg-orange-50' : 'border-gray-200 bg-white'
                    }`}
                  >
                    {/* TIDP Header */}
                    <div className="p-4 flex items-center justify-between">
                      <div className="flex items-center gap-3 flex-1">
                        <button
                          onClick={() => toggleTidpExpanded(tidp.tidpId)}
                          className="text-gray-600 hover:text-gray-900 transition-colors"
                        >
                          {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                        </button>

                        {tidp.isSynced ? (
                          <CheckCircle size={20} className="text-green-600 flex-shrink-0" />
                        ) : (
                          <AlertCircle size={20} className="text-orange-600 flex-shrink-0" />
                        )}

                        <div className="flex-1">
                          <div className="font-medium text-gray-900">{tidp.tidpName}</div>
                          <div className="text-sm text-gray-600 mt-0.5">
                            {tidp.containersCount} containers, {tidp.syncedDeliverablesCount} synced deliverables
                          </div>
                        </div>

                        <div className="flex items-center gap-2">
                          {tidp.isSynced ? (
                            <span className="px-3 py-1 bg-green-100 text-green-800 rounded text-sm font-medium">
                              In Sync
                            </span>
                          ) : (
                            <span className="px-3 py-1 bg-orange-100 text-orange-800 rounded text-sm font-medium">
                              Needs Sync
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center gap-2 ml-4">
                        <button
                          onClick={() => handleSyncSingle(tidp.tidpId)}
                          disabled={syncing || loading}
                          className="p-2 text-blue-600 hover:bg-blue-50 rounded transition-colors disabled:opacity-50"
                          title="Sync this TIDP"
                        >
                          <RefreshCw size={18} />
                        </button>
                        {tidp.syncedDeliverablesCount > 0 && (
                          <button
                            onClick={() => handleUnsync(tidp.tidpId)}
                            disabled={syncing || loading}
                            className="p-2 text-red-600 hover:bg-red-50 rounded transition-colors disabled:opacity-50"
                            title="Remove synced deliverables"
                          >
                            <Trash2 size={18} />
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Expanded Details */}
                    {isExpanded && (
                      <div className="px-4 pb-4 border-t border-gray-200 bg-gray-50">
                        <div className="grid grid-cols-2 gap-4 mt-4 text-sm">
                          <div>
                            <span className="text-gray-600">TIDP ID:</span>
                            <div className="font-mono text-xs text-gray-800 mt-1">{tidp.tidpId}</div>
                          </div>
                          <div>
                            <span className="text-gray-600">Containers:</span>
                            <div className="font-semibold text-gray-900 mt-1">{tidp.containersCount}</div>
                          </div>
                          <div>
                            <span className="text-gray-600">Synced Deliverables:</span>
                            <div className="font-semibold text-gray-900 mt-1">{tidp.syncedDeliverablesCount}</div>
                          </div>
                          <div>
                            <span className="text-gray-600">Sync Status:</span>
                            <div className={`mt-1 font-medium ${tidp.isSynced ? 'text-green-600' : 'text-orange-600'}`}>
                              {tidp.isSynced ? 'Synchronized' : 'Out of Sync'}
                            </div>
                          </div>
                        </div>

                        {tidp.needsSync && (
                          <div className="mt-4 bg-orange-100 border border-orange-200 rounded-lg p-3 flex items-start gap-2">
                            <AlertCircle size={16} className="text-orange-600 flex-shrink-0 mt-0.5" />
                            <div className="text-sm text-orange-800">
                              <p className="font-medium">Action Required</p>
                              <p className="mt-1">
                                This TIDP has {tidp.containersCount - tidp.syncedDeliverablesCount} containers that
                                haven't been synced to the deliverables matrix. Click "Sync" to update.
                              </p>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Footer Info */}
        <div className="bg-gray-50 px-6 py-4 border-t">
          <div className="flex items-start gap-3">
            <AlertCircle size={18} className="text-blue-600 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-gray-700">
              <p className="font-medium text-gray-900 mb-1">About TIDP Synchronization</p>
              <ul className="space-y-1 text-gray-600">
                <li>• Auto-populated deliverables are linked to their source TIDP containers</li>
                <li>• Re-syncing will update auto-populated deliverables with latest TIDP data</li>
                <li>• Manually edited deliverables are preserved unless "overwrite" is checked</li>
                <li>• Deleted containers will remove their corresponding auto-populated deliverables</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TIDPSyncPanel;
