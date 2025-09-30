import React, { useState, useEffect, useCallback } from 'react';
import { TrendingUp, Users, Clock, BarChart3, AlertTriangle } from 'lucide-react';
import ApiService from '../services/apiService';

const MIDPEvolutionDashboard = ({ midpId, onClose }) => {
  const [evolution, setEvolution] = useState(null);
  const [deliverables, setDeliverables] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeView, setActiveView] = useState('overview');
  const [alerts, setAlerts] = useState([]);

  const loadDashboardData = useCallback(async () => {
    setLoading(true);
    try {
      const [evolutionData, deliverablesData, midpData] = await Promise.all([
        ApiService.getMIDPEvolution(midpId),
        ApiService.getMIDPDeliverablesDashboard(midpId),
        ApiService.getMIDP(midpId)
      ]);

      setEvolution(evolutionData.data);
      setDeliverables(deliverablesData.data);

      // Check for delay alerts
      const midp = midpData.data;
      const newAlerts = [];
      if (midp.aggregatedData?.milestones) {
        midp.aggregatedData.milestones.forEach(milestone => {
          if (milestone.delayImpact) {
            newAlerts.push(`Delay detected in milestone: ${milestone.name}`);
          }
        });
      }
      setAlerts(newAlerts);
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  }, [midpId]);

  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString();
  };

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'completed': return 'text-green-600 bg-green-100';
      case 'in progress': case 'inprogress': return 'text-blue-600 bg-blue-100';
      case 'delayed': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const OverviewView = () => (
    <div className="space-y-6">
      {/* Delay Alerts */}
      {alerts.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertTriangle className="w-5 h-5 text-red-600 mr-2" />
            <h3 className="text-lg font-semibold text-red-800">Delay Alerts</h3>
          </div>
          <ul className="mt-2 list-disc list-inside text-red-700">
            {alerts.map((alert, index) => (
              <li key={index}>{alert}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Evolution Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Total TIDPs</h3>
              <p className="text-3xl font-bold text-blue-600">{evolution?.current?.tidpCount || 0}</p>
            </div>
            <Users className="w-8 h-8 text-blue-600" />
          </div>
          <div className="mt-2 text-sm text-gray-600">
            {evolution?.historical?.length > 0 && (
              <span className="flex items-center">
                <TrendingUp className="w-4 h-4 mr-1 text-green-500" />
                +{(evolution.current.tidpCount - evolution.historical[0].tidpCount)} since start
              </span>
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Total Deliverables</h3>
              <p className="text-3xl font-bold text-green-600">{evolution?.current?.totalContainers || 0}</p>
            </div>
            <BarChart3 className="w-8 h-8 text-green-600" />
          </div>
          <div className="mt-2 text-sm text-gray-600">
            {evolution?.historical?.length > 0 && (
              <span className="flex items-center">
                <TrendingUp className="w-4 h-4 mr-1 text-green-500" />
                +{(evolution.current.totalContainers - evolution.historical[0].totalContainers)} since start
              </span>
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Estimated Hours</h3>
              <p className="text-3xl font-bold text-purple-600">{evolution?.current?.totalEstimatedHours || 0}</p>
            </div>
            <Clock className="w-8 h-8 text-purple-600" />
          </div>
          <div className="mt-2 text-sm text-gray-600">
            {evolution?.historical?.length > 0 && (
              <span className="flex items-center">
                <TrendingUp className="w-4 h-4 mr-1 text-green-500" />
                +{(evolution.current.totalEstimatedHours - evolution.historical[0].totalEstimatedHours)} since start
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Milestones Overview */}
      {evolution?.current?.deliverablesByMilestone && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Deliverables by Milestone</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(evolution.current.deliverablesByMilestone).map(([milestone, data]) => (
              <div key={milestone} className="border rounded-lg p-4">
                <h4 className="font-medium text-gray-900">{milestone}</h4>
                <div className="mt-2 space-y-1 text-sm text-gray-600">
                  <div>Deliverables: {data.count}</div>
                  <div>Hours: {data.estimatedHours}</div>
                  <div>Teams: {data.teams?.length || 0}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Completion Status */}
      {evolution?.current?.completionStatus && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Completion Status</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(evolution.current.completionStatus).map(([status, count]) => (
              <div key={status} className="text-center">
                <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(status)}`}>
                  {status.charAt(0).toUpperCase() + status.slice(1)}
                </div>
                <div className="mt-2 text-2xl font-bold text-gray-900">{count}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  const DeliverablesView = () => (
    <div className="bg-white rounded-lg shadow">
      <div className="p-6">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-lg font-semibold text-gray-900">All Deliverables</h3>
          <div className="text-sm text-gray-600">
            Total: {deliverables?.deliverables?.length || 0} deliverables
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-blue-50 rounded-lg p-4">
            <h4 className="font-medium text-blue-900">By Status</h4>
            <div className="mt-2 space-y-1 text-sm">
              {deliverables?.summary?.byStatus && Object.entries(deliverables.summary.byStatus).map(([status, count]) => (
                <div key={status} className="flex justify-between">
                  <span className="text-blue-700">{status}:</span>
                  <span className="font-medium text-blue-900">{count}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-green-50 rounded-lg p-4">
            <h4 className="font-medium text-green-900">By Milestone</h4>
            <div className="mt-2 space-y-1 text-sm">
              {deliverables?.summary?.byMilestone && Object.entries(deliverables.summary.byMilestone).slice(0, 5).map(([milestone, count]) => (
                <div key={milestone} className="flex justify-between">
                  <span className="text-green-700 truncate">{milestone}:</span>
                  <span className="font-medium text-green-900">{count}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-purple-50 rounded-lg p-4">
            <h4 className="font-medium text-purple-900">By Discipline</h4>
            <div className="mt-2 space-y-1 text-sm">
              {deliverables?.summary?.byDiscipline && Object.entries(deliverables.summary.byDiscipline).map(([discipline, count]) => (
                <div key={discipline} className="flex justify-between">
                  <span className="text-purple-700">{discipline}:</span>
                  <span className="font-medium text-purple-900">{count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Deliverables Table */}
        <div className="overflow-x-auto">
          <table className="min-w-full border border-gray-200 rounded-lg">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-2 text-left text-sm font-medium text-gray-700 border-b">Container</th>
                <th className="px-4 py-2 text-left text-sm font-medium text-gray-700 border-b">Team</th>
                <th className="px-4 py-2 text-left text-sm font-medium text-gray-700 border-b">Discipline</th>
                <th className="px-4 py-2 text-left text-sm font-medium text-gray-700 border-b">Type</th>
                <th className="px-4 py-2 text-left text-sm font-medium text-gray-700 border-b">Milestone</th>
                <th className="px-4 py-2 text-left text-sm font-medium text-gray-700 border-b">Due Date</th>
                <th className="px-4 py-2 text-left text-sm font-medium text-gray-700 border-b">Status</th>
                <th className="px-4 py-2 text-left text-sm font-medium text-gray-700 border-b">Time</th>
              </tr>
            </thead>
            <tbody>
              {deliverables?.deliverables?.map((deliverable, index) => (
                <tr key={deliverable.id || index} className="border-b hover:bg-gray-50">
                  <td className="px-4 py-2 text-sm text-gray-900">{deliverable.containerName}</td>
                  <td className="px-4 py-2 text-sm text-gray-600">{deliverable.team}</td>
                  <td className="px-4 py-2 text-sm text-gray-600">{deliverable.discipline}</td>
                  <td className="px-4 py-2 text-sm text-gray-600">{deliverable.type}</td>
                  <td className="px-4 py-2 text-sm text-gray-600">{deliverable.milestone}</td>
                  <td className="px-4 py-2 text-sm text-gray-600">{formatDate(deliverable.dueDate)}</td>
                  <td className="px-4 py-2">
                    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(deliverable.status)}`}>
                      {deliverable.status}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-sm text-gray-600">{deliverable.estimatedTime}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl p-8">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <p className="mt-2 text-gray-600">Loading dashboard...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-100 rounded-lg shadow-xl max-w-7xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="bg-white border-b px-6 py-4">
          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold text-gray-900">MIDP Evolution Dashboard</h2>
            <button onClick={onClose} className="text-gray-500 hover:text-gray-700">✕</button>
          </div>

          {/* Navigation Tabs */}
          <div className="mt-4">
            <nav className="flex space-x-8">
              {[
                { id: 'overview', label: 'Overview', icon: TrendingUp },
                { id: 'deliverables', label: 'Deliverables', icon: BarChart3 }
              ].map(({ id, label, icon: Icon }) => (
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
        </div>

        <div className="p-6">
          {activeView === 'overview' && <OverviewView />}
          {activeView === 'deliverables' && <DeliverablesView />}
        </div>
      </div>
    </div>
  );
};

export default MIDPEvolutionDashboard;