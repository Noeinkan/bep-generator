import React from 'react';
import { Plus, Download, TrendingUp, Calendar } from 'lucide-react';

const MIDPsView = ({
  midps,
  loading,
  searchTerm,
  onAutoGenerate,
  onViewDetails,
  onViewEvolution,
  onDownloadMidp
}) => {
  const filteredMidps = midps.filter(midp => {
    return !searchTerm ||
      midp.projectName?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      midp.description?.toLowerCase().includes(searchTerm.toLowerCase());
  });

  return (
    <div className="space-y-6">
      {/* MIDPs Header */}
      <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-8 mb-8">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Master Information Delivery Plans</h2>
            <p className="text-gray-600 text-lg">Aggregated project delivery schedules from multiple team plans</p>
          </div>
          <button
            onClick={onAutoGenerate}
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
          {filteredMidps.map((midp, index) => (
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
                  <button
                    onClick={() => onViewDetails(midp.id)}
                    className="flex-1 bg-green-600 text-white font-semibold py-3 px-4 rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-all duration-200 hover:shadow-md"
                  >
                    View Details
                  </button>
                  <button
                    onClick={() => onViewEvolution(midp.id)}
                    className="bg-blue-600 text-white p-3 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200"
                    title="Evolution Dashboard"
                  >
                    <TrendingUp className="w-5 h-5" />
                  </button>
                  <button
                    onClick={() => onDownloadMidp(midp)}
                    className="bg-gray-100 text-gray-700 p-3 rounded-lg hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-all duration-200"
                    title="Download MIDP Report"
                  >
                    <Download className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {filteredMidps.length === 0 && !loading && (
        <div className="text-center py-16">
          <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-8">
            <Calendar className="w-10 h-10 text-gray-400" />
          </div>
          <h3 className="text-2xl font-bold text-gray-900 mb-4">No MIDPs created yet</h3>
          <p className="text-gray-600 text-lg mb-8 max-w-md mx-auto">Generate your first Master Information Delivery Plan by aggregating existing TIDPs.</p>
          <button
            onClick={onAutoGenerate}
            className="inline-flex items-center px-8 py-4 bg-green-600 text-white font-semibold text-lg rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-all duration-200 hover:shadow-lg"
          >
            <Plus className="w-6 h-6 mr-3" />
            Generate First MIDP
          </button>
        </div>
      )}
    </div>
  );
};

export default MIDPsView;
