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
    <div className="space-y-8">
      {/* MIDPs Header */}
      <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border-2 border-green-200 shadow-lg p-8">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
          <div>
            <h2 className="text-3xl font-bold text-gray-900 mb-3">Master Information Delivery Plans</h2>
            <p className="text-gray-700 text-lg font-medium">Aggregated project delivery schedules from multiple team plans</p>
          </div>
          <button
            onClick={onAutoGenerate}
            className="inline-flex items-center px-8 py-4 border border-transparent rounded-xl shadow-md text-base font-bold text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-all duration-300 hover:shadow-xl lg:shrink-0 transform hover:-translate-y-0.5"
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
            <div key={i} className="bg-white rounded-xl border-2 border-gray-200 shadow-lg p-8 animate-pulse">
              <div className="flex items-start justify-between mb-6">
                <div className="flex-1">
                  <div className="h-7 bg-gray-200 rounded-lg w-3/4 mb-3"></div>
                  <div className="h-5 bg-gray-200 rounded-lg w-1/2"></div>
                </div>
                <div className="h-7 bg-gray-200 rounded-full w-20"></div>
              </div>
              <div className="h-5 bg-gray-200 rounded-lg mb-6"></div>
              <div className="space-y-4 mb-8">
                <div className="flex justify-between">
                  <div className="h-5 bg-gray-200 rounded-lg w-28"></div>
                  <div className="h-5 bg-gray-200 rounded-lg w-10"></div>
                </div>
                <div className="flex justify-between">
                  <div className="h-5 bg-gray-200 rounded-lg w-32"></div>
                  <div className="h-5 bg-gray-200 rounded-lg w-12"></div>
                </div>
                <div className="flex justify-between">
                  <div className="h-5 bg-gray-200 rounded-lg w-24"></div>
                  <div className="h-5 bg-gray-200 rounded-lg w-20"></div>
                </div>
              </div>
              <div className="flex space-x-3">
                <div className="flex-1 h-12 bg-gray-200 rounded-xl"></div>
                <div className="h-12 w-12 bg-gray-200 rounded-xl"></div>
                <div className="h-12 w-12 bg-gray-200 rounded-xl"></div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {filteredMidps.map((midp, index) => (
            <div key={midp.id || index} className="bg-white rounded-xl border-2 border-gray-200 shadow-lg hover:shadow-2xl hover:border-green-400 transition-all duration-300 group transform hover:-translate-y-1">
              <div className="p-8">
                <div className="flex items-start justify-between mb-6">
                  <div className="flex-1">
                    <h3 className="text-2xl font-bold text-gray-900 mb-2 group-hover:text-green-600 transition-colors leading-tight">{midp.projectName || `MIDP ${index + 1}`}</h3>
                    <p className="text-base text-gray-600 font-semibold">{midp.includedTIDPs?.length || 0} TIDPs included</p>
                  </div>
                  <span className="inline-flex items-center px-4 py-1.5 rounded-full text-sm font-bold bg-green-100 text-green-800 shadow-sm">
                    {midp.status || 'Active'}
                  </span>
                </div>

                <p className="text-gray-700 text-base mb-6 leading-relaxed line-clamp-2">
                  {midp.description || 'Master information delivery plan aggregating multiple team plans for comprehensive project coordination.'}
                </p>

                <div className="space-y-4 mb-8 bg-gray-50 rounded-xl p-5">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 font-semibold text-base">Total Deliverables:</span>
                    <span className="font-bold text-gray-900 text-lg">{midp.aggregatedData?.totalContainers || 0}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 font-semibold text-base">Estimated Hours:</span>
                    <span className="font-bold text-gray-900 text-lg">{midp.aggregatedData?.totalEstimatedHours || 0}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 font-semibold text-base">Last Updated:</span>
                    <span className="font-bold text-gray-900 text-base">{new Date(midp.updatedAt).toLocaleDateString()}</span>
                  </div>
                </div>

                <div className="flex space-x-3">
                  <button
                    onClick={() => onViewDetails(midp.id)}
                    className="flex-1 bg-green-600 text-white font-bold py-3.5 px-4 rounded-xl hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-all duration-300 hover:shadow-lg transform hover:scale-105"
                  >
                    View Details
                  </button>
                  <button
                    onClick={() => onViewEvolution(midp.id)}
                    className="bg-blue-600 text-white p-3.5 rounded-xl hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-300 transform hover:scale-105"
                    title="Evolution Dashboard"
                  >
                    <TrendingUp className="w-5 h-5" />
                  </button>
                  <button
                    onClick={() => onDownloadMidp(midp)}
                    className="bg-gray-100 text-gray-700 p-3.5 rounded-xl hover:bg-green-100 hover:text-green-600 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-all duration-300 transform hover:scale-105"
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
