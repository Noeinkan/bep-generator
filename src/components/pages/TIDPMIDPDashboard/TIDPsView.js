import React from 'react';
import { Plus, Download, Search, Users } from 'lucide-react';

const TIDPsView = ({
  tidps,
  loading,
  searchTerm,
  onSearchChange,
  filterDiscipline,
  onFilterChange,
  disciplines,
  onCreateNew,
  onDownloadTemplate,
  onViewDetails,
  onDownloadTidp
}) => {
  const filteredTidps = tidps.filter(tidp => {
    const matchesSearch = !searchTerm ||
      tidp.teamName?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      tidp.description?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      tidp.discipline?.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesDiscipline = filterDiscipline === 'all' || tidp.discipline === filterDiscipline;

    return matchesSearch && matchesDiscipline;
  });

  return (
    <div className="space-y-6">
      {/* Filters and Search */}
      <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-8 mb-8">
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="flex-1">
            <label className="block text-sm font-semibold text-gray-700 mb-2">Search TIDPs</label>
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search by team name, description, or discipline..."
                value={searchTerm}
                onChange={(e) => onSearchChange(e.target.value)}
                className="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-base"
              />
            </div>
          </div>

          <div className="flex items-end space-x-4">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">Filter by Discipline</label>
              <select
                value={filterDiscipline}
                onChange={(e) => onFilterChange(e.target.value)}
                className="border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent text-base min-w-48"
              >
                <option value="all">All Disciplines</option>
                {disciplines.map(discipline => (
                  <option key={discipline} value={discipline}>{discipline}</option>
                ))}
              </select>
            </div>

            <button
              onClick={onCreateNew}
              className="inline-flex items-center px-8 py-3 border border-transparent rounded-lg shadow-sm text-base font-semibold text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 hover:shadow-md"
            >
              <Plus className="w-5 h-5 mr-3" />
              New TIDP
            </button>

            <button
              onClick={onDownloadTemplate}
              className="inline-flex items-center px-8 py-3 border border-purple-600 rounded-lg shadow-sm text-base font-semibold text-purple-600 bg-white hover:bg-purple-50 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 transition-all duration-200 hover:shadow-md"
              title="Download a CSV template with sample TIDP deliverables to fill and import"
            >
              <Download className="w-5 h-5 mr-3" />
              Download CSV Template
            </button>
          </div>
        </div>
      </div>

      {/* TIDPs Grid */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="bg-white rounded-lg border border-gray-200 shadow-sm p-8 animate-pulse">
              <div className="flex items-start justify-between mb-6">
                <div className="flex-1">
                  <div className="h-6 bg-gray-200 rounded w-3/4 mb-2"></div>
                  <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                </div>
                <div className="h-6 bg-gray-200 rounded w-16"></div>
              </div>
              <div className="space-y-2 mb-6">
                <div className="h-4 bg-gray-200 rounded"></div>
                <div className="h-4 bg-gray-200 rounded w-5/6"></div>
              </div>
              <div className="space-y-3 mb-8">
                <div className="flex justify-between">
                  <div className="h-4 bg-gray-200 rounded w-20"></div>
                  <div className="h-4 bg-gray-200 rounded w-8"></div>
                </div>
                <div className="flex justify-between">
                  <div className="h-4 bg-gray-200 rounded w-16"></div>
                  <div className="h-4 bg-gray-200 rounded w-12"></div>
                </div>
              </div>
              <div className="flex space-x-3">
                <div className="flex-1 h-10 bg-gray-200 rounded-lg"></div>
                <div className="h-10 w-10 bg-gray-200 rounded-lg"></div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {filteredTidps.map((tidp, index) => (
            <div key={tidp.id || index} className="bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-lg hover:border-blue-300 transition-all duration-200 group">
              <div className="p-8">
                <div className="flex items-start justify-between mb-6">
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-gray-900 mb-2 group-hover:text-blue-700 transition-colors">{tidp.teamName || `TIDP ${index + 1}`}</h3>
                    <p className="text-gray-600 font-medium">{tidp.discipline}</p>
                  </div>
                  <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold bg-blue-100 text-blue-800">
                    {tidp.status || 'Draft'}
                  </span>
                </div>

                <p className="text-gray-700 text-base mb-6 line-clamp-3 leading-relaxed">
                  {tidp.description || tidp.responsibilities || 'Task information delivery plan for team coordination and deliverables management.'}
                </p>

                <div className="space-y-3 mb-8">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 font-medium">Deliverables:</span>
                    <span className="font-bold text-gray-900">{tidp.containers?.length || 0}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 font-medium">Team Leader:</span>
                    <span className="font-bold text-gray-900">{tidp.leader || 'TBD'}</span>
                  </div>
                </div>

                <div className="flex space-x-3">
                  <button
                    onClick={() => onViewDetails(tidp.id)}
                    className="flex-1 bg-blue-600 text-white font-semibold py-3 px-4 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 hover:shadow-md"
                  >
                    View Details
                  </button>
                  <button
                    onClick={() => onDownloadTidp(tidp)}
                    className="bg-gray-100 text-gray-700 p-3 rounded-lg hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-all duration-200"
                    title="Download TIDP as CSV"
                  >
                    <Download className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {filteredTidps.length === 0 && !loading && (
        <div className="text-center py-16">
          <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-8">
            <Users className="w-10 h-10 text-gray-400" />
          </div>
          <h3 className="text-2xl font-bold text-gray-900 mb-4">No TIDPs found</h3>
          <p className="text-gray-600 text-lg mb-8 max-w-md mx-auto">Try adjusting your search terms or filters, or create a new TIDP to get started.</p>
          <button
            onClick={onCreateNew}
            className="inline-flex items-center px-8 py-4 bg-blue-600 text-white font-semibold text-lg rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 hover:shadow-lg"
          >
            <Plus className="w-6 h-6 mr-3" />
            Create New TIDP
          </button>
        </div>
      )}
    </div>
  );
};

export default TIDPsView;
