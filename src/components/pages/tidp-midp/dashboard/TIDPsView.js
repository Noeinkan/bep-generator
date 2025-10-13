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
    <div className="space-y-8">
      {/* Filters and Search */}
      <div className="bg-white rounded-xl border-2 border-gray-300 shadow p-6">
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="flex-1">
            <label className="block text-base font-bold text-gray-900 mb-3">Search TIDPs</label>
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search by team name, description, or discipline..."
                value={searchTerm}
                onChange={(e) => onSearchChange(e.target.value)}
                aria-label="Search TIDPs"
                className="w-full pl-12 pr-4 py-3.5 border-2 border-gray-300 rounded-xl focus:outline-none focus:ring-3 focus:ring-blue-200 focus:border-blue-500 text-base shadow-sm transition-all duration-200"
              />
            </div>
          </div>

          <div className="flex items-end space-x-4">
            <div>
              <label className="block text-base font-bold text-gray-900 mb-3">Filter by Discipline</label>
              <select
                value={filterDiscipline}
                onChange={(e) => onFilterChange(e.target.value)}
                className="border-2 border-gray-300 rounded-xl px-5 py-3.5 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-base min-w-48 font-medium shadow-sm transition-all duration-200"
              >
                <option value="all">All Disciplines</option>
                {disciplines.map(discipline => (
                  <option key={discipline} value={discipline}>{discipline}</option>
                ))}
              </select>
            </div>

            <button
              onClick={onCreateNew}
              className="inline-flex items-center px-6 py-3 border border-transparent rounded-xl shadow text-base font-bold text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-200 focus:ring-offset-2 transition-all duration-300 hover:shadow-lg transform hover:-translate-y-0.5"
              aria-label="Create new TIDP"
            >
              <Plus className="w-5 h-5 mr-3" />
              Create New TIDP
            </button>

            <button
              onClick={onDownloadTemplate}
              className="inline-flex items-center px-8 py-3.5 border-2 border-purple-600 rounded-xl shadow-md text-base font-bold text-purple-600 bg-white hover:bg-purple-50 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 transition-all duration-300 hover:shadow-xl transform hover:-translate-y-0.5"
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
            <div key={i} className="bg-white rounded-xl border-2 border-gray-300 shadow p-6 animate-pulse">
              <div className="flex items-start justify-between mb-6">
                <div className="flex-1">
                  <div className="h-7 bg-gray-200 rounded-lg w-3/4 mb-3"></div>
                  <div className="h-5 bg-gray-200 rounded-lg w-1/2"></div>
                </div>
                <div className="h-7 bg-gray-200 rounded-full w-20"></div>
              </div>
              <div className="space-y-3 mb-6">
                <div className="h-5 bg-gray-200 rounded-lg"></div>
                <div className="h-5 bg-gray-200 rounded-lg w-5/6"></div>
              </div>
              <div className="space-y-4 mb-8">
                <div className="flex justify-between">
                  <div className="h-5 bg-gray-200 rounded-lg w-24"></div>
                  <div className="h-5 bg-gray-200 rounded-lg w-10"></div>
                </div>
                <div className="flex justify-between">
                  <div className="h-5 bg-gray-200 rounded-lg w-20"></div>
                  <div className="h-5 bg-gray-200 rounded-lg w-16"></div>
                </div>
              </div>
              <div className="flex space-x-3">
                <div className="flex-1 h-12 bg-gray-200 rounded-xl"></div>
                <div className="h-12 w-12 bg-gray-200 rounded-xl"></div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {filteredTidps.map((tidp, index) => (
            <div
              key={tidp.id || index}
              role="button"
              tabIndex="0"
              aria-label={`Apri dettagli per ${tidp.teamName || `TIDP ${index + 1}`}`}
              onClick={() => onViewDetails(tidp.id)}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onViewDetails(tidp.id); }}
              className="bg-white rounded-xl border-2 border-gray-300 shadow-sm hover:shadow-2xl hover:border-blue-400 transition-all duration-300 group transform hover:-translate-y-1 focus:outline-none focus:ring-4 focus:ring-blue-200 focus:ring-offset-2 cursor-pointer"
            >
              <div className="p-6">
                <div className="flex items-start justify-between mb-6">
                  <div className="flex-1">
                    <h3 className="text-3xl font-extrabold text-gray-900 mb-2 group-hover:text-blue-600 transition-colors leading-tight">{tidp.teamName || `TIDP ${index + 1}`}</h3>
                    <p className="text-base text-gray-600 font-semibold">{tidp.discipline}</p>
                  </div>
                  <span className="inline-flex items-center px-4 py-1.5 rounded-full text-sm font-bold bg-blue-100 text-blue-800 shadow-sm">
                    {tidp.status || 'Draft'}
                  </span>
                </div>

                <p className="text-gray-700 text-base mb-6 line-clamp-3 leading-relaxed">
                  {tidp.description || tidp.responsibilities || 'Task information delivery plan for team coordination and deliverables management.'}
                </p>

                <div className="space-y-4 mb-8 bg-gray-50 rounded-xl p-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 font-semibold text-base">Deliverables:</span>
                    <span className="font-bold text-gray-900 text-lg">{tidp.containers?.length || 0}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 font-semibold text-base">Team Leader:</span>
                    <span className="font-bold text-gray-900 text-base">{tidp.leader || 'TBD'}</span>
                  </div>
                </div>

                <div className="flex space-x-3">
                  <button
                    onClick={(e) => { e.stopPropagation(); onViewDetails(tidp.id); }}
                    className="flex-1 bg-blue-600 text-white font-bold py-3 px-4 rounded-xl hover:bg-blue-700 focus:outline-none focus:ring-3 focus:ring-blue-200 focus:ring-offset-2 transition-all duration-300 hover:shadow-lg transform hover:scale-105"
                    aria-label={`Select the TIDP ${tidp.teamName || index + 1}`}
                  >
                    Select this TIDP
                  </button>
                  <button
                    onClick={(e) => { e.stopPropagation(); onDownloadTidp(tidp); }}
                    className="bg-gray-100 text-gray-700 p-3.5 rounded-xl hover:bg-blue-100 hover:text-blue-600 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-all duration-300 transform hover:scale-105"
                    title="Download TIDP as CSV"
                    aria-label={`Download ${tidp.teamName || `TIDP ${index + 1}`}`}
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
          <h3 className="text-3xl font-extrabold text-gray-900 mb-4">No TIDPs found</h3>
          <p className="text-gray-600 text-base mb-8 max-w-md mx-auto">Try adjusting your search terms or filters. You can also create a new TIDP to get started.</p>
          <div>
            <button
              onClick={onCreateNew}
              className="inline-flex items-center px-5 py-2 bg-blue-600 text-white font-semibold text-base rounded-md hover:bg-blue-700 focus:outline-none focus:ring-3 focus:ring-blue-200 focus:ring-offset-2 transition-all duration-200"
              aria-label="Create new TIDP"
            >
              <Plus className="w-5 h-5 mr-2" />
              Create New TIDP
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default TIDPsView;
