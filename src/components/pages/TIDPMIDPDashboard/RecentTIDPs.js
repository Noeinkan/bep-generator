import React from 'react';
import { Users, Plus } from 'lucide-react';

const RecentTIDPs = ({ tidps, onCreateNew }) => {
  if (tidps.length === 0) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Recent TIDPs</h2>
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <Users className="w-8 h-8 text-gray-400" />
          </div>
          <p className="text-xl font-semibold text-gray-900 mb-2">No TIDPs created yet</p>
          <p className="text-gray-600 mb-6">Get started by creating your first Team Information Delivery Plan</p>
          <button
            onClick={onCreateNew}
            className="inline-flex items-center px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200"
          >
            <Plus className="w-5 h-5 mr-2" />
            Create your first TIDP
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-8">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Recent TIDPs</h2>
      <div className="space-y-4">
        {tidps.slice(0, 5).map((tidp, index) => (
          <div key={tidp.id || index} className="flex items-center justify-between p-6 border border-gray-200 rounded-lg hover:border-blue-300 hover:shadow-sm transition-all duration-200">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                <Users className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <h3 className="font-bold text-gray-900 text-lg">{tidp.teamName || `TIDP ${index + 1}`}</h3>
                <p className="text-gray-600">{tidp.discipline} • {tidp.containers?.length || 0} deliverables</p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm font-medium text-gray-900">
                {new Date(tidp.updatedAt).toLocaleDateString()}
              </div>
              <div className="text-sm text-gray-500">Last updated</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RecentTIDPs;
