import React from 'react';
import { Users, Calendar, FileText, TrendingUp } from 'lucide-react';

const StatisticsCards = ({ stats, loading }) => {
  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-white rounded-lg border border-gray-200 shadow-sm p-6 animate-pulse">
            <div className="flex items-center">
              <div className="w-12 h-12 bg-gray-200 rounded-lg"></div>
              <div className="ml-4 flex-1">
                <div className="h-8 bg-gray-200 rounded w-16 mb-2"></div>
                <div className="h-4 bg-gray-200 rounded w-20"></div>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  const cards = [
    {
      icon: Users,
      value: stats.totalTidps,
      label: 'TIDPs',
      colorClass: 'blue'
    },
    {
      icon: Calendar,
      value: stats.totalMidps,
      label: 'MIDPs',
      colorClass: 'green'
    },
    {
      icon: FileText,
      value: stats.totalDeliverables,
      label: 'Deliverables',
      colorClass: 'purple'
    },
    {
      icon: TrendingUp,
      value: stats.activeMilestones,
      label: 'Milestones',
      colorClass: 'orange'
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {cards.map(({ icon: Icon, value, label, colorClass }) => (
        <div
          key={label}
          className={`bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-all duration-200 p-6 hover:border-${colorClass}-300`}
        >
          <div className="flex items-center">
            <div className={`p-3 bg-${colorClass}-50 rounded-lg`}>
              <Icon className={`w-8 h-8 text-${colorClass}-600`} />
            </div>
            <div className="ml-4">
              <p className="text-3xl font-bold text-gray-900">{value}</p>
              <p className="text-gray-600 font-medium">{label}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default StatisticsCards;
