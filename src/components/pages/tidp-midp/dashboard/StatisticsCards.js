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

  const colorStyles = {
    blue: {
      border: 'border-blue-200',
      hoverBorder: 'hover:border-blue-400',
      bg: 'bg-blue-50',
      text: 'text-blue-600',
      hoverShadow: 'hover:shadow-blue-100/50'
    },
    green: {
      border: 'border-green-200',
      hoverBorder: 'hover:border-green-400',
      bg: 'bg-green-50',
      text: 'text-green-600',
      hoverShadow: 'hover:shadow-green-100/50'
    },
    purple: {
      border: 'border-purple-200',
      hoverBorder: 'hover:border-purple-400',
      bg: 'bg-purple-50',
      text: 'text-purple-600',
      hoverShadow: 'hover:shadow-purple-100/50'
    },
    orange: {
      border: 'border-orange-200',
      hoverBorder: 'hover:border-orange-400',
      bg: 'bg-orange-50',
      text: 'text-orange-600',
      hoverShadow: 'hover:shadow-orange-100/50'
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {cards.map(({ icon: Icon, value, label, colorClass }) => {
        const styles = colorStyles[colorClass];
        return (
          <div
            key={label}
            className={`bg-white rounded-xl border-2 ${styles.border} ${styles.hoverBorder} shadow-md hover:shadow-xl ${styles.hoverShadow} transition-all duration-300 p-6 transform hover:-translate-y-1`}
          >
            <div className="flex items-center">
              <div className={`p-4 ${styles.bg} rounded-xl shadow-sm`}>
                <Icon className={`w-8 h-8 ${styles.text}`} />
              </div>
              <div className="ml-5">
                <p className="text-3xl font-bold text-gray-900 tracking-tight">{value}</p>
                <p className="text-base text-gray-600 font-semibold mt-1">{label}</p>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default StatisticsCards;
