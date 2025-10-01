import React from 'react';
import { Users, Calendar, Upload } from 'lucide-react';

const QuickActions = ({ onViewTIDPs, onViewMIDPs, onImport, showImport = true }) => {
  const actions = [
    {
      icon: Users,
      title: 'Manage TIDPs',
      description: 'Create and edit team plans',
      onClick: onViewTIDPs,
      colorClass: 'blue'
    },
    {
      icon: Calendar,
      title: 'View MIDP',
      description: 'Monitor master plan',
      onClick: onViewMIDPs,
      colorClass: 'green'
    }
  ];

  // Only include the Import action when allowed (e.g., not on the main dashboard)
  if (showImport) {
    actions.push({
      icon: Upload,
      title: 'Import Data',
      description: 'Import from Excel/CSV',
      onClick: onImport,
      colorClass: 'purple'
    });
  }

  const colorStyles = {
    blue: {
      border: 'border-blue-200',
      hoverBorder: 'hover:border-blue-500',
      hoverBg: 'hover:bg-blue-50',
      iconBg: 'bg-blue-100',
      iconHoverBg: 'group-hover:bg-blue-500',
      iconText: 'text-blue-600',
      iconHoverText: 'group-hover:text-white',
      ring: 'focus:ring-blue-500'
    },
    green: {
      border: 'border-green-200',
      hoverBorder: 'hover:border-green-500',
      hoverBg: 'hover:bg-green-50',
      iconBg: 'bg-green-100',
      iconHoverBg: 'group-hover:bg-green-500',
      iconText: 'text-green-600',
      iconHoverText: 'group-hover:text-white',
      ring: 'focus:ring-green-500'
    },
    purple: {
      border: 'border-purple-200',
      hoverBorder: 'hover:border-purple-500',
      hoverBg: 'hover:bg-purple-50',
      iconBg: 'bg-purple-100',
      iconHoverBg: 'group-hover:bg-purple-500',
      iconText: 'text-purple-600',
      iconHoverText: 'group-hover:text-white',
      ring: 'focus:ring-purple-500'
    }
  };

  return (
    <div className="bg-white rounded-xl border-2 border-gray-200 shadow-lg p-8">
      <h2 className="text-3xl font-bold text-gray-900 mb-8">Quick Actions</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {actions.map(({ icon: Icon, title, description, onClick, colorClass }) => {
          const styles = colorStyles[colorClass];
          return (
            <button
              key={title}
              onClick={onClick}
              className={`group flex items-center p-6 border-2 ${styles.border} ${styles.hoverBorder} ${styles.hoverBg} rounded-xl shadow-md hover:shadow-xl transition-all duration-300 focus:outline-none ${styles.ring} focus:ring-2 focus:ring-offset-2 transform hover:-translate-y-1`}
            >
              <div className={`p-4 ${styles.iconBg} ${styles.iconHoverBg} rounded-xl transition-all duration-300 shadow-sm`}>
                <Icon className={`w-7 h-7 ${styles.iconText} ${styles.iconHoverText} transition-colors duration-300`} />
              </div>
              <div className="ml-5 text-left">
                <p className="font-bold text-gray-900 text-lg mb-1">{title}</p>
                <p className="text-gray-600 text-base leading-relaxed">{description}</p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default QuickActions;
