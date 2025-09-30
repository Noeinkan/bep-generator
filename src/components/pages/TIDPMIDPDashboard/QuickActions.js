import React from 'react';
import { Users, Calendar, Upload } from 'lucide-react';

const QuickActions = ({ onViewTIDPs, onViewMIDPs, onImport }) => {
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
    },
    {
      icon: Upload,
      title: 'Import Data',
      description: 'Import from Excel/CSV',
      onClick: onImport,
      colorClass: 'purple'
    }
  ];

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-8">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Quick Actions</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {actions.map(({ icon: Icon, title, description, onClick, colorClass }) => (
          <button
            key={title}
            onClick={onClick}
            className={`group flex items-center p-6 border-2 border-gray-200 rounded-lg hover:border-${colorClass}-400 hover:bg-${colorClass}-50 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-${colorClass}-500 focus:border-transparent`}
          >
            <div className={`p-3 bg-${colorClass}-100 rounded-lg group-hover:bg-${colorClass}-200 transition-colors`}>
              <Icon className={`w-6 h-6 text-${colorClass}-600`} />
            </div>
            <div className="ml-4 text-left">
              <p className="font-bold text-gray-900 text-lg">{title}</p>
              <p className="text-gray-600 mt-1">{description}</p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default QuickActions;
