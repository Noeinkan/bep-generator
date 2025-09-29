import React from 'react';

const Toast = ({ open, message, type = 'info', onClose }) => {
  if (!open) return null;
  let bg = 'bg-blue-600';
  if (type === 'error') bg = 'bg-red-600';
  if (type === 'success') bg = 'bg-green-600';
  return (
    <div className={`fixed bottom-6 right-6 z-50 px-6 py-3 rounded shadow-lg text-white ${bg}`}>
      <div className="flex items-center space-x-3">
        <span>{message}</span>
        <button onClick={onClose} className="ml-2 text-white hover:text-gray-200">&times;</button>
      </div>
    </div>
  );
};

export default Toast;
