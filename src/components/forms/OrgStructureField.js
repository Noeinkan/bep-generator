
import React, { useState } from 'react';
import OrgStructureChart from '../visual/OrgStructureChart';


const OrgStructureField = ({ value, onChange, field, intro, onIntroChange }) => {


  return (
  <div className="w-full" style={{ minWidth: 0, maxWidth: '100%' }}>
      <label className="block text-sm font-medium mb-2">
        {field.label} {field.required && '*'}
      </label>
      <textarea
        className="mb-3 w-full border rounded p-2 text-gray-700 text-sm"
        value={intro}
        onChange={e => onIntroChange && onIntroChange(e.target.value)}
        rows={3}
        placeholder="Enter introductory passage..."
        style={{ resize: 'vertical' }}
      />
      <OrgStructureChart data={value} onChange={onChange} editable />
    </div>
  );
};

export default OrgStructureField;
