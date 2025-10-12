
import React from 'react';
import OrgStructureChart from './OrgStructureChart';



const OrgStructureField = ({ value, onChange, field, intro, onIntroChange, formData }) => {
  // Pass the full formData object to OrgStructureChart for real data
  return (
    <div className="w-full" style={{ minWidth: 0, maxWidth: '100%' }}>
      <label className="block text-sm font-medium mb-2">
        {field.label} {field.required && '*'}
      </label>
      <OrgStructureChart data={formData} onChange={onChange} editable />
    </div>
  );
};

export default OrgStructureField;
