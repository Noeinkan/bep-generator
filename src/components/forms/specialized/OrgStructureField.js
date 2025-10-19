
import React from 'react';
import OrgStructureChart from '../diagrams/OrgStructureChart';
import FieldHeader from '../base/FieldHeader';



const OrgStructureField = ({ value, onChange, field, intro, onIntroChange, formData }) => {
  // Pass the full formData object to OrgStructureChart for real data
  return (
    <div className="w-full" style={{ minWidth: 0, maxWidth: '100%' }}>
      <FieldHeader 
        fieldName={field.name}
        label={field.label}
        number={field.number}
        required={field.required}
      />
      <OrgStructureChart data={formData} onChange={onChange} editable />
    </div>
  );
};

export default OrgStructureField;
