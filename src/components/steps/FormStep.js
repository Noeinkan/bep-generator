import React from 'react';
import InputField from '../forms/base/InputField';
import CONFIG from '../../config/bepConfig';

// Field types that should span full width (both columns) in the grid layout
const FULL_WIDTH_FIELD_TYPES = [
  'textarea',
  'checkbox',
  'table',
  'introTable',
  'fileStructure',
  'cdeDiagram',
  'mindmap',
  'orgchart',
  'orgstructure-data-table',
  // Step 5 specialized types
  'milestones-table',
  'tidp-reference',
  'tidp-section',
  'deliverables-matrix',
  'im-activities-matrix',
  'im-strategy-builder',
  'naming-conventions'
];

// Field types that should span all 3 columns
const THREE_COLUMN_FIELD_TYPES = [
  'standardsTable'
];

const FormStep = ({ stepIndex, formData, updateFormData, errors, bepType }) => {
  // Safety check - ensure we have the required props
  if (!formData || !bepType) {
    return <div>Loading...</div>;
  }

  const stepConfig = CONFIG.getFormFields(bepType, stepIndex);

  if (!stepConfig) {
    return <div>No configuration found for step {stepIndex}</div>;
  }

  if (!stepConfig.fields) {
    return <div>No fields configured for this step</div>;
  }

  // Determine grid layout based on step
  // Appendices step (13) uses 3-column layout for landscape A4 equivalent
  // All other steps use 2-column layout for portrait A4 equivalent
  const isAppendicesStep = stepIndex === 13;
  const gridColsClass = isAppendicesStep ? 'md:grid-cols-3' : 'md:grid-cols-2';

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold">{stepConfig.number} {stepConfig.title}</h3>

      <div className={`grid grid-cols-1 ${gridColsClass} gap-4`}>
        {stepConfig.fields.map(field => (
          <div key={field.name} className={
            isAppendicesStep ? 'md:col-span-3' :
            THREE_COLUMN_FIELD_TYPES.includes(field.type) ? 'md:col-span-3' :
            FULL_WIDTH_FIELD_TYPES.includes(field.type) ? 'md:col-span-2' : ''
          }>
            <InputField
              field={field}
              value={formData ? formData[field.name] : ''}
              onChange={updateFormData}
              error={errors ? errors[field.name] : ''}
              formData={formData || {}}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default FormStep;