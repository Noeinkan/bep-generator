import React from 'react';
import InputField from '../forms/InputField';
import CONFIG from '../../config/bepConfig';
import InformationDeliveryPlanning from '../pages/InformationDeliveryPlanning';

// Field types that should span full width (both columns) in the grid layout
const FULL_WIDTH_FIELD_TYPES = [
  'textarea',
  'checkbox',
  'table',
  'introTable',
  'fileStructure',
  'cdeDiagram',
  'mindmap',
  'orgchart'
];

const FormStep = ({ stepIndex, formData, updateFormData, errors, bepType }) => {
  // Safety check - ensure we have the required props
  if (!formData || !bepType) {
    return <div>Loading...</div>;
  }

  // Step 5 is Information Delivery Planning - use specialized component
  if (stepIndex === 5) {
    return (
      <InformationDeliveryPlanning
        formData={formData}
        updateFormData={updateFormData}
        errors={errors}
        bepType={bepType}
      />
    );
  }

  const stepConfig = CONFIG.getFormFields(bepType, stepIndex);

  if (!stepConfig) {
    return <div>No configuration found for step {stepIndex}</div>;
  }

  if (!stepConfig.fields) {
    return <div>No fields configured for this step</div>;
  }

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold">{stepConfig.title}</h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {stepConfig.fields.map(field => (
          <div key={field.name} className={FULL_WIDTH_FIELD_TYPES.includes(field.type) ? 'md:col-span-2' : ''}>
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