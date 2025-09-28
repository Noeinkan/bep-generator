import React from 'react';
import InputField from '../forms/InputField';
import CONFIG from '../../config/bepConfig';

const FormStep = React.memo(({ stepIndex, formData, updateFormData, errors, bepType }) => {
  const stepConfig = CONFIG.getFormFields(bepType, stepIndex);
  if (!stepConfig) return null;

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold">{stepConfig.title}</h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {stepConfig.fields.map(field => (
          <div key={field.name} className={field.type === 'textarea' || field.type === 'checkbox' || field.type === 'table' || field.type === 'fileStructure' || field.type === 'cdeDiagram' || field.type === 'mindmap' || field.type === 'orgchart' ? 'md:col-span-2' : ''}>
            <InputField
              field={field}
              value={formData[field.name]}
              onChange={updateFormData}
              error={errors[field.name]}
            />
          </div>
        ))}
      </div>
    </div>
  );
});

export default FormStep;