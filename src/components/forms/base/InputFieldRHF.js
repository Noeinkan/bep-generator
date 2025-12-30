import React from 'react';
import { Controller, useFormContext } from 'react-hook-form';
import InputField from './InputField';

/**
 * Wrapper around InputField that integrates with React Hook Form
 * Uses Controller to manage the field value and validation
 */
const InputFieldRHF = ({ field, error }) => {
  const { control, watch, setValue } = useFormContext();

  // Watch all form data to pass to InputField (some components need full formData)
  const formData = watch();

  // Safety check: if field doesn't have a name, render nothing
  if (!field || !field.name) {
    console.warn('InputFieldRHF: field or field.name is missing', field);
    return null;
  }

  // Determine default value based on field type
  const getDefaultValue = () => {
    if (field.type === 'checkbox') return [];
    if (field.type === 'orgchart') {
      return {
        id: 'appointing_default',
        name: 'Appointing Party',
        role: 'Appointing Party',
        leadGroups: []
      };
    }
    return '';
  };

  return (
    <Controller
      name={field.name}
      control={control}
      defaultValue={getDefaultValue()}
      render={({ field: { value, onChange: rhfOnChange } }) => {
        // Create onChange handler that works with both signatures
        // and supports updating multiple fields (for orgchart)
        const handleChange = (fieldNameOrValue, fieldValue) => {
          // If called with two arguments (fieldName, value)
          if (fieldValue !== undefined) {
            // Use setValue to update a different field
            setValue(fieldNameOrValue, fieldValue, { shouldDirty: true, shouldValidate: true });
          } else {
            // Single argument - update the current field
            rhfOnChange(fieldNameOrValue);
          }
        };

        return (
          <InputField
            field={field}
            value={value}
            onChange={handleChange}
            error={error}
            formData={formData}
          />
        );
      }}
    />
  );
};

export default InputFieldRHF;
