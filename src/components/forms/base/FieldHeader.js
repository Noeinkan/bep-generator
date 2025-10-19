import React from 'react';
import FieldHelpTooltip from '../controls/FieldHelpTooltip';
import { getHelpContent } from '../../../data/helpContent';

/**
 * Standardized field header component with label, number, required indicator, and help tooltip
 * Used across InputField and IntroTableField for consistency
 */
const FieldHeader = ({ 
  fieldName, 
  label, 
  number, 
  required = false, 
  className = "block text-sm font-medium mb-2",
  htmlFor 
}) => {
  const helpContent = getHelpContent(fieldName);

  return (
    <div className="flex items-center gap-2 mb-2">
      <label htmlFor={htmlFor} className={className}>
        {number ? `${number} ` : ''}{label} {required && <span className="text-red-500">*</span>}
      </label>
      {helpContent && (
        <FieldHelpTooltip fieldName={fieldName} helpContent={helpContent} />
      )}
    </div>
  );
};

export default FieldHeader;
