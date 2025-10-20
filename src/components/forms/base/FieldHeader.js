import React from 'react';
import FieldHelpTooltip from '../controls/FieldHelpTooltip';
import { getHelpContent } from '../../../data/helpContent';

/**
 * Standardized field header component with label, number, required indicator, and help tooltip
 * Used across InputField and IntroTableField for consistency
 *
 * When asSectionHeader is true, renders with prominent section header styling
 */
const FieldHeader = ({
  fieldName,
  label,
  number,
  required = false,
  className = "block text-sm font-medium mb-2",
  htmlFor,
  asSectionHeader = false
}) => {
  const helpContent = getHelpContent(fieldName);

  // Section header style (for subsections like 9.2.1, 9.2.2, etc.)
  if (asSectionHeader) {
    return (
      <div className="border-b border-gray-200 pb-3 mb-4">
        <div className="flex items-center gap-2">
          <h4 className="text-base font-semibold text-gray-900">
            {number && <span className="text-blue-600">{number} </span>}
            {label}
            {required && <span className="text-red-500 ml-1">*</span>}
          </h4>
          {helpContent && (
            <FieldHelpTooltip fieldName={fieldName} helpContent={helpContent} />
          )}
        </div>
      </div>
    );
  }

  // Standard field header style
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
