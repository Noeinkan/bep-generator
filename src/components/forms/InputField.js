import OrgStructureField from './OrgStructureField';
// ...existing code...
import React from 'react';
import CONFIG from '../../config/bepConfig';
import EditableTable from './EditableTable';
import IntroTableField from './IntroTableField';
import FileStructureDiagram from './FileStructureDiagram';
import CDEDiagramBuilder from './CDEDiagramBuilder';
import VolumeStrategyMindmap from './VolumeStrategyMindmap';
import TipTapEditor from './TipTapEditor';
import TimelineInput from './TimelineInput';
import BudgetInput from './BudgetInput';
import FieldHelpTooltip from './FieldHelpTooltip';
import StandardsTable from '../StandardsTable';
import HELP_CONTENT from '../../data/helpContentData';

const InputField = React.memo(({ field, value, onChange, error, formData = {} }) => {
  const { name, label, type, required, rows, placeholder, options: fieldOptions } = field;
  const optionsList = fieldOptions ? CONFIG.options[fieldOptions] : null;
  const helpContent = HELP_CONTENT[name]; // Get help content for this field

  const baseClasses = "w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500";

  const handleCheckboxChange = (option) => {
    const current = Array.isArray(value) ? value : [];
    const updated = current.includes(option)
      ? current.filter(item => item !== option)
      : [...current, option];
    onChange(name, updated);
  };

  // Helper component for label with optional help tooltip
  const FieldLabel = ({ htmlFor, children, className = "block text-sm font-medium mb-2" }) => (
    <div className="flex items-center gap-2 mb-2">
      <label htmlFor={htmlFor} className={className}>
        {children}
      </label>
      {helpContent && (
        <FieldHelpTooltip fieldName={name} helpContent={helpContent} />
      )}
    </div>
  );

  switch (type) {
    case 'timeline':
      return (
        <TimelineInput
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'budget':
      return (
        <BudgetInput
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'orgchart':
      return (
        <OrgStructureField
          field={field}
          value={value}
          onChange={(v) => {
            // If the org chart component emits an object with leadAppointedParty and finalizedParties,
            // persist those into separate form fields expected elsewhere in the app.
            if (v && typeof v === 'object') {
              if (v.leadAppointedParty !== undefined) {
                onChange('leadAppointedParty', v.leadAppointedParty);
              }
              if (v.finalizedParties !== undefined) {
                onChange('finalizedParties', v.finalizedParties);
              }
              // Also keep the organizationalStructure field (the org tree) for compatibility
              if (v.tree !== undefined) {
                onChange(name, v.tree);
              } else {
                onChange(name, v);
              }
            } else {
              onChange(name, v);
            }
          }}
          formData={formData}
        />
      );
    case 'table':
      return (
        <EditableTable
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'standardsTable':
      return (
        <StandardsTable
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'introTable':
      return (
        <IntroTableField
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'fileStructure':
      return (
        <FileStructureDiagram
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'cdeDiagram':
      return (
        <CDEDiagramBuilder
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'mindmap':
      return (
        <VolumeStrategyMindmap
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'textarea':
      return (
        <div>
          <FieldLabel htmlFor={name}>
            {label} {required && '*'}
          </FieldLabel>
          <TipTapEditor
            id={name}
            aria-required={required}
            value={value || ''}
            onChange={(newValue) => onChange(name, newValue)}
            className=""
            placeholder={placeholder || `Enter ${label.toLowerCase()}...`}
            minHeight={`${(rows || 3) * 24}px`}
            autoSaveKey={`tiptap-${name}`}
            fieldName={name}
          />
          {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
        </div>
      );

    case 'select':
      return (
        <div>
          <FieldLabel htmlFor={name}>
            {label} {required && '*'}
          </FieldLabel>
          <select
            id={name}
            aria-required={required}
            value={value || ''}
            onChange={(e) => onChange(name, e.target.value)}
            className={baseClasses}
          >
            <option value="">Select {label.toLowerCase()}</option>
            {optionsList?.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
          {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
        </div>
      );

    case 'checkbox':
      return (
        <div>
          <FieldLabel>
            {label} {required && '*'}
          </FieldLabel>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-60 overflow-y-auto border rounded-lg p-3">
            {optionsList?.map(option => (
              <label key={option} htmlFor={`${name}-${option}`} className="flex items-center space-x-2 p-2 border rounded cursor-pointer hover:bg-gray-50">
                <input
                  id={`${name}-${option}`}
                  type="checkbox"
                  checked={(value || []).includes(option)}
                  onChange={() => handleCheckboxChange(option)}
                  className="rounded"
                />
                <span className="text-sm">{option}</span>
              </label>
            ))}
          </div>
          {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
        </div>
      );

    default:
      return (
        <div>
          <FieldLabel htmlFor={name}>
            {label} {required && '*'}
          </FieldLabel>
          <input
            id={name}
            aria-required={required}
            type="text"
            value={value || ''}
            onChange={(e) => onChange(name, e.target.value)}
            className={baseClasses}
            placeholder={placeholder || `Enter ${label.toLowerCase()}`}
          />
          {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
        </div>
      );
  }
});

export default InputField;