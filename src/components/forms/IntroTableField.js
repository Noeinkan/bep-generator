import React from 'react';
import TipTapEditor from './TipTapEditor';
import EditableTable from './EditableTable';

const IntroTableField = React.memo(({ field, value, onChange, error }) => {
  const { name, label, required, introPlaceholder, tableColumns } = field;

  // The value should be an object with 'intro' and 'table' properties
  const currentValue = value || { intro: '', table: [] };
  const introValue = currentValue.intro || '';
  const tableValue = currentValue.table || [];

  const handleIntroChange = (newIntro) => {
    onChange(name, {
      ...currentValue,
      intro: newIntro
    });
  };

  const handleTableChange = (_, newTable) => {
    onChange(name, {
      ...currentValue,
      table: newTable
    });
  };

  return (
    <div className="w-full">
      <label className="block text-lg font-semibold mb-4 text-gray-800">
        {label} {required && <span className="text-red-500">*</span>}
      </label>

      {/* Intro Text Editor */}
      <div className="mb-4 w-full">
        <TipTapEditor
          value={introValue}
          onChange={handleIntroChange}
          className="w-full"
          placeholder={introPlaceholder || 'Enter introductory text...'}
          minHeight="72px"
          autoSaveKey={`intro-table-${name}-intro`}
          fieldName={`${name}-intro`}
        />
      </div>

      {/* Table - The EditableTable component has its own mb-8 */}
      <EditableTable
        field={{
          name: `${name}-table`,
          label: '',
          columns: tableColumns
        }}
        value={tableValue}
        onChange={handleTableChange}
        error={null}
      />

      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
});

export default IntroTableField;
