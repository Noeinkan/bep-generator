import React, { useState } from 'react';
import { FileText, Plus, Trash2, ChevronDown, ChevronUp, Info } from 'lucide-react';
import TipTapEditor from '../editors/TipTapEditor';

/**
 * NamingConventionBuilder
 * Component for Section 4.8.3: Naming and File Structure Standards
 * 
 * Provides a structured interface for defining:
 * 1. Naming Convention Fields - Components of the file naming pattern
 * 2. Deliverable Attributes - Metadata and properties associated with deliverables
 * 3. Folder Structure - Directory organization standards
 */
const NamingConventionBuilder = ({ field, value = {}, onChange, error, disabled = false }) => {
  // React hooks must be called before any conditional returns
  const [expandedSections, setExpandedSections] = useState({
    overview: true,
    namingFields: true,
    pattern: false,
    attributes: false,
    folderStructure: false
  });

  // Safety check (after all hooks)
  if (!field) {
    return <div className="text-red-600">Error: Field configuration is missing</div>;
  }

  const { name } = field;

  // Initialize with default structure
  const defaultValue = {
    overview: '<p>File naming follows <strong>ISO 19650-2</strong> convention to ensure consistency, traceability, and efficient information management across all project deliverables.</p>',
    namingFields: [
      {
        fieldName: 'Project Code',
        exampleValue: 'PRJ001',
        description: 'Unique project identifier assigned by the appointing party'
      },
      {
        fieldName: 'Originator',
        exampleValue: 'ARC',
        description: 'Organization/discipline creating the information (e.g., ARC=Architecture, STR=Structural, MEP=MEP)'
      },
      {
        fieldName: 'Volume/System',
        exampleValue: 'XX',
        description: 'Building zone, system, or spatial reference (XX if not applicable)'
      },
      {
        fieldName: 'Level/Location',
        exampleValue: 'GF',
        description: 'Floor level or location code (e.g., GF=Ground Floor, B1=Basement 1)'
      },
      {
        fieldName: 'Type',
        exampleValue: 'M3',
        description: 'Information type (e.g., M3=Model, DR=Drawing, SP=Specification, SC=Schedule)'
      },
      {
        fieldName: 'Role',
        exampleValue: 'ARC',
        description: 'Discipline or role responsible for the content'
      },
      {
        fieldName: 'Number',
        exampleValue: '0001',
        description: 'Sequential number for the deliverable'
      },
      {
        fieldName: 'Revision',
        exampleValue: 'P01',
        description: 'Revision status (e.g., P01=First Issue, C01=First Construction Issue)'
      }
    ],
    namingPattern: '<p><strong>Pattern:</strong> [Project Code]-[Originator]-[Volume/System]-[Level/Location]-[Type]-[Role]-[Number]-[Revision]</p><p><strong>Example:</strong> <code>PRJ001-ARC-XX-GF-M3-ARC-0001-P01.rvt</code></p>',
    deliverableAttributes: [
      {
        attributeName: 'File Format',
        exampleValue: '.rvt, .dwg, .pdf, .ifc',
        description: 'Acceptable file formats for each deliverable type'
      },
      {
        attributeName: 'Classification System',
        exampleValue: 'Uniclass 2015',
        description: 'Classification framework for organizing information'
      },
      {
        attributeName: 'Level of Information Need',
        exampleValue: 'LOD 300',
        description: 'Required level of detail/information for the deliverable'
      },
      {
        attributeName: 'Security Classification',
        exampleValue: 'Confidential',
        description: 'Information security level (e.g., Public, Internal, Confidential)'
      },
      {
        attributeName: 'Suitability Code',
        exampleValue: 'S2 - Suitable for Information',
        description: 'Document status/suitability per ISO 19650'
      }
    ],
    folderStructure: '<ul><li><strong>00_WIP</strong> - Work in Progress (active development, not shared)</li><li><strong>01_SHARED</strong> - Shared for review and coordination</li><li><strong>02_PUBLISHED</strong> - Published/Approved information</li><li><strong>03_ARCHIVE</strong> - Superseded versions and historical records</li></ul><p>Each folder follows the CDE workflow states aligned with ISO 19650-2 information container strategy.</p>'
  };

  // Handle different value types
  let currentValue = defaultValue;
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    currentValue = {
      overview: value.overview || defaultValue.overview,
      namingFields: value.namingFields || defaultValue.namingFields,
      namingPattern: value.namingPattern || defaultValue.namingPattern,
      deliverableAttributes: value.deliverableAttributes || defaultValue.deliverableAttributes,
      folderStructure: value.folderStructure || defaultValue.folderStructure
    };
  }

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  // Handle overview change
  const handleOverviewChange = (newValue) => {
    onChange(name, {
      ...currentValue,
      overview: newValue
    });
  };

  // Handle naming pattern change
  const handlePatternChange = (newValue) => {
    onChange(name, {
      ...currentValue,
      namingPattern: newValue
    });
  };

  // Handle folder structure change
  const handleFolderStructureChange = (newValue) => {
    onChange(name, {
      ...currentValue,
      folderStructure: newValue
    });
  };

  // Add new naming field
  const addNamingField = () => {
    onChange(name, {
      ...currentValue,
      namingFields: [
        ...currentValue.namingFields,
        { fieldName: '', exampleValue: '', description: '' }
      ]
    });
  };

  // Update naming field
  const updateNamingField = (index, key, value) => {
    const updatedFields = [...currentValue.namingFields];
    updatedFields[index] = { ...updatedFields[index], [key]: value };
    onChange(name, {
      ...currentValue,
      namingFields: updatedFields
    });
  };

  // Remove naming field
  const removeNamingField = (index) => {
    onChange(name, {
      ...currentValue,
      namingFields: currentValue.namingFields.filter((_, i) => i !== index)
    });
  };

  // Add new deliverable attribute
  const addAttribute = () => {
    onChange(name, {
      ...currentValue,
      deliverableAttributes: [
        ...currentValue.deliverableAttributes,
        { attributeName: '', exampleValue: '', description: '' }
      ]
    });
  };

  // Update deliverable attribute
  const updateAttribute = (index, key, value) => {
    const updatedAttrs = [...currentValue.deliverableAttributes];
    updatedAttrs[index] = { ...updatedAttrs[index], [key]: value };
    onChange(name, {
      ...currentValue,
      deliverableAttributes: updatedAttrs
    });
  };

  // Remove deliverable attribute
  const removeAttribute = (index) => {
    onChange(name, {
      ...currentValue,
      deliverableAttributes: currentValue.deliverableAttributes.filter((_, i) => i !== index)
    });
  };

  // Render collapsible section header
  const renderSectionHeader = (sectionKey, title, icon, description) => {
    const isExpanded = expandedSections[sectionKey];
    const Icon = icon;

    return (
      <button
        type="button"
        onClick={() => toggleSection(sectionKey)}
        disabled={disabled}
        className={`w-full flex items-center justify-between p-4 text-left transition-colors border-b ${
          disabled ? 'cursor-not-allowed opacity-60' : 'hover:bg-gray-50'
        }`}
      >
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${isExpanded ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-600'}`}>
            <Icon className="w-5 h-5" />
          </div>
          <div>
            <h4 className="font-semibold text-gray-900">{title}</h4>
            <p className="text-sm text-gray-500 mt-0.5">{description}</p>
          </div>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-gray-400" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-400" />
        )}
      </button>
    );
  };

  return (
    <div className="space-y-4">
      {/* Info Banner */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <h4 className="font-semibold text-blue-900 mb-1">
              Naming and File Structure Standards
            </h4>
            <p className="text-sm text-blue-800">
              Define standardized naming conventions and file structure to ensure consistency, 
              traceability, and efficient information management per ISO 19650-2.
            </p>
          </div>
        </div>
      </div>

      <div className="border border-gray-200 rounded-lg bg-white">
        {/* Overview Section */}
        {renderSectionHeader('overview', 'Overview', FileText, 'General approach to naming conventions')}
        {expandedSections.overview && (
          <div className="p-4">
            <TipTapEditor
              id="naming-overview"
              value={currentValue.overview || ''}
              onChange={handleOverviewChange}
              placeholder="Describe the overall approach to file naming and structure..."
              minHeight="80px"
              autoSaveKey="naming-overview"
              fieldName="namingOverview"
            />
          </div>
        )}

        {/* Naming Convention Fields Section */}
        {renderSectionHeader('namingFields', 'Naming Convention Fields', FileText, 'Define each component of the file naming pattern')}
        {expandedSections.namingFields && (
          <div className="p-4 space-y-3">
            {currentValue.namingFields.map((field, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4 bg-gray-50">
                <div className="flex items-start justify-between mb-3">
                  <h5 className="font-medium text-gray-700">Field {index + 1}</h5>
                  <button
                    type="button"
                    onClick={() => removeNamingField(index)}
                    disabled={disabled}
                    className="text-red-600 hover:text-red-700 disabled:opacity-50"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">
                      Field Name
                    </label>
                    <input
                      type="text"
                      value={field.fieldName}
                      onChange={(e) => updateNamingField(index, 'fieldName', e.target.value)}
                      disabled={disabled}
                      placeholder="e.g., Project Code"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">
                      Example Value
                    </label>
                    <input
                      type="text"
                      value={field.exampleValue}
                      onChange={(e) => updateNamingField(index, 'exampleValue', e.target.value)}
                      disabled={disabled}
                      placeholder="e.g., PRJ001"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">
                      Description
                    </label>
                    <input
                      type="text"
                      value={field.description}
                      onChange={(e) => updateNamingField(index, 'description', e.target.value)}
                      disabled={disabled}
                      placeholder="e.g., Unique project identifier"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                </div>
              </div>
            ))}
            <button
              type="button"
              onClick={addNamingField}
              disabled={disabled}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-blue-400 hover:text-blue-600 transition-colors disabled:opacity-50"
            >
              <Plus className="w-4 h-4" />
              <span className="text-sm font-medium">Add Naming Field</span>
            </button>
          </div>
        )}

        {/* Naming Pattern Section */}
        {renderSectionHeader('pattern', 'Naming Pattern & Example', FileText, 'Complete pattern and example filename')}
        {expandedSections.pattern && (
          <div className="p-4">
            <TipTapEditor
              id="naming-pattern"
              value={currentValue.namingPattern || ''}
              onChange={handlePatternChange}
              placeholder="Define the complete naming pattern and provide examples..."
              minHeight="100px"
              autoSaveKey="naming-pattern"
              fieldName="namingPattern"
            />
          </div>
        )}

        {/* Deliverable Attributes Section */}
        {renderSectionHeader('attributes', 'Deliverable Attributes', FileText, 'Metadata and properties associated with deliverables')}
        {expandedSections.attributes && (
          <div className="p-4 space-y-3">
            {currentValue.deliverableAttributes.map((attr, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4 bg-gray-50">
                <div className="flex items-start justify-between mb-3">
                  <h5 className="font-medium text-gray-700">Attribute {index + 1}</h5>
                  <button
                    type="button"
                    onClick={() => removeAttribute(index)}
                    disabled={disabled}
                    className="text-red-600 hover:text-red-700 disabled:opacity-50"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">
                      Attribute Name
                    </label>
                    <input
                      type="text"
                      value={attr.attributeName}
                      onChange={(e) => updateAttribute(index, 'attributeName', e.target.value)}
                      disabled={disabled}
                      placeholder="e.g., File Format"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">
                      Example Value
                    </label>
                    <input
                      type="text"
                      value={attr.exampleValue}
                      onChange={(e) => updateAttribute(index, 'exampleValue', e.target.value)}
                      disabled={disabled}
                      placeholder="e.g., .rvt, .dwg, .pdf"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">
                      Description
                    </label>
                    <input
                      type="text"
                      value={attr.description}
                      onChange={(e) => updateAttribute(index, 'description', e.target.value)}
                      disabled={disabled}
                      placeholder="e.g., Acceptable file formats"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                </div>
              </div>
            ))}
            <button
              type="button"
              onClick={addAttribute}
              disabled={disabled}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-blue-400 hover:text-blue-600 transition-colors disabled:opacity-50"
            >
              <Plus className="w-4 h-4" />
              <span className="text-sm font-medium">Add Deliverable Attribute</span>
            </button>
          </div>
        )}

        {/* Folder Structure Section */}
        {renderSectionHeader('folderStructure', 'Folder Structure', FileText, 'Directory organization and CDE workflow alignment')}
        {expandedSections.folderStructure && (
          <div className="p-4">
            <TipTapEditor
              id="folder-structure"
              value={currentValue.folderStructure || ''}
              onChange={handleFolderStructureChange}
              placeholder="Define the folder structure and organization strategy..."
              minHeight="120px"
              autoSaveKey="folder-structure"
              fieldName="folderStructure"
            />
          </div>
        )}
      </div>

      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
};

export default NamingConventionBuilder;
