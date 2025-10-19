import React, { useState } from 'react';
import { 
  ChevronDown, 
  ChevronUp, 
  Calendar, 
  Users, 
  FileText, 
  CheckCircle, 
  GraduationCap, 
  BarChart3, 
  Target
} from 'lucide-react';
import EditableTable from '../base/EditableTable';
import FieldHeader from '../base/FieldHeader';
import TipTapEditor from '../editors/TipTapEditor';
import RACIReferenceBuilder from './RACIReferenceBuilder';
import NamingConventionBuilder from './NamingConventionBuilder';

/**
 * IMStrategyBuilder - Information Management Strategy Builder
 * Custom component for Section 4.8: Approach to Facilitating Information Management Goals
 * 
 * Provides a structured interface for defining 7 key aspects of IM strategy per ISO 19650-2:
 * 1. Coordination Meeting Schedule
 * 2. RACI Responsibility Matrices
 * 3. Naming and File Structure Standards
 * 4. Quality Checking Tools
 * 5. Training and Competency Requirements
 * 6. Performance Monitoring and KPIs
 * 7. Ongoing Alignment Maintenance Strategy
 */
const IMStrategyBuilder = ({ field, value = {}, onChange, error, disabled = false }) => {
  // Track which sections are expanded (hook must be called before any returns)
  const [expandedSections, setExpandedSections] = useState({
    meetingSchedule: true,
    raciReference: false,
    namingStandards: false,
    qualityTools: false,
    trainingPlan: false,
    kpis: false,
    alignmentStrategy: false
  });

  // Safety check for field prop (after all hooks)
  if (!field) {
    return <div className="text-red-600">Error: Field configuration is missing</div>;
  }

  const { name } = field;
  
  // Initialize with default structure if empty
  const defaultValue = {
    meetingSchedule: { columns: [], data: [] },
    raciReference: {
      referenceText: 'Detailed RACI (Responsible, Accountable, Consulted, Informed) matrices are defined in Section 6.6 (Information Deliverables Responsibility Matrix) and Section 6.7 (Information Management Activities Responsibility Matrix per ISO 19650-2 Annex A). These matrices establish clear accountability for all information production, coordination, approval, and delivery activities throughout the project lifecycle.',
      keyDecisionPoints: { 
        columns: ['Key Decision/Activity', 'Accountable', 'Responsible', 'Consulted', 'Informed'],
        data: [
          {
            'Key Decision/Activity': 'Model Federation Approval',
            'Accountable': 'Lead BIM Coordinator',
            'Responsible': 'Discipline Coordinators',
            'Consulted': 'Design Team',
            'Informed': 'Client'
          },
          {
            'Key Decision/Activity': 'Design Coordination Sign-off',
            'Accountable': 'Design Manager',
            'Responsible': 'Discipline Leads',
            'Consulted': 'BIM Manager',
            'Informed': 'Project Director'
          },
          {
            'Key Decision/Activity': 'Information Delivery Approval',
            'Accountable': 'Information Manager',
            'Responsible': 'BIM Manager',
            'Consulted': 'Task Team',
            'Informed': 'Client Representative'
          },
          {
            'Key Decision/Activity': 'CDE Access Management',
            'Accountable': 'Information Manager',
            'Responsible': 'CDE Administrator',
            'Consulted': 'IT Security',
            'Informed': 'Project Team'
          },
          {
            'Key Decision/Activity': 'Change Request Processing',
            'Accountable': 'Project Manager',
            'Responsible': 'Design Manager',
            'Consulted': 'Affected Disciplines',
            'Informed': 'All Stakeholders'
          }
        ]
      }
    },
    namingStandards: {
      overview: '<p>File naming follows <strong>ISO 19650-2</strong> convention to ensure consistency, traceability, and efficient information management across all project deliverables.</p>',
      namingFields: [],
      namingPattern: '<p><strong>Pattern:</strong> [Project Code]-[Originator]-[Volume/System]-[Level/Location]-[Type]-[Role]-[Number]-[Revision]</p><p><strong>Example:</strong> <code>PRJ001-ARC-XX-GF-M3-ARC-0001-P01.rvt</code></p>',
      deliverableAttributes: [],
      folderStructure: '<ul><li><strong>00_WIP</strong> - Work in Progress (active development, not shared)</li><li><strong>01_SHARED</strong> - Shared for review and coordination</li><li><strong>02_PUBLISHED</strong> - Published/Approved information</li><li><strong>03_ARCHIVE</strong> - Superseded versions and historical records</li></ul><p>Each folder follows the CDE workflow states aligned with ISO 19650-2 information container strategy.</p>'
    },
    qualityTools: { columns: [], data: [] },
    trainingPlan: { columns: [], data: [] },
    kpis: { columns: [], data: [] },
    alignmentStrategy: ''
  };

  // Deep merge to handle existing values while preserving default structure
  const currentValue = { ...defaultValue };
  if (value && typeof value === 'object') {
    Object.keys(value).forEach(key => {
      if (key === 'raciReference' || key === 'namingStandards') {
        // Special handling for complex objects - ensure they're objects
        if (value[key] && typeof value[key] === 'object' && !Array.isArray(value[key])) {
          currentValue[key] = value[key];
        } else {
          // If it's a string or other type, use default
          currentValue[key] = defaultValue[key];
        }
      } else {
        currentValue[key] = value[key];
      }
    });
  }

  // Toggle section expansion
  const toggleSection = (sectionKey) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionKey]: !prev[sectionKey]
    }));
  };

  // Handle changes to any section
  const handleSectionChange = (sectionKey, newValue) => {
    const updatedValue = {
      ...currentValue,
      [sectionKey]: newValue
    };
    onChange(name, updatedValue);
  };

  // Section definitions with metadata
  const sections = [
    {
      key: 'meetingSchedule',
      number: '4.8.1',
      title: 'Coordination Meeting Schedule',
      icon: Calendar,
      description: 'Define regular coordination meetings for information management',
      type: 'table',
      columns: [
        { key: 'meetingType', label: 'Meeting Type' },
        { key: 'frequency', label: 'Frequency' },
        { key: 'participants', label: 'Key Participants' },
        { key: 'agenda', label: 'Standard Agenda Items' },
        { key: 'duration', label: 'Duration' }
      ],
      placeholder: 'Add coordination meeting...'
    },
    {
      key: 'raciReference',
      number: '4.8.2',
      title: 'RACI Responsibility Matrices',
      icon: Users,
      description: 'Reference to responsibility assignment matrices and key decision points',
      type: 'raci-reference'
    },
    {
      key: 'namingStandards',
      number: '4.8.3',
      title: 'Naming and File Structure Standards',
      icon: FileText,
      description: 'Standardized conventions for files, folders, and information containers',
      type: 'naming-convention'
    },
    {
      key: 'qualityTools',
      number: '4.8.4',
      title: 'Automated Quality Checking Tools',
      icon: CheckCircle,
      description: 'Tools and processes for automated quality assurance',
      type: 'table',
      columns: [
        { key: 'toolName', label: 'Tool/Software' },
        { key: 'checkType', label: 'Check Type' },
        { key: 'frequency', label: 'Check Frequency' },
        { key: 'responsible', label: 'Responsible Role' },
        { key: 'action', label: 'Action on Failure' }
      ],
      placeholder: 'Add quality tool...'
    },
    {
      key: 'trainingPlan',
      number: '4.8.5',
      title: 'Training and Competency Requirements',
      icon: GraduationCap,
      description: 'Training programs and competency verification processes',
      type: 'table',
      columns: [
        { key: 'role', label: 'Role/Personnel' },
        { key: 'trainingTopic', label: 'Training Topic' },
        { key: 'provider', label: 'Provider/Method' },
        { key: 'timeline', label: 'Timeline' },
        { key: 'verification', label: 'Competency Verification' }
      ],
      placeholder: 'Add training requirement...'
    },
    {
      key: 'kpis',
      number: '4.8.6',
      title: 'Performance Monitoring and KPIs',
      icon: BarChart3,
      description: 'Key Performance Indicators for information management effectiveness',
      type: 'table',
      columns: [
        { key: 'kpiName', label: 'KPI Name' },
        { key: 'metric', label: 'Measurement Metric' },
        { key: 'target', label: 'Target Value' },
        { key: 'frequency', label: 'Monitoring Frequency' },
        { key: 'owner', label: 'Owner' }
      ],
      placeholder: 'Add KPI...'
    },
    {
      key: 'alignmentStrategy',
      number: '4.8.7',
      title: 'Ongoing Alignment Maintenance Strategy',
      icon: Target,
      description: 'Approach to maintaining alignment throughout project lifecycle',
      type: 'textarea',
      rows: 4,
      placeholder: 'Example: Alignment will be maintained through: 1) Monthly BIM coordination reviews with all disciplines, 2) Quarterly stakeholder workshops to validate information requirements, 3) Continuous monitoring of KPIs with corrective actions for deviations >10%, 4) Regular updates to EIR alignment matrix, 5) Change management process for scope variations...'
    }
  ];

  // Render a single section
  const renderSection = (section) => {
    const isExpanded = expandedSections[section.key];
    const Icon = section.icon;
    const sectionValue = currentValue[section.key];

    return (
      <div key={section.key} className="border border-gray-200 rounded-lg mb-3 bg-white">
        {/* Section Header */}
        <button
          type="button"
          onClick={() => toggleSection(section.key)}
          disabled={disabled}
          className={`w-full flex items-center justify-between p-4 text-left transition-colors ${
            disabled ? 'cursor-not-allowed opacity-60' : 'hover:bg-gray-50'
          }`}
        >
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${isExpanded ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-600'}`}>
              <Icon className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">
                {section.number && <span className="text-blue-600 mr-2">{section.number}</span>}
                {section.title}
              </h3>
              <p className="text-sm text-gray-500 mt-0.5">{section.description}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {/* Completion indicator */}
            {section.type === 'table' && sectionValue?.data?.length > 0 && (
              <span className="text-xs font-medium text-green-600 bg-green-50 px-2 py-1 rounded">
                {sectionValue.data.length} {sectionValue.data.length === 1 ? 'item' : 'items'}
              </span>
            )}
            {section.type === 'textarea' && sectionValue && sectionValue.trim().length > 0 && (
              <span className="text-xs font-medium text-green-600 bg-green-50 px-2 py-1 rounded">
                ✓ Completed
              </span>
            )}
            {isExpanded ? (
              <ChevronUp className="w-5 h-5 text-gray-400" />
            ) : (
              <ChevronDown className="w-5 h-5 text-gray-400" />
            )}
          </div>
        </button>

        {/* Section Content */}
        {isExpanded && (
          <div className="px-4 pb-4 pt-2 border-t border-gray-100">
            {section.type === 'table' ? (
              <EditableTable
                field={{
                  name: section.key,
                  label: section.title,
                  columns: section.columns
                }}
                value={sectionValue}
                onChange={(fieldName, newValue) => {
                  handleSectionChange(section.key, newValue);
                }}
                error={null}
              />
            ) : section.type === 'raci-reference' ? (
              <RACIReferenceBuilder
                field={{
                  name: section.key,
                  label: section.title
                }}
                value={sectionValue}
                onChange={(fieldName, newValue) => {
                  handleSectionChange(section.key, newValue);
                }}
                error={null}
                disabled={disabled}
              />
            ) : section.type === 'naming-convention' ? (
              <NamingConventionBuilder
                field={{
                  name: section.key,
                  label: section.title
                }}
                value={sectionValue}
                onChange={(fieldName, newValue) => {
                  handleSectionChange(section.key, newValue);
                }}
                error={null}
                disabled={disabled}
              />
            ) : (
              <TipTapEditor
                id={`im-strategy-${section.key}`}
                value={sectionValue || ''}
                onChange={(newValue) => handleSectionChange(section.key, newValue)}
                placeholder={section.placeholder}
                minHeight={`${(section.rows || 3) * 24}px`}
                autoSaveKey={`im-strategy-${section.key}`}
                fieldName={section.key}
                className=""
              />
            )}
          </div>
        )}
      </div>
    );
  };

  // Calculate completion status
  const completedSections = sections.filter(section => {
    const sectionValue = currentValue[section.key];
    if (section.type === 'table') {
      return sectionValue?.data?.length > 0;
    }
    if (section.type === 'raci-reference') {
      return (sectionValue?.referenceText?.trim().length > 0) || 
             (sectionValue?.keyDecisionPoints?.data?.length > 0);
    }
    if (section.type === 'naming-convention') {
      return (sectionValue?.overview?.trim().length > 0) ||
             (sectionValue?.namingFields?.length > 0) ||
             (sectionValue?.deliverableAttributes?.length > 0);
    }
    return sectionValue && sectionValue.trim().length > 0;
  }).length;

  const completionPercentage = Math.round((completedSections / sections.length) * 100);

  return (
    <div className="space-y-4">
      {/* Field Header */}
      <FieldHeader 
        fieldName={name}
        label={field.label}
        number={field.number}
        required={field.required}
      />
      
      {/* Header with progress indicator */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start justify-between mb-2">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Information Management Strategy</h2>
            <p className="text-sm text-gray-600 mt-1">
              Define your approach to facilitating information management goals per ISO 19650-2
            </p>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-blue-600">{completionPercentage}%</div>
            <div className="text-xs text-gray-500">{completedSections} of {sections.length} completed</div>
          </div>
        </div>
        
        {/* Progress bar */}
        <div className="w-full bg-gray-200 rounded-full h-2 mt-3">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${completionPercentage}%` }}
          />
        </div>
      </div>

      {/* Quick actions */}
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => {
            const allExpanded = {};
            sections.forEach(s => allExpanded[s.key] = true);
            setExpandedSections(allExpanded);
          }}
          className="px-3 py-1.5 text-sm font-medium text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
        >
          Expand All
        </button>
        <button
          type="button"
          onClick={() => {
            const allCollapsed = {};
            sections.forEach(s => allCollapsed[s.key] = false);
            setExpandedSections(allCollapsed);
          }}
          className="px-3 py-1.5 text-sm font-medium text-gray-600 hover:bg-gray-50 rounded-lg transition-colors"
        >
          Collapse All
        </button>
      </div>

      {/* Sections */}
      <div className="space-y-3">
        {sections.map(section => renderSection(section))}
      </div>

      {/* Footer helper text */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm text-amber-800">
        <strong>ISO 19650-2 Compliance:</strong> This strategy should demonstrate how you will maintain alignment 
        between the appointing party's information requirements and your team's information management processes 
        throughout the project lifecycle.
      </div>
    </div>
  );
};

export default IMStrategyBuilder;
