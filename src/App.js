import React, { useState, useRef } from 'react';
import { ChevronRight, ChevronLeft, Download, FileText, Users, Settings, CheckCircle, AlertCircle, Building, Zap, Shield, Database, Calendar, Target, BookOpen, Monitor, Eye, FileType, Printer } from 'lucide-react';

// Configurazione centralizzata
const CONFIG = {
  categories: {
    Commercial: 'bg-blue-100 text-blue-800',
    Management: 'bg-green-100 text-green-800',
    Technical: 'bg-purple-100 text-purple-800'
  },
  
  options: {
    bimUses: ['Design Authoring', 'Design Reviews', '3D Coordination', 'Clash Detection', 'Quantity Take-off', '4D Planning', '5D Cost Management', 'Asset Management', 'Construction Sequencing', 'Facility Management Handover', 'Energy Analysis', 'Code Validation', 'Space Planning', 'Site Analysis', 'Structural Analysis', 'MEP Analysis', 'Lighting Analysis', 'Acoustical Analysis', 'Other Analysis'],
    
    informationPurposes: ['Design Development', 'Construction Planning', 'Quantity Surveying', 'Cost Estimation', 'Facility Management', 'Asset Management', 'Carbon Footprint Analysis', 'Fire Strategy', 'Structural Analysis', 'MEP Coordination', 'Space Management', 'Maintenance Planning', 'Energy Performance', 'Code Compliance', 'Safety Planning', 'Sustainability Assessment'],
    
    software: ['Autodesk Revit', 'ArchiCAD', 'Tekla Structures', 'Bentley MicroStation', 'Bentley AECOsim', 'SketchUp Pro', 'Rhino', 'Navisworks', 'Solibri Model Checker', 'BIM 360', 'Trimble Connect', 'Synchro Pro', 'Vico Office', 'CostX', 'Innovaya', 'dRofus', 'BIMcollab', 'Aconex', 'PowerBI', 'Tableau', 'FME', 'Safe Software', 'Other'],
    
    fileFormats: ['IFC 2x3', 'IFC 4', 'IFC 4.1', 'IFC 4.3', 'DWG', 'DXF', 'PDF', 'PDF/A', 'BCF 2.1', 'BCF 3.0', 'NWD', 'NWC', 'NWF', 'RVT', 'PLN', 'DGN', 'SKP', 'COBie', 'XML', 'JSON', 'CSV', 'XLS', 'XLSX'],
    
    projectTypes: ['Commercial Building', 'Residential', 'Infrastructure', 'Industrial', 'Healthcare', 'Education', 'Mixed Use', 'Renovation/Retrofit']
  },

  steps: [
    { title: 'BEP Type & Project Info', icon: Building, description: 'Define BEP type and basic project information', category: 'Commercial' },
    { title: 'Stakeholders & Roles', icon: Users, description: 'Define project stakeholders and responsibilities', category: 'Commercial' },
    { title: 'BIM Goals & Uses', icon: Target, description: 'Define BIM objectives and applications', category: 'Commercial' },
    { title: 'Level of Information Need', icon: Database, description: 'Specify LOIN requirements and content', category: 'Management' },
    { title: 'Information Delivery Planning', icon: Calendar, description: 'MIDP, TIDPs and delivery schedules', category: 'Management' },
    { title: 'Common Data Environment', icon: Monitor, description: 'CDE specification and workflows', category: 'Technical' },
    { title: 'Technology Requirements', icon: Settings, description: 'Software, hardware and technical specs', category: 'Technical' },
    { title: 'Information Production', icon: FileText, description: 'Methods, standards and procedures', category: 'Management' },
    { title: 'Quality Assurance', icon: CheckCircle, description: 'QA framework and validation processes', category: 'Management' },
    { title: 'Security & Privacy', icon: Shield, description: 'Information security and privacy measures', category: 'Management' },
    { title: 'Training & Competency', icon: BookOpen, description: 'Training requirements and competency levels', category: 'Management' },
    { title: 'Coordination & Risk', icon: AlertCircle, description: 'Collaboration procedures and risk management', category: 'Management' }
  ],

  formFields: {
    0: {
      title: 'Project Information and Objectives',
      fields: [
        { name: 'projectName', label: 'Project Name', required: true, type: 'text' },
        { name: 'projectNumber', label: 'Project Number', type: 'text' },
        { name: 'projectType', label: 'Project Type', required: true, type: 'select', options: 'projectTypes' },
        { name: 'appointingParty', label: 'Appointing Party', required: true, type: 'text' },
        { name: 'projectTimeline', label: 'Project Timeline', type: 'text' },
        { name: 'projectBudget', label: 'Project Budget', type: 'text' },
        { name: 'projectDescription', label: 'Project Description', type: 'textarea', rows: 4 }
      ]
    },
    1: {
      title: 'Stakeholders and Responsibilities',
      fields: [
        { name: 'leadAppointedParty', label: 'Lead Appointed Party', required: true, type: 'text' },
        { name: 'informationManager', label: 'Information Manager', required: true, type: 'text' },
        { name: 'taskTeamLeaders', label: 'Task Team Leaders', type: 'textarea', rows: 3 },
        { name: 'appointedParties', label: 'Appointed Parties', type: 'textarea', rows: 4 }
      ]
    },
    2: {
      title: 'BIM Goals and Objectives',
      fields: [
        { name: 'bimGoals', label: 'BIM Goals', required: true, type: 'textarea', rows: 4 },
        { name: 'primaryObjectives', label: 'Primary Objectives', type: 'textarea', rows: 3 },
        { name: 'bimUses', label: 'BIM Uses', required: true, type: 'checkbox', options: 'bimUses' }
      ]
    },
    3: {
      title: 'Level of Information Need (LOIN)',
      fields: [
        { name: 'informationPurposes', label: 'Information Purposes', required: true, type: 'checkbox', options: 'informationPurposes' },
        { name: 'geometricalInfo', label: 'Geometrical Information Requirements', type: 'textarea', rows: 3 },
        { name: 'alphanumericalInfo', label: 'Alphanumerical Information Requirements', type: 'textarea', rows: 3 },
        { name: 'documentationInfo', label: 'Documentation Requirements', type: 'textarea', rows: 3 },
        { name: 'informationFormats', label: 'Information Formats', type: 'checkbox', options: 'fileFormats' }
      ]
    },
    4: {
      title: 'Information Delivery Planning',
      fields: [
        { name: 'midpDescription', label: 'Master Information Delivery Plan (MIDP)', required: true, type: 'textarea', rows: 4 },
        { name: 'keyMilestones', label: 'Key Information Delivery Milestones', required: true, type: 'textarea', rows: 4 },
        { name: 'deliverySchedule', label: 'Delivery Schedule', type: 'textarea', rows: 3 },
        { name: 'tidpRequirements', label: 'Task Information Delivery Plans (TIDPs)', type: 'textarea', rows: 3 }
      ]
    },
    5: {
      title: 'Common Data Environment (CDE)',
      fields: [
        { name: 'cdeProvider', label: 'CDE Provider', required: true, type: 'text' },
        { name: 'cdePlatform', label: 'CDE Platform Version', type: 'text' },
        { name: 'workflowStates', label: 'Workflow States', required: true, type: 'textarea', rows: 4 },
        { name: 'accessControl', label: 'Access Control', type: 'textarea', rows: 3 },
        { name: 'securityMeasures', label: 'Security Measures', type: 'textarea', rows: 3 },
        { name: 'backupProcedures', label: 'Backup Procedures', type: 'textarea', rows: 3 }
      ]
    },
    6: {
      title: 'Technology and Software Requirements',
      fields: [
        { name: 'bimSoftware', label: 'BIM Software Applications', required: true, type: 'checkbox', options: 'software' },
        { name: 'fileFormats', label: 'File Formats', required: true, type: 'checkbox', options: 'fileFormats' },
        { name: 'hardwareRequirements', label: 'Hardware Requirements', type: 'textarea', rows: 3 },
        { name: 'networkRequirements', label: 'Network Requirements', type: 'textarea', rows: 3 },
        { name: 'interoperabilityNeeds', label: 'Interoperability Requirements', type: 'textarea', rows: 3 }
      ]
    },
    7: {
      title: 'Information Production Methods and Procedures',
      fields: [
        { name: 'modelingStandards', label: 'Modeling Standards', required: true, type: 'textarea', rows: 4 },
        { name: 'namingConventions', label: 'Naming Conventions', required: true, type: 'textarea', rows: 3 },
        { name: 'fileStructure', label: 'File Structure', type: 'textarea', rows: 3 },
        { name: 'versionControl', label: 'Version Control', type: 'textarea', rows: 3 },
        { name: 'dataExchangeProtocols', label: 'Data Exchange Protocols', type: 'textarea', rows: 3 }
      ]
    },
    8: {
      title: 'Quality Assurance and Control',
      fields: [
        { name: 'qaFramework', label: 'Quality Assurance Framework', required: true, type: 'textarea', rows: 4 },
        { name: 'modelValidation', label: 'Model Validation Procedures', required: true, type: 'textarea', rows: 4 },
        { name: 'reviewProcesses', label: 'Review Processes', type: 'textarea', rows: 3 },
        { name: 'approvalWorkflows', label: 'Approval Workflows', type: 'textarea', rows: 3 },
        { name: 'complianceVerification', label: 'Compliance Verification', type: 'textarea', rows: 3 }
      ]
    },
    9: {
      title: 'Information Security and Privacy',
      fields: [
        { name: 'dataClassification', label: 'Data Classification', required: true, type: 'textarea', rows: 3 },
        { name: 'accessPermissions', label: 'Access Permissions', required: true, type: 'textarea', rows: 3 },
        { name: 'encryptionRequirements', label: 'Encryption Requirements', type: 'textarea', rows: 3 },
        { name: 'dataTransferProtocols', label: 'Data Transfer Protocols', type: 'textarea', rows: 3 },
        { name: 'privacyConsiderations', label: 'Privacy Considerations', type: 'textarea', rows: 3 }
      ]
    },
    10: {
      title: 'Training and Competency',
      fields: [
        { name: 'bimCompetencyLevels', label: 'BIM Competency Levels', required: true, type: 'textarea', rows: 4 },
        { name: 'trainingRequirements', label: 'Training Requirements', type: 'textarea', rows: 3 },
        { name: 'certificationNeeds', label: 'Certification Requirements', type: 'textarea', rows: 3 },
        { name: 'projectSpecificTraining', label: 'Project-Specific Training', type: 'textarea', rows: 3 }
      ]
    },
    11: {
      title: 'Coordination, Collaboration & Risk Management',
      fields: [
        { name: 'coordinationMeetings', label: 'Coordination Meetings', required: true, type: 'textarea', rows: 3 },
        { name: 'clashDetectionWorkflow', label: 'Clash Detection Workflow', type: 'textarea', rows: 3 },
        { name: 'issueResolution', label: 'Issue Resolution Process', type: 'textarea', rows: 3 },
        { name: 'communicationProtocols', label: 'Communication Protocols', type: 'textarea', rows: 3 },
        { name: 'federationStrategy', label: 'Model Federation Strategy', type: 'textarea', rows: 3 },
        { name: 'informationRisks', label: 'Information-Related Risks', required: true, type: 'textarea', rows: 4 },
        { name: 'technologyRisks', label: 'Technology-Related Risks', type: 'textarea', rows: 3 },
        { name: 'riskMitigation', label: 'Risk Mitigation Strategies', type: 'textarea', rows: 3 },
        { name: 'contingencyPlans', label: 'Contingency Plans', type: 'textarea', rows: 3 },
        { name: 'performanceMetrics', label: 'Performance Metrics and KPIs', type: 'textarea', rows: 3 },
        { name: 'monitoringProcedures', label: 'Monitoring Procedures', type: 'textarea', rows: 3 },
        { name: 'auditTrails', label: 'Audit Trails', type: 'textarea', rows: 3 },
        { name: 'updateProcesses', label: 'Update Processes', type: 'textarea', rows: 3 }
      ]
    }
  }
};

// Dati iniziali di esempio
const INITIAL_DATA = {
  projectName: 'New Office Complex Development',
  projectNumber: 'NOC-2025-001',
  projectDescription: 'A modern 15-story office complex with retail spaces on the ground floor, underground parking, and sustainable building systems. The project includes advanced MEP systems, curtain wall facades, and LEED Gold certification requirements.',
  projectType: 'Commercial Building',
  projectTimeline: '36 months (March 2025 - February 2028)',
  projectBudget: '£45M - £52M',
  appointingParty: 'Metropolitan Development Corp.',
  leadAppointedParty: 'Global Construction Ltd.',
  appointedParties: 'Architecture: Modern Design Associates\nStructural: Engineering Excellence Ltd.\nMEP: Advanced Systems Group\nQuantity Surveyor: Cost Management Partners\nSpecialist Facades: Curtain Wall Experts Ltd.',
  informationManager: 'Sarah Johnson, BIM Manager - Global Construction Ltd.',
  taskTeamLeaders: 'Architecture: John Smith (Modern Design Associates)\nStructural: Emily Chen (Engineering Excellence Ltd.)\nMEP: Michael Rodriguez (Advanced Systems Group)\nFacades: David Wilson (Curtain Wall Experts Ltd.)',
  bimGoals: 'Implement a collaborative BIM workflow to improve design coordination, reduce construction conflicts, optimize project delivery timelines, and establish a comprehensive digital asset for facility management handover.',
  bimUses: ['Design Authoring', '3D Coordination', 'Clash Detection', 'Quantity Take-off', '4D Planning'],
  primaryObjectives: 'Achieve zero design conflicts at construction stage, reduce RFIs by 40%, improve construction efficiency, and deliver comprehensive FM data for operations.',
  informationPurposes: ['Design Development', 'Construction Planning', 'Quantity Surveying', 'Facility Management'],
  geometricalInfo: 'LOD 350 for construction documentation phase, with dimensional accuracy of ±10mm for structural elements and ±5mm for MEP coordination points.',
  alphanumericalInfo: 'All building elements must include material specifications, performance data, manufacturer information, maintenance requirements, and warranty details.',
  documentationInfo: 'Construction drawings, specifications, schedules, O&M manuals, warranty documents, and asset registers in digital format.',
  informationFormats: ['IFC 4', 'PDF', 'BCF 2.1', 'DWG', 'COBie'],
  midpDescription: 'The MIDP coordinates all discipline-specific TIDPs into a unified delivery schedule aligned with RIBA stages and construction milestones. Information exchanges occur at stage gates with formal approval processes.',
  keyMilestones: 'Stage 2 (Concept Design): Basic geometry and spatial coordination\nStage 3 (Spatial Coordination): Full coordination model\nStage 4 (Technical Design): Construction-ready information\nStage 5 (Manufacturing): Production information\nStage 6 (Handover): As-built models and FM data',
  deliverySchedule: 'Monthly model updates during design phases, weekly coordination cycles during construction documentation, and daily updates during critical construction phases.',
  tidpRequirements: 'Each task team must produce TIDPs detailing their information deliverables, responsibilities, quality requirements, and delivery schedules in alignment with project milestones.',
  cdeProvider: 'Autodesk BIM 360',
  cdePlatform: 'BIM 360 Design & Docs - Enterprise Version 2024',
  workflowStates: 'Work in Progress (WIP): Active development by task teams\nShared: Available for coordination and review\nPublished: Approved for use by the project team\nArchived: Historical versions for reference',
  accessControl: 'Role-based access with project administrator, discipline leads, team members, and read-only stakeholder levels. Multi-factor authentication required for all users.',
  securityMeasures: 'ISO 27001 compliant platform with end-to-end encryption, regular security audits, data residency controls, and comprehensive audit logging.',
  backupProcedures: 'Automated daily backups with 30-day retention, weekly full system backups, geographic redundancy, and quarterly disaster recovery testing.',
  bimSoftware: ['Autodesk Revit', 'Navisworks', 'Solibri Model Checker', 'BIM 360'],
  fileFormats: ['IFC 4', 'DWG', 'PDF', 'BCF 2.1', 'NWD'],
  hardwareRequirements: 'Minimum: Intel i7 or equivalent, 32GB RAM, dedicated graphics card (RTX 3060 or higher), 1TB SSD storage, dual monitors recommended.',
  networkRequirements: 'High-speed internet connection (minimum 100 Mbps), VPN access for remote working, secure cloud connectivity to CDE platform.',
  interoperabilityNeeds: 'Seamless data exchange between Revit disciplines, coordination in Navisworks, model checking in Solibri, and cloud collaboration through BIM 360.',
  modelingStandards: 'UK BIM Alliance standards, Uniclass 2015 classification system, LOD specification based on AIA guidelines, and company-specific modeling conventions.',
  namingConventions: 'Project code: NOC, Originator codes by discipline (ARC, STR, MEP), Volume/Level codes, Type classifications following BS 1192 naming convention.',
  fileStructure: 'Organized by discipline and project phase with clear folder hierarchies, version control through file naming, and linked file management protocols.',
  versionControl: 'Sequential numbering (P01, P02, etc.) for WIP, formal revision codes (A, B, C, etc.) for issued drawings, with comprehensive revision tracking.',
  dataExchangeProtocols: 'Weekly IFC exports for coordination, BCF workflow for issue management, and formal information exchanges at project milestones.',
  qaFramework: 'Comprehensive QA process including automated model checking, manual design reviews, coordination clash detection, and compliance verification against project standards.',
  modelValidation: 'Automated checking using Solibri Model Checker for geometric accuracy, completeness, and standard compliance. Manual reviews for design intent and buildability.',
  reviewProcesses: 'Stage gate reviews at each RIBA stage, weekly coordination reviews, monthly progress reviews, and formal design freeze approvals.',
  approvalWorkflows: 'Task team lead approval, discipline coordination review, project manager authorization, and client sign-off for major milestones.',
  complianceVerification: 'Regular audits against ISO 19650 requirements, BIM standards compliance checks, and quality metrics monitoring.',
  dataClassification: 'Public: Marketing materials\nInternal: Design development work\nConfidential: Commercial information\nRestricted: Security-sensitive building systems',
  accessPermissions: 'Granular permissions based on project roles, need-to-know basis for sensitive information, regular access reviews, and immediate revocation upon project completion.',
  encryptionRequirements: 'AES-256 encryption for data at rest, TLS 1.3 for data in transit, encrypted email for sensitive communications, and secure file transfer protocols.',
  dataTransferProtocols: 'Secure cloud transfer through approved CDE, encrypted email for sensitive documents, secure FTP for large files, and audit trails for all transfers.',
  privacyConsiderations: 'GDPR compliance for all personal data, data retention policies, right to erasure procedures, and privacy impact assessments for data processing.',
  bimCompetencyLevels: 'Level 1 (Awareness): All project staff\nLevel 2 (Knowledge): Discipline leads and coordinators\nLevel 3 (Competence): BIM specialists and managers\nLevel 4 (Expertise): Information manager and senior BIM roles',
  trainingRequirements: 'Software proficiency certification, ISO 19650 awareness training, project-specific BIM procedures, and CDE platform training for all users.',
  certificationNeeds: 'BIM certification for key personnel, software vendor certifications, ISO 19650 practitioner certification, and ongoing professional development.',
  projectSpecificTraining: 'Project induction covering BIM requirements, CDE usage training, modeling standards workshop, and regular update sessions for process changes.',
  coordinationMeetings: 'Weekly BIM coordination meetings, monthly progress reviews, quarterly stakeholder updates, and ad-hoc sessions for critical issues.',
  clashDetectionWorkflow: 'Automated daily clash detection in Navisworks, weekly clash reports, prioritized resolution tracking, and formal sign-off on cleared clashes.',
  issueResolution: 'BCF-based issue tracking, responsibility assignment, deadline management, escalation procedures, and resolution verification process.',
  communicationProtocols: 'Project collaboration platform for daily communication, formal reporting channels, escalation matrix, and documented decision-making process.',
  federationStrategy: 'Central federated model in Navisworks updated weekly, discipline model linking protocols, version synchronization, and coordination point management.',
  informationRisks: 'Data loss through inadequate backup procedures, information security breaches, quality issues from insufficient checking, interoperability failures between software platforms.',
  technologyRisks: 'Software compatibility issues, hardware failures affecting productivity, network connectivity problems, cloud service outages, and version control conflicts.',
  riskMitigation: 'Robust backup strategies, comprehensive security measures, regular quality audits, software compatibility testing, and redundant system capabilities.',
  contingencyPlans: 'Alternative CDE platforms identified, backup workflow procedures, emergency communication protocols, and rapid response teams for critical issues.',
  performanceMetrics: 'Model quality scores, coordination efficiency metrics, information delivery timeline adherence, and stakeholder satisfaction ratings.',
  monitoringProcedures: 'Monthly performance reviews, automated quality checking, delivery milestone tracking, and continuous improvement feedback loops.',
  auditTrails: 'Comprehensive logging of all CDE activities, version history tracking, approval records, and change management documentation.',
  updateProcesses: 'Quarterly BEP reviews, change request procedures, stakeholder approval for modifications, and continuous alignment with project requirements.'
};

// Componenti riutilizzabili
const InputField = ({ field, value, onChange, options }) => {
  const { name, label, type, required, rows, placeholder } = field;
  const optionsList = options ? CONFIG.options[options] : null;

  const baseClasses = "w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500";

  switch (type) {
    case 'textarea':
      return (
        <div>
          <label className="block text-sm font-medium mb-2">
            {label} {required && '*'}
          </label>
          <textarea
            value={value || ''}
            onChange={(e) => onChange(name, e.target.value)}
            rows={rows || 3}
            className={baseClasses}
            placeholder={placeholder || `Enter ${label.toLowerCase()}...`}
          />
        </div>
      );

    case 'select':
      return (
        <div>
          <label className="block text-sm font-medium mb-2">
            {label} {required && '*'}
          </label>
          <select
            value={value || ''}
            onChange={(e) => onChange(name, e.target.value)}
            className={baseClasses}
          >
            <option value="">Select {label.toLowerCase()}</option>
            {optionsList?.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>
      );

    case 'checkbox':
      return (
        <div>
          <label className="block text-sm font-medium mb-2">
            {label} {required && '*'}
          </label>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-60 overflow-y-auto border rounded-lg p-3">
            {optionsList?.map(option => (
              <label key={option} className="flex items-center space-x-2 p-2 border rounded cursor-pointer hover:bg-gray-50">
                <input
                  type="checkbox"
                  checked={(value || []).includes(option)}
                  onChange={() => {
                    const current = value || [];
                    const updated = current.includes(option)
                      ? current.filter(item => item !== option)
                      : [...current, option];
                    onChange(name, updated);
                  }}
                  className="rounded"
                />
                <span className="text-sm">{option}</span>
              </label>
            ))}
          </div>
        </div>
      );

    default:
      return (
        <div>
          <label className="block text-sm font-medium mb-2">
            {label} {required && '*'}
          </label>
          <input
            type="text"
            value={value || ''}
            onChange={(e) => onChange(name, e.target.value)}
            className={baseClasses}
            placeholder={placeholder || `Enter ${label.toLowerCase()}`}
          />
        </div>
      );
  }
};

const ProgressSidebar = ({ steps, currentStep, completedSections, onStepClick, validateStep }) => (
  <div className="bg-white rounded-lg shadow-sm p-6 sticky top-8">
    <h2 className="text-lg font-semibold mb-4">Progress Overview</h2>
    <div className="space-y-3">
      {steps.map((step, index) => {
        const isComplete = completedSections.has(index);
        const isValid = validateStep(index);
        const isCurrent = currentStep === index;
        
        return (
          <div
            key={index}
            className={`flex items-start space-x-3 p-3 rounded-lg cursor-pointer transition-colors
              ${isCurrent ? 'bg-blue-50 border border-blue-200' : 
                isComplete ? 'bg-green-50 border border-green-200' : 'hover:bg-gray-50'}`}
            onClick={() => onStepClick(index)}
          >
            <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center
              ${isCurrent ? 'bg-blue-600 text-white' : 
                isComplete ? 'bg-green-600 text-white' : 'bg-gray-200 text-gray-600'}`}>
              {isComplete ? <CheckCircle className="w-4 h-4" /> : <step.icon className="w-4 h-4" />}
            </div>
            <div className="flex-1 min-w-0">
              <p className={`text-sm font-medium ${
                isCurrent ? 'text-blue-900' : isComplete ? 'text-green-900' : 'text-gray-900'
              }`}>
                {step.title}
              </p>
              <p className="text-xs text-gray-500 mt-1">{step.description}</p>
              <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium mt-1 ${
                CONFIG.categories[step.category]
              }`}>
                {step.category}
              </span>
            </div>
            {!isValid && index !== currentStep && (
              <AlertCircle className="w-4 h-4 text-orange-500 flex-shrink-0" />
            )}
          </div>
        );
      })}
    </div>

    <div className="mt-6 pt-4 border-t">
      <div className="text-sm text-gray-600 mb-2">
        Completion: {Math.round((completedSections.size / steps.length) * 100)}%
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div 
          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
          style={{ width: `${(completedSections.size / steps.length) * 100}%` }}
        />
      </div>
    </div>

    <div className="mt-4 pt-4 border-t">
      <div className="text-xs text-gray-500 space-y-1">
        {Object.keys(CONFIG.categories).map(category => (
          <div key={category} className="flex justify-between">
            <span>{category}:</span>
            <span>
              {steps.filter((s, i) => s.category === category && completedSections.has(i)).length}/
              {steps.filter(s => s.category === category).length}
            </span>
          </div>
        ))}
      </div>
    </div>
  </div>
);

const BepTypeSelector = ({ bepType, setBepType }) => (
  <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
    <h3 className="text-lg font-semibold text-blue-900 mb-3">BEP Type Selection</h3>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {[
        { value: 'pre-appointment', title: 'Pre-Appointment BEP', description: 'Demonstrates capability and proposed approach during tender phase' },
        { value: 'post-appointment', title: 'Post-Appointment BEP', description: 'Confirms delivery approach with detailed planning and schedules' }
      ].map(option => (
        <label key={option.value} className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
          bepType === option.value ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
        }`}>
          <input
            type="radio"
            value={option.value}
            checked={bepType === option.value}
            onChange={(e) => setBepType(e.target.value)}
            className="sr-only"
          />
          <div className="font-medium text-gray-900">{option.title}</div>
          <div className="text-sm text-gray-600 mt-1">{option.description}</div>
        </label>
      ))}
    </div>
  </div>
);

const ProfessionalBEPGenerator = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [bepType, setBepType] = useState('pre-appointment');
  const [formData, setFormData] = useState(INITIAL_DATA);
  const [completedSections, setCompletedSections] = useState(new Set());
  const [exportFormat, setExportFormat] = useState('html');

  const updateFormData = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const validateStep = (stepIndex) => {
    const stepConfig = CONFIG.formFields[stepIndex];
    if (!stepConfig) return true;
    
    const requiredFields = stepConfig.fields.filter(field => field.required);
    return requiredFields.every(field => {
      const value = formData[field.name];
      return Array.isArray(value) ? value.length > 0 : value && value.trim() !== '';
    });
  };

  const nextStep = () => {
    if (validateStep(currentStep)) {
      setCompletedSections(prev => new Set([...prev, currentStep]));
    }
    if (currentStep < CONFIG.steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const generateBEPContent = () => {
    const currentDate = new Date();
    const formattedDate = currentDate.toLocaleDateString();
    const formattedTime = currentDate.toLocaleTimeString();

    const sections = [
      { category: 'COMMERCIAL ASPECTS', items: [
        { title: '1. PROJECT INFORMATION AND OBJECTIVES', fields: [
          { label: 'Project Name', value: formData.projectName },
          { label: 'Project Number', value: formData.projectNumber },
          { label: 'Project Type', value: formData.projectType },
          { label: 'Timeline', value: formData.projectTimeline },
          { label: 'Budget', value: formData.projectBudget },
          { label: 'Project Description', value: formData.projectDescription, isText: true }
        ]},
        { title: '2. STAKEHOLDERS AND RESPONSIBILITIES', fields: [
          { label: 'Appointing Party', value: formData.appointingParty },
          { label: 'Lead Appointed Party', value: formData.leadAppointedParty },
          { label: 'Information Manager', value: formData.informationManager },
          { label: 'Task Team Leaders', value: formData.taskTeamLeaders, isText: true },
          { label: 'Appointed Parties', value: formData.appointedParties, isText: true }
        ]},
        { title: '3. BIM GOALS AND OBJECTIVES', fields: [
          { label: 'BIM Goals', value: formData.bimGoals, isText: true },
          { label: 'Primary Objectives', value: formData.primaryObjectives, isText: true },
          { label: 'BIM Uses', value: formData.bimUses, isList: true }
        ]}
      ]},
      { category: 'MANAGEMENT ASPECTS', items: [
        { title: '4. LEVEL OF INFORMATION NEED (LOIN)', fields: [
          { label: 'Information Purposes', value: formData.informationPurposes, isList: true },
          { label: 'Geometrical Information Requirements', value: formData.geometricalInfo, isText: true },
          { label: 'Alphanumerical Information Requirements', value: formData.alphanumericalInfo, isText: true },
          { label: 'Documentation Requirements', value: formData.documentationInfo, isText: true },
          { label: 'Information Formats', value: formData.informationFormats, isList: true }
        ]},
        { title: '5. INFORMATION DELIVERY PLANNING', fields: [
          { label: 'Master Information Delivery Plan (MIDP)', value: formData.midpDescription, isText: true },
          { label: 'Key Information Delivery Milestones', value: formData.keyMilestones, isText: true },
          { label: 'Delivery Schedule', value: formData.deliverySchedule, isText: true },
          { label: 'Task Information Delivery Plans (TIDPs)', value: formData.tidpRequirements, isText: true }
        ]},
        { title: '6. INFORMATION PRODUCTION METHODS AND PROCEDURES', fields: [
          { label: 'Modeling Standards', value: formData.modelingStandards, isText: true },
          { label: 'Naming Conventions', value: formData.namingConventions, isText: true },
          { label: 'File Structure', value: formData.fileStructure, isText: true },
          { label: 'Version Control', value: formData.versionControl, isText: true },
          { label: 'Data Exchange Protocols', value: formData.dataExchangeProtocols, isText: true }
        ]},
        { title: '7. QUALITY ASSURANCE AND CONTROL', fields: [
          { label: 'Quality Assurance Framework', value: formData.qaFramework, isText: true },
          { label: 'Model Validation Procedures', value: formData.modelValidation, isText: true },
          { label: 'Review Processes', value: formData.reviewProcesses, isText: true },
          { label: 'Approval Workflows', value: formData.approvalWorkflows, isText: true },
          { label: 'Compliance Verification', value: formData.complianceVerification, isText: true }
        ]},
        { title: '8. INFORMATION SECURITY AND PRIVACY', fields: [
          { label: 'Data Classification', value: formData.dataClassification, isText: true },
          { label: 'Access Permissions', value: formData.accessPermissions, isText: true },
          { label: 'Encryption Requirements', value: formData.encryptionRequirements, isText: true },
          { label: 'Data Transfer Protocols', value: formData.dataTransferProtocols, isText: true },
          { label: 'Privacy Considerations', value: formData.privacyConsiderations, isText: true }
        ]},
        { title: '9. TRAINING AND COMPETENCY', fields: [
          { label: 'BIM Competency Levels', value: formData.bimCompetencyLevels, isText: true },
          { label: 'Training Requirements', value: formData.trainingRequirements, isText: true },
          { label: 'Certification Requirements', value: formData.certificationNeeds, isText: true },
          { label: 'Project-Specific Training', value: formData.projectSpecificTraining, isText: true }
        ]}
      ]},
      { category: 'TECHNICAL ASPECTS', items: [
        { title: '10. COMMON DATA ENVIRONMENT (CDE)', fields: [
          { label: 'CDE Provider', value: formData.cdeProvider },
          { label: 'CDE Platform', value: formData.cdePlatform },
          { label: 'Workflow States', value: formData.workflowStates, isText: true },
          { label: 'Access Control', value: formData.accessControl, isText: true },
          { label: 'Security Measures', value: formData.securityMeasures, isText: true },
          { label: 'Backup Procedures', value: formData.backupProcedures, isText: true }
        ]},
        { title: '11. TECHNOLOGY AND SOFTWARE REQUIREMENTS', fields: [
          { label: 'BIM Software Applications', value: formData.bimSoftware, isList: true },
          { label: 'File Formats', value: formData.fileFormats, isList: true },
          { label: 'Hardware Requirements', value: formData.hardwareRequirements, isText: true },
          { label: 'Network Requirements', value: formData.networkRequirements, isText: true },
          { label: 'Interoperability Requirements', value: formData.interoperabilityNeeds, isText: true }
        ]},
        { title: '12. COORDINATION AND COLLABORATION PROCEDURES', fields: [
          { label: 'Coordination Meetings', value: formData.coordinationMeetings, isText: true },
          { label: 'Clash Detection Workflow', value: formData.clashDetectionWorkflow, isText: true },
          { label: 'Issue Resolution Process', value: formData.issueResolution, isText: true },
          { label: 'Communication Protocols', value: formData.communicationProtocols, isText: true },
          { label: 'Model Federation Strategy', value: formData.federationStrategy, isText: true }
        ]},
        { title: '13. RISK MANAGEMENT', fields: [
          { label: 'Information-Related Risks', value: formData.informationRisks, isText: true },
          { label: 'Technology-Related Risks', value: formData.technologyRisks, isText: true },
          { label: 'Risk Mitigation Strategies', value: formData.riskMitigation, isText: true },
          { label: 'Contingency Plans', value: formData.contingencyPlans, isText: true }
        ]},
        { title: '14. COMPLIANCE AND MONITORING', fields: [
          { label: 'Performance Metrics and KPIs', value: formData.performanceMetrics, isText: true },
          { label: 'Monitoring Procedures', value: formData.monitoringProcedures, isText: true },
          { label: 'Audit Trails', value: formData.auditTrails, isText: true },
          { label: 'Update Processes', value: formData.updateProcesses, isText: true }
        ]}
      ]}
    ];

    const renderField = (field) => {
      if (!field.value) return '';
      
      if (field.isList && Array.isArray(field.value)) {
        return `<h3>${field.label}</h3><ul>${field.value.map(item => `<li>${item}</li>`).join('')}</ul>`;
      }
      
      if (field.isText) {
        return `<h3>${field.label}</h3><p>${field.value}</p>`;
      }
      
      return `<tr><td class="label">${field.label}:</td><td>${field.value}</td></tr>`;
    };

    return `
      <!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BIM Execution Plan - ${formData.projectName}</title>
        <style>
          body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }
          .header { text-align: center; border-bottom: 3px solid #2563eb; padding-bottom: 20px; margin-bottom: 30px; }
          h1 { color: #1e40af; font-size: 28px; margin-bottom: 10px; }
          .subtitle { color: #059669; font-size: 18px; font-weight: bold; margin-bottom: 5px; }
          .bep-type { background: #dbeafe; padding: 10px; border-radius: 8px; display: inline-block; margin: 10px 0; font-weight: bold; }
          h2 { color: #1e40af; margin-top: 35px; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; font-size: 20px; }
          h3 { color: #374151; margin-top: 25px; font-size: 16px; border-left: 4px solid #2563eb; padding-left: 12px; }
          .section { margin: 25px 0; }
          .info-box { background-color: #f8fafc; padding: 20px; border-left: 4px solid #2563eb; margin: 20px 0; border-radius: 0 8px 8px 0; }
          .category-header { background: linear-gradient(135deg, #2563eb, #1e40af); color: white; padding: 15px; margin: 30px 0 20px 0; border-radius: 8px; font-weight: bold; font-size: 18px; }
          ul { margin: 10px 0; padding-left: 25px; }
          li { margin: 8px 0; }
          table { width: 100%; border-collapse: collapse; margin: 20px 0; }
          th, td { padding: 12px; border-bottom: 1px solid #e5e7eb; text-align: left; }
          th { background-color: #f8fafc; font-weight: bold; color: #374151; }
          .label { font-weight: bold; width: 250px; color: #4b5563; }
          .compliance-box { background: #ecfdf5; border: 2px solid #10b981; padding: 20px; border-radius: 8px; margin: 20px 0; }
          .footer { margin-top: 50px; padding-top: 25px; border-top: 2px solid #e5e7eb; background: #f9fafb; padding: 25px; border-radius: 8px; }
        </style>
      </head>
      <body>
        <div class="header">
          <h1>BIM EXECUTION PLAN (BEP)</h1>
          <div class="subtitle">ISO 19650-2 Compliant</div>
          <div class="bep-type">${bepType === 'pre-appointment' ? 'Pre-Appointment BEP' : 'Post-Appointment BEP'}</div>
        </div>

        <div class="compliance-box">
          <h3>Document Information</h3>
          <table>
            <tr><td class="label">Document Type:</td><td>${bepType === 'pre-appointment' ? 'Pre-Appointment BEP' : 'Post-Appointment BEP'}</td></tr>
            <tr><td class="label">Project Name:</td><td>${formData.projectName || 'Not specified'}</td></tr>
            <tr><td class="label">Project Number:</td><td>${formData.projectNumber || 'Not specified'}</td></tr>
            <tr><td class="label">Generated Date:</td><td>${formattedDate} at ${formattedTime}</td></tr>
            <tr><td class="label">Status:</td><td>Work in Progress</td></tr>
          </table>
        </div>

        ${sections.map(section => `
          <div class="category-header">${section.category}</div>
          ${section.items.map(item => `
            <div class="section">
              <h2>${item.title}</h2>
              <div class="info-box">
                ${item.fields.some(f => !f.isText && !f.isList) ? `<table>${item.fields.filter(f => !f.isText && !f.isList).map(renderField).join('')}</table>` : ''}
                ${item.fields.filter(f => f.isText || f.isList).map(renderField).join('')}
              </div>
            </div>
          `).join('')}
        `).join('')}

        <div class="footer">
          <h3>Document Control Information</h3>
          <table>
            <tr><td class="label">Document Type:</td><td>BIM Execution Plan (BEP)</td></tr>
            <tr><td class="label">ISO Standard:</td><td>ISO 19650-2:2018</td></tr>
            <tr><td class="label">Document Status:</td><td>Work in Progress</td></tr>
            <tr><td class="label">Generated By:</td><td>Professional BEP Generator Tool</td></tr>
            <tr><td class="label">Generated Date:</td><td>${formattedDate}</td></tr>
            <tr><td class="label">Generated Time:</td><td>${formattedTime}</td></tr>
          </table>
        </div>
      </body>
      </html>
    `;
  };

  const downloadBEP = () => {
    const content = generateBEPContent();
    const currentDate = new Date().toISOString().split('T')[0];
    const fileName = `Professional_BEP_${formData.projectName || 'Project'}_${currentDate}`;
    
    if (exportFormat === 'pdf') {
      const printWindow = window.open('', '_blank');
      printWindow.document.write(content);
      printWindow.document.close();
      printWindow.focus();
      setTimeout(() => printWindow.print(), 500);
      return;
    }
    
    const blob = new Blob([content], { type: 'text/html;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName + (exportFormat === 'word' ? '.doc' : '.html');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const previewBEP = () => {
    const content = generateBEPContent();
    const previewWindow = window.open('', '_blank', 'width=1200,height=800');
    previewWindow.document.write(content);
    previewWindow.document.close();
  };

  const renderStepContent = () => {
    const stepConfig = CONFIG.formFields[currentStep];
    if (!stepConfig) return null;

    return (
      <div className="space-y-6">
        {currentStep === 0 && <BepTypeSelector bepType={bepType} setBepType={setBepType} />}
        
        <h3 className="text-xl font-semibold">{stepConfig.title}</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {stepConfig.fields.map(field => (
            <div key={field.name} className={field.type === 'textarea' || field.type === 'checkbox' ? 'md:col-span-2' : ''}>
              <InputField
                field={field}
                value={formData[field.name]}
                onChange={updateFormData}
                options={field.options}
              />
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2">
                <Zap className="w-8 h-8 text-blue-600" />
                <h1 className="text-2xl font-bold text-gray-900">Professional BEP Generator</h1>
              </div>
              <span className="text-sm text-gray-500">ISO 19650-2 Compliant</span>
            </div>
            <div className="text-sm text-gray-600">
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                {bepType === 'pre-appointment' ? 'Pre-Appointment BEP' : 'Post-Appointment BEP'}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          <div className="lg:col-span-1">
            <ProgressSidebar
              steps={CONFIG.steps}
              currentStep={currentStep}
              completedSections={completedSections}
              onStepClick={setCurrentStep}
              validateStep={validateStep}
            />
          </div>

          <div className="lg:col-span-3">
            <div className="bg-white rounded-lg shadow-sm p-8">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">{CONFIG.steps[currentStep].title}</h2>
                  <p className="text-gray-600 mt-1">{CONFIG.steps[currentStep].description}</p>
                </div>
                <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                  CONFIG.categories[CONFIG.steps[currentStep].category]
                }`}>
                  {CONFIG.steps[currentStep].category} Aspects
                </span>
              </div>

              {renderStepContent()}

              <div className="flex justify-between items-center mt-8 pt-6 border-t">
                <button
                  onClick={prevStep}
                  disabled={currentStep === 0}
                  className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <ChevronLeft className="w-4 h-4" />
                  <span>Previous</span>
                </button>

                <div className="text-sm text-gray-500">
                  Step {currentStep + 1} of {CONFIG.steps.length}
                </div>

                <div className="flex space-x-3">
                  {currentStep === CONFIG.steps.length - 1 ? (
                    <div className="flex flex-col space-y-4">
                      <div className="flex items-center space-x-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                        <span className="text-sm font-medium text-blue-900">Export Format:</span>
                        <div className="flex space-x-3">
                          {[
                            { value: 'html', icon: FileType, label: 'HTML' },
                            { value: 'pdf', icon: Printer, label: 'PDF' },
                            { value: 'word', icon: FileText, label: 'Word' }
                          ].map(format => (
                            <label key={format.value} className="flex items-center space-x-2 cursor-pointer">
                              <input
                                type="radio"
                                value={format.value}
                                checked={exportFormat === format.value}
                                onChange={(e) => setExportFormat(e.target.value)}
                                className="text-blue-600"
                              />
                              <format.icon className="w-4 h-4 text-blue-600" />
                              <span className="text-sm text-blue-900">{format.label}</span>
                            </label>
                          ))}
                        </div>
                      </div>
                      
                      <div className="flex space-x-3">
                        <button
                          onClick={previewBEP}
                          className="flex items-center space-x-2 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg transition-all shadow-lg"
                        >
                          <Eye className="w-5 h-5" />
                          <span>Preview BEP</span>
                        </button>
                        
                        <button
                          onClick={downloadBEP}
                          className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-8 py-3 rounded-lg transition-all transform hover:scale-105 shadow-lg"
                        >
                          <Download className="w-5 h-5" />
                          <span>Download Professional BEP</span>
                        </button>
                      </div>
                    </div>
                  ) : (
                    <button
                      onClick={nextStep}
                      className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors"
                    >
                      <span>Next</span>
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfessionalBEPGenerator;