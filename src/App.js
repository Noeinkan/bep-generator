import React, { useState, useEffect, useCallback, useMemo, createContext, useContext } from 'react';
import { ChevronRight, ChevronLeft, Download, FileText, Users, Settings, CheckCircle, AlertCircle, Building, Zap, Shield, Database, Calendar, Target, BookOpen, Monitor, Eye, FileType, Printer } from 'lucide-react';
import jsPDF from 'jspdf';
import { Packer, Document, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, AlignmentType } from 'docx';
import DOMPurify from 'dompurify';

// Authentication Context
const AuthContext = createContext();

const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const savedUser = localStorage.getItem('bepUser');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
    }
    setLoading(false);
  }, []);

  const login = (userData) => {
    setUser(userData);
    localStorage.setItem('bepUser', JSON.stringify(userData));
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('bepUser');
  };

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

const Login = () => {
  const { login } = useAuth();
  const [formData, setFormData] = useState({ name: '', company: '', email: '' });

  const handleSubmit = (e) => {
    e.preventDefault();
    if (formData.name && formData.company) {
      login({
        id: Date.now(),
        name: formData.name,
        company: formData.company,
        email: formData.email
      });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-md p-8">
        <div className="text-center mb-8">
          <Zap className="w-16 h-16 text-blue-600 mx-auto mb-4" />
          <h1 className="text-3xl font-bold text-gray-900 mb-2">BEP Generator</h1>
          <p className="text-gray-600">Professional BIM Execution Plans</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Full Name *
            </label>
            <input
              type="text"
              required
              value={formData.name}
              onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              placeholder="Your full name"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Company *
            </label>
            <input
              type="text"
              required
              value={formData.company}
              onChange={(e) => setFormData(prev => ({ ...prev, company: e.target.value }))}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              placeholder="Your company name"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Email
            </label>
            <input
              type="email"
              value={formData.email}
              onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              placeholder="your.email@company.com"
            />
          </div>

          <button
            type="submit"
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition-colors"
          >
            Start Creating BEP
          </button>
        </form>

        <div className="mt-8 text-center text-sm text-gray-500">
          <p>ISO 19650-2 Compliant • Professional Templates</p>
        </div>
      </div>
    </div>
  );
};



// Configurazione centralizzata
const CONFIG = {
  categories: {
    Commercial: { name: 'COMMERCIAL ASPECTS', bg: 'bg-blue-100 text-blue-800' },
    Management: { name: 'MANAGEMENT ASPECTS', bg: 'bg-green-100 text-green-800' },
    Technical: { name: 'TECHNICAL ASPECTS', bg: 'bg-purple-100 text-purple-800' }
  },

  bepTypeDefinitions: {
    'pre-appointment': {
      title: 'Pre-Appointment BEP',
      subtitle: 'Tender Phase Document',
      description: 'A document outlining the prospective delivery team\'s proposed approach, capability, and capacity to meet the appointing party\'s exchange information requirements (EIRs). It demonstrates to the client that the potential delivery team has the ability to handle project data according to any assigned information criteria.',
      purpose: 'Demonstrates capability during tender phase',
      focus: 'Proposed approach and team capability',
      language: 'We propose to...  Our capability includes...  We would implement...',
      icon: Building,
      color: 'blue',
      bgClass: 'bg-blue-50',
      borderClass: 'border-blue-200',
      textClass: 'text-blue-900'
    },
    'post-appointment': {
      title: 'Post-Appointment BEP',
      subtitle: 'Project Execution Document',
      description: 'Confirms the delivery team\'s information management approach and includes detailed planning and schedules. It offers a delivery instrument that the appointed delivery team will use to produce, manage and exchange project information during the appointment.',
      purpose: 'Delivery instrument during project execution',
      focus: 'Confirmed approach with detailed planning',
      language: 'We will deliver...  The assigned team will...  Implementation schedule is...',
      icon: CheckCircle,
      color: 'green',
      bgClass: 'bg-green-50',
      borderClass: 'border-green-200',
      textClass: 'text-green-900'
    }
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
    'pre-appointment': {
      0: {
        title: 'Project Information and Proposed Approach',
        fields: [
          { name: 'projectName', label: 'Project Name', required: true, type: 'text' },
          { name: 'projectNumber', label: 'Project Number', type: 'text' },
          { name: 'projectType', label: 'Project Type', required: true, type: 'select', options: 'projectTypes' },
          { name: 'appointingParty', label: 'Appointing Party', required: true, type: 'text' },
          { name: 'proposedTimeline', label: 'Proposed Project Timeline', type: 'text' },
          { name: 'estimatedBudget', label: 'Estimated Project Budget', type: 'text' },
          { name: 'projectDescription', label: 'Project Description', type: 'textarea', rows: 4 },
          { name: 'tenderApproach', label: 'Our Proposed Approach', type: 'textarea', rows: 3, placeholder: 'Describe your proposed approach to the project...' }
        ]
      },
      1: {
        title: 'Proposed Team and Capabilities',
        fields: [
          { name: 'proposedLead', label: 'Proposed Lead Appointed Party', required: true, type: 'text' },
          { name: 'proposedInfoManager', label: 'Proposed Information Manager', required: true, type: 'text' },
          { name: 'proposedTeamLeaders', label: 'Proposed Task Team Leaders', type: 'table', columns: ['Discipline', 'Name & Title', 'Company', 'Experience'] },
          { name: 'teamCapabilities', label: 'Team Capabilities and Experience', type: 'textarea', rows: 4, placeholder: 'Describe your team\'s relevant experience and capabilities...' },
          { name: 'subcontractors', label: 'Proposed Subcontractors/Partners', type: 'table', columns: ['Role/Service', 'Company Name', 'Certification', 'Contact'] }
        ]
      },
      2: {
        title: 'Proposed BIM Goals and Objectives',
        fields: [
          { name: 'proposedBimGoals', label: 'Proposed BIM Goals', required: true, type: 'textarea', rows: 4, placeholder: 'Our proposed BIM goals for this project include...' },
          { name: 'proposedObjectives', label: 'Proposed Primary Objectives', type: 'textarea', rows: 3 },
          { name: 'intendedBimUses', label: 'Intended BIM Uses', required: true, type: 'checkbox', options: 'bimUses' }
        ]
      }
    },
    'post-appointment': {
      0: {
        title: 'Project Information and Confirmed Objectives',
        fields: [
          { name: 'projectName', label: 'Project Name', required: true, type: 'text' },
          { name: 'projectNumber', label: 'Project Number', type: 'text' },
          { name: 'projectType', label: 'Project Type', required: true, type: 'select', options: 'projectTypes' },
          { name: 'appointingParty', label: 'Appointing Party', required: true, type: 'text' },
          { name: 'confirmedTimeline', label: 'Confirmed Project Timeline', type: 'text' },
          { name: 'confirmedBudget', label: 'Confirmed Project Budget', type: 'text' },
          { name: 'projectDescription', label: 'Project Description', type: 'textarea', rows: 4 },
          { name: 'deliveryApproach', label: 'Confirmed Delivery Approach', type: 'textarea', rows: 3, placeholder: 'Our confirmed approach to project delivery...' }
        ]
      },
      1: {
        title: 'Confirmed Team and Responsibilities',
        fields: [
          { name: 'leadAppointedParty', label: 'Lead Appointed Party', required: true, type: 'text' },
          { name: 'informationManager', label: 'Information Manager', required: true, type: 'text' },
          { name: 'assignedTeamLeaders', label: 'Assigned Task Team Leaders', type: 'table', columns: ['Discipline', 'Name & Title', 'Company', 'Role Details'] },
          { name: 'finalizedParties', label: 'Finalized Appointed Parties', type: 'table', columns: ['Role/Service', 'Company Name', 'Lead Contact', 'Contract Value'] },
          { name: 'resourceAllocation', label: 'Resource Allocation', type: 'textarea', rows: 3, placeholder: 'Detailed resource allocation and assignments...' }
        ]
      },
      2: {
        title: 'Confirmed BIM Goals and Implementation',
        fields: [
          { name: 'confirmedBimGoals', label: 'Confirmed BIM Goals', required: true, type: 'textarea', rows: 4, placeholder: 'The confirmed BIM goals for this project are...' },
          { name: 'implementationObjectives', label: 'Implementation Objectives', type: 'textarea', rows: 3 },
          { name: 'finalBimUses', label: 'Final BIM Uses', required: true, type: 'checkbox', options: 'bimUses' }
        ]
      }
    }
  },

  // Keep the original shared formFields for sections 3-11 that are common to both
  sharedFormFields: {
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
        { name: 'keyMilestones', label: 'Key Information Delivery Milestones', required: true, type: 'table', columns: ['Stage/Phase', 'Milestone Description', 'Deliverables', 'Due Date'] },
        { name: 'deliverySchedule', label: 'Delivery Schedule', type: 'textarea', rows: 3 },
        { name: 'tidpRequirements', label: 'Task Information Delivery Plans (TIDPs)', type: 'textarea', rows: 3 }
      ]
    },
    5: {
      title: 'Common Data Environment (CDE)',
      fields: [
        { name: 'cdeProvider', label: 'CDE Provider', required: true, type: 'text' },
        { name: 'cdePlatform', label: 'CDE Platform Version', type: 'text' },
        { name: 'workflowStates', label: 'Workflow States', required: true, type: 'table', columns: ['State Name', 'Description', 'Access Level', 'Next State'] },
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
        { name: 'modelingStandards', label: 'Modeling Standards', required: true, type: 'table', columns: ['Standard/Guideline', 'Version', 'Application Area', 'Compliance Level'] },
        { name: 'namingConventions', label: 'Naming Conventions', required: true, type: 'textarea', rows: 3 },
        { name: 'fileStructure', label: 'File Structure', type: 'textarea', rows: 3 },
        { name: 'versionControl', label: 'Version Control', type: 'table', columns: ['Document Type', 'Version Format', 'Approval Process', 'Archive Location'] },
        { name: 'dataExchangeProtocols', label: 'Data Exchange Protocols', type: 'table', columns: ['Exchange Type', 'Format', 'Frequency', 'Delivery Method'] }
      ]
    },
    8: {
      title: 'Quality Assurance and Control',
      fields: [
        { name: 'qaFramework', label: 'Quality Assurance Framework', required: true, type: 'table', columns: ['QA Activity', 'Responsibility', 'Frequency', 'Tools/Methods'] },
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
  },

  // Function to get appropriate form fields based on BEP type and step
  getFormFields: (bepType, stepIndex) => {
    // For steps 0-2, use BEP type specific fields
    if (stepIndex <= 2 && CONFIG.formFields[bepType] && CONFIG.formFields[bepType][stepIndex]) {
      return CONFIG.formFields[bepType][stepIndex];
    }
    // For steps 3-11, use shared fields
    if (stepIndex >= 3 && CONFIG.sharedFormFields[stepIndex]) {
      return CONFIG.sharedFormFields[stepIndex];
    }
    return null;
  }
};

// Dati iniziali di esempio
const INITIAL_DATA = {
  // Common fields for both BEP types
  projectName: 'New Office Complex Development',
  projectNumber: 'NOC-2025-001',
  projectDescription: 'A modern 15-story office complex with retail spaces on the ground floor, underground parking, and sustainable building systems. The project includes advanced MEP systems, curtain wall facades, and LEED Gold certification requirements.',
  projectType: 'Commercial Building',
  appointingParty: 'Metropolitan Development Corp.',

  // Pre-appointment specific fields
  proposedTimeline: '36 months (March 2025 - February 2028)',
  estimatedBudget: '£45M - £52M (preliminary estimate based on current scope)',
  tenderApproach: 'Our proposed approach focuses on collaborative BIM implementation from day one, utilizing cloud-based platforms for real-time coordination, implementing clash detection protocols, and establishing clear information exchange procedures to ensure seamless project delivery.',
  proposedLead: 'Global Construction Ltd. (Lead Contractor with 15+ years BIM experience)',
  proposedInfoManager: 'Sarah Johnson, BIM Manager - Global Construction Ltd. (ISO 19650 Lead Practitioner)',
  proposedTeamLeaders: [
    { 'Discipline': 'Architecture', 'Name & Title': 'John Smith, Director', 'Company': 'Modern Design Associates', 'Experience': '12 years BIM experience, 50+ projects' },
    { 'Discipline': 'Structural', 'Name & Title': 'Emily Chen, Senior Engineer', 'Company': 'Engineering Excellence Ltd.', 'Experience': '10 years structural BIM, P.Eng' },
    { 'Discipline': 'MEP', 'Name & Title': 'Michael Rodriguez, BIM Coordinator', 'Company': 'Advanced Systems Group', 'Experience': '8 years MEP coordination experience' },
    { 'Discipline': 'Facades', 'Name & Title': 'David Wilson, Technical Director', 'Company': 'Curtain Wall Experts Ltd.', 'Experience': '15 years facade design, BIM certified' }
  ],
  teamCapabilities: 'Our team brings together over 45 years of combined BIM experience across all disciplines. We have successfully delivered 50+ projects using collaborative BIM workflows, including 3 similar high-rise office complexes. Our capabilities include advanced parametric modeling, 4D/5D simulation, clash detection, and FM data preparation.',
  subcontractors: [
    { 'Role/Service': 'MEP Services', 'Company Name': 'Advanced Systems Group', 'Certification': 'ISO 19650 certified', 'Contact': 'info@advancedsystems.com' },
    { 'Role/Service': 'Curtain Wall', 'Company Name': 'Specialist Facades Ltd.', 'Certification': 'BIM Level 2 compliant', 'Contact': 'projects@specialistfacades.com' },
    { 'Role/Service': 'Landscaping', 'Company Name': 'Green Spaces Design', 'Certification': 'Autodesk certified', 'Contact': 'design@greenspaces.com' }
  ],
  proposedBimGoals: 'We propose to implement a collaborative BIM workflow that will improve design coordination by 60%, reduce construction conflicts by 90%, optimize project delivery timelines by 20%, and establish a comprehensive digital asset for facility management handover.',
  proposedObjectives: 'Our proposed objectives include achieving zero design conflicts at construction stage, reducing RFIs by 40%, improving construction efficiency by 25%, and delivering comprehensive FM data for operations.',
  intendedBimUses: ['Design Authoring', '3D Coordination', 'Clash Detection', 'Quantity Take-off', '4D Planning'],

  // Post-appointment specific fields
  confirmedTimeline: '36 months (March 2025 - February 2028) - Contract confirmed',
  confirmedBudget: '£47.5M - Final contract value',
  deliveryApproach: 'Our confirmed delivery approach implements the collaborative BIM workflow as agreed in contract, utilizing Autodesk BIM 360 for cloud-based coordination, weekly clash detection reviews, and structured information exchanges at key project milestones.',
  leadAppointedParty: 'Global Construction Ltd.',
  informationManager: 'Sarah Johnson, BIM Manager - Global Construction Ltd.',
  assignedTeamLeaders: [
    { 'Discipline': 'Architecture', 'Name & Title': 'John Smith, Project Director', 'Company': 'Modern Design Associates', 'Role Details': 'Overall design coordination and client liaison' },
    { 'Discipline': 'Structural', 'Name & Title': 'Emily Chen, Senior Engineer', 'Company': 'Engineering Excellence Ltd.', 'Role Details': 'Structural design and analysis coordination' },
    { 'Discipline': 'MEP', 'Name & Title': 'Michael Rodriguez, BIM Coordinator', 'Company': 'Advanced Systems Group', 'Role Details': 'MEP systems integration and clash detection' },
    { 'Discipline': 'Facades', 'Name & Title': 'David Wilson, Technical Director', 'Company': 'Curtain Wall Experts Ltd.', 'Role Details': 'Facade design and performance optimization' }
  ],
  finalizedParties: [
    { 'Role/Service': 'Architecture', 'Company Name': 'Modern Design Associates', 'Lead Contact': 'John Smith - j.smith@mda.com', 'Contract Value': '£2.1M' },
    { 'Role/Service': 'Structural Engineering', 'Company Name': 'Engineering Excellence Ltd.', 'Lead Contact': 'Emily Chen - e.chen@engexcel.com', 'Contract Value': '£1.8M' },
    { 'Role/Service': 'MEP Engineering', 'Company Name': 'Advanced Systems Group', 'Lead Contact': 'Michael Rodriguez - m.rodriguez@asg.com', 'Contract Value': '£3.2M' },
    { 'Role/Service': 'Quantity Surveying', 'Company Name': 'Cost Management Partners', 'Lead Contact': 'Sarah Williams - s.williams@cmp.com', 'Contract Value': '£0.3M' },
    { 'Role/Service': 'Specialist Facades', 'Company Name': 'Curtain Wall Experts Ltd.', 'Lead Contact': 'David Wilson - d.wilson@cwe.com', 'Contract Value': '£4.5M' }
  ],
  resourceAllocation: 'Project staffing confirmed: 2x Senior BIM Coordinators, 4x Discipline BIM Modelers, 1x Information Manager, 1x CDE Administrator. Weekly allocation: 40 hours coordination, 160 hours modeling, 20 hours QA/QC.',
  confirmedBimGoals: 'The confirmed BIM goals include implementing collaborative workflows to achieve improved design coordination, reduced construction conflicts, optimized delivery timelines, and comprehensive digital asset creation for facility management.',
  implementationObjectives: 'Implementation objectives include zero design conflicts at construction, 40% reduction in RFIs, improved construction efficiency, and delivery of comprehensive FM data for operations.',
  finalBimUses: ['Design Authoring', '3D Coordination', 'Clash Detection', 'Quantity Take-off', '4D Planning'],

  // Legacy fields for backward compatibility
  bimGoals: 'Implement a collaborative BIM workflow to improve design coordination, reduce construction conflicts, optimize project delivery timelines, and establish a comprehensive digital asset for facility management handover.',
  bimUses: ['Design Authoring', '3D Coordination', 'Clash Detection', 'Quantity Take-off', '4D Planning'],
  primaryObjectives: 'Achieve zero design conflicts at construction stage, reduce RFIs by 40%, improve construction efficiency, and deliver comprehensive FM data for operations.',
  // Legacy fields for backward compatibility (converted from table format)
  taskTeamLeaders: 'Architecture: John Smith (Modern Design Associates)\nStructural: Emily Chen (Engineering Excellence Ltd.)\nMEP: Michael Rodriguez (Advanced Systems Group)\nFacades: David Wilson (Curtain Wall Experts Ltd.)',
  appointedParties: 'Architecture: Modern Design Associates\nStructural: Engineering Excellence Ltd.\nMEP: Advanced Systems Group\nQuantity Surveyor: Cost Management Partners\nSpecialist Facades: Curtain Wall Experts Ltd.',
  informationPurposes: ['Design Development', 'Construction Planning', 'Quantity Surveying', 'Facility Management'],
  geometricalInfo: 'LOD 350 for construction documentation phase, with dimensional accuracy of ±10mm for structural elements and ±5mm for MEP coordination points.',
  alphanumericalInfo: 'All building elements must include material specifications, performance data, manufacturer information, maintenance requirements, and warranty details.',
  documentationInfo: 'Construction drawings, specifications, schedules, O&M manuals, warranty documents, and asset registers in digital format.',
  informationFormats: ['IFC 4', 'PDF', 'BCF 2.1', 'DWG', 'COBie'],
  midpDescription: 'The MIDP coordinates all discipline-specific TIDPs into a unified delivery schedule aligned with RIBA stages and construction milestones. Information exchanges occur at stage gates with formal approval processes.',
  keyMilestones: [
    { 'Stage/Phase': 'Stage 2', 'Milestone Description': 'Concept Design Complete', 'Deliverables': 'Basic geometry and spatial coordination models', 'Due Date': 'Month 6' },
    { 'Stage/Phase': 'Stage 3', 'Milestone Description': 'Spatial Coordination', 'Deliverables': 'Full coordination model with clash detection', 'Due Date': 'Month 12' },
    { 'Stage/Phase': 'Stage 4', 'Milestone Description': 'Technical Design', 'Deliverables': 'Construction-ready information and documentation', 'Due Date': 'Month 18' },
    { 'Stage/Phase': 'Stage 5', 'Milestone Description': 'Manufacturing Support', 'Deliverables': 'Production information and fabrication models', 'Due Date': 'Month 24' },
    { 'Stage/Phase': 'Stage 6', 'Milestone Description': 'Handover', 'Deliverables': 'As-built models and FM data', 'Due Date': 'Month 36' }
  ],
  deliverySchedule: 'Monthly model updates during design phases, weekly coordination cycles during construction documentation, and daily updates during critical construction phases.',
  tidpRequirements: 'Each task team must produce TIDPs detailing their information deliverables, responsibilities, quality requirements, and delivery schedules in alignment with project milestones.',
  cdeProvider: 'Autodesk BIM 360',
  cdePlatform: 'BIM 360 Design & Docs - Enterprise Version 2024',
  workflowStates: [
    { 'State Name': 'Work in Progress (WIP)', 'Description': 'Active development by task teams', 'Access Level': 'Author only', 'Next State': 'Shared' },
    { 'State Name': 'Shared', 'Description': 'Available for coordination and review', 'Access Level': 'Team members', 'Next State': 'Published' },
    { 'State Name': 'Published', 'Description': 'Approved for use by the project team', 'Access Level': 'All stakeholders', 'Next State': 'Archived' },
    { 'State Name': 'Archived', 'Description': 'Historical versions for reference', 'Access Level': 'Read-only access', 'Next State': 'N/A' }
  ],
  accessControl: 'Role-based access with project administrator, discipline leads, team members, and read-only stakeholder levels. Multi-factor authentication required for all users.',
  securityMeasures: 'ISO 27001 compliant platform with end-to-end encryption, regular security audits, data residency controls, and comprehensive audit logging.',
  backupProcedures: 'Automated daily backups with 30-day retention, weekly full system backups, geographic redundancy, and quarterly disaster recovery testing.',
  bimSoftware: ['Autodesk Revit', 'Navisworks', 'Solibri Model Checker', 'BIM 360'],
  fileFormats: ['IFC 4', 'DWG', 'PDF', 'BCF 2.1', 'NWD'],
  hardwareRequirements: 'Minimum: Intel i7 or equivalent, 32GB RAM, dedicated graphics card (RTX 3060 or higher), 1TB SSD storage, dual monitors recommended.',
  networkRequirements: 'High-speed internet connection (minimum 100 Mbps), VPN access for remote working, secure cloud connectivity to CDE platform.',
  interoperabilityNeeds: 'Seamless data exchange between Revit disciplines, coordination in Navisworks, model checking in Solibri, and cloud collaboration through BIM 360.',
  modelingStandards: [
    { 'Standard/Guideline': 'UK BIM Alliance Standards', 'Version': 'v2.1', 'Application Area': 'General BIM practices', 'Compliance Level': 'Mandatory' },
    { 'Standard/Guideline': 'Uniclass 2015', 'Version': '2015', 'Application Area': 'Classification system', 'Compliance Level': 'Mandatory' },
    { 'Standard/Guideline': 'AIA LOD Specification', 'Version': '2019', 'Application Area': 'Level of development', 'Compliance Level': 'Mandatory' },
    { 'Standard/Guideline': 'Company Modeling Guide', 'Version': 'v3.2', 'Application Area': 'Internal procedures', 'Compliance Level': 'Required' }
  ],
  namingConventions: 'Project code: NOC, Originator codes by discipline (ARC, STR, MEP), Volume/Level codes, Type classifications following BS 1192 naming convention.',
  fileStructure: 'Organized by discipline and project phase with clear folder hierarchies, version control through file naming, and linked file management protocols.',
  versionControl: [
    { 'Document Type': 'Working Models', 'Version Format': 'P01, P02, P03...', 'Approval Process': 'Author approval only', 'Archive Location': 'WIP folder' },
    { 'Document Type': 'Issued Drawings', 'Version Format': 'A, B, C, D...', 'Approval Process': 'Discipline lead + PM', 'Archive Location': 'Published folder' },
    { 'Document Type': 'Coordination Models', 'Version Format': 'Weekly increments', 'Approval Process': 'BIM Coordinator', 'Archive Location': 'Shared folder' },
    { 'Document Type': 'Final Deliverables', 'Version Format': 'Client approval code', 'Approval Process': 'Client sign-off', 'Archive Location': 'Client portal' }
  ],
  dataExchangeProtocols: [
    { 'Exchange Type': 'IFC Coordination', 'Format': 'IFC 4.0', 'Frequency': 'Weekly', 'Delivery Method': 'BIM 360 upload' },
    { 'Exchange Type': 'Issue Management', 'Format': 'BCF 2.1', 'Frequency': 'Daily as needed', 'Delivery Method': 'BCF workflow' },
    { 'Exchange Type': 'Drawing Sets', 'Format': 'PDF + DWG', 'Frequency': 'At milestones', 'Delivery Method': 'Client portal' },
    { 'Exchange Type': 'FM Handover', 'Format': 'COBie + IFC', 'Frequency': 'Final delivery', 'Delivery Method': 'Secure transfer' }
  ],
  qaFramework: [
    { 'QA Activity': 'Automated Model Checking', 'Responsibility': 'BIM Coordinator', 'Frequency': 'Daily', 'Tools/Methods': 'Solibri Model Checker + custom rules' },
    { 'QA Activity': 'Manual Design Reviews', 'Responsibility': 'Discipline Leads', 'Frequency': 'Weekly', 'Tools/Methods': 'Navisworks review sessions' },
    { 'QA Activity': 'Clash Detection', 'Responsibility': 'BIM Coordinator', 'Frequency': 'Bi-weekly', 'Tools/Methods': 'Navisworks Manage + BCF reports' },
    { 'QA Activity': 'Standards Compliance', 'Responsibility': 'Information Manager', 'Frequency': 'Monthly', 'Tools/Methods': 'Compliance checklist + audit trail' },
    { 'QA Activity': 'Client Reviews', 'Responsibility': 'Project Manager', 'Frequency': 'At milestones', 'Tools/Methods': 'Formal review meetings + sign-off' }
  ],
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
const EditableTable = React.memo(({ field, value, onChange, error }) => {
  const { name, label, required, columns = ['Role/Discipline', 'Name/Company', 'Experience/Notes'] } = field;
  const tableData = Array.isArray(value) ? value : [];

  const addRow = () => {
    const newRow = columns.reduce((acc, col) => ({ ...acc, [col]: '' }), {});
    onChange(name, [...tableData, newRow]);
  };

  const removeRow = (index) => {
    const newData = tableData.filter((_, i) => i !== index);
    onChange(name, newData);
  };

  const updateCell = (rowIndex, column, cellValue) => {
    const newData = tableData.map((row, index) =>
      index === rowIndex ? { ...row, [column]: cellValue } : row
    );
    onChange(name, newData);
  };

  const moveRow = (fromIndex, toIndex) => {
    if (toIndex < 0 || toIndex >= tableData.length) return;
    const newData = [...tableData];
    const [movedRow] = newData.splice(fromIndex, 1);
    newData.splice(toIndex, 0, movedRow);
    onChange(name, newData);
  };

  return (
    <div className="mb-8">
      <label className="block text-lg font-semibold mb-4 text-gray-800">
        {label} {required && <span className="text-red-500">*</span>}
      </label>

      <div className="border rounded-xl overflow-hidden shadow-sm bg-white">
        <div className="bg-gradient-to-r from-gray-50 to-gray-100 px-6 py-4 border-b border-gray-200">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <span className="text-base font-semibold text-gray-800">
                {tableData.length} {tableData.length === 1 ? 'Entry' : 'Entries'}
              </span>
              {tableData.length > 0 && (
                <span className="text-sm text-gray-500">
                  Click and drag to reorder • Use textarea for multi-line content
                </span>
              )}
            </div>
            <button
              type="button"
              onClick={addRow}
              className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-all transform hover:scale-105 shadow-md"
            >
              <span className="text-lg">+</span>
              <span>Add Row</span>
            </button>
          </div>
        </div>

        {tableData.length === 0 ? (
          <div className="p-12 text-center text-gray-500">
            <Users className="w-16 h-16 mx-auto mb-4 text-gray-300" />
            <p className="text-lg">No entries yet. Click "Add Row" to get started.</p>
          </div>
        ) : (
          <div className="overflow-x-auto bg-white">
            <table className="w-full min-w-full table-fixed">
              <thead className="bg-gray-50">
                <tr>
                  <th className="w-16 px-2 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                    Order
                  </th>
                  {columns.map((column, index) => (
                    <th key={column} className={`px-3 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider ${
                      columns.length === 4 ? 'w-1/4' :
                      columns.length === 3 ? 'w-1/3' :
                      'w-auto'
                    }`}>
                      {column}
                    </th>
                  ))}
                  <th className="w-16 px-2 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {tableData.map((row, rowIndex) => (
                  <tr key={rowIndex} className="hover:bg-gray-50 transition-colors">
                    <td className="px-2 py-2">
                      <div className="flex flex-col items-center space-y-1">
                        <button
                          type="button"
                          onClick={() => moveRow(rowIndex, rowIndex - 1)}
                          disabled={rowIndex === 0}
                          className="w-6 h-6 flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs"
                          title="Move up"
                        >
                          ↑
                        </button>
                        <span className="text-xs font-medium text-gray-600 bg-gray-100 px-1 py-0.5 rounded">{rowIndex + 1}</span>
                        <button
                          type="button"
                          onClick={() => moveRow(rowIndex, rowIndex + 1)}
                          disabled={rowIndex === tableData.length - 1}
                          className="w-6 h-6 flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs"
                          title="Move down"
                        >
                          ↓
                        </button>
                      </div>
                    </td>
                    {columns.map(column => (
                      <td key={column} className="px-1 py-2">
                        <textarea
                          value={row[column] || ''}
                          onChange={(e) => updateCell(rowIndex, column, e.target.value)}
                          className="w-full min-h-[100px] p-2 border border-gray-300 rounded focus:ring-1 focus:ring-blue-500 focus:border-blue-500 text-sm resize-y"
                          placeholder={`Enter ${column.toLowerCase()}...`}
                          rows={4}
                        />
                      </td>
                    ))}
                    <td className="px-2 py-2">
                      <button
                        type="button"
                        onClick={() => removeRow(rowIndex)}
                        className="w-8 h-8 flex items-center justify-center text-red-600 hover:text-red-800 hover:bg-red-50 rounded transition-colors font-medium text-sm"
                        title="Remove row"
                      >
                        ✕
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
});

const InputField = React.memo(({ field, value, onChange, error }) => {
  const { name, label, type, required, rows, placeholder, options: fieldOptions } = field;
  const optionsList = fieldOptions ? CONFIG.options[fieldOptions] : null;

  const baseClasses = "w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500";

  const handleCheckboxChange = (option) => {
    const current = Array.isArray(value) ? value : [];
    const updated = current.includes(option)
      ? current.filter(item => item !== option)
      : [...current, option];
    onChange(name, updated);
  };

  switch (type) {
    case 'table':
      return (
        <EditableTable
          field={field}
          value={value}
          onChange={onChange}
          error={error}
        />
      );

    case 'textarea':
      return (
        <div>
          <label htmlFor={name} className="block text-sm font-medium mb-2">
            {label} {required && '*'}
          </label>
          <textarea
            id={name}
            aria-required={required}
            value={value || ''}
            onChange={(e) => onChange(name, e.target.value)}
            rows={rows || 3}
            className={baseClasses}
            placeholder={placeholder || `Enter ${label.toLowerCase()}...`}
          />
          {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
        </div>
      );

    case 'select':
      return (
        <div>
          <label htmlFor={name} className="block text-sm font-medium mb-2">
            {label} {required && '*'}
          </label>
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
          <label className="block text-sm font-medium mb-2">
            {label} {required && '*'}
          </label>
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
          <label htmlFor={name} className="block text-sm font-medium mb-2">
            {label} {required && '*'}
          </label>
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

const ProgressSidebar = React.memo(({ steps, currentStep, completedSections, onStepClick, validateStep }) => (
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
                CONFIG.categories[step.category].bg
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
));

const EnhancedBepTypeSelector = ({ bepType, setBepType, onProceed }) => (
  <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
    <div className="bg-white rounded-2xl shadow-xl w-full max-w-4xl p-8">
      <div className="text-center mb-8">
        <Zap className="w-16 h-16 text-blue-600 mx-auto mb-4" />
        <h1 className="text-3xl font-bold text-gray-900 mb-2">BIM Execution Plan Generator</h1>
        <p className="text-gray-600 mb-6">Choose your BEP type to begin the tailored workflow</p>
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-left">
          <h3 className="font-semibold text-yellow-800 mb-2">What is a BEP?</h3>
          <p className="text-sm text-yellow-700">
            A BIM Execution Plan (BEP) explains how the information management aspects of the appointment will be carried out by the delivery team.
            It sets out how information requirements are managed and delivered collectively by all parties involved in the project.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {Object.entries(CONFIG.bepTypeDefinitions).map(([key, definition]) => {
          const IconComponent = definition.icon;
          const isSelected = bepType === key;

          return (
            <div
              key={key}
              className={`relative p-6 border-2 rounded-xl cursor-pointer transition-all transform hover:scale-105 ${
                isSelected
                  ? `border-${definition.color}-500 ${definition.bgClass} shadow-lg`
                  : 'border-gray-200 bg-white hover:border-gray-300 shadow-md'
              }`}
              onClick={() => setBepType(key)}
            >
              <div className="flex items-start space-x-4">
                <div className={`p-3 rounded-lg ${
                  isSelected ? `bg-${definition.color}-100` : 'bg-gray-100'
                }`}>
                  <IconComponent className={`w-8 h-8 ${
                    isSelected ? `text-${definition.color}-600` : 'text-gray-600'
                  }`} />
                </div>

                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-2">
                    <h3 className={`text-xl font-bold ${
                      isSelected ? definition.textClass : 'text-gray-900'
                    }`}>
                      {definition.title}
                    </h3>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      isSelected
                        ? `bg-${definition.color}-100 text-${definition.color}-700`
                        : 'bg-gray-100 text-gray-600'
                    }`}>
                      {definition.subtitle}
                    </span>
                  </div>

                  <p className="text-sm text-gray-600 mb-4 leading-relaxed">
                    {definition.description}
                  </p>

                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <Target className="w-4 h-4 text-gray-500" />
                      <span className="text-sm font-medium text-gray-700">Purpose:</span>
                      <span className="text-sm text-gray-600">{definition.purpose}</span>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Eye className="w-4 h-4 text-gray-500" />
                      <span className="text-sm font-medium text-gray-700">Focus:</span>
                      <span className="text-sm text-gray-600">{definition.focus}</span>
                    </div>

                    <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                      <span className="text-xs font-medium text-gray-700 block mb-1">Language Style:</span>
                      <span className="text-xs text-gray-600 italic">{definition.language}</span>
                    </div>
                  </div>
                </div>
              </div>

              {isSelected && (
                <div className="absolute top-3 right-3">
                  <CheckCircle className={`w-6 h-6 text-${definition.color}-600`} />
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="text-center">
        <button
          onClick={onProceed}
          disabled={!bepType}
          className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium px-8 py-3 rounded-lg transition-all transform hover:scale-105 shadow-lg mx-auto disabled:transform-none disabled:cursor-not-allowed"
        >
          <span>Proceed with {bepType ? CONFIG.bepTypeDefinitions[bepType].title : 'Selected BEP Type'}</span>
          <ChevronRight className="w-5 h-5" />
        </button>

        {bepType && (
          <p className="mt-3 text-sm text-gray-600">
            You've selected: <span className="font-medium">{CONFIG.bepTypeDefinitions[bepType].title}</span>
          </p>
        )}
      </div>
    </div>
  </div>
);

const FormStep = React.memo(({ stepIndex, formData, updateFormData, errors, bepType }) => {
  const stepConfig = CONFIG.getFormFields(bepType, stepIndex);
  if (!stepConfig) return null;

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold">{stepConfig.title}</h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {stepConfig.fields.map(field => (
          <div key={field.name} className={field.type === 'textarea' || field.type === 'checkbox' || field.type === 'table' ? 'md:col-span-2' : ''}>
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

const PreviewExportPage = ({ generateBEPContent, exportFormat, setExportFormat, previewBEP, downloadBEP, isExporting }) => {
  const content = generateBEPContent();
  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold">Preview & Export</h3>
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
          disabled={isExporting}
          className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-8 py-3 rounded-lg transition-all transform hover:scale-105 shadow-lg disabled:opacity-50"
        >
          <Download className="w-5 h-5" />
          <span>{isExporting ? 'Exporting...' : 'Download Professional BEP'}</span>
        </button>
      </div>

      <iframe
        srcDoc={content}
        title="BEP Preview"
        className="w-full border rounded-lg"
        style={{ height: '600px' }}
      />
    </div>
  );
};

const AppContent = () => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Zap className="w-12 h-12 text-blue-600 animate-pulse mx-auto mb-4" />
          <p className="text-gray-600">Loading BEP Generator...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return <Login />;
  }

  return <ProfessionalBEPGenerator user={user} />;
};

const ProfessionalBEPGenerator = ({ user }) => {
  const { logout } = useAuth();
  const [currentStep, setCurrentStep] = useState(0);
  const [bepType, setBepType] = useState('');
  const [formData, setFormData] = useState(INITIAL_DATA);
  const [completedSections, setCompletedSections] = useState(new Set());
  const [exportFormat, setExportFormat] = useState('html');
  const [errors, setErrors] = useState({});
  const [isExporting, setIsExporting] = useState(false);
  const [showBepTypeSelector, setShowBepTypeSelector] = useState(true);

  useEffect(() => {
    const savedData = localStorage.getItem(`bepData_${user.id}`);
    if (savedData) {
      try {
        const parsedData = JSON.parse(savedData);
        // Merge saved data with INITIAL_DATA to ensure new fields have example values
        setFormData({ ...INITIAL_DATA, ...parsedData });
      } catch (error) {
        console.error('Error parsing saved data:', error);
        // If there's an error, use INITIAL_DATA
        setFormData(INITIAL_DATA);
      }
    }
  }, [user.id]);

  const debounce = (func, delay) => {
    let timeoutId;
    return (...args) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => func(...args), delay);
    };
  };

  useEffect(() => {
    const debouncedSave = debounce(() => {
      localStorage.setItem(`bepData_${user.id}`, JSON.stringify(formData));
    }, 500);
    debouncedSave();
  }, [formData, user.id]);

  const updateFormData = useCallback((field, value) => {
    const sanitizedValue = typeof value === 'string' ? DOMPurify.sanitize(value) : value;
    setFormData(prev => ({ ...prev, [field]: sanitizedValue }));
    const stepConfig = CONFIG.getFormFields(bepType, currentStep);
    const fieldConfig = stepConfig?.fields.find(f => f.name === field);
    if (fieldConfig) {
      const error = validateField(field, sanitizedValue, fieldConfig.required);
      setErrors(prev => ({ ...prev, [field]: error }));
    }
  }, [currentStep, bepType]);

  const validateField = (name, value, required) => {
    if (required && (!value || (Array.isArray(value) && value.length === 0) || (typeof value === 'string' && value.trim() === ''))) {
      return `${name.replace(/([A-Z])/g, ' $1').trim()} is required`;
    }
    return null;
  };

  const validateStep = useCallback((stepIndex) => {
    const stepConfig = CONFIG.getFormFields(bepType, stepIndex);
    if (!stepConfig) return true;

    return stepConfig.fields.every(field => {
      const value = formData[field.name];
      return !field.required || (value && (Array.isArray(value) ? value.length > 0 : value.trim() !== ''));
    });
  }, [formData, bepType]);

  const validatedSteps = useMemo(() => {
    return CONFIG.steps.map((_, index) => validateStep(index));
  }, [validateStep]);

  const validateCurrentStep = () => {
    const stepConfig = CONFIG.getFormFields(bepType, currentStep);
    if (!stepConfig) return true;

    const newErrors = {};
    let isValid = true;

    stepConfig.fields.forEach(field => {
      const error = validateField(field.name, formData[field.name], field.required);
      if (error) {
        newErrors[field.name] = error;
        isValid = false;
      }
    });

    setErrors(newErrors);
    return isValid;
  };

  const nextStep = () => {
    if (validateCurrentStep()) {
      setCompletedSections(prev => new Set([...prev, currentStep]));
      if (currentStep < CONFIG.steps.length - 1) {
        setCurrentStep(currentStep + 1);
      }
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const onStepClick = useCallback((index) => setCurrentStep(index), []);

  const goToPreview = () => {
    if (validateCurrentStep()) {
      setCompletedSections(prev => new Set([...prev, currentStep]));
      setCurrentStep(CONFIG.steps.length);
    }
  };

  const generateBEPContent = () => {
    const currentDate = new Date();
    const formattedDate = currentDate.toLocaleDateString();
    const formattedTime = currentDate.toLocaleTimeString();

    // Group steps by category
    const groupedSteps = CONFIG.steps.reduce((acc, step, index) => {
      const cat = step.category;
      if (!acc[cat]) acc[cat] = [];
      const stepConfig = CONFIG.getFormFields(bepType, index);
      if (stepConfig) {
        acc[cat].push({ index, title: `${acc[cat].length + 1}. ${stepConfig.title.toUpperCase()}`, fields: stepConfig.fields });
      }
      return acc;
    }, {});

    const renderField = (field) => {
      let value = formData[field.name];
      if (!value) return '';

      if (field.type === 'checkbox' && Array.isArray(value)) {
        return `<h3>${field.label}</h3><ul>${value.map(item => `<li>${DOMPurify.sanitize(item)}</li>`).join('')}</ul>`;
      }

      if (field.type === 'table' && Array.isArray(value)) {
        if (value.length === 0) return '';

        const columns = field.columns || ['Role/Discipline', 'Name/Company', 'Experience/Notes'];
        let tableHtml = `<h3>${field.label}</h3>`;
        tableHtml += '<table class="table-data" style="width: 100%; border-collapse: collapse; margin: 10px 0;">';

        // Header
        tableHtml += '<thead><tr style="background: #f3f4f6;">';
        columns.forEach(col => {
          tableHtml += `<th style="border: 1px solid #d1d5db; padding: 8px; text-align: left; font-weight: bold;">${col}</th>`;
        });
        tableHtml += '</tr></thead>';

        // Rows
        tableHtml += '<tbody>';
        value.forEach((row, index) => {
          tableHtml += `<tr style="background: ${index % 2 === 0 ? '#ffffff' : '#f9fafb'};">`;
          columns.forEach(col => {
            const cellValue = DOMPurify.sanitize(row[col] || '');
            tableHtml += `<td style="border: 1px solid #d1d5db; padding: 8px;">${cellValue}</td>`;
          });
          tableHtml += '</tr>';
        });
        tableHtml += '</tbody></table>';

        return tableHtml;
      }

      if (field.type === 'textarea') {
        return `<h3>${field.label}</h3><p>${DOMPurify.sanitize(value)}</p>`;
      }

      return `<tr><td class="label">${field.label}:</td><td>${DOMPurify.sanitize(value)}</td></tr>`;
    };

    const sectionsHtml = Object.entries(groupedSteps).map(([cat, items]) => [
      `<div class="category-header">${CONFIG.categories[cat].name}</div>`,
      items.map(item => {
        const fields = item.fields;
        const tableFields = fields.filter(f => f.type !== 'textarea' && f.type !== 'checkbox');
        const otherFields = fields.filter(f => f.type === 'textarea' || f.type === 'checkbox');
        return `
          <div class="section">
            <h2>${item.title}</h2>
            <div class="info-box">
              ${tableFields.length > 0 ? `<table>${tableFields.map(renderField).join('')}</table>` : ''}
              ${otherFields.map(renderField).join('')}
            </div>
          </div>
        `;
      }).join('')
    ]).flat().join('');

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
          <div class="bep-type">${CONFIG.bepTypeDefinitions[bepType].title}</div>
          <div class="bep-purpose" style="background: #f3f4f6; padding: 10px; border-radius: 8px; margin: 10px 0; font-style: italic; color: #6b7280;">
            ${CONFIG.bepTypeDefinitions[bepType].description}
          </div>
        </div>

        <div class="compliance-box">
          <h3>Document Information</h3>
          <table>
            <tr><td class="label">Document Type:</td><td>${CONFIG.bepTypeDefinitions[bepType].title}</td></tr>
            <tr><td class="label">Document Purpose:</td><td>${CONFIG.bepTypeDefinitions[bepType].purpose}</td></tr>
            <tr><td class="label">Project Name:</td><td>${formData.projectName || 'Not specified'}</td></tr>
            <tr><td class="label">Project Number:</td><td>${formData.projectNumber || 'Not specified'}</td></tr>
            <tr><td class="label">Generated Date:</td><td>${formattedDate} at ${formattedTime}</td></tr>
            <tr><td class="label">Status:</td><td>${bepType === 'pre-appointment' ? 'Tender Submission' : 'Working Document'}</td></tr>
          </table>
        </div>

        ${sectionsHtml}

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

  const generateDocx = async () => {
    const currentDate = new Date();
    const formattedDate = currentDate.toLocaleDateString();
    const formattedTime = currentDate.toLocaleTimeString();

    const sections = [];

    // Header
    sections.push(
      new Paragraph({
        text: "BIM EXECUTION PLAN (BEP)",
        heading: HeadingLevel.HEADING_1,
        alignment: AlignmentType.CENTER
      }),
      new Paragraph({
        text: "ISO 19650-2 Compliant",
        heading: HeadingLevel.HEADING_2,
        alignment: AlignmentType.CENTER
      }),
      new Paragraph({
        text: CONFIG.bepTypeDefinitions[bepType].title,
        alignment: AlignmentType.CENTER
      }),
      new Paragraph({
        text: CONFIG.bepTypeDefinitions[bepType].description,
        alignment: AlignmentType.CENTER
      })
    );

    // Document Information Table
    sections.push(
      new Paragraph({
        text: "Document Information",
        heading: HeadingLevel.HEADING_3
      }),
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        rows: [
          new TableRow({
            children: [
              new TableCell({ children: [new Paragraph("Document Type:")], width: { size: 50, type: WidthType.PERCENTAGE } }),
              new TableCell({ children: [new Paragraph(CONFIG.bepTypeDefinitions[bepType].title)] })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ children: [new Paragraph("Document Purpose:")] }),
              new TableCell({ children: [new Paragraph(CONFIG.bepTypeDefinitions[bepType].purpose)] })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ children: [new Paragraph("Project Name:")] }),
              new TableCell({ children: [new Paragraph(formData.projectName || 'Not specified')] })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ children: [new Paragraph("Project Number:")] }),
              new TableCell({ children: [new Paragraph(formData.projectNumber || 'Not specified')] })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ children: [new Paragraph("Generated Date:")] }),
              new TableCell({ children: [new Paragraph(`${formattedDate} at ${formattedTime}`)] })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ children: [new Paragraph("Status:")] }),
              new TableCell({ children: [new Paragraph(bepType === 'pre-appointment' ? 'Tender Submission' : 'Working Document')] })
            ]
          })
        ]
      })
    );

    // Group steps by category
    const groupedSteps = CONFIG.steps.reduce((acc, step, index) => {
      const cat = step.category;
      if (!acc[cat]) acc[cat] = [];
      const stepConfig = CONFIG.getFormFields(bepType, index);
      if (stepConfig) {
        acc[cat].push({ index, title: `${acc[cat].length + 1}. ${stepConfig.title.toUpperCase()}`, fields: stepConfig.fields });
      }
      return acc;
    }, {});

    Object.entries(groupedSteps).forEach(([cat, items]) => {
      sections.push(
        new Paragraph({
          text: CONFIG.categories[cat].name,
          heading: HeadingLevel.HEADING_1
        })
      );

      items.forEach(item => {
        sections.push(
          new Paragraph({
            text: item.title,
            heading: HeadingLevel.HEADING_2
          })
        );

        const fields = item.fields;
        const tableFields = fields.filter(f => f.type !== 'textarea' && f.type !== 'checkbox');
        const otherFields = fields.filter(f => f.type === 'textarea' || f.type === 'checkbox');

        if (tableFields.length > 0) {
          const tableRows = tableFields.map(field => 
            new TableRow({
              children: [
                new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: field.label + ":", bold: true }) ] })] }),
                new TableCell({ children: [new Paragraph(formData[field.name] || '')] })
              ]
            })
          );

          sections.push(
            new Table({
              width: { size: 100, type: WidthType.PERCENTAGE },
              rows: tableRows
            })
          );
        }

        otherFields.forEach(field => {
          sections.push(
            new Paragraph({
              text: field.label,
              heading: HeadingLevel.HEADING_3
            })
          );

          const value = formData[field.name];
          if (field.type === 'checkbox' && Array.isArray(value)) {
            value.forEach(item => {
              sections.push(
                new Paragraph({
                  text: item,
                  bullet: { level: 0 }
                })
              );
            });
          } else if (field.type === 'textarea') {
            sections.push(
              new Paragraph(value || '')
            );
          }
        });
      });
    });

    // Footer
    sections.push(
      new Paragraph({
        text: "Document Control Information",
        heading: HeadingLevel.HEADING_3
      }),
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        rows: [
          new TableRow({
            children: [
              new TableCell({ children: [new Paragraph("Document Type:")] }),
              new TableCell({ children: [new Paragraph("BIM Execution Plan (BEP)")] })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ children: [new Paragraph("ISO Standard:")] }),
              new TableCell({ children: [new Paragraph("ISO 19650-2:2018")] })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ children: [new Paragraph("Document Status:")] }),
              new TableCell({ children: [new Paragraph("Work in Progress")] })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ children: [new Paragraph("Generated By:")] }),
              new TableCell({ children: [new Paragraph("Professional BEP Generator Tool")] })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ children: [new Paragraph("Generated Date:")] }),
              new TableCell({ children: [new Paragraph(formattedDate)] })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ children: [new Paragraph("Generated Time:")] }),
              new TableCell({ children: [new Paragraph(formattedTime)] })
            ]
          })
        ]
      })
    );

    const doc = new Document({
      sections: [{
        properties: {},
        children: sections,
      }],
    });

    return doc;
  };

  const generatePDF = () => {
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4'
    });
    let y = 10;
    const margin = 10;
    const pageWidth = pdf.internal.pageSize.getWidth();
    const maxLineWidth = pageWidth - 2 * margin;
    const lineHeight = 6;

    const addText = (text, size, bold = false, align = 'left') => {
      pdf.setFontSize(size);
      pdf.setFont('helvetica', bold ? 'bold' : 'normal');
      const lines = text.split('\n').flatMap(line => pdf.splitTextToSize(line, maxLineWidth));
      lines.forEach(line => {
        pdf.text(line, margin, y, { align });
        y += lineHeight;
        if (y > 270) {
          pdf.addPage();
          y = margin;
        }
      });
    };

    const addTable = (rows) => {
      rows.forEach(([label, value]) => {
        addText(label + ':', 10, true);
        addText(value, 10);
        y += lineHeight / 2;
      });
    };

    const addTableData = (field) => {
      let value = formData[field.name];
      if (!value) return;

      addText(field.label + ':', 12, true);
      y += lineHeight / 2;

      if (field.type === 'table' && Array.isArray(value)) {
        if (value.length === 0) return;

        const columns = field.columns || ['Role/Discipline', 'Name/Company', 'Experience/Notes'];

        // Add table header
        let headerText = columns.join(' | ');
        addText(headerText, 9, true);
        addText('-'.repeat(headerText.length), 9);

        // Add table rows
        value.forEach((row, index) => {
          let rowText = columns.map(col => row[col] || '').join(' | ');
          addText(rowText, 9);
        });
        y += lineHeight;
      } else if (field.type === 'checkbox' && Array.isArray(value)) {
        addText(value.join(', '), 10);
      } else if (typeof value === 'string') {
        addText(value, 10);
      }
      y += lineHeight;
    };

    const currentDate = new Date();
    const formattedDate = currentDate.toLocaleDateString();
    const formattedTime = currentDate.toLocaleTimeString();

    // Header
    addText('BIM EXECUTION PLAN (BEP)', 18, true, 'center');
    y += lineHeight;
    addText('ISO 19650-2 Compliant', 14, true, 'center');
    y += lineHeight;
    addText(CONFIG.bepTypeDefinitions[bepType].title, 12, true, 'center');
    y += lineHeight;
    addText(CONFIG.bepTypeDefinitions[bepType].description, 10, false, 'center');
    y += lineHeight * 2;

    // Document Information
    addText('Document Information', 12, true);
    y += lineHeight;
    addTable([
      ['Document Type', CONFIG.bepTypeDefinitions[bepType].title],
      ['Document Purpose', CONFIG.bepTypeDefinitions[bepType].purpose],
      ['Project Name', formData.projectName || 'Not specified'],
      ['Project Number', formData.projectNumber || 'Not specified'],
      ['Generated Date', `${formattedDate} at ${formattedTime}`],
      ['Status', bepType === 'pre-appointment' ? 'Tender Submission' : 'Working Document']
    ]);
    y += lineHeight * 2;

    // Group steps by category
    const groupedSteps = CONFIG.steps.reduce((acc, step, index) => {
      const cat = step.category;
      if (!acc[cat]) acc[cat] = [];
      const stepConfig = CONFIG.getFormFields(bepType, index);
      if (stepConfig) {
        acc[cat].push({ index, title: `${acc[cat].length + 1}. ${stepConfig.title.toUpperCase()}`, fields: stepConfig.fields });
      }
      return acc;
    }, {});

    Object.entries(groupedSteps).forEach(([cat, items]) => {
      addText(CONFIG.categories[cat].name, 16, true);
      y += lineHeight;

      items.forEach(item => {
        addText(item.title, 14, true);
        y += lineHeight;

        item.fields.forEach(field => {
          if (field.type === 'table') {
            addTableData(field);
          } else {
            const value = formData[field.name] || '';
            addText(field.label, 12, true);
            y += lineHeight / 2;

            if (field.type === 'checkbox' && Array.isArray(value)) {
              value.forEach(item => {
                addText('- ' + item, 10);
              });
            } else {
              addText(value, 10);
            }
            y += lineHeight / 2;
          }
        });
        y += lineHeight;
      });
    });

    // Footer
    y += lineHeight * 2;
    addText('Document Control Information', 12, true);
    y += lineHeight;
    addTable([
      ['Document Type', 'BIM Execution Plan (BEP)'],
      ['ISO Standard', 'ISO 19650-2:2018'],
      ['Document Status', 'Work in Progress'],
      ['Generated By', 'Professional BEP Generator Tool'],
      ['Generated Date', formattedDate],
      ['Generated Time', formattedTime]
    ]);

    return pdf;
  };

  const downloadBEP = async () => {
    setIsExporting(true);
    const content = generateBEPContent();
    const currentDate = new Date().toISOString().split('T')[0];
    const fileName = `Professional_BEP_${formData.projectName || 'Project'}_${currentDate}`;

    try {
      if (exportFormat === 'html') {
        const blob = new Blob([content], { type: 'text/html;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${fileName}.html`;
        a.click();
        URL.revokeObjectURL(url);
      } else if (exportFormat === 'pdf') {
        const pdf = generatePDF();
        pdf.save(`${fileName}.pdf`);
      } else if (exportFormat === 'word') {
        const doc = await generateDocx();
        const blob = await Packer.toBlob(doc);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${fileName}.docx`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Export error:', error);
    } finally {
      setIsExporting(false);
    }
  };

  const previewBEP = () => {
    const content = generateBEPContent();
    const previewWindow = window.open('', '_blank', 'width=1200,height=800');
    previewWindow.document.write(content);
    previewWindow.document.close();
  };

  const handleBepTypeProceed = () => {
    setShowBepTypeSelector(false);
    setCurrentStep(0);
  };

  // Helper function to reset data to initial values (useful for testing)
  const resetToInitialData = () => {
    setFormData(INITIAL_DATA);
    localStorage.setItem(`bepData_${user.id}`, JSON.stringify(INITIAL_DATA));
  };

  // Show BEP type selector if no type is selected
  if (showBepTypeSelector || !bepType) {
    return (
      <EnhancedBepTypeSelector
        bepType={bepType}
        setBepType={setBepType}
        onProceed={handleBepTypeProceed}
      />
    );
  }

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
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-600">
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                  {bepType === 'pre-appointment' ? 'Pre-Appointment BEP' : 'Post-Appointment BEP'}
                </span>
              </div>
              <div className="flex items-center space-x-3">
                <span className="text-sm text-gray-600">
                  Welcome, {user.name}
                </span>
                <button
                  onClick={logout}
                  className="text-sm text-gray-500 hover:text-gray-700 underline"
                >
                  Logout
                </button>
              </div>
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
              onStepClick={onStepClick}
              validateStep={(index) => validatedSteps[index]}
            />
          </div>

          <div className="lg:col-span-3">
            <div className="bg-white rounded-lg shadow-sm p-8">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">{currentStep < CONFIG.steps.length ? CONFIG.steps[currentStep].title : 'Preview & Export'}</h2>
                  <p className="text-gray-600 mt-1">{currentStep < CONFIG.steps.length ? CONFIG.steps[currentStep].description : 'Preview and export the generated BEP'}</p>
                </div>
                {currentStep < CONFIG.steps.length && (
                  <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                    CONFIG.categories[CONFIG.steps[currentStep].category].bg
                  }`}>
                    {CONFIG.steps[currentStep].category} Aspects
                  </span>
                )}
              </div>

              {currentStep < CONFIG.steps.length ? (
                <FormStep
                  stepIndex={currentStep}
                  formData={formData}
                  updateFormData={updateFormData}
                  errors={errors}
                  bepType={bepType}
                />
              ) : (
                <PreviewExportPage 
                  generateBEPContent={generateBEPContent}
                  exportFormat={exportFormat}
                  setExportFormat={setExportFormat}
                  previewBEP={previewBEP}
                  downloadBEP={downloadBEP}
                  isExporting={isExporting}
                />
              )}

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
                  Step {currentStep + 1} of {CONFIG.steps.length + (currentStep >= CONFIG.steps.length ? 1 : 0)}
                </div>

                <div className="flex space-x-3">
                  {currentStep < CONFIG.steps.length - 1 ? (
                    <button
                      onClick={nextStep}
                      className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors"
                    >
                      <span>Next</span>
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  ) : currentStep === CONFIG.steps.length - 1 ? (
                    <button
                      onClick={goToPreview}
                      className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors"
                    >
                      <span>Preview & Export</span>
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  ) : null}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const App = () => {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
};

export default App;