import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { ChevronRight, ChevronLeft, Download, FileText, CheckCircle, Zap, Target, Eye, FileType, Printer } from 'lucide-react';
import jsPDF from 'jspdf';
import { Packer, Document, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, AlignmentType } from 'docx';
import DOMPurify from 'dompurify';

// Import separated components
import { AuthProvider, useAuth } from './contexts/AuthContext';
import Login from './components/Login';
import InputField from './components/forms/InputField';
import ProgressSidebar from './components/ui/ProgressSidebar';
import CONFIG from './config/bepConfig';


// Dati iniziali di esempio
const INITIAL_DATA = {
  // Common fields for both BEP types
  projectName: 'Greenfield Office Complex Phase 2',
  projectNumber: 'GF-2024-017',
  projectDescription: 'A modern 8-story office complex featuring sustainable design principles, flexible workspace layouts, and integrated smart building technologies. The building will accommodate 800+ employees across multiple tenants with shared amenities including conference facilities, cafeteria, and underground parking for 200 vehicles.',
  projectType: 'Commercial Building',
  appointingParty: 'ABC Development Corporation',

  // Pre-appointment specific fields
  proposedTimeline: '24 months (Jan 2025 - Dec 2026)',
  estimatedBudget: '£12.5 million',
  tenderApproach: 'Our approach emphasizes collaborative design coordination through advanced BIM workflows, early stakeholder engagement, and integrated sustainability analysis. We propose a phased delivery strategy with continuous value engineering and risk mitigation throughout all project stages.',
  proposedLead: 'Smith & Associates Architects Ltd.',
  proposedInfoManager: 'Sarah Johnson, BIM Manager (RICS Certified, ISO 19650 Lead)',

  // Executive Summary fields
  projectContext: 'This BEP outlines our comprehensive approach to delivering the Greenfield Office Complex using advanced BIM methodologies. Our strategy emphasizes collaborative design coordination, data-driven decision making, and seamless information handover to support long-term facility management. The project will serve as a flagship example of sustainable commercial development in the region.',
  bimStrategy: 'Our BIM strategy centers on early clash detection, integrated 4D/5D modeling for construction sequencing and cost control, and comprehensive digital twin creation for facilities management. We will utilize federated models across all disciplines with real-time collaboration through cloud-based platforms, ensuring design quality and construction efficiency while reducing project risks.',
  keyCommitments: 'We commit to full ISO 19650-2:2018 compliance throughout all project phases. Key deliverables include: coordinated federated models at each design milestone, comprehensive COBie data for asset handover, 4D construction sequences for all major building elements, and a complete digital twin with integrated IoT sensor data. All information will be delivered through our cloud-based CDE with full audit trails and version control.',
  keyContacts: [
    { 'Role': 'Project Director', 'Name': 'John Smith', 'Company': 'Smith & Associates Architects Ltd.', 'Contact Details': 'j.smith@smithassociates.com | +44 20 1234 5678' },
    { 'Role': 'BIM Manager', 'Name': 'Sarah Johnson', 'Company': 'Smith & Associates Architects Ltd.', 'Contact Details': 's.johnson@smithassociates.com | +44 20 1234 5679' },
    { 'Role': 'Client Representative', 'Name': 'David Brown', 'Company': 'ABC Development Corporation', 'Contact Details': 'd.brown@abcdev.com | +44 20 9876 5432' }
  ],
  valueProposition: 'Our BIM approach will deliver 15% reduction in construction costs through early clash detection, 25% faster design coordination, and comprehensive lifecycle cost analysis enabling informed material selections. The digital twin will provide 30% operational cost savings through predictive maintenance and space optimization, while the structured data handover ensures seamless facilities management integration.',

  proposedTeamLeaders: [
    { 'Discipline': 'Architecture', 'Name & Title': 'John Smith, Director', 'Company': 'Modern Design Associates', 'Experience': '12 years BIM experience, 50+ projects' },
    { 'Discipline': 'Structural', 'Name & Title': 'Emily Chen, Senior Engineer', 'Company': 'Engineering Excellence Ltd.', 'Experience': '10 years structural BIM, P.Eng' },
    { 'Discipline': 'MEP', 'Name & Title': 'Michael Rodriguez, BIM Coordinator', 'Company': 'Advanced Systems Group', 'Experience': '8 years MEP coordination experience' },
    { 'Discipline': 'Facades', 'Name & Title': 'David Wilson, Technical Director', 'Company': 'Curtain Wall Experts Ltd.', 'Experience': '15 years facade design, BIM certified' }
  ],
  teamCapabilities: 'Our multidisciplinary team brings 15+ years of BIM implementation experience across £500M+ of commercial projects. Key capabilities include: ISO 19650 certified information management, advanced parametric design using Revit/Grasshopper, integrated MEP coordination, 4D/5D modeling expertise, and digital twin development. Recent projects include the award-winning Tech Hub (£25M) and Riverside Commercial Center (£18M).',
  subcontractors: [
    { 'Role/Service': 'MEP Services', 'Company Name': 'Advanced Systems Group', 'Certification': 'ISO 19650 certified', 'Contact': 'info@advancedsystems.com' },
    { 'Role/Service': 'Curtain Wall', 'Company Name': 'Specialist Facades Ltd.', 'Certification': 'BIM Level 2 compliant', 'Contact': 'projects@specialistfacades.com' },
    { 'Role/Service': 'Landscaping', 'Company Name': 'Green Spaces Design', 'Certification': 'Autodesk certified', 'Contact': 'design@greenspaces.com' }
  ],
  proposedBimGoals: 'We propose to implement a collaborative BIM workflow that will improve design coordination by 60%, reduce construction conflicts by 90%, optimize project delivery timelines by 20%, and establish a comprehensive digital asset for facility management handover.',
  proposedObjectives: 'Our proposed objectives include achieving zero design conflicts at construction stage, reducing RFIs by 40%, improving construction efficiency by 25%, and delivering comprehensive FM data for operations.',
  intendedBimUses: ['Design Authoring', '3D Coordination', 'Clash Detection', 'Quantity Take-off', '4D Planning'],

  // Post-appointment specific fields
  confirmedTimeline: '24 months (Jan 2025 - Dec 2026)',
  confirmedBudget: '£12.5 million',
  deliveryApproach: 'Our delivery approach implements collaborative design coordination through advanced BIM workflows, stakeholder integration at key milestones, and continuous value engineering. We will execute a phased delivery strategy with integrated sustainability analysis and proactive risk management throughout all project stages to ensure on-time, on-budget completion.',
  referencedMaterial: 'This BEP references: Exchange Information Requirements (EIR) v2.1, Project Information Requirements (PIR) dated March 2024, ISO 19650-2:2018, BS 1192:2007+A2:2016, PAS 1192-2:2013, Client BIM Standards Manual v3.0, Health & Safety Information Requirements, and all applicable RIBA Plan of Work 2020 deliverables.',
  leadAppointedParty: 'Smith & Associates Architects Ltd.',
  informationManager: 'Sarah Johnson, BIM Manager (RICS Certified, ISO 19650 Lead)',
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
  informationManagementResponsibilities: 'Sarah Johnson, Information Manager, oversees all information production, validation, and exchange protocols in full compliance with ISO 19650-2:2018. Key responsibilities include establishing CDE governance structures, coordinating Task Information Delivery Plans (TIDPs) across all disciplines, ensuring model federation quality and consistency, implementing information security protocols including access controls and audit procedures, conducting weekly quality audits of information deliverables, facilitating cross-disciplinary coordination meetings, managing version control and approval workflows, monitoring compliance with established naming conventions and standards, coordinating client information exchanges and milestone reviews, and providing regular progress reports to project leadership on information delivery performance.',
  organizationalStructure: 'The delivery team operates under a Lead Appointed Party structure with Smith & Associates Architects as the primary coordinator reporting directly to ABC Development Corporation. The organizational hierarchy includes: Project Director (Michael Thompson - overall project governance), Information Manager (Sarah Johnson - ISO 19650 compliance and data coordination), Design Team Coordinator (James Wilson - discipline coordination), and four Task Team Leaders representing Architecture (Emma Davis), Structural Engineering (Robert Chen), MEP Engineering (Lisa Rodriguez), and Quantity Surveying (David Kumar). Supporting specialists include Sustainability Consultant (Green Building Associates), Facade Engineer (Advanced Envelope Solutions), and Cost Manager (Value Engineering Partners). All parties maintain direct contractual relationships with the client while operating through established collaboration agreements, shared CDE protocols, and unified project communication channels to ensure seamless information exchange and coordinated delivery.',
  taskTeamsBreakdown: [
    { 'Task Team': 'Architecture', 'Leader': 'Emma Davis (Smith & Associates)', 'Members': '6 architects, 2 BIM specialists, 1 visualization expert', 'Responsibilities': 'Design development, spatial coordination, building envelope design, interior layouts, accessibility compliance, planning submission drawings, and architectural specification preparation' },
    { 'Task Team': 'Structural Engineering', 'Leader': 'Robert Chen (Engineering Excellence)', 'Members': '4 structural engineers, 1 BIM coordinator, 1 analysis specialist', 'Responsibilities': 'Structural design and analysis, foundation design, steel/concrete detailing, connection design, loading calculations, construction sequence planning, and structural model coordination' },
    { 'Task Team': 'MEP Engineering', 'Leader': 'Lisa Rodriguez (Advanced Systems)', 'Members': '5 MEP engineers, 2 BIM modelers, 1 sustainability engineer', 'Responsibilities': 'HVAC system design, electrical distribution, plumbing design, fire protection systems, building automation integration, energy modeling, and MEP coordination with architecture' },
    { 'Task Team': 'Quantity Surveying', 'Leader': 'David Kumar (Cost Management Partners)', 'Members': '2 quantity surveyors, 1 cost planner, 1 data analyst', 'Responsibilities': 'Cost planning and control, quantity extraction from BIM models, value engineering analysis, tender documentation, contract administration, and lifecycle cost assessment' },
    { 'Task Team': 'Information Management', 'Leader': 'Sarah Johnson (BIM Manager)', 'Members': '2 information coordinators, 1 CDE administrator, 1 quality controller', 'Responsibilities': 'CDE management, model federation, quality assurance, standards compliance, information delivery coordination, and client liaison for information requirements' }
  ],
  confirmedBimGoals: 'The confirmed BIM goals include implementing collaborative workflows to achieve improved design coordination, reduced construction conflicts, optimized delivery timelines, and comprehensive digital asset creation for facility management.',
  implementationObjectives: 'Implementation objectives include zero design conflicts at construction, 40% reduction in RFIs, improved construction efficiency, and delivery of comprehensive FM data for operations.',
  finalBimUses: ['Design Authoring', '3D Coordination', 'Clash Detection', 'Quantity Take-off', '4D Planning'],

  // Legacy fields for backward compatibility
  bimUses: ['Design Authoring', '3D Coordination', 'Clash Detection', 'Quantity Take-off', '4D Planning'],
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
  workflowStates: [
    { 'State Name': 'Work in Progress (WIP)', 'Description': 'Active development by task teams', 'Access Level': 'Author only', 'Next State': 'Shared' },
    { 'State Name': 'Shared', 'Description': 'Available for coordination and review', 'Access Level': 'Team members', 'Next State': 'Published' },
    { 'State Name': 'Published', 'Description': 'Approved for use by the project team', 'Access Level': 'All stakeholders', 'Next State': 'Archived' },
    { 'State Name': 'Archived', 'Description': 'Historical versions for reference', 'Access Level': 'Read-only access', 'Next State': 'N/A' }
  ],
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
  updateProcesses: 'Quarterly BEP reviews, change request procedures, stakeholder approval for modifications, and continuous alignment with project requirements.',

  // Additional shared fields
  bimGoals: 'The BIM goals for this project are to enhance design coordination through clash detection reducing RFIs by 40%, improve construction sequencing through 4D modeling resulting in 20% schedule compression, enable accurate cost forecasting through 5D integration achieving ±2% budget variance, and deliver comprehensive digital asset information for lifecycle management supporting 25% reduction in operational costs over the first 5 years.',
  primaryObjectives: 'Primary objectives include: eliminating design conflicts before construction through rigorous clash detection protocols, optimising building performance through integrated analysis and simulation, enabling efficient construction through accurate quantity extraction and sequencing models, supporting sustainability targets through embedded carbon analysis and energy modeling, and facilitating seamless handover with structured asset data for predictive maintenance and space management.',
  cdeProvider: 'Autodesk Construction Cloud',
  cdePlatform: 'BIM 360 Design v2024.1',
  accessControl: 'Role-based access control with Project Administrator, Design Team, Review Team, and Client View permissions. Multi-factor authentication required for all users. Project folders restricted by discipline with read/write permissions assigned per project phase. Guest access limited to 30-day periods with approval workflows.',
  securityMeasures: 'End-to-end encryption for data in transit and at rest using AES-256 standards. SSL/TLS certificates for secure connections. Regular security audits and penetration testing. ISO 27001 certified cloud infrastructure. Automated malware scanning for all uploads. Data residency compliance with UK GDPR requirements.',
  backupProcedures: 'Automated daily backups with 30-day retention policy. Weekly full system backups with 12-month retention. Geo-redundant storage across multiple UK data centres. 99.9% uptime SLA with disaster recovery protocols. Regular backup integrity testing and documented restoration procedures. Monthly backup verification reports.',

  // Volume Strategy and Classification Systems
  volumeStrategy: 'Model breakdown strategy by building zones and disciplines: ARC (Architecture) models by floor levels (L00-L08), STR (Structural) models by structural zones (SZ01-SZ04), MEP (Mechanical/Electrical/Plumbing) models by service zones (MZ01-MZ03), Site models (SITE) for external works, and multi-zone models (ZZ) for coordination across boundaries. Each volume maintains consistent spatial relationships and coordinate systems for effective federation and clash detection.',
  classificationSystems: [
    { 'Classification System': 'Uniclass 2015', 'Application Area': 'Building Elements', 'Code Format': 'Ss_25_30_05', 'Responsibility': 'All Disciplines' },
    { 'Classification System': 'Uniclass 2015', 'Application Area': 'MEP Equipment', 'Code Format': 'Pr_35_31_26', 'Responsibility': 'MEP Engineers' },
    { 'Classification System': 'Uniclass 2015', 'Application Area': 'Architectural Elements', 'Code Format': 'Ac_45_10_12', 'Responsibility': 'Architects' },
    { 'Classification System': 'SfB/Uniclass', 'Application Area': 'Space Classification', 'Code Format': '(21) Office Areas', 'Responsibility': 'Space Planning Team' },
    { 'Classification System': 'COBie Classification', 'Application Area': 'Asset Data', 'Code Format': 'Type.Component.Asset', 'Responsibility': 'Information Manager' }
  ],
  classificationStandards: 'Implementation of Uniclass 2015 classification system for all building elements and spaces. Element codes follow format: Ss_25_30_05 for structural concrete elements, Pr_35_31_26 for MEP equipment, and Ac_45_10_12 for architectural finishes. Space classification using SfB/Uniclass codes for consistent asset data preparation and facilities management integration. All team members trained on classification requirements with quality checking procedures to ensure compliance.',

  // BIM Value Applications
  bimValueApplications: 'BIM will maximize project value through: 4D scheduling for time optimization reducing construction duration by 15%, energy modeling for sustainability compliance achieving BREEAM Excellent rating, life-cycle costing analysis enabling informed material selections with 20-year cost projections, design alternative evaluations through parametric modeling supporting value engineering decisions, pre-fabrication coordination reducing on-site assembly time by 30%, stakeholder visualization for enhanced buy-in and reduced change orders, and comprehensive digital asset creation supporting £2M+ operational cost savings over building lifecycle.',
  valueMetrics: [
    { 'Value Area': 'Schedule Optimization', 'Target Metric': '15% reduction in construction duration', 'Measurement Method': '4D model analysis vs baseline schedule', 'Baseline/Benchmark': '24-month traditional schedule' },
    { 'Value Area': 'Cost Reduction', 'Target Metric': '£500k savings through clash elimination', 'Measurement Method': 'Clash detection reports and change order tracking', 'Baseline/Benchmark': 'Industry average 3% RFI costs' },
    { 'Value Area': 'Sustainability Performance', 'Target Metric': 'BREEAM Excellent rating achievement', 'Measurement Method': 'Energy modeling validation', 'Baseline/Benchmark': 'Building Regulations Part L compliance' },
    { 'Value Area': 'Operational Efficiency', 'Target Metric': '25% reduction in FM costs', 'Measurement Method': 'Digital twin performance monitoring', 'Baseline/Benchmark': 'Industry benchmark £150/m²/year' },
    { 'Value Area': 'Quality Improvement', 'Target Metric': '40% reduction in RFIs', 'Measurement Method': 'Design coordination metrics', 'Baseline/Benchmark': 'Previous project average 120 RFIs' }
  ],
  strategicAlignment: 'BIM strategy directly supports client objectives including: 15% reduction in total project delivery time through optimized sequencing, achievement of net-zero carbon targets through integrated energy modeling, enhanced asset performance through digital twin implementation, improved tenant satisfaction via optimized space planning and MEP design, future-proofing for smart building integration, and comprehensive data foundation for predictive maintenance reducing operational costs by 25% annually.',

  // Appendices Data
  responsibilityMatrix: [
    { 'Role/Task': 'Model Authoring - Architecture', 'Responsible': 'Lead Architect', 'Accountable': 'Project Director', 'Consulted': 'Planning Consultant', 'Informed': 'Client, Construction Team' },
    { 'Role/Task': 'Model Authoring - Structure', 'Responsible': 'Structural Engineer', 'Accountable': 'Engineering Manager', 'Consulted': 'Architect, MEP Engineer', 'Informed': 'QS, Construction Team' },
    { 'Role/Task': 'Model Authoring - MEP', 'Responsible': 'MEP Engineer', 'Accountable': 'MEP Manager', 'Consulted': 'Architect, Structural', 'Informed': 'FM Team, Client' },
    { 'Role/Task': 'Clash Detection', 'Responsible': 'BIM Manager', 'Accountable': 'Information Manager', 'Consulted': 'All Disciplines', 'Informed': 'Project Team' },
    { 'Role/Task': 'Model Federation', 'Responsible': 'Information Manager', 'Accountable': 'Project Director', 'Consulted': 'BIM Manager', 'Informed': 'Stakeholders' }
  ],
  cobieRequirements: [
    { 'Component Type': 'Doors', 'Required Parameters': 'Fire Rating, U-Value, Warranty Period, Manufacturer, Model', 'Data Source': 'Architectural Model + Specification', 'Validation Method': 'Automated checking + Manual review' },
    { 'Component Type': 'Windows', 'Required Parameters': 'U-Value, Solar Heat Gain, Acoustic Rating, Warranty, Installation Date', 'Data Source': 'Architectural Model + Product Data', 'Validation Method': 'IFC export validation + COBie reports' },
    { 'Component Type': 'HVAC Equipment', 'Required Parameters': 'Capacity, Energy Rating, Maintenance Schedule, Serial Number, Commissioning Date', 'Data Source': 'MEP Model + Equipment Schedules', 'Validation Method': 'Equipment database validation' },
    { 'Component Type': 'Lighting Fixtures', 'Required Parameters': 'Wattage, Light Output, Control System, Replacement Cycle, Warranty', 'Data Source': 'MEP Model + Lighting Schedules', 'Validation Method': 'Automated parameter checking' },
    { 'Component Type': 'Structural Elements', 'Required Parameters': 'Material Grade, Load Capacity, Fire Rating, Installation Date, Inspection Schedule', 'Data Source': 'Structural Model + Material Data', 'Validation Method': 'Structural analysis integration' }
  ],
  fileNamingExamples: 'Comprehensive file naming examples:\n\nProject Models:\nGF-SAA-L02-ARC-001 (Greenfield-Smith Associates-Level 02-Architecture-Model 001)\nGF-JEL-SZ1-STR-002 (Greenfield-Jones Engineering-Structural Zone 1-Structure-Model 002)\nGF-TSS-MZ2-MEP-003 (Greenfield-TechServ Solutions-MEP Zone 2-Services-Model 003)\n\nDrawings:\nGF-SAA-ZZ-ARC-DR-A-1001 (General Arrangement Plans)\nGF-JEL-ZZ-STR-DR-S-2001 (Structural General Arrangement)\nGF-TSS-L03-MEP-DR-M-3001 (Level 3 Mechanical Plans)\n\nDocuments:\nGF-SAA-ZZ-ARC-SP-001 (Architectural Specification)\nGF-CMP-ZZ-QS-RP-001 (Cost Report)\nGF-ALL-ZZ-PM-MR-001 (Project Meeting Minutes)',
  exchangeWorkflow: [
    { 'Exchange Point': 'Design Development Review', 'Information Required': 'Coordinated discipline models, clash reports, design drawings', 'Format': 'IFC 4, BCF 2.1, PDF', 'Quality Checks': 'Model validation, clash detection, drawing coordination', 'Approval Process': 'Discipline lead review, IM approval, client sign-off' },
    { 'Exchange Point': 'Technical Design Milestone', 'Information Required': 'Construction-ready models, specifications, quantities', 'Format': 'IFC 4, native files, schedules', 'Quality Checks': 'Construction readiness check, quantity validation', 'Approval Process': 'Technical review, QS validation, project director approval' },
    { 'Exchange Point': 'Construction Handover', 'Information Required': 'As-built models, COBie data, O&M information', 'Format': 'IFC 4, COBie, PDF manuals', 'Quality Checks': 'As-built verification, data completeness check', 'Approval Process': 'Construction team verification, client acceptance' },
    { 'Exchange Point': 'Facilities Management Handover', 'Information Required': 'Digital twin, asset data, maintenance schedules', 'Format': 'COBie, digital twin platform, maintenance systems', 'Quality Checks': 'Data integration testing, system functionality', 'Approval Process': 'FM team acceptance, operational readiness confirmation' }
  ]
};

// Componenti riutilizzabili

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

              {/* Top Navigation Bar */}
              <div className="flex justify-between items-center mb-6 pb-4 border-b bg-gray-50 rounded-lg p-4">
                <button
                  onClick={prevStep}
                  disabled={currentStep === 0}
                  className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeft className="w-4 h-4" />
                  <span>Previous</span>
                </button>

                <div className="text-sm text-gray-600 font-medium">
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
                      <Eye className="w-4 h-4" />
                    </button>
                  ) : null}
                </div>
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