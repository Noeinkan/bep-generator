// Dati iniziali di esempio per il BEP Generator
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
  proposedLeadAndInfoManagers: {
    columns: ['Lead Appointed Party', 'Information Manager'],
    data: [
      { 'Lead Appointed Party': 'Smith & Associates Architects Ltd.', 'Information Manager': 'Sarah Johnson, BIM Manager (RICS Certified, ISO 19650 Lead)' },
      { 'Lead Appointed Party': 'Jones Engineering Consultants', 'Information Manager': 'Michael Chen, Information Coordinator (ISO 19650-2 Certified)' }
    ]
  },

  // Executive Summary fields
  projectContext: 'This BEP outlines our comprehensive approach to delivering the Greenfield Office Complex using advanced BIM methodologies. Our strategy emphasizes collaborative design coordination, data-driven decision making, and seamless information handover to support long-term facility management. The project will serve as a flagship example of sustainable commercial development in the region.',
  bimStrategy: 'Our BIM strategy centers on early clash detection, integrated 4D/5D modeling for construction sequencing and cost control, and comprehensive digital twin creation for facilities management. We will utilize federated models across all disciplines with real-time collaboration through cloud-based platforms, ensuring design quality and construction efficiency while reducing project risks.',
  keyCommitments: {
    intro: 'We commit to full ISO 19650-2:2018 compliance throughout all project phases. Key deliverables include:',
    table: [
      { 'Deliverable': 'Coordinated Federated Models', 'Description': 'Multi-discipline coordinated models at each design milestone with clash detection', 'Due Date': 'Each milestone' },
      { 'Deliverable': 'COBie Data Handover', 'Description': 'Comprehensive asset data for facility management integration', 'Due Date': 'Project completion' },
      { 'Deliverable': '4D Construction Sequences', 'Description': 'Time-phased construction models for all major building elements', 'Due Date': 'Pre-construction' },
      { 'Deliverable': 'Digital Twin with IoT', 'Description': 'Complete digital twin with integrated IoT sensor data', 'Due Date': 'Project handover' }
    ]
  },
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
  referencedMaterial: {
    intro: 'This BEP references and responds to the following documents and standards:',
    table: [
      { 'Document/Standard': 'Exchange Information Requirements (EIR)', 'Version/Date': 'v2.1 - March 2024', 'Relevance/Purpose': 'Defines client information requirements and delivery expectations' },
      { 'Document/Standard': 'Project Information Requirements (PIR)', 'Version/Date': 'March 2024', 'Relevance/Purpose': 'Specifies project-specific information delivery standards' },
      { 'Document/Standard': 'ISO 19650-2:2018', 'Version/Date': '2018', 'Relevance/Purpose': 'Information management framework for delivery phase' },
      { 'Document/Standard': 'BS 1192:2007+A2:2016', 'Version/Date': '2007+A2:2016', 'Relevance/Purpose': 'Collaborative production of architectural, engineering and construction information' },
      { 'Document/Standard': 'PAS 1192-2:2013', 'Version/Date': '2013', 'Relevance/Purpose': 'Specification for information management for the capital/delivery phase' },
      { 'Document/Standard': 'Client BIM Standards Manual', 'Version/Date': 'v3.0', 'Relevance/Purpose': 'Client-specific BIM standards and procedures' },
      { 'Document/Standard': 'Health & Safety Information Requirements', 'Version/Date': 'Current', 'Relevance/Purpose': 'CDM regulations and safety information requirements' },
      { 'Document/Standard': 'RIBA Plan of Work 2020', 'Version/Date': '2020', 'Relevance/Purpose': 'Project stage definitions and deliverable milestones' }
    ]
  },
  leadAppointedParty: 'Smith & Associates Architects Ltd.',
  informationManager: 'Sarah Johnson, BIM Manager (RICS Certified, ISO 19650 Lead)',
  leadAndInfoManagers: {
    columns: ['Lead Appointed Party', 'Information Manager'],
    data: [
      { 'Lead Appointed Party': 'Smith & Associates Architects Ltd.', 'Information Manager': 'Sarah Johnson, BIM Manager (RICS Certified, ISO 19650 Lead)' },
      { 'Lead Appointed Party': 'Jones Engineering Consultants', 'Information Manager': 'Michael Chen, Information Coordinator (ISO 19650-2 Certified)' }
    ]
  },
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
  loinIntroduction: `Introduction to Level of Information Need (LOIN) - Define the information requirements that specify what information is needed, at what level of detail, and for what purposes.

This section establishes the foundation for information delivery by linking to the Project Information Requirements (PIR) and Employer's Information Requirements (EIR) defined earlier in the BEP.`,
  informationPurposes: ['Design Development', 'Construction Planning', 'Quantity Surveying', 'Facility Management'],
  geometricalInfo: {
    intro: `Describe the purpose of geometrical requirements (e.g., "Support interdisciplinary coordination and visualization"). Specify the LOD standard reference (e.g., UNI EN 17412-1, NBS BIM Object Standard).`,
    table: [
      { 'Element': 'Structures (beams, columns)', 'Phase (RIBA)': 'Stage 2 (Concept)', 'LOD': 'LOD 100', 'Purpose': 'Preliminary visualization', 'Verification': 'Basic dimensional control', 'Delivery Format': 'IFC 4.0' },
      { 'Element': 'MEP systems (ducts)', 'Phase (RIBA)': 'Stage 4 (Technical Design)', 'LOD': 'LOD 300', 'Purpose': 'Clash detection', 'Verification': 'Solibri audit (tolerance ±5mm)', 'Delivery Format': 'IFC 4.0, Revit 2025' },
      { 'Element': 'Facades', 'Phase (RIBA)': 'Stage 3 (Developed Design)', 'LOD': 'LOD 200', 'Purpose': 'Energy analysis', 'Verification': 'Geometric compliance', 'Delivery Format': 'IFC 4.0' }
    ]
  },
  alphanumericalInfo: {
    intro: `General Description: Introduce the purpose of alphanumerical attributes (e.g., "Support facility management and regulatory compliance"). Cite standards like COBie or BS 1192-4.`,
    table: [
      { 'Element': 'HVAC Pump', 'Phase (RIBA)': 'Stage 5 (Construction)', 'Attributes (LOI)': 'Model, Power, InstallationDate, WarrantyDuration', 'Purpose': 'Facility Management', 'Verification': 'COBie validation in Excel', 'Delivery Format': 'COBie XLS' },
      { 'Element': 'Walls', 'Phase (RIBA)': 'Stage 3 (Developed Design)', 'Attributes (LOI)': 'Material, Thickness, UniclassCode', 'Purpose': 'Cost analysis', 'Verification': 'Attribute check in CDE', 'Delivery Format': 'BIM Database' },
      { 'Element': 'Windows', 'Phase (RIBA)': 'Stage 6 (Handover)', 'Attributes (LOI)': 'U-Value, MaintenanceSchedule', 'Purpose': 'Maintenance', 'Verification': 'Validation report', 'Delivery Format': 'COBie XLS' }
    ]
  },
  documentationInfo: {
    intro: `General Description: Introduce the purpose of documents (e.g., "Provide maintenance manuals or compliance reports"). Cite standards (e.g., PDF/A for long-term archiving).`,
    table: [
      { 'Document': 'O&M Manual', 'Phase (RIBA)': 'Stage 6 (Handover)', 'Detail': 'Maintenance instructions, technical data sheets', 'Purpose': 'Facility Management', 'Verification': 'Completeness check', 'Delivery Format': 'PDF/A, max 10MB' },
      { 'Document': 'Clash Detection Report', 'Phase (RIBA)': 'Stage 4 (Technical Design)', 'Detail': 'List of resolved clashes', 'Purpose': 'Coordination', 'Verification': 'Solibri validation', 'Delivery Format': 'PDF' },
      { 'Document': 'Material Specification Sheet', 'Phase (RIBA)': 'Stage 3 (Developed Design)', 'Detail': 'Material specifications', 'Purpose': 'Cost estimation', 'Verification': 'EIR compliance', 'Delivery Format': 'PDF' }
    ]
  },
  informationFormats: ['IFC 4', 'PDF', 'BCF 2.1', 'DWG', 'COBie'],
  projectInformationRequirements: `Define the Project Information Requirements (PIR) - the information needed to support asset management and operational objectives beyond project delivery.

Address:
• Asset management system integration
• Space management and occupancy data
• Energy monitoring and performance tracking
• Maintenance planning and scheduling
• Digital twin connectivity
• Building performance analytics
• Compliance and regulatory reporting`,
  loinTransition: `The LOIN requirements defined here will be satisfied through the delivery plans described in the Information Delivery Planning Sect., with MIDP and TIDPs that map each deliverable to specific LOD/LOIN.`,
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
  mobilisationPlan: 'Project mobilisation occurs over 4 weeks: Week 1 includes CDE setup, template development, and team onboarding; Week 2 involves standards training, tool deployment, and workflow testing; Week 3 encompasses pilot model creation, federation testing, and quality checking procedures; Week 4 includes final system validation, team competency verification, and formal project launch. All team members complete ISO 19650 certification and project-specific training before accessing the CDE and commencing information production activities.',
  teamCapabilitySummary: 'The delivery team provides comprehensive BIM capabilities across all disciplines: 15 certified BIM professionals with ISO 19650 competency, advanced modeling expertise in Revit, Tekla, and specialist analysis software, 5+ years experience delivering federated models for commercial projects £10M+, proven track record in clash detection reducing construction issues by 60%, established workflows for 4D/5D integration, and dedicated quality assurance resources ensuring deliverable compliance. Team capacity supports peak deployment of 35 specialists during technical design phases.',
  taskTeamExchange: 'Information exchange protocols establish: weekly model federation with automated clash detection reports, fortnightly design coordination meetings with federated model reviews, monthly design freeze periods for cross-disciplinary validation, standardised BCF workflows for issue resolution, real-time model access through shared CDE workspace, automated notification systems for model updates and issue assignments, and formal sign-off procedures for milestone deliverables ensuring all disciplines approve federated models before progression to next design stage.',
  modelReferencing3d: 'Model referencing procedures ensure consistent spatial coordination: shared coordinate system established from Ordnance Survey grid references, standardised origin points and level datums across all disciplines, automated reference model linking through shared CDE folders, version control protocols preventing out-of-date reference usage, clash detection workflows identifying reference model conflicts, weekly reference model updates with team notifications, and quality gates preventing model publication without current reference verification ensuring geometric consistency throughout the federated environment.',
  milestoneInformation: [
    { 'Milestone': 'Stage 2 - Concept Design', 'Information Required': 'Basic geometry models, spatial arrangements, outline specifications', 'Format': 'IFC 4, PDF drawings, Excel schedules', 'Quality Level': 'LOD 200, outline accuracy ±50mm' },
    { 'Milestone': 'Stage 3 - Spatial Coordination', 'Information Required': 'Coordinated discipline models, clash detection reports, design drawings', 'Format': 'IFC 4, BCF 2.1, PDF, DWG', 'Quality Level': 'LOD 300, coordination accuracy ±25mm' },
    { 'Milestone': 'Stage 4 - Technical Design', 'Information Required': 'Construction-ready models, detailed specifications, quantity schedules', 'Format': 'IFC 4, native files, PDF, Excel', 'Quality Level': 'LOD 350, construction accuracy ±10mm' },
    { 'Milestone': 'Stage 5 - Manufacturing', 'Information Required': 'Fabrication models, assembly sequences, installation guides', 'Format': 'IFC 4, manufacturer formats, 4D models', 'Quality Level': 'LOD 400, fabrication accuracy ±5mm' },
    { 'Milestone': 'Stage 6 - Handover', 'Information Required': 'As-built models, COBie data, O&M manuals, warranties', 'Format': 'IFC 4, COBie, PDF, digital documents', 'Quality Level': 'LOD 500, as-built verification complete' }
  ],
  informationRiskRegister: [
    { 'Risk Description': 'Model coordination conflicts due to discipline isolation', 'Impact': 'High - Construction delays and rework', 'Probability': 'Medium', 'Mitigation': 'Weekly federated model reviews, automated clash detection, early coordination protocols' },
    { 'Risk Description': 'Data loss or corruption in CDE platform', 'Impact': 'High - Project delays and data recreation costs', 'Probability': 'Low', 'Mitigation': 'Daily automated backups, geo-redundant storage, version control, disaster recovery procedures' },
    { 'Risk Description': 'Software interoperability failures between disciplines', 'Impact': 'Medium - Information exchange delays', 'Probability': 'Medium', 'Mitigation': 'Standardized IFC workflows, software compatibility testing, alternative exchange formats' },
    { 'Risk Description': 'Inconsistent information delivery by task teams', 'Impact': 'Medium - Project coordination issues', 'Probability': 'Medium', 'Mitigation': 'Clear TIDP requirements, regular progress monitoring, milestone quality gates' },
    { 'Risk Description': 'Team member turnover affecting BIM competency', 'Impact': 'Medium - Knowledge loss and training delays', 'Probability': 'Low', 'Mitigation': 'Cross-training programs, documented procedures, knowledge management systems' },
    { 'Risk Description': 'Client changes affecting information requirements', 'Impact': 'Medium - Scope creep and delivery delays', 'Probability': 'High', 'Mitigation': 'Change control procedures, impact assessments, flexible workflow systems' }
  ],
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
  informationBreakdownStrategy: 'Information breakdown follows ISO 19650 principles with clear segregation by discipline, building zone, and information type. Architecture models are subdivided by floor levels (L00-L08), structural models by structural zones (SZ01-SZ04), and MEP models by service zones (MZ01-MZ03). Each volume maintains consistent coordinate systems and spatial relationships. Cross-zone coordination models ensure seamless integration whilst enabling parallel development and efficient collaboration workflows across all disciplines.',
  federationProcess: 'Model federation occurs weekly through automated processes in Navisworks Manage, with daily clash detection runs for active design areas. Each discipline publishes IFC models to designated CDE folders following standardized naming conventions. The federation workflow includes automated quality checking, spatial coordination verification, and clash detection reporting. Federated models undergo review and approval before release to the wider project team, ensuring consistency and reliability of coordination information.',
  softwareHardwareInfrastructure: [
    { 'Category': 'BIM Authoring', 'Item/Component': 'Autodesk Revit Architecture 2024', 'Specification': 'Licensed seats: 12, Cloud entitlement', 'Purpose': 'Architectural design and documentation' },
    { 'Category': 'BIM Authoring', 'Item/Component': 'Autodesk Revit Structure 2024', 'Specification': 'Licensed seats: 8, Cloud entitlement', 'Purpose': 'Structural design and analysis integration' },
    { 'Category': 'BIM Authoring', 'Item/Component': 'Autodesk Revit MEP 2024', 'Specification': 'Licensed seats: 10, Cloud entitlement', 'Purpose': 'MEP systems design and coordination' },
    { 'Category': 'Model Coordination', 'Item/Component': 'Navisworks Manage 2024', 'Specification': 'Licensed seats: 6, Freedom entitlement', 'Purpose': 'Model federation and clash detection' },
    { 'Category': 'Quality Assurance', 'Item/Component': 'Solibri Model Checker v9.12', 'Specification': 'Licensed seats: 4, Annual subscription', 'Purpose': 'Model validation and rule checking' },
    { 'Category': 'Cloud Platform', 'Item/Component': 'Autodesk Construction Cloud', 'Specification': 'Premium plan, 500GB storage', 'Purpose': 'Collaborative design and data sharing' },
    { 'Category': 'Hardware Workstations', 'Item/Component': 'Dell Precision 5000 Series', 'Specification': 'Intel i7-13700K, 32GB RAM, RTX 4070, 1TB NVMe SSD', 'Purpose': 'BIM modeling and coordination workstations' },
    { 'Category': 'Network Infrastructure', 'Item/Component': 'Fiber Internet Connection', 'Specification': '1Gbps symmetric, 99.9% uptime SLA', 'Purpose': 'High-speed cloud collaboration and file sync' },
    { 'Category': 'Data Storage', 'Item/Component': 'Network Attached Storage (NAS)', 'Specification': 'Synology DS1821+, 64TB capacity, RAID 6', 'Purpose': 'Local backup and file server functionality' },
    { 'Category': 'Security', 'Item/Component': 'Multi-Factor Authentication', 'Specification': 'Microsoft Authenticator, SMS backup', 'Purpose': 'Enhanced security for all cloud platforms' }
  ],
  documentControlInfo: 'This BIM Execution Plan is prepared in accordance with ISO 19650-2:2018 standards for information management during the delivery phase of construction projects. Document classification: CONFIDENTIAL - Project delivery team access only. Version control follows established procedures with formal review and approval workflows. Document generated using Professional BEP Generator Tool v1.0, ensuring compliance with current industry standards and best practices. Regular updates and revisions will be managed through the project CDE with full audit trails maintained for all changes.',
  modelingStandards: [
    { 'Standard/Guideline': 'UK BIM Alliance Standards', 'Version': 'v2.1', 'Application Area': 'General BIM practices', 'Compliance Level': 'Mandatory' },
    { 'Standard/Guideline': 'Uniclass 2015', 'Version': '2015', 'Application Area': 'Classification system', 'Compliance Level': 'Mandatory' },
    { 'Standard/Guideline': 'AIA LOD Specification', 'Version': '2019', 'Application Area': 'Level of development', 'Compliance Level': 'Mandatory' },
    { 'Standard/Guideline': 'Company Modeling Guide', 'Version': 'v3.2', 'Application Area': 'Internal procedures', 'Compliance Level': 'Required' }
  ],
  namingConventions: [
    { 'Element Type': 'Project Models', 'Naming Format': 'PP-OO-VV-DD-###', 'Example': 'GF-SAA-L02-ARC-001', 'Description': 'PP=Project, OO=Originator, VV=Volume/Level, DD=Discipline, ###=Sequential' },
    { 'Element Type': 'Drawings', 'Naming Format': 'PP-OO-VV-DD-DR-T-####', 'Example': 'GF-SAA-ZZ-ARC-DR-A-1001', 'Description': 'Additional DR=Drawing, T=Type (A/S/M), ####=Drawing number' },
    { 'Element Type': 'Documents', 'Naming Format': 'PP-OO-VV-DD-TT-###', 'Example': 'GF-SAA-ZZ-ARC-SP-001', 'Description': 'TT=Document type (SP=Specification, RP=Report, etc.)' },
    { 'Element Type': 'MEP Equipment', 'Naming Format': 'PP-OO-VV-MEP-EQ-###', 'Example': 'GF-TSS-L03-MEP-EQ-001', 'Description': 'EQ=Equipment designation with sequential numbering' }
  ],
  fileStructure: 'Organized by discipline and project phase with clear folder hierarchies, version control through file naming, and linked file management protocols.',
  fileStructureDiagram: '📁 WIP (Work in Progress)\n📁 SHARED (Coordination)\n📁 PUBLISHED (Approved)',
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
  dataClassification: [
    { 'Classification Level': 'Public', 'Description': 'Information that can be freely shared with external parties', 'Examples': 'Marketing materials, project brochures, general site photos', 'Access Controls': 'No access restrictions, publicly available' },
    { 'Classification Level': 'Internal', 'Description': 'Information for internal project team use only', 'Examples': 'Design development work, meeting minutes, progress reports', 'Access Controls': 'Project team members only, authenticated access required' },
    { 'Classification Level': 'Confidential', 'Description': 'Sensitive business information requiring protection', 'Examples': 'Commercial pricing, tender information, contractual details', 'Access Controls': 'Senior team members only, need-to-know basis, encryption required' },
    { 'Classification Level': 'Restricted', 'Description': 'Highly sensitive information with security implications', 'Examples': 'Security-sensitive building systems, access control details, critical infrastructure plans', 'Access Controls': 'Authorized personnel only, enhanced security measures, audit trails mandatory' },
    { 'Classification Level': 'Commercial-in-Confidence', 'Description': 'Commercially sensitive information affecting business operations', 'Examples': 'Cost breakdowns, supplier agreements, procurement strategies', 'Access Controls': 'Commercial team only, encrypted storage, controlled distribution' },
    { 'Classification Level': 'Technical-Confidential', 'Description': 'Technical information requiring specialized protection', 'Examples': 'Detailed BIM models, structural calculations, MEP system designs', 'Access Controls': 'Discipline experts only, version control, watermarked documents' }
  ],
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
  collaborativeProductionGoals: 'Collaborative production goals focus on establishing unified data standards across all disciplines, implementing real-time model coordination through federated workflows, ensuring consistent information delivery at all project milestones, maintaining version control integrity throughout design development, facilitating transparent communication through shared visualisation platforms, and creating comprehensive audit trails for decision-making accountability whilst adhering to ISO 19650 information management principles.',
  alignmentStrategy: 'Our alignment strategy implements weekly coordination meetings with federated model reviews, establishes clear responsibility matrices for information production and validation, deploys standardised naming conventions and file structures across all disciplines, utilises automated quality checking workflows to ensure compliance, maintains continuous training programmes for team competency development, and implements performance monitoring through defined KPIs including model accuracy, delivery timeliness, and information completeness metrics.',
  cdeStrategy: 'The project employs a federated CDE approach utilizing multiple specialized platforms to optimize workflow efficiency and data management across different information types and project phases. Each platform is selected for its specific strengths while maintaining seamless integration and unified information governance through standardized protocols, consistent naming conventions, and automated synchronization processes.',
  cdePlatforms: [
    { 'Platform/Service': 'Microsoft SharePoint Online', 'Usage/Purpose': 'WIP collaboration and document management', 'Information Types': 'MS Office documents, specifications, reports, meeting minutes, correspondence', 'Workflow States': 'WIP, Review, Approved', 'Access Control': 'Team member authentication, version control, document check-out/in' },
    { 'Platform/Service': 'Autodesk Construction Cloud', 'Usage/Purpose': 'BIM model sharing and design coordination', 'Information Types': 'Native BIM models (RVT, DWG), IFC files, federated models, clash reports', 'Workflow States': 'WIP, Shared, Published', 'Access Control': 'Role-based access, model viewing permissions, download restrictions' },
    { 'Platform/Service': 'Bentley ProjectWise', 'Usage/Purpose': 'Engineering deliverables and technical drawings', 'Information Types': 'CAD drawings, engineering calculations, technical specifications, design reports', 'Workflow States': 'Draft, Review, Approved, Issued', 'Access Control': 'Discipline-based folders, approval workflows, audit trails' },
    { 'Platform/Service': 'Esri ArcGIS Online', 'Usage/Purpose': 'Geospatial data management and GIS analysis', 'Information Types': 'Survey data, site analysis, environmental data, location intelligence', 'Workflow States': 'Draft, Published, Archived', 'Access Control': 'GIS team access, public viewing portals, data export controls' },
    { 'Platform/Service': 'Aconex (Oracle)', 'Usage/Purpose': 'Client communication and formal submissions', 'Information Types': 'Official correspondence, RFIs, submittals, progress reports, certificates', 'Workflow States': 'Draft, Submitted, Under Review, Approved', 'Access Control': 'Client portal access, formal approval workflows, notification systems' }
  ],
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
    { 'Classification System': 'COBie', 'Application Area': 'Asset Data', 'Code Format': 'Type.Component.Asset', 'Responsibility': 'Information Manager' }
  ],
  classificationStandards: [
    { 'Element Category': 'Structural Elements', 'Classification System': 'Uniclass 2015', 'Code Format': 'Ss_25_30_05', 'Example Code': 'Ss_25_30_05', 'Description': 'Structural concrete elements' },
    { 'Element Category': 'MEP Equipment', 'Classification System': 'Uniclass 2015', 'Code Format': 'Pr_35_31_26', 'Example Code': 'Pr_35_31_26', 'Description': 'MEP equipment systems' },
    { 'Element Category': 'Architectural Elements', 'Classification System': 'Uniclass 2015', 'Code Format': 'Ac_45_10_12', 'Example Code': 'Ac_45_10_12', 'Description': 'Architectural finishes' },
    { 'Element Category': 'Space Classification', 'Classification System': 'SfB/Uniclass', 'Code Format': '(21) Office Areas', 'Example Code': '(21) Office Areas', 'Description': 'Space classification codes' },
    { 'Element Category': 'Asset Data', 'Classification System': 'COBie', 'Code Format': 'Type.Component.Asset', 'Example Code': 'Type.Component.Asset', 'Description': 'Asset management data' },
    { 'Element Category': 'Building Services', 'Classification System': 'Uniclass 2015', 'Code Format': 'Ss_25_40_20', 'Example Code': 'Ss_25_40_20', 'Description': 'HVAC distribution systems' },
    { 'Element Category': 'Construction Products', 'Classification System': 'Uniclass 2015', 'Code Format': 'Pr_20_93_45', 'Example Code': 'Pr_20_93_45', 'Description': 'Insulation materials and products' },
    { 'Element Category': 'Work Results', 'Classification System': 'Uniclass 2015', 'Code Format': 'Zz_25_10_35', 'Example Code': 'Zz_25_10_35', 'Description': 'Concrete work execution standards' }
  ],


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
    { 'Task/Activity': 'Model Authoring - Architecture', 'Responsible Party': 'Lead Architect (Emma Davis)', 'Accountable Party': 'Project Director (Michael Thompson)', 'Support/Input': 'Planning Consultant, Interior Designer' },
    { 'Task/Activity': 'Model Authoring - Structural', 'Responsible Party': 'Structural Engineer (Robert Chen)', 'Accountable Party': 'Engineering Manager', 'Support/Input': 'Architect, MEP Engineer, Geotechnical' },
    { 'Task/Activity': 'Model Authoring - MEP', 'Responsible Party': 'MEP Engineer (Lisa Rodriguez)', 'Accountable Party': 'MEP Manager', 'Support/Input': 'Architect, Structural, Commissioning' },
    { 'Task/Activity': 'Clash Detection & Resolution', 'Responsible Party': 'BIM Manager (Sarah Johnson)', 'Accountable Party': 'Information Manager', 'Support/Input': 'All Discipline Leaders' },
    { 'Task/Activity': 'Model Federation', 'Responsible Party': 'Information Manager (Sarah Johnson)', 'Accountable Party': 'Project Director', 'Support/Input': 'BIM Coordinator, Quality Controller' },
    { 'Task/Activity': 'Quality Assurance', 'Responsible Party': 'Quality Controller', 'Accountable Party': 'Information Manager', 'Support/Input': 'Discipline Leads, Standards Compliance' },
    { 'Task/Activity': 'Information Exchange', 'Responsible Party': 'CDE Administrator', 'Accountable Party': 'Information Manager', 'Support/Input': 'IT Support, Security Team' },
    { 'Task/Activity': 'Client Reporting', 'Responsible Party': 'Project Manager', 'Accountable Party': 'Project Director', 'Support/Input': 'Information Manager, QS' }
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

export default INITIAL_DATA;