import { Building, CheckCircle, Users, Target, Database, Calendar, Monitor, Settings, FileText, Shield, BookOpen, AlertCircle } from 'lucide-react';

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
    { number: 1, title: 'BEP Type & Project Info', icon: Building, description: 'Define BEP type and basic project information', category: 'Commercial' },
    { number: 2, title: 'Executive Summary', icon: FileText, description: 'High-level overview and key commitments', category: 'Commercial' },
    { number: 3, title: 'Stakeholders & Roles', icon: Users, description: 'Define project stakeholders and responsibilities', category: 'Commercial' },
    { number: 4, title: 'BIM Goals & Uses', icon: Target, description: 'Define BIM objectives and applications', category: 'Commercial' },
    { number: 5, title: 'Level of Information Need', icon: Database, description: 'Specify LOIN requirements and content', category: 'Management' },
    { number: 6, title: 'Information Delivery Planning', icon: Calendar, description: 'MIDP, TIDPs and delivery schedules', category: 'Management' },
    { number: 7, title: 'Common Data Environment', icon: Monitor, description: 'CDE specification and workflows', category: 'Technical' },
    { number: 8, title: 'Technology Requirements', icon: Settings, description: 'Software, hardware and technical specs', category: 'Technical' },
    { number: 9, title: 'Information Production', icon: FileText, description: 'Methods, standards and procedures', category: 'Management' },
    { number: 10, title: 'Quality Assurance', icon: CheckCircle, description: 'QA framework and validation processes', category: 'Management' },
    { number: 11, title: 'Security & Privacy', icon: Shield, description: 'Information security and privacy measures', category: 'Management' },
    { number: 12, title: 'Training & Competency', icon: BookOpen, description: 'Training requirements and competency levels', category: 'Management' },
    { number: 13, title: 'Coordination & Risk', icon: AlertCircle, description: 'Collaboration procedures and risk management', category: 'Management' },
    { number: 14, title: 'Appendices', icon: FileText, description: 'Supporting materials and templates', category: 'Management' }
  ],

  formFields: {
    'pre-appointment': {
      0: {
        number: '1',
        title: 'Project Information and Proposed Approach',
        fields: [
          { number: '1.1', name: 'projectName', label: 'Project Name', required: true, type: 'text', placeholder: 'Greenfield Office Complex Phase 2' },
          { number: '1.1', name: 'projectNumber', label: 'Project Number', type: 'text', placeholder: 'GF-2024-017' },
          { number: '1.1', name: 'projectType', label: 'Project Type', required: true, type: 'select', options: 'projectTypes' },
          { number: '1.1', name: 'appointingParty', label: 'Appointing Party', required: true, type: 'text', placeholder: 'ABC Development Corporation' },
          { number: '1.1', name: 'proposedTimeline', label: 'Proposed Project Timeline', type: 'timeline', placeholder: '24 months (Jan 2025 - Dec 2026)' },
          { number: '1.1', name: 'estimatedBudget', label: 'Estimated Project Budget', type: 'budget', placeholder: '£12.5 million' },
          { number: '1.1', name: 'projectDescription', label: 'Project Description', type: 'textarea', rows: 4, placeholder: 'A modern 8-story office complex featuring sustainable design principles, flexible workspace layouts, and integrated smart building technologies. The building will accommodate 800+ employees across multiple tenants with shared amenities including conference facilities, cafeteria, and underground parking for 200 vehicles.' },
          { number: '1.1', name: 'tenderApproach', label: 'Our Proposed Approach', type: 'textarea', rows: 3, placeholder: 'Our approach emphasizes collaborative design coordination through advanced BIM workflows, early stakeholder engagement, and integrated sustainability analysis. We propose a phased delivery strategy with continuous value engineering and risk mitigation throughout all project stages.' }
        ]
      },
      1: {
        number: '2',
        title: 'Executive Summary',
        fields: [
          { number: '2.1', name: 'projectContext', label: 'Project Context and Overview', required: true, type: 'textarea', rows: 4, placeholder: 'This BEP outlines our comprehensive approach to delivering the Greenfield Office Complex using advanced BIM methodologies. Our strategy emphasizes collaborative design coordination, data-driven decision making, and seamless information handover to support long-term facility management. The project will serve as a flagship example of sustainable commercial development in the region.' },
          { number: '2.1', name: 'bimStrategy', label: 'BIM Strategy Summary', required: true, type: 'textarea', rows: 3, placeholder: 'Our BIM strategy centers on early clash detection, integrated 4D/5D modeling for construction sequencing and cost control, and comprehensive digital twin creation for facilities management. We will utilize federated models across all disciplines with real-time collaboration through cloud-based platforms, ensuring design quality and construction efficiency while reducing project risks.' },
          { number: '2.1', name: 'keyCommitments', label: 'Key Commitments and Deliverables', required: true, type: 'introTable', introPlaceholder: 'We commit to full ISO 19650-2:2018 compliance throughout all project phases. Key deliverables include:', tableColumns: ['Deliverable', 'Description', 'Due Date'] },
          { number: '2.1', name: 'keyContacts', label: 'Key Project Contacts', type: 'table', columns: ['Role', 'Name', 'Company', 'Email', 'Phone Number'] },
          { number: '2.1', name: 'valueProposition', label: 'Value Proposition', type: 'textarea', rows: 3, placeholder: 'Our BIM approach will deliver 15% reduction in construction costs through early clash detection, 25% faster design coordination, and comprehensive lifecycle cost analysis enabling informed material selections. The digital twin will provide 30% operational cost savings through predictive maintenance and space optimization, while the structured data handover ensures seamless facilities management integration.' }
        ]
      },
      2: {
        number: '3',
        title: 'Proposed Team and Capabilities',
        fields: [
          { number: '3.1', name: 'proposedLead', label: 'Proposed Lead Appointed Party', required: true, type: 'text', placeholder: 'Smith & Associates Architects Ltd.' },
          { number: '3.1', name: 'proposedInfoManager', label: 'Proposed Information Manager', required: true, type: 'text', placeholder: 'Sarah Johnson, BIM Manager (RICS Certified, ISO 19650 Lead)' },
          { number: '3.1', name: 'proposedTeamLeaders', label: 'Proposed Task Team Leaders', type: 'table', columns: ['Discipline', 'Name & Title', 'Company', 'Experience'] },
          { 
            number: '3.1', 
            name: 'proposedResourceAllocation', 
            label: 'Proposed Resource Allocation - Capability and Capacity', 
            type: 'table', 
            columns: ['Role', 'Proposed Personnel', 'Key Competencies/Experience', 'Anticipated Weekly Allocation (Hours)', 'Software/Hardware Requirements', 'Notes']
          },
          { number: '3.1', name: 'teamCapabilities', label: 'Team Capabilities and Experience', type: 'textarea', rows: 4, placeholder: 'Our multidisciplinary team brings 15+ years of BIM implementation experience across £500M+ of commercial projects. Key capabilities include: ISO 19650 certified information management, advanced parametric design using Revit/Grasshopper, integrated MEP coordination, 4D/5D modeling expertise, and digital twin development. Recent projects include the award-winning Tech Hub (£25M) and Riverside Commercial Center (£18M).' },
          { 
            number: '3.1', 
            name: 'proposedMobilizationPlan', 
            label: 'Proposed Mobilization Plan', 
            type: 'textarea', 
            rows: 3, 
            placeholder: 'Upon appointment, our mobilization plan includes: Week 1 - Team onboarding and ISO 19650 training; Week 2 - IT infrastructure setup and software licensing (Revit 2024, Navisworks); Week 3 - Capability verification through pilot models and CDE testing. Risk mitigation strategies include access to specialist consultants and contingency resource pools to address potential skill gaps or capacity constraints aligned with ISO 19650-2 clauses 5.3.3-5.3.5.' 
          },
          { number: '3.1', name: 'subcontractors', label: 'Proposed Subcontractors/Partners', type: 'table', columns: ['Role/Service', 'Company Name', 'Certification', 'Contact'] }
        ]
      }
    },
    'post-appointment': {
      0: {
        number: '1',
        title: 'Project Information and Confirmed Objectives',
        fields: [
          { number: '1.1', name: 'projectName', label: 'Project Name', required: true, type: 'text', placeholder: 'Greenfield Office Complex Phase 2' },
          { number: '', name: 'projectNumber', label: 'Project Number', type: 'text', placeholder: 'GF-2024-017' },
          { number: '', name: 'projectType', label: 'Project Type', required: true, type: 'select', options: 'projectTypes' },
          { number: '', name: 'appointingParty', label: 'Appointing Party', required: true, type: 'text', placeholder: 'ABC Development Corporation' },
          { number: '', name: 'confirmedTimeline', label: 'Confirmed Project Timeline', type: 'timeline', placeholder: '24 months (Jan 2025 - Dec 2026)' },
          { number: '', name: 'confirmedBudget', label: 'Confirmed Project Budget', type: 'budget', placeholder: '£12.5 million' },
          { number: '1.2', name: 'projectDescription', label: 'Project Description', type: 'textarea', rows: 4, placeholder: 'A modern 8-storey office complex featuring sustainable design principles, flexible workspace layouts, and integrated smart building technologies. The building will accommodate 800+ employees across multiple tenants with shared amenities including conference facilities, cafeteria, and underground parking for 200 vehicles.' },
          { number: '1.3', name: 'deliveryApproach', label: 'Confirmed Delivery Approach', type: 'textarea', rows: 3, placeholder: 'Our delivery approach implements collaborative design coordination through advanced BIM workflows, stakeholder integration at key milestones, and continuous value engineering. We will execute a phased delivery strategy with integrated sustainability analysis and proactive risk management throughout all project stages to ensure on-time, on-budget completion.' }
        ]
      },
      1: {
        number: '2',
        title: 'Executive Summary',
        fields: [
          { number: '2.1', name: 'projectContext', label: 'Project Context and Overview', required: true, type: 'textarea', rows: 4, placeholder: 'This BEP confirms our comprehensive approach to delivering the Greenfield Office Complex using advanced BIM methodologies. Our strategy emphasises collaborative design coordination, data-driven decision making, and seamless information handover to support long-term facility management. The project will serve as a flagship example of sustainable commercial development in the region, incorporating smart building technologies and flexible workspace design.' },
          { number: '2.2', name: 'bimStrategy', label: 'BIM Strategy Summary', required: true, type: 'textarea', rows: 3, placeholder: 'Our confirmed BIM strategy centres on early clash detection, integrated 4D/5D modelling for construction sequencing and cost control, and comprehensive digital twin creation for facilities management. We will utilise federated models across all disciplines with real-time collaboration through cloud-based platforms, ensuring design quality and construction efficiency whilst reducing project risks and enabling predictive maintenance capabilities.' },
          { number: '2.3', name: 'keyCommitments', label: 'Key Commitments and Deliverables', required: true, type: 'introTable', introPlaceholder: 'We are committed to full ISO 19650-2:2018 compliance throughout all project phases. Key deliverables include:', tableColumns: ['Deliverable', 'Description', 'Due Date'] },
          { number: '2.4', name: 'keyContacts', label: 'Key Project Contacts', type: 'table', columns: ['Role', 'Name', 'Company', 'Email', 'Phone Number'] },
          { number: '2.5', name: 'valueProposition', label: 'Value Proposition', type: 'textarea', rows: 3, placeholder: 'Our BIM approach will deliver 15% reduction in construction costs through early clash detection, 25% faster design coordination, and comprehensive lifecycle cost analysis enabling informed material selections. The digital twin will provide 30% operational cost savings through predictive maintenance and space optimisation, whilst the structured data handover ensures seamless facilities management integration and supports the client\'s sustainability targets through enhanced building performance monitoring.' }
        ]
      },
      2: {
        number: '3',
        title: 'Confirmed Team and Responsibilities',
        fields: [
          { number: '3.1', name: 'organizationalStructure', label: 'Delivery Team\'s Organisational Structure and Composition', type: 'orgchart' },
          { number: '3.2', name: 'leadAppointedPartiesTable', label: 'Lead Appointed Parties and Information Managers', type: 'orgstructure-data-table', readOnly: true },
          { number: '3.3', name: 'taskTeamsBreakdown', label: 'Task Teams', type: 'table', columns: ['Task Team', 'Discipline', 'Leader', 'Leader Contact', 'Company'] },
          { 
            number: '3.4', 
            name: 'resourceAllocationTable', 
            label: 'Resource Allocation - Capability and Capacity Assessment', 
            type: 'table', 
            columns: ['Role', 'Assigned Personnel', 'Key Competencies/Experience', 'Weekly Allocation (Hours)', 'Software/Hardware Requirements', 'Notes']
          },
          { 
            number: '3.5', 
            name: 'mobilizationPlan', 
            label: 'Mobilization Plan and Risk Mitigation', 
            type: 'textarea', 
            rows: 6, 
            placeholder: 'PHASED MOBILIZATION TIMELINE\n\nWeek 1 - Onboarding and Training:\n  - Team orientation and project kickoff meeting\n  - ISO 19650-2:2018 training (2-day workshop)\n  - Information security briefings and CDE access provisioning\n\nWeek 2 - IT Infrastructure Setup:\n  - Workstation configuration (Revit 2024, Navisworks, AutoCAD)\n  - Software licensing and cloud storage allocation\n  - CDE platform configuration with permissions\n\nWeek 3 - Capability Verification:\n  - Pilot model production and federation testing\n  - IFC export testing for interoperability\n  - CDE submission procedures and quality checks\n\nRISK MITIGATION STRATEGY\n\nResource capacity risks documented in project risk register per ISO 19650-2 clause 5.3.6. Contingency plans include specialist consultants, backup connectivity, and escalation protocols via MIDP notifications.' 
          },
          { number: '3.6', name: 'informationManagementResponsibilities', label: 'Information Management Responsibilities', type: 'textarea', rows: 3, placeholder: 'The Information Manager oversees all aspects of information production, validation, and exchange in accordance with ISO 19650-2:2018. Responsibilities include: establishing and maintaining the CDE, coordinating task information delivery plans (TIDPs), ensuring model federation quality, managing information security protocols, conducting regular audits of information deliverables, and facilitating cross-disciplinary information exchanges. The IM reports directly to the project director and client representative.' }
        ]
      }
    }
  },

  sharedFormFields: {
    3: {
      number: '4',
      title: 'BIM Goals and Objectives',
      fields: [
        { number: '4.1', name: 'bimGoals', label: 'BIM Goals', required: true, type: 'textarea', rows: 4, placeholder: 'The BIM goals for this project are to enhance design coordination through clash detection reducing RFIs by 40%, improve construction sequencing through 4D modeling resulting in 20% schedule compression, enable accurate cost forecasting through 5D integration achieving ±2% budget variance, and deliver comprehensive digital asset information for lifecycle management supporting 25% reduction in operational costs over the first 5 years.' },
        { number: '4.2', name: 'primaryObjectives', label: 'Primary Objectives', type: 'textarea', rows: 3, placeholder: 'Primary objectives include: eliminating design conflicts before construction through rigorous clash detection protocols, optimising building performance through integrated analysis and simulation, enabling efficient construction through accurate quantity extraction and sequencing models, supporting sustainability targets through embedded carbon analysis and energy modeling, and facilitating seamless handover with structured asset data for predictive maintenance and space management.' },
        { number: '4.3', name: 'bimUses', label: 'BIM Uses', required: true, type: 'checkbox', options: 'bimUses' },
        { number: '4.4', name: 'bimValueApplications', label: 'BIM Applications for Project Value', required: true, type: 'textarea', rows: 4, placeholder: 'BIM will maximize project value through: 4D scheduling for time optimization reducing construction duration by 15%, energy modeling for sustainability compliance achieving BREEAM Excellent rating, life-cycle costing analysis enabling informed material selections with 20-year cost projections, design alternative evaluations through parametric modeling supporting value engineering decisions, pre-fabrication coordination reducing on-site assembly time by 30%, stakeholder visualization for enhanced buy-in and reduced change orders, and comprehensive digital asset creation supporting £2M+ operational cost savings over building lifecycle.' },
        { number: '4.5', name: 'valueMetrics', label: 'Success Metrics and Value Measurement', required: true, type: 'table', columns: ['Value Area', 'Target Metric', 'Measurement Method', 'Baseline/Benchmark'] },
        { number: '4.6', name: 'strategicAlignment', label: 'Alignment with Client Strategic Objectives', type: 'textarea', rows: 3, placeholder: 'BIM strategy directly supports client objectives including: 15% reduction in total project delivery time through optimized sequencing, achievement of net-zero carbon targets through integrated energy modeling, enhanced asset performance through digital twin implementation, improved tenant satisfaction via optimized space planning and MEP design, future-proofing for smart building integration, and comprehensive data foundation for predictive maintenance reducing operational costs by 25% annually.' },
        { number: '4.7', name: 'collaborativeProductionGoals', label: 'Objectives/Goals for the Collaborative Production of Information', type: 'textarea', rows: 4, placeholder: 'Collaborative production goals focus on establishing unified data standards across all disciplines, implementing real-time model coordination through federated workflows, ensuring consistent information delivery at all project milestones, maintaining version control integrity throughout design development, facilitating transparent communication through shared visualisation platforms, and creating comprehensive audit trails for decision-making accountability whilst adhering to ISO 19650 information management principles.' },
        { number: '4.8', name: 'alignmentStrategy', label: 'Approach to Facilitating Information Management Goals', type: 'im-strategy-builder' }
      ]
    },
    4: {
      number: '5',
      title: 'Level of Information Need (LOIN)',
      fields: [
        { number: '5.1', name: 'informationPurposes', label: 'Information Purposes', required: true, type: 'checkbox', options: 'informationPurposes' },
        { number: '5.2', name: 'geometricalInfo', label: 'Geometrical Information Requirements', type: 'textarea', rows: 3, placeholder: 'Geometrical information requirements include: LOD 300 for all structural elements during design development, LOD 400 for MEP systems and connections, LOD 350 for architectural elements including doors, windows, and fixtures, accurate site survey data with ±5mm tolerance, as-built survey verification for existing structures, detailed connection details for all structural joints, and precise spatial coordination with surveyed existing conditions to ensure clash-free installation.' },
        { number: '5.3', name: 'alphanumericalInfo', label: 'Alphanumerical Information Requirements', type: 'textarea', rows: 3, placeholder: 'Alphanumerical information requirements encompass: complete material specifications with thermal and fire ratings, manufacturer part numbers and product data sheets, cost data linked to quantity schedules, maintenance schedules and warranty information, energy performance data for all building systems, space programming and occupancy requirements, structural load calculations and safety factors, MEP capacity and performance specifications, and comprehensive asset data preparation for COBie handover including serial numbers and commissioning dates.' },
        { number: '5.4', name: 'documentationInfo', label: 'Documentation Requirements', type: 'textarea', rows: 3, placeholder: 'Documentation requirements include: technical specification documents for all building systems, operation and maintenance manuals, health and safety file documentation, commissioning reports and test certificates, warranties and guarantees documentation, training materials for building operators, emergency procedures and evacuation plans, sustainability certifications and energy assessments, planning compliance certificates, and comprehensive as-built drawings with redline markups incorporated and verified against site conditions.' },
        { number: '5.5', name: 'informationFormats', label: 'Information Formats', type: 'checkbox', options: 'fileFormats' },
        { number: '5.6', name: 'projectInformationRequirements', label: 'Project Information Requirements (PIR)', type: 'textarea', rows: 4, placeholder: 'Project Information Requirements specify deliverable information to support asset management objectives: integrated 3D models with embedded property data for space management systems, energy consumption monitoring through IoT sensor integration, preventive maintenance scheduling with equipment lifecycle data, tenant fit-out guidelines with services capacity information, building performance analytics for continuous optimisation, digital twin connectivity for predictive maintenance, compliance monitoring systems for regulatory reporting, and structured data formats supporting client\'s existing CAFM systems and sustainability reporting requirements.' }
      ]
    },
    5: {
      number: '6',
      title: 'Information Delivery Planning',
      fields: [
        // 6.1 - Key Milestones (Milestone chiave con deliverable e date)
        { number: '6.1', name: 'keyMilestones', label: 'Key Information Delivery Milestones', required: true, type: 'milestones-table', columns: ['Stage/Phase', 'Milestone Description', 'Deliverables', 'Due Date'] },

        // 6.2 - Delivery Schedule (Schedule dettagliato)
        { number: '6.2', name: 'deliverySchedule', label: 'Delivery Schedule', type: 'textarea', rows: 3, placeholder: 'Information delivery follows a phased approach: Months 1-3 (project mobilisation and concept design models), Months 4-8 (spatial coordination and developed design models), Months 9-14 (technical design and construction documentation), Months 15-20 (construction phase information and progress models), Months 21-24 (commissioning data, as-built verification, and handover documentation). Weekly model federation occurs throughout with formal milestone reviews at stage gates requiring client approval before progression.' },

        // 6.3 - TIDP Requirements (Requisiti TIDP con lista esistenti)
        { number: '6.3', name: 'tidpRequirements', label: 'Task Information Delivery Plans (TIDPs)', type: 'tidp-reference', placeholder: 'TIDPs define discipline-specific delivery requirements: Architecture TIDP delivers spatial models and specification schedules biweekly, Structural TIDP provides analysis models and connection details monthly, MEP TIDP delivers coordinated services models and equipment schedules fortnightly, Quantity Surveying TIDP extracts cost data and material quantities monthly, and Sustainability TIDP delivers performance analysis and compliance reports at stage gates. Each TIDP includes quality checking procedures, approval workflows, and integration requirements with the federated model.' },

        // 6.4 - TIDP Description (Campo testo aggiuntivo)
        { number: '6.4', name: 'tidpDescription', label: 'TIDP Description and Notes', type: 'textarea', rows: 3, placeholder: 'Additional notes about TIDPs, coordination requirements, or specific delivery protocols...' },

        // 6.5 - MIDP Description (Spostato qui da 6.1)
        { number: '6.5', name: 'midpDescription', label: 'Master Information Delivery Plan (MIDP) Description', required: true, type: 'textarea', rows: 4, placeholder: 'The MIDP establishes a structured schedule for information delivery aligned with RIBA Plan of Work 2020 stages. Key deliverables include: Stage 3 coordinated federated models by Month 8, Stage 4 construction-ready models with full MEP coordination by Month 14, Stage 5 as-built verification models by Month 22, and handover documentation including COBie data, O&M manuals, and digital twin integration by Month 24. Each delivery milestone includes quality gates with defined acceptance criteria and client approval processes.' },

        // 6.6 - Information Deliverables Matrix (era 6.5.1)
        { number: '6.6', name: 'informationDeliverablesMatrix', label: 'Information Deliverables Responsibility Matrix', type: 'deliverables-matrix', matrixType: 'deliverables', placeholder: 'Deliverables schedule with responsibilities, due dates, and formats. Auto-syncs from TIDPs.' },

        // 6.7 - Information Management Activities Matrix (era 6.5)
        { number: '6.7', name: 'informationManagementMatrix', label: 'Information Management Activities Responsibility Matrix (ISO 19650-2 Annex A)', type: 'im-activities-matrix', matrixType: 'im-activities', placeholder: 'RACI matrix for information management activities per ISO 19650-2. Click to open the Responsibility Matrix Manager.' },

        // 6.8 - Mobilisation Plan (Piano mobilizzazione)
        { number: '6.8', name: 'mobilisationPlan', label: 'Mobilisation Plan', type: 'textarea', rows: 3, placeholder: 'Project mobilisation occurs over 4 weeks: Week 1 includes CDE setup, template development, and team onboarding; Week 2 involves standards training, tool deployment, and workflow testing; Week 3 encompasses pilot model creation, federation testing, and quality checking procedures; Week 4 includes final system validation, team competency verification, and formal project launch. All team members complete ISO 19650 certification and project-specific training before accessing the CDE and commencing information production activities.' },

        // 6.9 - Team Capability Summary (Capacità team)
        { number: '6.9', name: 'teamCapabilitySummary', label: 'Delivery Team Capability & Capacity Summary', type: 'textarea', rows: 3, placeholder: 'The delivery team provides comprehensive BIM capabilities across all disciplines: 15 certified BIM professionals with ISO 19650 competency, advanced modeling expertise in Revit, Tekla, and specialist analysis software, 5+ years experience delivering federated models for commercial projects £10M+, proven track record in clash detection reducing construction issues by 60%, established workflows for 4D/5D integration, and dedicated quality assurance resources ensuring deliverable compliance. Team capacity supports peak deployment of 35 specialists during technical design phases.' },

        // 6.10 - Information Risk Register (Registro rischi)
        { number: '6.10', name: 'informationRiskRegister', label: 'Delivery Team\'s Information Risk Register', type: 'table', columns: ['Risk Description', 'Impact', 'Probability', 'Mitigation'] },

        // 6.11 - Task Team Exchange (Scambio informazioni)
        { number: '6.11', name: 'taskTeamExchange', label: 'Exchange of Information Between Task Teams', type: 'textarea', rows: 3, placeholder: 'Information exchange protocols establish: weekly model federation with automated clash detection reports, fortnightly design coordination meetings with federated model reviews, monthly design freeze periods for cross-disciplinary validation, standardised BCF workflows for issue resolution, real-time model access through shared CDE workspace, automated notification systems for model updates and issue assignments, and formal sign-off procedures for milestone deliverables ensuring all disciplines approve federated models before progression to next design stage.' },

        // 6.12 - Model Referencing 3D (Referenziazione modelli)
        { number: '6.12', name: 'modelReferencing3d', label: 'Referencing of 3D Information Models', type: 'textarea', rows: 3, placeholder: 'Model referencing procedures ensure consistent spatial coordination: shared coordinate system established from Ordnance Survey grid references, standardised origin points and level datums across all disciplines, automated reference model linking through shared CDE folders, version control protocols preventing out-of-date reference usage, clash detection workflows identifying reference model conflicts, weekly reference model updates with team notifications, and quality gates preventing model publication without current reference verification ensuring geometric consistency throughout the federated environment.' }
      ]
    },
    6: {
      number: '7',
      title: 'Common Data Environment (CDE)',
      fields: [
        { number: '7.1', name: 'cdeStrategy', label: 'Multi-Platform CDE Strategy', type: 'cdeDiagram', placeholder: 'The project employs a federated CDE approach utilizing multiple specialized platforms to optimize workflow efficiency and data management across different information types and project phases. Each platform is selected for its specific strengths while maintaining seamless integration and unified information governance.' },
        { number: '7.2', name: 'cdePlatforms', label: 'CDE Platform Matrix', required: true, type: 'table', columns: ['Platform/Service', 'Usage/Purpose', 'Information Types', 'Workflow States', 'Access Control'] },
        { number: '7.3', name: 'workflowStates', label: 'Unified Workflow States', required: true, type: 'table', columns: ['State Name', 'Description', 'Access Level', 'Next State'] },
        { number: '7.4', name: 'accessControl', label: 'Integrated Access Control', type: 'textarea', rows: 3, placeholder: 'Unified role-based access control across all CDE platforms with Single Sign-On (SSO) integration. Project Administrator, Design Team, Review Team, and Client View permissions maintained consistently. Multi-factor authentication required for all platforms. Cross-platform folder synchronization with discipline-specific read/write permissions. Guest access limited to 30-day periods with approval workflows across all systems.' },
        { number: '7.5', name: 'securityMeasures', label: 'Multi-Platform Security Framework', type: 'textarea', rows: 3, placeholder: 'End-to-end encryption for data in transit and at rest using AES-256 standards across all platforms. SSL/TLS certificates for secure connections. Regular security audits and penetration testing for each platform. ISO 27001 certified cloud infrastructure. Automated malware scanning for all uploads. Data residency compliance with UK GDPR requirements. Cross-platform security monitoring and incident response procedures.' },
        { number: '7.6', name: 'backupProcedures', label: 'Comprehensive Backup Strategy', type: 'textarea', rows: 3, placeholder: 'Automated daily backups with 30-day retention policy across all CDE platforms. Weekly full system backups with 12-month retention. Geo-redundant storage across multiple UK data centres. 99.9% uptime SLA with disaster recovery protocols. Regular backup integrity testing and documented restoration procedures. Cross-platform data synchronization and recovery protocols. Monthly backup verification reports for all systems.' }
      ]
    },
    7: {
      number: '8',
      title: 'Technology and Software Requirements',
      fields: [
        { number: '8.1', name: 'bimSoftware', label: 'BIM Software Applications', required: true, type: 'checkbox', options: 'software' },
        { number: '8.2', name: 'fileFormats', label: 'File Formats', required: true, type: 'checkbox', options: 'fileFormats' },
        { number: '8.3', name: 'hardwareRequirements', label: 'Hardware Requirements', type: 'textarea', rows: 3 },
        { number: '8.4', name: 'networkRequirements', label: 'Network Requirements', type: 'textarea', rows: 3 },
        { number: '8.5', name: 'interoperabilityNeeds', label: 'Interoperability Requirements', type: 'textarea', rows: 3 },
        { number: '8.6', name: 'federationStrategy', label: 'Federation Strategy', type: 'textarea', rows: 3, placeholder: 'Strategy details...' },
        { number: '8.7', name: 'informationBreakdownStrategy', label: 'Information Breakdown Strategy', type: 'textarea', rows: 3, placeholder: 'Information breakdown strategy...' },
        { number: '8.8', name: 'federationProcess', label: 'Federation Process', type: 'textarea', rows: 3, placeholder: 'Federation process details...' },
        { number: '8.9', name: 'softwareHardwareInfrastructure', label: 'Software, Hardware and IT Infrastructure', type: 'table', columns: ['Category', 'Item/Component', 'Specification', 'Purpose'] },
        { number: '8.10', name: 'documentControlInfo', label: 'Document Control Information', type: 'textarea', rows: 4, placeholder: 'Document type, ISO standards, status, generator details...' }
      ]
    },
    8: {
      number: '9',
      title: 'Information Production Methods and Procedures',
      fields: [
        { number: '9.1', name: 'modelingStandards', label: 'Modeling Standards', required: true, type: 'table', columns: ['Standard/Guideline', 'Version', 'Application Area', 'Compliance Level'] },
        { number: '9.2', name: 'namingConventions', label: 'Naming Conventions', required: true, type: 'table', columns: ['Element Type', 'Naming Format', 'Example', 'Description'] },
        { number: '9.3', name: 'fileStructure', label: 'File Structure', type: 'textarea', rows: 3 },
        { number: '9.4', name: 'fileStructureDiagram', label: 'File Structure Diagram', type: 'fileStructure' },
        { number: '9.5', name: 'volumeStrategy', label: 'Volume Strategy and Model Breakdown', required: true, type: 'mindmap' },
        { number: '9.6', name: 'classificationSystems', label: 'Classification Systems and Coding', required: true, type: 'table', columns: ['Classification System', 'Application Area', 'Code Format', 'Responsibility'] },
        { number: '9.7', name: 'classificationStandards', label: 'Classification Standards Implementation', type: 'table', columns: ['Element Category', 'Classification System', 'Code Format', 'Example Code', 'Description'] },
        { number: '9.8', name: 'dataExchangeProtocols', label: 'Data Exchange Protocols', type: 'table', columns: ['Exchange Type', 'Format', 'Frequency', 'Delivery Method'] }
      ]
    },
    9: {
      number: '10',
      title: 'Quality Assurance and Control',
      fields: [
        { number: '10.1', name: 'qaFramework', label: 'Quality Assurance Framework', required: true, type: 'table', columns: ['QA Activity', 'Responsibility', 'Frequency', 'Tools/Methods'] },
        { number: '10.2', name: 'modelValidation', label: 'Model Validation Procedures', required: true, type: 'textarea', rows: 4 },
        { number: '10.3', name: 'reviewProcesses', label: 'Review Processes', type: 'textarea', rows: 3 },
        { number: '10.4', name: 'approvalWorkflows', label: 'Approval Workflows', type: 'textarea', rows: 3 },
        { number: '10.5', name: 'complianceVerification', label: 'Compliance Verification', type: 'textarea', rows: 3 },
        { number: '10.6', name: 'modelReviewAuthorisation', label: 'Information Model Review and Authorisation', type: 'textarea', rows: 3, placeholder: 'Review and authorisation procedures...' }
      ]
    },
    10: {
      number: '11',
      title: 'Information Security and Privacy',
      fields: [
        { number: '11.1', name: 'dataClassification', label: 'Data Classification', required: true, type: 'table', columns: ['Classification Level', 'Description', 'Examples', 'Access Controls'] },
        { number: '11.2', name: 'accessPermissions', label: 'Access Permissions', required: true, type: 'textarea', rows: 3 },
        { number: '11.3', name: 'encryptionRequirements', label: 'Encryption Requirements', type: 'textarea', rows: 3 },
        { number: '11.4', name: 'dataTransferProtocols', label: 'Data Transfer Protocols', type: 'textarea', rows: 3 },
        { number: '11.5', name: 'privacyConsiderations', label: 'Privacy Considerations', type: 'textarea', rows: 3 }
      ]
    },
    11: {
      number: '12',
      title: 'Training and Competency',
      fields: [
        { number: '12.1', name: 'bimCompetencyLevels', label: 'BIM Competency Levels', required: true, type: 'textarea', rows: 4 },
        { number: '12.2', name: 'trainingRequirements', label: 'Training Requirements', type: 'textarea', rows: 3 },
        { number: '12.3', name: 'certificationNeeds', label: 'Certification Requirements', type: 'textarea', rows: 3 },
        { number: '12.4', name: 'projectSpecificTraining', label: 'Project-Specific Training', type: 'textarea', rows: 3 }
      ]
    },
    12: {
      number: '13',
      title: 'Coordination, Collaboration & Risk Management',
      fields: [
        { number: '13.1', name: 'coordinationMeetings', label: 'Coordination Meetings', required: true, type: 'textarea', rows: 3 },
        { number: '13.2', name: 'clashDetectionWorkflow', label: 'Clash Detection Workflow', type: 'textarea', rows: 3 },
        { number: '13.3', name: 'issueResolution', label: 'Issue Resolution Process', type: 'textarea', rows: 3 },
        { number: '13.4', name: 'communicationProtocols', label: 'Communication Protocols', type: 'textarea', rows: 3 },
        { number: '13.5', name: 'federationStrategy', label: 'Model Federation Strategy', type: 'textarea', rows: 3 },
        { number: '13.6', name: 'informationRisks', label: 'Information-Related Risks', required: true, type: 'textarea', rows: 4 },
        { number: '13.7', name: 'technologyRisks', label: 'Technology-Related Risks', type: 'textarea', rows: 3 },
        { number: '13.8', name: 'riskMitigation', label: 'Risk Mitigation Strategies', type: 'textarea', rows: 3 },
        { number: '13.9', name: 'contingencyPlans', label: 'Contingency Plans', type: 'textarea', rows: 3 },
        { number: '13.10', name: 'performanceMetrics', label: 'Performance Metrics and KPIs', type: 'textarea', rows: 3 },
        { number: '13.11', name: 'monitoringProcedures', label: 'Monitoring Procedures', type: 'textarea', rows: 3 },
        { number: '13.12', name: 'auditTrails', label: 'Audit Trails', type: 'textarea', rows: 3 },
        { number: '13.13', name: 'updateProcesses', label: 'Update Processes', type: 'textarea', rows: 3 },
        { number: '13.14', name: 'projectKpis', label: 'Project Key Performance Indicators (KPIs)', type: 'table', columns: ['KPI Name', 'Target Value', 'Measurement Method', 'Responsibility'] }
      ]
    },
    13: {
      number: '14',
      title: 'Appendices',
      fields: [
        { name: 'responsibilityMatrix', label: 'Appendix A: Responsibility Matrix Template', required: true, type: 'table', columns: ['Role/Task', 'Responsible', 'Accountable', 'Consulted', 'Informed'] },
        { name: 'cobieRequirements', label: 'Appendix B: COBie Data Requirements', required: true, type: 'table', columns: ['Component Type', 'Required Parameters', 'Data Source', 'Validation Method'] },
        { name: 'fileNamingExamples', label: 'Appendix C: File Naming Convention Examples', required: true, type: 'textarea', rows: 6, placeholder: 'Comprehensive file naming examples:\n\nProject Models:\nGF-SAA-L02-ARC-001 (Greenfield-Smith Associates-Level 02-Architecture-Model 001)\nGF-JEL-SZ1-STR-002 (Greenfield-Jones Engineering-Structural Zone 1-Structure-Model 002)\nGF-TSS-MZ2-MEP-003 (Greenfield-TechServ Solutions-MEP Zone 2-Services-Model 003)\n\nDrawings:\nGF-SAA-ZZ-ARC-DR-A-1001 (General Arrangement Plans)\nGF-JEL-ZZ-STR-DR-S-2001 (Structural General Arrangement)\nGF-TSS-L03-MEP-DR-M-3001 (Level 3 Mechanical Plans)\n\nDocuments:\nGF-SAA-ZZ-ARC-SP-001 (Architectural Specification)\nGF-CMP-ZZ-QS-RP-001 (Cost Report)\nGF-ALL-ZZ-PM-MR-001 (Project Meeting Minutes)' },
        { name: 'exchangeWorkflow', label: 'Appendix D: Information Exchange Workflow Template', required: true, type: 'table', columns: ['Exchange Point', 'Information Required', 'Format', 'Quality Checks', 'Approval Process'] },
        { name: 'modelCheckingCriteria', label: 'Appendix E: Model Quality Checking Criteria', type: 'table', columns: ['Check Type', 'Acceptance Criteria', 'Tools Used', 'Frequency'] },
        { name: 'softwareVersionMatrix', label: 'Appendix F: Software Version Compatibility Matrix', type: 'table', columns: ['Software', 'Version', 'File Formats Supported', 'Interoperability Notes'] },
        { name: 'deliverableTemplates', label: 'Appendix G: Information Deliverable Templates', type: 'textarea', rows: 4, placeholder: 'Standard templates and schedules for key deliverables:\n\n- Task Information Delivery Plan (TIDP) Template\n- Model Element Checklist Template\n- Quality Assurance Report Template\n- Clash Detection Report Template\n- Progress Report Template\n- Information Exchange Record Template\n- Asset Data Handover Template\n\nAll templates available in project CDE Templates folder with version control and approval workflows.' },
        { name: 'referencedDocuments', label: 'Appendix H: Referenced Documents and Standards', type: 'standardsTable' }
      ]
    }
  },

  // Function to get appropriate form fields based on BEP type and step
  getFormFields: (bepType, stepIndex) => {
    // For steps 0-2, use BEP type specific fields
    if (stepIndex <= 2 && CONFIG.formFields[bepType] && CONFIG.formFields[bepType][stepIndex]) {
      return CONFIG.formFields[bepType][stepIndex];
    }
    // For steps 3-13, use shared fields
    if (stepIndex >= 3 && CONFIG.sharedFormFields[stepIndex]) {
      return CONFIG.sharedFormFields[stepIndex];
    }
    return null;
  }
};

export default CONFIG;