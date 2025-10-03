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
    { title: 'BEP Type & Project Info', icon: Building, description: 'Define BEP type and basic project information', category: 'Commercial' },
    { title: 'Executive Summary', icon: FileText, description: 'Project overview and key commitments', category: 'Commercial' },
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
    { title: 'Coordination & Risk', icon: AlertCircle, description: 'Collaboration procedures and risk management', category: 'Management' },
    { title: 'Appendices', icon: FileText, description: 'Reference materials and supporting documentation', category: 'Management' }
  ],

  formFields: {
    'pre-appointment': {
      0: {
        title: 'Project Information and Proposed Approach',
        fields: [
          { name: 'projectName', label: 'Project Name', required: true, type: 'text', placeholder: 'Greenfield Office Complex Phase 2' },
          { name: 'projectNumber', label: 'Project Number', type: 'text', placeholder: 'GF-2024-017' },
          { name: 'projectType', label: 'Project Type', required: true, type: 'select', options: 'projectTypes' },
          { name: 'appointingParty', label: 'Appointing Party', required: true, type: 'text', placeholder: 'ABC Development Corporation' },
          { name: 'proposedTimeline', label: 'Proposed Project Timeline', type: 'timeline', placeholder: '24 months (Jan 2025 - Dec 2026)' },
          { name: 'estimatedBudget', label: 'Estimated Project Budget', type: 'budget', placeholder: '£12.5 million' },
          { name: 'projectDescription', label: 'Project Description', type: 'textarea', rows: 4, placeholder: 'A modern 8-story office complex featuring sustainable design principles, flexible workspace layouts, and integrated smart building technologies. The building will accommodate 800+ employees across multiple tenants with shared amenities including conference facilities, cafeteria, and underground parking for 200 vehicles.' },
          { name: 'tenderApproach', label: 'Our Proposed Approach', type: 'textarea', rows: 3, placeholder: 'Our approach emphasizes collaborative design coordination through advanced BIM workflows, early stakeholder engagement, and integrated sustainability analysis. We propose a phased delivery strategy with continuous value engineering and risk mitigation throughout all project stages.' }
        ]
      },
      1: {
        title: 'Executive Summary',
        fields: [
          { name: 'projectContext', label: 'Project Context and Overview', required: true, type: 'textarea', rows: 4, placeholder: 'This BEP outlines our comprehensive approach to delivering the Greenfield Office Complex using advanced BIM methodologies. Our strategy emphasizes collaborative design coordination, data-driven decision making, and seamless information handover to support long-term facility management. The project will serve as a flagship example of sustainable commercial development in the region.' },
          { name: 'bimStrategy', label: 'BIM Strategy Summary', required: true, type: 'textarea', rows: 3, placeholder: 'Our BIM strategy centers on early clash detection, integrated 4D/5D modeling for construction sequencing and cost control, and comprehensive digital twin creation for facilities management. We will utilize federated models across all disciplines with real-time collaboration through cloud-based platforms, ensuring design quality and construction efficiency while reducing project risks.' },
          { name: 'keyCommitments', label: 'Key Commitments and Deliverables', required: true, type: 'introTable', introPlaceholder: 'We commit to full ISO 19650-2:2018 compliance throughout all project phases. Key deliverables include:', tableColumns: ['Deliverable', 'Description', 'Due Date'] },
          { name: 'keyContacts', label: 'Key Project Contacts', type: 'table', columns: ['Role', 'Name', 'Company', 'Email', 'Phone Number'] },
          { name: 'valueProposition', label: 'Value Proposition', type: 'textarea', rows: 3, placeholder: 'Our BIM approach will deliver 15% reduction in construction costs through early clash detection, 25% faster design coordination, and comprehensive lifecycle cost analysis enabling informed material selections. The digital twin will provide 30% operational cost savings through predictive maintenance and space optimization, while the structured data handover ensures seamless facilities management integration.' }
        ]
      },
      2: {
        title: 'Proposed Team and Capabilities',
        fields: [
          { name: 'proposedLeadAndInfoManagers', label: 'Lead Appointed Party', required: true, type: 'table', columns: ['Lead Appointed Party', 'Information Manager'] },
          { name: 'proposedTeamLeaders', label: 'Proposed Task Team Leaders', type: 'table', columns: ['Discipline', 'Name & Title', 'Company', 'Experience'] },
          { name: 'teamCapabilities', label: 'Team Capabilities and Experience', type: 'textarea', rows: 4, placeholder: 'Our multidisciplinary team brings 15+ years of BIM implementation experience across £500M+ of commercial projects. Key capabilities include: ISO 19650 certified information management, advanced parametric design using Revit/Grasshopper, integrated MEP coordination, 4D/5D modeling expertise, and digital twin development. Recent projects include the award-winning Tech Hub (£25M) and Riverside Commercial Center (£18M).' },
          { name: 'subcontractors', label: 'Proposed Subcontractors/Partners', type: 'table', columns: ['Role/Service', 'Company Name', 'Certification', 'Contact'] }
        ]
      }
    },
    'post-appointment': {
      0: {
        title: 'Project Information and Confirmed Objectives',
        fields: [
          { name: 'projectName', label: 'Project Name', required: true, type: 'text', placeholder: 'Greenfield Office Complex Phase 2' },
          { name: 'projectNumber', label: 'Project Number', type: 'text', placeholder: 'GF-2024-017' },
          { name: 'projectType', label: 'Project Type', required: true, type: 'select', options: 'projectTypes' },
          { name: 'appointingParty', label: 'Appointing Party', required: true, type: 'text', placeholder: 'ABC Development Corporation' },
          { name: 'confirmedTimeline', label: 'Confirmed Project Timeline', type: 'timeline', placeholder: '24 months (Jan 2025 - Dec 2026)' },
          { name: 'confirmedBudget', label: 'Confirmed Project Budget', type: 'budget', placeholder: '£12.5 million' },
          { name: 'projectDescription', label: 'Project Description', type: 'textarea', rows: 4, placeholder: 'A modern 8-storey office complex featuring sustainable design principles, flexible workspace layouts, and integrated smart building technologies. The building will accommodate 800+ employees across multiple tenants with shared amenities including conference facilities, cafeteria, and underground parking for 200 vehicles.' },
          { name: 'deliveryApproach', label: 'Confirmed Delivery Approach', type: 'textarea', rows: 3, placeholder: 'Our delivery approach implements collaborative design coordination through advanced BIM workflows, stakeholder integration at key milestones, and continuous value engineering. We will execute a phased delivery strategy with integrated sustainability analysis and proactive risk management throughout all project stages to ensure on-time, on-budget completion.' },
          { name: 'referencedMaterial', label: 'Referenced Documents and Standards', type: 'introTable', introPlaceholder: 'This BEP references and responds to the following documents and standards:', tableColumns: ['Document/Standard', 'Version/Date', 'Relevance/Purpose'] }
        ]
      },
      1: {
        title: 'Executive Summary',
        fields: [
          { name: 'projectContext', label: 'Project Context and Overview', required: true, type: 'textarea', rows: 4, placeholder: 'This BEP confirms our comprehensive approach to delivering the Greenfield Office Complex using advanced BIM methodologies. Our strategy emphasises collaborative design coordination, data-driven decision making, and seamless information handover to support long-term facility management. The project will serve as a flagship example of sustainable commercial development in the region, incorporating smart building technologies and flexible workspace design.' },
          { name: 'bimStrategy', label: 'BIM Strategy Summary', required: true, type: 'textarea', rows: 3, placeholder: 'Our confirmed BIM strategy centres on early clash detection, integrated 4D/5D modelling for construction sequencing and cost control, and comprehensive digital twin creation for facilities management. We will utilise federated models across all disciplines with real-time collaboration through cloud-based platforms, ensuring design quality and construction efficiency whilst reducing project risks and enabling predictive maintenance capabilities.' },
          { name: 'keyCommitments', label: 'Key Commitments and Deliverables', required: true, type: 'introTable', introPlaceholder: 'We are committed to full ISO 19650-2:2018 compliance throughout all project phases. Key deliverables include:', tableColumns: ['Deliverable', 'Description', 'Due Date'] },
          { name: 'keyContacts', label: 'Key Project Contacts', type: 'table', columns: ['Role', 'Name', 'Company', 'Email', 'Phone Number'] },
          { name: 'valueProposition', label: 'Value Proposition', type: 'textarea', rows: 3, placeholder: 'Our BIM approach will deliver 15% reduction in construction costs through early clash detection, 25% faster design coordination, and comprehensive lifecycle cost analysis enabling informed material selections. The digital twin will provide 30% operational cost savings through predictive maintenance and space optimisation, whilst the structured data handover ensures seamless facilities management integration and supports the client\'s sustainability targets through enhanced building performance monitoring.' }
        ]
      },
      2: {
        title: 'Confirmed Team and Responsibilities',
        fields: [
          { name: 'organizationalStructure', label: 'Delivery Team\'s Organisational Structure and Composition', type: 'orgchart' },
          { name: 'leadAndInfoManagers', label: 'Lead Appointed Party', required: true, type: 'table', columns: ['Lead Appointed Party', 'Information Manager'] },
          { name: 'taskTeamsBreakdown', label: 'Task Teams', type: 'table', columns: ['Task Team', 'Discipline', 'Leader', 'Leader Contact', 'Company'] },
          { name: 'resourceAllocation', label: 'Resource Allocation', type: 'textarea', rows: 3, placeholder: 'The confirmed delivery team comprises 45 specialists across all disciplines: 12 architects, 8 structural engineers, 10 MEP engineers, 6 quantity surveyors, 4 project managers, and 5 BIM specialists. Resource allocation follows RIBA stages with peak deployment during Stage 4 (Technical Design) requiring 35 FTE personnel. Specialist consultants for sustainability and facade engineering will be engaged for 6-month periods during critical design phases.' },
          { name: 'informationManagementResponsibilities', label: 'Information Management Responsibilities', type: 'textarea', rows: 3, placeholder: 'The Information Manager oversees all aspects of information production, validation, and exchange in accordance with ISO 19650-2:2018. Responsibilities include: establishing and maintaining the CDE, coordinating task information delivery plans (TIDPs), ensuring model federation quality, managing information security protocols, conducting regular audits of information deliverables, and facilitating cross-disciplinary information exchanges. The IM reports directly to the project director and client representative.' }
        ]
      }
    }
  },

  sharedFormFields: {
    3: {
      title: 'BIM Goals and Objectives',
      fields: [
        { name: 'bimGoals', label: 'BIM Goals', required: true, type: 'textarea', rows: 4, placeholder: 'The BIM goals for this project are to enhance design coordination through clash detection reducing RFIs by 40%, improve construction sequencing through 4D modeling resulting in 20% schedule compression, enable accurate cost forecasting through 5D integration achieving ±2% budget variance, and deliver comprehensive digital asset information for lifecycle management supporting 25% reduction in operational costs over the first 5 years.' },
        { name: 'primaryObjectives', label: 'Primary Objectives', type: 'textarea', rows: 3, placeholder: 'Primary objectives include: eliminating design conflicts before construction through rigorous clash detection protocols, optimising building performance through integrated analysis and simulation, enabling efficient construction through accurate quantity extraction and sequencing models, supporting sustainability targets through embedded carbon analysis and energy modeling, and facilitating seamless handover with structured asset data for predictive maintenance and space management.' },
        { name: 'bimUses', label: 'BIM Uses', required: true, type: 'checkbox', options: 'bimUses' },
        { name: 'bimValueApplications', label: 'BIM Applications for Project Value', required: true, type: 'textarea', rows: 4, placeholder: 'BIM will maximize project value through: 4D scheduling for time optimization reducing construction duration by 15%, energy modeling for sustainability compliance achieving BREEAM Excellent rating, life-cycle costing analysis enabling informed material selections with 20-year cost projections, design alternative evaluations through parametric modeling supporting value engineering decisions, pre-fabrication coordination reducing on-site assembly time by 30%, stakeholder visualization for enhanced buy-in and reduced change orders, and comprehensive digital asset creation supporting £2M+ operational cost savings over building lifecycle.' },
        { name: 'valueMetrics', label: 'Success Metrics and Value Measurement', required: true, type: 'table', columns: ['Value Area', 'Target Metric', 'Measurement Method', 'Baseline/Benchmark'] },
        { name: 'strategicAlignment', label: 'Alignment with Client Strategic Objectives', type: 'textarea', rows: 3, placeholder: 'BIM strategy directly supports client objectives including: 15% reduction in total project delivery time through optimized sequencing, achievement of net-zero carbon targets through integrated energy modeling, enhanced asset performance through digital twin implementation, improved tenant satisfaction via optimized space planning and MEP design, future-proofing for smart building integration, and comprehensive data foundation for predictive maintenance reducing operational costs by 25% annually.' },
        { name: 'collaborativeProductionGoals', label: 'Objectives/Goals for the Collaborative Production of Information', type: 'textarea', rows: 4, placeholder: 'Collaborative production goals focus on establishing unified data standards across all disciplines, implementing real-time model coordination through federated workflows, ensuring consistent information delivery at all project milestones, maintaining version control integrity throughout design development, facilitating transparent communication through shared visualisation platforms, and creating comprehensive audit trails for decision-making accountability whilst adhering to ISO 19650 information management principles.' },
        { name: 'alignmentStrategy', label: 'Approach to Facilitating Information Management Goals', type: 'textarea', rows: 3, placeholder: 'Our alignment strategy implements weekly coordination meetings with federated model reviews, establishes clear responsibility matrices for information production and validation, deploys standardised naming conventions and file structures across all disciplines, utilises automated quality checking workflows to ensure compliance, maintains continuous training programmes for team competency development, and implements performance monitoring through defined KPIs including model accuracy, delivery timeliness, and information completeness metrics.' }
      ]
    },
    4: {
      title: 'Level of Information Need (LOIN)',
      fields: [
        { name: 'loinIntroduction', label: 'Introduction and Context', type: 'textarea', rows: 2, placeholder: 'Introduction to Level of Information Need (LOIN) - Define the information requirements that specify what information is needed, at what level of detail, and for what purposes. This section establishes the foundation for information delivery by linking to the Project Information Requirements (PIR) and Employer\'s Information Requirements (EIR) defined earlier in the BEP.' },
        { name: 'informationPurposes', label: 'Information Purposes', required: true, type: 'checkbox', options: 'informationPurposes' },
        { name: 'geometricalInfo', label: 'Geometrical Information Requirements', type: 'introTable', introPlaceholder: 'Describe the purpose of geometrical requirements (e.g., "Support interdisciplinary coordination and visualization"). Specify the LOD standard reference (e.g., UNI EN 17412-1, NBS BIM Object Standard).', tableColumns: ['Element', 'Phase (RIBA)', 'LOD', 'Purpose', 'Verification', 'Delivery Format'] },
        { name: 'alphanumericalInfo', label: 'Alphanumerical Information Requirements', type: 'introTable', introPlaceholder: 'General Description: Introduce the purpose of alphanumerical attributes (e.g., "Support facility management and regulatory compliance"). Cite standards like COBie or BS 1192-4.', tableColumns: ['Element', 'Phase (RIBA)', 'Attributes (LOI)', 'Purpose', 'Verification', 'Delivery Format'] },
        { name: 'documentationInfo', label: 'Documentation Requirements', type: 'introTable', introPlaceholder: 'General Description: Introduce the purpose of documents (e.g., "Provide maintenance manuals or compliance reports"). Cite standards (e.g., PDF/A for long-term archiving).', tableColumns: ['Document', 'Phase (RIBA)', 'Detail', 'Purpose', 'Verification', 'Delivery Format'] },
        { name: 'informationFormats', label: 'Information Formats', type: 'checkbox', options: 'fileFormats' },
        { name: 'projectInformationRequirements', label: 'Project Information Requirements (PIR)', type: 'textarea', rows: 4, placeholder: 'Project Information Requirements specify deliverable information to support asset management objectives: integrated 3D models with embedded property data for space management systems, energy consumption monitoring through IoT sensor integration, preventive maintenance scheduling with equipment lifecycle data, tenant fit-out guidelines with services capacity information, building performance analytics for continuous optimisation, digital twin connectivity for predictive maintenance, compliance monitoring systems for regulatory reporting, and structured data formats supporting client\'s existing CAFM systems and sustainability reporting requirements.' },
        { name: 'loinTransition', label: 'Transition to Delivery Plans', type: 'textarea', rows: 2, placeholder: 'The LOIN requirements defined here will be satisfied through the delivery plans described in the Information Delivery Planning Sect., with MIDP and TIDPs that map each deliverable to specific LOD/LOIN.' }
      ]
    },
    5: {
      title: 'Information Delivery Planning',
      fields: [
        { name: 'deliveryIntroduction', label: 'Introduction and Context', type: 'textarea', rows: 4, placeholder: 'This section establishes the comprehensive framework for information delivery throughout the project lifecycle, defining the schedule, responsibilities, and processes for delivering information in accordance with the Level of Information Need (LOIN) requirements specified in Section 4. The delivery planning ensures that all project stakeholders understand what information is required, when it must be delivered, who is responsible for its production, and how it will be validated against the defined LOIN criteria.' },
        { name: 'loinReference', label: 'Cross-Reference to LOIN Requirements', type: 'textarea', rows: 3, placeholder: 'All information deliverables defined in this section are directly linked to the LOIN requirements established in Section 4. Geometrical information will comply with LOD specifications (Section 4.3.1), alphanumerical attributes will follow the defined LOI standards (Section 4.3.2), and documentation will meet the requirements outlined in Section 4.3.3. Each MIDP and TIDP entry references the specific LOIN requirement it addresses.' },
        { name: 'midpDescription', label: 'Master Information Delivery Plan (MIDP)', type: 'textarea', rows: 6, placeholder: 'The MIDP establishes a structured schedule for information delivery aligned with RIBA Plan of Work 2020 stages and project milestones. It coordinates all Task Information Delivery Plans (TIDPs) and ensures that information is delivered at the right time, to the right quality level, and in the required format. Key deliverables include: Stage 3 coordinated federated models by Month 8, Stage 4 construction-ready models with full MEP coordination by Month 14, Stage 5 as-built verification models by Month 22, and handover documentation including COBie data, O&M manuals, and digital twin integration by Month 24. Each delivery milestone includes quality gates with defined acceptance criteria linked to LOIN requirements and client approval processes.' },
        { name: 'deliverySchedule', label: 'Delivery Schedule and Phasing', type: 'textarea', rows: 4, placeholder: 'Information delivery follows a phased approach aligned with project stages: Months 1-3 (project mobilisation, CDE setup, and concept design models at LOD 200), Months 4-8 (spatial coordination and developed design models at LOD 300), Months 9-14 (technical design and construction documentation at LOD 350), Months 15-20 (construction phase information and progress models at LOD 400), Months 21-24 (commissioning data, as-built verification at LOD 500, and handover documentation). Weekly model federation occurs throughout with formal milestone reviews at stage gates requiring validation against LOIN criteria and client approval before progression.' },
        { name: 'tidpOverview', label: 'Task Information Delivery Plans (TIDPs)', type: 'textarea', rows: 5, placeholder: 'TIDPs define discipline-specific delivery requirements, detailing how each task team will meet the overall MIDP objectives and LOIN requirements. Each TIDP specifies the information to be produced, delivery dates, quality levels (referencing LOIN geometrical and alphanumerical requirements), validation procedures, and integration points with other disciplines. TIDPs are living documents that are reviewed and updated at each project stage gate to ensure continued alignment with project requirements and LOIN specifications.' },
        { name: 'keyMilestonesIntro', label: 'Key Information Delivery Milestones', type: 'introTable', introPlaceholder: 'Information delivery is structured around key project milestones, each with defined acceptance criteria linked to LOIN requirements. The following milestones represent critical decision points and formal information exchanges:', tableColumns: ['Milestone', 'Stage/Date', 'Information Required', 'LOIN Reference', 'Acceptance Criteria', 'Approval Authority'] },
        { name: 'qualityGates', label: 'Quality Gates and Approval Workflows', type: 'textarea', rows: 3, placeholder: 'Each information delivery milestone includes quality gate reviews to verify compliance with LOIN requirements before information can progress to the next workflow state. Quality gates include: automated model validation checks (geometry, attributes, classification), manual design review by discipline leads, cross-discipline coordination reviews, client review and approval workflows, and formal sign-off documentation. Information failing quality gate criteria must be returned to the originating team with specific revision requirements referencing the LOIN standards (e.g., "Geometrical information does not meet LOD 350 requirements as defined in Section 4.3.1").' },
        { name: 'responsibilityMatrix', label: 'Information Delivery Responsibility Matrix (RACI)', type: 'table', columns: ['Information/Task', 'Responsible', 'Accountable', 'Consulted', 'Informed', 'LOIN Reference'] },
        { name: 'mobilisationPlan', label: 'Project Mobilisation Plan', type: 'textarea', rows: 3, placeholder: 'Project mobilisation occurs over 4 weeks: Week 1 includes CDE setup, LOIN requirement review and template development aligned to Section 4 specifications, and team onboarding; Week 2 involves standards training (ISO 19650, LOIN compliance), tool deployment, and workflow testing; Week 3 encompasses pilot model creation demonstrating LOIN compliance, federation testing, and quality checking procedures; Week 4 includes final system validation, team competency verification through LOIN compliance testing, and formal project launch. All team members complete ISO 19650 certification and project-specific LOIN training before accessing the CDE and commencing information production activities.' },
        { name: 'coordinationIntro', label: 'Coordination and Information Exchange', type: 'textarea', rows: 4, placeholder: 'Effective coordination ensures that information produced by different task teams is compatible, consistent, and meets the collective project requirements. Coordination processes include: weekly model federation meetings with automated clash detection analysis, bi-weekly cross-disciplinary design coordination sessions reviewing federated models against LOIN requirements, monthly design freeze periods for cross-disciplinary validation and quality assurance, standardised BCF workflows for issue resolution with LOIN compliance tracking, real-time model access through shared CDE workspace states, automated notification systems for model updates and issue assignments, and formal sign-off procedures for milestone deliverables ensuring all disciplines validate information against LOIN specifications before progression to next design stage.' },
        { name: 'informationExchange', label: 'Information Exchange Protocols Between Task Teams', type: 'introTable', introPlaceholder: 'Information exchange between task teams follows structured protocols to ensure consistency and LOIN compliance:', tableColumns: ['Exchange Type', 'From → To', 'Information Content', 'LOIN Compliance Check', 'Frequency', 'Format/Method'] },
        { name: 'federationCoordination', label: 'Model Federation and Coordination Procedures', type: 'textarea', rows: 3, placeholder: 'Model federation procedures ensure spatial coordination and LOIN compliance across all disciplines: shared coordinate system established from Ordnance Survey grid references, standardised origin points and level datums across all disciplines, automated reference model linking through shared CDE folders with version control, weekly federation cycles with automated LOIN validation (geometry, attributes, classification), clash detection workflows identifying spatial conflicts and LOIN requirement gaps, cross-reference validation ensuring all models meet Section 4 geometrical and alphanumerical requirements, and quality gates preventing model publication without current reference verification and LOIN compliance confirmation.' },
        { name: 'riskRegisterDelivery', label: 'Information Delivery Risk Register', type: 'introTable', introPlaceholder: 'Key risks associated with information delivery and their mitigation strategies:', tableColumns: ['Risk Description', 'Impact on LOIN/Delivery', 'Probability', 'Mitigation Strategy', 'Contingency Plan', 'Owner'] },
        { name: 'deliveryConclusion', label: 'Commitments and Next Steps', type: 'textarea', rows: 3, placeholder: 'The delivery team commits to: (1) Delivering all information in accordance with the MIDP schedule and LOIN requirements specified in Section 4, (2) Maintaining continuous LOIN compliance monitoring and reporting at each milestone, (3) Conducting weekly progress reviews against TIDP schedules with proactive risk mitigation, (4) Ensuring all information passes defined quality gates before client submission, (5) Providing transparent communication and early notification of any delivery challenges. Next steps include: finalising all TIDPs within 2 weeks of project mobilisation, conducting LOIN compliance training for all team members, establishing CDE workflows and quality checking automation, and scheduling the first coordination review meeting for Week 3 of project commencement.' }
      ]
    },
    6: {
      title: 'Common Data Environment (CDE)',
      fields: [
        { name: 'cdeStrategy', label: 'Multi-Platform CDE Strategy', type: 'textarea', rows: 3, placeholder: 'The project employs a federated CDE approach utilizing multiple specialized platforms to optimize workflow efficiency and data management across different information types and project phases. Each platform is selected for its specific strengths while maintaining seamless integration and unified information governance.' },
        { name: 'cdeFlowDiagram', label: 'CDE Platform Workflow Diagram', type: 'cdeFlowDiagram' },
        { name: 'cdePlatforms', label: 'CDE Platform Matrix', required: true, type: 'table', columns: ['Platform/Service', 'Usage/Purpose', 'Information Types', 'Workflow States', 'Access Control'] },
        { name: 'workflowStates', label: 'Unified Workflow States', required: true, type: 'table', columns: ['State Name', 'Description', 'Access Level', 'Next State'] },
        { name: 'accessControl', label: 'Integrated Access Control', type: 'textarea', rows: 3, placeholder: 'Unified role-based access control across all CDE platforms with Single Sign-On (SSO) integration. Project Administrator, Design Team, Review Team, and Client View permissions maintained consistently. Multi-factor authentication required for all platforms. Cross-platform folder synchronization with discipline-specific read/write permissions. Guest access limited to 30-day periods with approval workflows across all systems.' },
        { name: 'securityMeasures', label: 'Multi-Platform Security Framework', type: 'textarea', rows: 3, placeholder: 'End-to-end encryption for data in transit and at rest using AES-256 standards across all platforms. SSL/TLS certificates for secure connections. Regular security audits and penetration testing for each platform. ISO 27001 certified cloud infrastructure. Automated malware scanning for all uploads. Data residency compliance with UK GDPR requirements. Cross-platform security monitoring and incident response procedures.' },
        { name: 'backupProcedures', label: 'Comprehensive Backup Strategy', type: 'textarea', rows: 3, placeholder: 'Automated daily backups with 30-day retention policy across all CDE platforms. Weekly full system backups with 12-month retention. Geo-redundant storage across multiple UK data centres. 99.9% uptime SLA with disaster recovery protocols. Regular backup integrity testing and documented restoration procedures. Cross-platform data synchronization and recovery protocols. Monthly backup verification reports for all systems.' }
      ]
    },
    7: {
      title: 'Technology and Software Requirements',
      fields: [
        { name: 'bimSoftware', label: 'BIM Software Applications', required: true, type: 'checkbox', options: 'software' },
        { name: 'fileFormats', label: 'File Formats', required: true, type: 'checkbox', options: 'fileFormats' },
        { name: 'hardwareRequirements', label: 'Hardware Requirements', type: 'textarea', rows: 3 },
        { name: 'networkRequirements', label: 'Network Requirements', type: 'textarea', rows: 3 },
        { name: 'interoperabilityNeeds', label: 'Interoperability Requirements', type: 'textarea', rows: 3 },
        { name: 'federationStrategy', label: 'Federation Strategy', type: 'textarea', rows: 3, placeholder: 'Strategy details...' },
        { name: 'informationBreakdownStrategy', label: 'Information Breakdown Strategy', type: 'textarea', rows: 3, placeholder: 'Information breakdown strategy...' },
        { name: 'federationProcess', label: 'Federation Process', type: 'textarea', rows: 3, placeholder: 'Federation process details...' },
        { name: 'softwareHardwareInfrastructure', label: 'Software, Hardware and IT Infrastructure', type: 'table', columns: ['Category', 'Item/Component', 'Specification', 'Purpose'] },
        { name: 'documentControlInfo', label: 'Document Control Information', type: 'textarea', rows: 4, placeholder: 'Document type, ISO standards, status, generator details...' }
      ]
    },
    8: {
      title: 'Information Production Methods and Procedures',
      fields: [
        { name: 'modelingStandards', label: 'Modeling Standards', required: true, type: 'table', columns: ['Standard/Guideline', 'Version', 'Application Area', 'Compliance Level'] },
        { name: 'namingConventions', label: 'Naming Conventions', required: true, type: 'table', columns: ['Element Type', 'Naming Format', 'Example', 'Description'] },
        { name: 'fileStructure', label: 'File Structure', type: 'textarea', rows: 3 },
        { name: 'fileStructureDiagram', label: 'File Structure Diagram', type: 'fileStructure' },
        { name: 'volumeStrategy', label: 'Volume Strategy and Model Breakdown', required: true, type: 'mindmap' },
        { name: 'classificationSystems', label: 'Classification Systems and Coding', required: true, type: 'table', columns: ['Classification System', 'Application Area', 'Code Format', 'Responsibility'] },
        { name: 'classificationStandards', label: 'Classification Standards Implementation', type: 'table', columns: ['Element Category', 'Classification System', 'Code Format', 'Example Code', 'Description'] },
        { name: 'dataExchangeProtocols', label: 'Data Exchange Protocols', type: 'table', columns: ['Exchange Type', 'Format', 'Frequency', 'Delivery Method'] }
      ]
    },
    9: {
      title: 'Quality Assurance and Control',
      fields: [
        { name: 'qaFramework', label: 'Quality Assurance Framework', required: true, type: 'table', columns: ['QA Activity', 'Responsibility', 'Frequency', 'Tools/Methods'] },
        { name: 'modelValidation', label: 'Model Validation Procedures', required: true, type: 'textarea', rows: 4 },
        { name: 'reviewProcesses', label: 'Review Processes', type: 'textarea', rows: 3 },
        { name: 'approvalWorkflows', label: 'Approval Workflows', type: 'textarea', rows: 3 },
        { name: 'complianceVerification', label: 'Compliance Verification', type: 'textarea', rows: 3 },
        { name: 'modelReviewAuthorisation', label: 'Information Model Review and Authorisation', type: 'textarea', rows: 3, placeholder: 'Review and authorisation procedures...' }
      ]
    },
    10: {
      title: 'Information Security and Privacy',
      fields: [
        { name: 'dataClassification', label: 'Data Classification', required: true, type: 'table', columns: ['Classification Level', 'Description', 'Examples', 'Access Controls'] },
        { name: 'accessPermissions', label: 'Access Permissions', required: true, type: 'textarea', rows: 3 },
        { name: 'encryptionRequirements', label: 'Encryption Requirements', type: 'textarea', rows: 3 },
        { name: 'dataTransferProtocols', label: 'Data Transfer Protocols', type: 'textarea', rows: 3 },
        { name: 'privacyConsiderations', label: 'Privacy Considerations', type: 'textarea', rows: 3 }
      ]
    },
    11: {
      title: 'Training and Competency',
      fields: [
        { name: 'bimCompetencyLevels', label: 'BIM Competency Levels', required: true, type: 'textarea', rows: 4 },
        { name: 'trainingRequirements', label: 'Training Requirements', type: 'textarea', rows: 3 },
        { name: 'certificationNeeds', label: 'Certification Requirements', type: 'textarea', rows: 3 },
        { name: 'projectSpecificTraining', label: 'Project-Specific Training', type: 'textarea', rows: 3 }
      ]
    },
    12: {
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
        { name: 'updateProcesses', label: 'Update Processes', type: 'textarea', rows: 3 },
        { name: 'projectKpis', label: 'Project Key Performance Indicators (KPIs)', type: 'table', columns: ['KPI Name', 'Target Value', 'Measurement Method', 'Responsibility'] }
      ]
    },
    13: {
      title: 'Appendices',
      fields: [
        { name: 'responsibilityMatrix', label: 'Appendix A: Responsibility Matrix Template', required: true, type: 'table', columns: ['Role/Task', 'Responsible', 'Accountable', 'Consulted', 'Informed'] },
        { name: 'cobieRequirements', label: 'Appendix B: COBie Data Requirements', required: true, type: 'table', columns: ['Component Type', 'Required Parameters', 'Data Source', 'Validation Method'] },
        { name: 'fileNamingExamples', label: 'Appendix C: File Naming Convention Examples', required: true, type: 'textarea', rows: 6, placeholder: 'Comprehensive file naming examples:\n\nProject Models:\nGF-SAA-L02-ARC-001 (Greenfield-Smith Associates-Level 02-Architecture-Model 001)\nGF-JEL-SZ1-STR-002 (Greenfield-Jones Engineering-Structural Zone 1-Structure-Model 002)\nGF-TSS-MZ2-MEP-003 (Greenfield-TechServ Solutions-MEP Zone 2-Services-Model 003)\n\nDrawings:\nGF-SAA-ZZ-ARC-DR-A-1001 (General Arrangement Plans)\nGF-JEL-ZZ-STR-DR-S-2001 (Structural General Arrangement)\nGF-TSS-L03-MEP-DR-M-3001 (Level 3 Mechanical Plans)\n\nDocuments:\nGF-SAA-ZZ-ARC-SP-001 (Architectural Specification)\nGF-CMP-ZZ-QS-RP-001 (Cost Report)\nGF-ALL-ZZ-PM-MR-001 (Project Meeting Minutes)' },
        { name: 'exchangeWorkflow', label: 'Appendix D: Information Exchange Workflow Template', required: true, type: 'table', columns: ['Exchange Point', 'Information Required', 'Format', 'Quality Checks', 'Approval Process'] },
        { name: 'modelCheckingCriteria', label: 'Appendix E: Model Quality Checking Criteria', type: 'table', columns: ['Check Type', 'Acceptance Criteria', 'Tools Used', 'Frequency'] },
        { name: 'softwareVersionMatrix', label: 'Appendix F: Software Version Compatibility Matrix', type: 'table', columns: ['Software', 'Version', 'File Formats Supported', 'Interoperability Notes'] },
        { name: 'deliverableTemplates', label: 'Appendix G: Information Deliverable Templates', type: 'textarea', rows: 4, placeholder: 'Standard templates and schedules for key deliverables:\n\n- Task Information Delivery Plan (TIDP) Template\n- Model Element Checklist Template\n- Quality Assurance Report Template\n- Clash Detection Report Template\n- Progress Report Template\n- Information Exchange Record Template\n- Asset Data Handover Template\n\nAll templates available in project CDE Templates folder with version control and approval workflows.' }
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