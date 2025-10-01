// Help content for BEP form fields
// This provides contextual guidance, ISO 19650 references, best practices, and examples

const HELP_CONTENT = {
  // ====================================================================
  // PROJECT INFORMATION FIELDS
  // ====================================================================

  projectDescription: {
    description: `Provide a comprehensive project overview including scope, scale, complexity, and key objectives. This should give readers a clear understanding of what the project entails, its unique characteristics, and why BIM is essential for its delivery.

Include:
• Project type and primary use
• Physical scale (area, height, number of buildings)
• Project value/budget range
• Key sustainability or certification targets (LEED, BREEAM, WELL, etc.)
• Unique technical or design challenges
• Main stakeholder requirements and expectations
• Site constraints or existing conditions that impact delivery`,

    iso19650: `ISO 19650-2:2018 Section 5.1.2 - Project Information

The standard requires clear project definition including information about the appointing party's requirements, project objectives, constraints, and any specific information management requirements. This helps all parties understand the context for information delivery.`,

    bestPractices: [
      'Start with project type and scale to set context immediately',
      'Quantify wherever possible (area, capacity, budget range, timeline)',
      'Explicitly mention sustainability targets (BREEAM, LEED, WELL, PassivHaus)',
      'Highlight complexity factors that justify the BIM approach',
      'Reference the client\'s strategic objectives when known',
      'Include site context if it impacts design or construction approach',
      'Mention number of disciplines for complex multi-disciplinary projects',
      'Keep it concise but comprehensive (200-400 words is ideal)'
    ],

    examples: {
      'Commercial Building': `A modern 15-storey Grade A office complex with retail on the ground floor, located in central Manchester. The 45,000m² development will accommodate 800+ office workers with flexible workspace layouts, integrated smart building technologies, and targeting BREEAM Excellent certification.

The project includes basement parking for 150 vehicles, rooftop plant areas, and a double-height reception atrium. Key challenges include integration with adjacent heritage buildings, complex MEP systems for smart building features, and achieving net-zero operational carbon. Budget: £65M. Timeline: 32 months (Design to Handover).`,

      'Infrastructure': `A 12km dual-carriageway bypass including three major grade-separated junctions, two river crossings, and associated drainage infrastructure serving the town of Westfield. The scheme will relieve congestion in the town centre and support planned residential development of 5,000 new homes.

Key technical challenges include environmentally sensitive river crossings, complex utilities diversions, and coordination with operational rail infrastructure. The project requires extensive stakeholder engagement with Network Rail, Environment Agency, and local communities. Budget: £180M. Programme: 48 months including 18-month design phase.`,

      'Healthcare': `Extension and refurbishment of St. Mary's District Hospital adding 200-bed capacity across a new 5-storey clinical block. The development includes 4 operating theatres, diagnostic imaging suite, intensive care unit, and supporting clinical facilities, all requiring integration with existing operational hospital systems.

Critical requirements include stringent infection control (HBN 04-01 compliance), complex MEP systems for medical gases and critical ventilation, maintaining hospital operations throughout construction, and achieving BREEAM Healthcare Excellent. Budget: £85M. Phased delivery over 36 months with operational hospital constraints.`,

      'Residential': `Mixed-use development of 280 residential units across two towers (18 and 22 storeys) with ground floor retail and 2 levels of basement parking. Located on a brownfield site in East London, targeting 35% affordable housing and BREEAM Communities Excellent.

The scheme includes a mix of 1, 2, and 3-bed apartments, communal amenity spaces, and landscaped courtyard. Technical challenges include ground conditions requiring piled foundations, tall building regulations compliance, and complex MEP distribution in high-rise residential. Budget: £92M. Programme: 30 months.`,

      'Education': `New-build secondary academy for 1,200 students (11-18 years) on a greenfield site including sports facilities, performing arts centre, and science laboratories. The school is designed to PassivHaus standards with natural ventilation strategy, achieving net-zero carbon in operation.

Facilities include 60 teaching spaces, assembly hall for 400, double sports hall, 3G sports pitch, and extensive STEM facilities. The project must achieve BREEAM Excellent and meet strict DfE output specifications. Budget: £35M. Programme: 24 months including summer handover for September opening.`
    },

    commonMistakes: [
      'Being too vague or generic - "A large office building" tells readers nothing useful',
      'Omitting key project metrics (size, value, timeline) that provide context',
      'No mention of sustainability targets or environmental certifications',
      'Failing to highlight what makes the project complex or challenging',
      'Not connecting project characteristics to BIM requirements',
      'Too much unnecessary detail about architectural aesthetics',
      'Missing stakeholder context or client strategic objectives',
      'Not mentioning site constraints that impact delivery approach'
    ],

    relatedFields: ['projectType', 'estimatedBudget', 'confirmedBudget', 'proposedTimeline', 'confirmedTimeline']
  },

  // ====================================================================

  bimStrategy: {
    description: `Describe your comprehensive BIM strategy for this project. Explain how BIM will be used throughout the project lifecycle to deliver value, reduce risk, improve collaboration, and meet the client's information requirements.

Cover:
• Primary BIM objectives and goals
• Key BIM uses (clash detection, 4D/5D, energy analysis, etc.)
• Technology and software approach
• Collaboration and coordination methods
• Information management approach
• Digital twin / FM handover strategy
• How BIM reduces project risks`,

    iso19650: `ISO 19650-2:2018 Section 5.1.4 - Mobilization of Resources

The BIM strategy should demonstrate how the delivery team will mobilize information management capabilities, establish information management processes, and deliver against the Exchange Information Requirements (EIR). It must show understanding of the appointing party's information requirements.`,

    bestPractices: [
      'Link BIM uses directly to client business objectives and project goals',
      'Be specific about which BIM processes will be used and why',
      'Mention federated model approach for multi-discipline coordination',
      'Reference clash detection protocols and expected clash reduction targets',
      'Include 4D sequencing for construction planning if applicable',
      '5D cost management integration for budget control and value engineering',
      'Explain digital twin approach for FM handover and lifecycle management',
      'Reference cloud-based collaboration platforms (BIM 360, Trimble Connect, etc.)',
      'Quantify expected benefits where possible (% clash reduction, time savings)'
    ],

    examples: {
      'Commercial Building': `Our confirmed BIM strategy centres on early clash detection to eliminate design conflicts, integrated 4D/5D modelling for construction sequencing and cost control, and comprehensive digital twin creation for facilities management. We will utilize federated models across all disciplines (Architecture, Structure, MEP, Civils) with real-time collaboration through BIM 360.

Key processes include weekly clash detection reducing RFIs by 40%, 4D sequencing for complex MEP installation coordination, 5D integration enabling continuous value engineering, and LOD 350 models for fabrication. The digital twin will include all MEP systems, space data, and equipment for predictive maintenance, reducing operational costs by 25% over 5 years.`,

      'Infrastructure': `Our BIM strategy emphasizes 3D design coordination for complex junction geometries, 4D sequencing for traffic management and utilities diversions, and integrated civil/structures/drainage models for clash-free construction documentation. We will use 12d Model for road/drainage design, Tekla for structures, and Navisworks for coordination.

Key benefits include clash-free design reducing on-site variations by 60%, 4D visualization for stakeholder engagement and planning approvals, quantity extraction for accurate cost estimates, and as-built model delivery for asset management. All design data will be delivered in IFC format for the client's asset management system.`,

      'Healthcare': `Our BIM strategy addresses the complexity of live hospital operations during construction, stringent infection control requirements, and integration with existing building systems. We will implement phased coordination models, detailed MEP clash detection for medical gases and critical ventilation, and 4D sequencing coordinated with hospital operations.

Specific processes include: room-by-room coordination to LOD 350 for clinical spaces, medical equipment spatial coordination, services coordination for complex ceiling voids, and detailed handover models including all MEP systems with maintenance data for the Trust's CAFM system. Target: zero clashes in critical clinical areas before construction.`
    },

    commonMistakes: [
      'Generic statements like "We will use BIM" without specific processes',
      'No connection between BIM uses and project-specific challenges',
      'Failing to quantify expected benefits or improvements',
      'Not mentioning specific software or technology platforms to be used',
      'Omitting the digital twin / FM handover strategy',
      'No reference to level of information need (LOIN) or LOD requirements',
      'Missing coordination and clash detection protocols',
      'Not addressing how BIM reduces specific project risks'
    ],

    relatedFields: ['bimGoals', 'primaryObjectives', 'bimUses']
  },

  // ====================================================================

  keyCommitments: {
    description: `List your firm commitments and key deliverables for this project. These are the specific, measurable outcomes you will deliver using BIM processes. This section demonstrates accountability and sets clear expectations.

Include:
• ISO 19650 compliance commitment
• Specific model deliverables (federated models, LOD, frequency)
• Data deliverables (COBie, asset data, O&M information)
• 4D/5D deliverables if applicable
• Coordination and clash detection commitments
• CDE and information management commitments
• Quality assurance and validation processes
• Digital twin / FM handover commitments`,

    iso19650: `ISO 19650-2:2018 Section 5.4 - Information Delivery Planning

The delivery team must commit to specific information deliverables aligned with the project's Information Delivery Plan (IDP). These commitments should be measurable, time-bound, and directly responsive to the Exchange Information Requirements (EIR).`,

    bestPractices: [
      'Start with ISO 19650-2:2018 compliance as primary commitment',
      'Be specific and measurable - "Coordinated model every 2 weeks" not "regular models"',
      'Include LOD specifications for each project stage (RIBA 3: LOD 300, RIBA 4: LOD 350, etc.)',
      'Commit to specific clash detection frequency and resolution targets',
      'Reference COBie or other structured data format for FM handover',
      'Mention CDE implementation with role-based access and audit trails',
      'Include quality checking protocols (Solibri, Navisworks validation)',
      'State information security and data protection commitments',
      'Quantify where possible (95% clash-free, < 24hr issue resolution, etc.)'
    ],

    examples: {
      'Commercial Building': `We are committed to full ISO 19650-2:2018 compliance throughout all project phases. Key deliverables include:

• Coordinated federated models at each RIBA stage milestone (LOD 300 at Stage 3, LOD 350 at Stage 4)
• Bi-weekly clash detection with 95% clash resolution before construction
• Comprehensive COBie 2.4 data for all MEP systems and building components at handover
• 4D construction sequences for all major building elements and MEP installations
• 5D cost model integrated with design for continuous value engineering
• Complete digital twin with IoT sensor integration and equipment data
• All information delivered through BIM 360 CDE with full audit trails and version control
• Monthly model quality validation using Solibri and Navisworks`,

      'Infrastructure': `We commit to delivering the scheme in full accordance with ISO 19650-2:2018 and the client's BIM Protocol. Specific commitments include:

• Coordinated 3D design models for all disciplines (Highway, Structures, Drainage, Utilities) at each design stage
• Weekly clash detection during detailed design with clash-free issue for construction
• 4D construction programme models for stakeholder visualization and planning consent
• Quantity schedules extracted from models for all major elements (earthworks, structures, pavements)
• As-built models delivered in IFC 4 format for the client's asset management system
• Full compliance with UK BIM Framework and National Highways BIM requirements
• GIS integration for linear asset data management`,

      'Healthcare': `Our key commitments for this healthcare project include:

• Full ISO 19650-2:2018 and HBN 00-07 (Planning for a Resilient Healthcare Estate) compliance
• Room-by-room coordination models to LOD 350 for all clinical spaces
• Medical equipment spatial coordination models before procurement
• Clash-free MEP models for medical gases, critical ventilation, and electrical systems
• Detailed handover models with full COBie data for integration with the Trust's CAFM system
• O&M manuals linked to model components for maintenance planning
• Phased construction models coordinated with operational hospital constraints
• Information security compliant with NHS Data Security Standards`
    },

    commonMistakes: [
      'Vague commitments like "we will deliver high quality models"',
      'No specific LOD or level of information need mentioned',
      'Missing frequency or timing of deliverables',
      'No mention of structured data (COBie) for FM handover',
      'Omitting CDE implementation and information management',
      'Not addressing quality assurance or validation processes',
      'Failing to reference ISO 19650 or other applicable standards',
      'No quantified targets or success criteria'
    ],

    relatedFields: ['bimStrategy', 'informationManagementResponsibilities']
  },

  // ====================================================================

  teamCapabilities: {
    description: `Describe your team's BIM capabilities, experience, and track record. This demonstrates that you have the right skills, certifications, and experience to deliver the project successfully using BIM.

Cover:
• Years of BIM implementation experience
• Team size and composition (BIM Manager, Coordinators, Modellers)
• Relevant certifications (ISO 19650, BIM Level 2, etc.)
• Software expertise and licenses
• Relevant project experience (similar type, scale, complexity)
• Value delivered on past projects (cost savings, clash reduction, etc.)
• Training and continuous professional development`,

    iso19650: `ISO 19650-2:2018 Section 5.1.3 - Capability and Capacity

The delivery team must demonstrate capability (skills, knowledge, certifications) and capacity (resources, availability) to meet the information management requirements. This includes evidence of past performance on similar projects.`,

    bestPractices: [
      'Lead with years of experience and total project value delivered using BIM',
      'Mention key certifications (ISO 19650 Lead, BIM Level 2, RICS, etc.)',
      'Quantify team composition (e.g., 5 BIM coordinators, 15 modellers)',
      'List core software competencies (Revit, Navisworks, Solibri, etc.)',
      'Reference 2-3 recent similar projects with measurable outcomes',
      'Include value delivered (% cost savings, clash reduction, time saved)',
      'Mention any awards or recognition for BIM excellence',
      'Reference continuous training and development programs',
      'Highlight unique capabilities (Dynamo scripting, computational design, etc.)'
    ],

    examples: {
      'Commercial Building': `Our multidisciplinary team brings 15+ years of BIM implementation experience across £500M+ of commercial projects. Key capabilities include:

• ISO 19650-2:2018 certified information management with dedicated BIM Manager (RICS Certified)
• Team of 25+ including 5 BIM Coordinators and 20 discipline modellers
• Advanced parametric design using Revit and Grasshopper for complex facade geometries
• Integrated MEP coordination expertise delivering 95% clash-free models
• 4D/5D modelling using Synchro and CostX for programme and cost integration
• Digital twin development with IoT integration experience
• Software licenses: Autodesk Construction Cloud (50 users), Solibri, Navisworks, Dynamo

Recent projects include the award-winning Tech Hub (£25M, 2022 - 40% clash reduction, 3-month programme saving) and Riverside Commercial Centre (£18M, 2023 - BREEAM Excellent achieved through BIM-enabled sustainability analysis).`,

      'Infrastructure': `Our infrastructure BIM team has delivered £2B+ of major projects over 12 years, specializing in highways, structures, and rail infrastructure. Core capabilities:

• Certified to PAS 1192-2:2013 and ISO 19650-2:2018 with dedicated Information Manager
• Team of 30+ including highway designers, structural engineers, and BIM specialists
• Expertise in 12d Model, Civil 3D, Tekla Structures, and Bentley systems
• Track record of delivering complex junction geometries and bridge structures
• 4D planning expertise for traffic management and construction sequencing
• GIS integration for linear asset management and handover
• Experience with Network Rail, Highways England, and HS2 BIM requirements

Recent schemes: A45 Junction Improvement (£85M, 2022 - 8 months early through 4D optimization) and Westside Rail Bridge (£32M, 2023 - zero site clashes through detailed coordination).`,

      'Healthcare': `Specialized healthcare BIM team with 10+ years NHS project experience totaling £350M+ of hospital and healthcare facilities. Key strengths:

• ISO 19650 and HBN 00-07 compliance with healthcare-sector BIM Manager
• Team of 18 including medical planning specialists and MEP coordinators
• Deep expertise in HBN/HTM standards and healthcare-specific requirements
• Medical equipment coordination and infection control modeling
• Complex MEP systems for medical gases, critical ventilation, and electrical resilience
• Experience maintaining operational hospitals during construction through phased BIM
• CAFM integration expertise for NHS Trusts

Recent projects: St. James's Hospital Extension (£55M, 2022 - full digital twin delivered), Regional Diagnostic Centre (£28M, 2023 - zero clashes in clinical areas, COBie handover 2 weeks early).`
    },

    commonMistakes: [
      'Generic claims without quantifiable evidence or metrics',
      'Not mentioning specific certifications or professional qualifications',
      'Missing team size and composition details',
      'No reference to past project examples with measurable outcomes',
      'Failing to link capabilities to project-specific requirements',
      'Not mentioning software licenses or technical infrastructure',
      'Omitting continuous development and training programs',
      'No differentiation - sounds like every other BIM team'
    ],

    relatedFields: ['proposedInfoManager', 'informationManager', 'resourceAllocation']
  },

  // ====================================================================

  bimGoals: {
    description: `Define the overarching BIM goals for this project. These are the high-level objectives that BIM will help achieve - they should align with the client's business objectives and project success criteria.

Typical BIM goals include:
• Improve design coordination and reduce conflicts
• Enhance construction efficiency and reduce waste
• Enable accurate cost forecasting and value engineering
• Support sustainability and environmental targets
• Facilitate asset management and operational efficiency
• Reduce project risks and improve predictability
• Enable better stakeholder communication and engagement

Make goals SMART: Specific, Measurable, Achievable, Relevant, Time-bound.`,

    iso19650: `ISO 19650-2:2018 Section 5.1.1 - Assessment and Need

BIM goals should be established based on the appointing party's information requirements and the project's business objectives. They provide the framework for defining BIM uses and information exchange requirements.`,

    bestPractices: [
      'Align BIM goals with client business objectives (cost, quality, time, sustainability)',
      'Make goals measurable with quantified targets (reduce clashes by 40%, etc.)',
      'Include both design phase and construction/FM lifecycle goals',
      'Reference sustainability targets (carbon reduction, energy performance)',
      'Connect goals to risk reduction and quality improvement',
      'Mention stakeholder communication and approval processes',
      'Include digital twin / asset management goals for long-term value',
      'State goals that address project-specific challenges',
      'Keep to 3-5 primary goals - avoid laundry lists'
    ],

    examples: {
      'Commercial Building': `The BIM goals for this project are to:

1. Enhance design coordination through rigorous clash detection, reducing RFIs by 40% and eliminating on-site design conflicts that typically delay programme by 3-4 weeks

2. Improve construction sequencing through integrated 4D modeling, resulting in 20% schedule compression and optimized site logistics for the constrained urban site

3. Enable accurate cost forecasting through 5D integration, achieving ±2% budget variance and supporting continuous value engineering throughout design development

4. Support net-zero carbon targets through integrated energy analysis and embedded carbon calculations, ensuring BREEAM Excellent certification

5. Deliver comprehensive digital asset information for lifecycle management, supporting 25% reduction in operational costs over the first 5 years through predictive maintenance and space optimization`,

      'Infrastructure': `Our BIM goals for this bypass scheme are:

1. Eliminate design coordination conflicts between highway, structures, drainage, and utilities disciplines through federated model coordination, preventing costly on-site clashes

2. Optimize construction programme through 4D sequencing, reducing traffic management impacts by 30% and enabling faster delivery of benefits to the community

3. Enable accurate quantity extraction and cost control, maintaining budget certainty and supporting value engineering through design development

4. Support environmental consent through visualization and 3D analysis, demonstrating minimal impact on sensitive river crossings and protected species habitats

5. Deliver complete as-built asset information for the highway authority's asset management system, enabling predictive maintenance and lifecycle cost optimization`
    },

    commonMistakes: [
      'Vague goals like "improve collaboration" without measurable outcomes',
      'Not connecting BIM goals to client business objectives',
      'Missing quantified targets or success criteria',
      'Focusing only on design phase, ignoring construction and FM',
      'Not addressing project-specific challenges or constraints',
      'Too many goals - dilutes focus (keep to 3-5 primary goals)',
      'Goals that aren\'t actually achievable through BIM processes',
      'No connection to sustainability or environmental targets'
    ],

    relatedFields: ['bimStrategy', 'primaryObjectives', 'bimUses']
  },

  // ====================================================================

  primaryObjectives: {
    description: `State the primary objectives that will be achieved through BIM implementation. These are the specific, actionable outcomes that support the overarching BIM goals.

Objectives should be concrete and tactical, such as:
• Clash-free design coordination for MEP systems
• 4D construction sequencing for complex areas
• Accurate quantity take-off for cost control
• Energy analysis to meet performance targets
• COBie data for FM handover
• Virtual reality walkthroughs for client approvals
• Safety planning through 4D visualization

Be specific about which BIM processes will be used to achieve each objective.`,

    iso19650: `ISO 19650-2:2018 Section 5.1.6 - Collaborative Production of Information

Primary objectives should demonstrate how the delivery team will collaboratively produce, manage, and exchange information to meet the project's information requirements. They translate high-level goals into specific delivery activities.`,

    bestPractices: [
      'Make objectives specific and actionable - state exactly what will be done',
      'Link each objective to a specific BIM process or use case',
      'Include objectives across the full project lifecycle (design, construction, handover)',
      'Reference quality standards and acceptance criteria',
      'Mention coordination frequency (weekly clash detection, bi-weekly coordination)',
      'Include data and information deliverables (COBie, asset data, IFC exports)',
      'Address stakeholder communication and approval processes',
      'Specify software tools that will be used for each objective',
      'Keep objectives aligned with project priorities and constraints'
    ],

    examples: {
      'Commercial Building': `Primary objectives include:

1. Eliminate design conflicts before construction through rigorous weekly clash detection protocols, achieving 95% clash-free models before tender with < 24hr clash resolution SLA

2. Optimize building performance through integrated energy modeling and CFD analysis, ensuring compliance with net-zero carbon targets and validating passive design strategies

3. Enable efficient construction through accurate quantity extraction and 4D sequencing models, supporting just-in-time procurement and optimized site logistics for constrained urban site

4. Support sustainability targets through embedded carbon analysis and material lifecycle assessment, enabling informed material selections that reduce whole-life carbon by 30%

5. Deliver comprehensive digital twin with structured asset data (COBie 2.4), equipment specifications, and O&M information for seamless FM integration and predictive maintenance`,

      'Healthcare': `Primary objectives for this hospital extension:

1. Achieve zero clashes in critical clinical areas through room-by-room coordination to LOD 350, with specific focus on medical gases, critical ventilation, and ceiling void coordination

2. Coordinate medical equipment spatially with all building services before procurement, ensuring adequate clearances, power supplies, and servicing access for all clinical equipment

3. Validate infection control design through airflow analysis and room pressure validation, ensuring compliance with HBN 04-01 and achieving required air change rates

4. Support phased construction planning through 4D models coordinated with operational hospital constraints, minimizing disruption to live clinical services

5. Deliver complete asset data integrated with the Trust's CAFM system, including all MEP systems, medical equipment, and maintenance schedules for lifecycle management`
    },

    commonMistakes: [
      'Objectives too vague - "improve coordination" rather than specific clash detection protocols',
      'Not specifying which BIM processes or tools will be used',
      'Missing quality standards or acceptance criteria',
      'No connection to project-specific challenges',
      'Focusing only on design, ignoring construction and handover objectives',
      'Objectives that aren\'t measurable or verifiable',
      'Not addressing information deliverables (COBie, IFC, etc.)',
      'Missing objectives for stakeholder communication and approvals'
    ],

    relatedFields: ['bimGoals', 'bimStrategy', 'bimUses']
  },

  // ====================================================================

  valueProposition: {
    description: `Articulate the value that your BIM approach will deliver to the client and project. This is your opportunity to sell the benefits - explain how BIM will save money, reduce risk, improve quality, accelerate delivery, and support long-term asset management.

Quantify value wherever possible:
• Cost savings through clash reduction and value engineering
• Time savings through better coordination and planning
• Quality improvements through visualization and analysis
• Risk reduction through early problem identification
• Operational benefits through digital twin and asset data
• Sustainability benefits through analysis and optimization

Make it compelling and client-focused.`,

    iso19650: `ISO 19650-1:2018 Section 5.2 - Value and Benefits

Information management processes should deliver tangible value to all project stakeholders. The value proposition demonstrates return on investment for BIM implementation and aligns with the appointing party's business case.`,

    bestPractices: [
      'Lead with quantified cost and time savings where possible',
      'Connect value to client\'s strategic objectives and pain points',
      'Include both short-term (design/construction) and long-term (operations) value',
      'Reference risk reduction and quality improvements',
      'Mention sustainability and environmental benefits',
      'Include stakeholder communication and decision-making improvements',
      'Support claims with evidence from past projects if possible',
      'Make it client-focused - "you will benefit from" not "we will deliver"',
      'Keep it concise but impactful - focus on the top 3-5 value drivers'
    ],

    examples: {
      'Commercial Building': `Our BIM approach will deliver significant value across design, construction, and operational phases:

Cost Savings: 15% reduction in construction costs through early clash detection (saving £9.75M on £65M project), eliminating costly on-site rework and programme delays. Continuous 5D value engineering will identify £2-3M of savings opportunities during design development.

Time Benefits: 25% faster design coordination through real-time federated models and cloud collaboration, compressing design programme by 8 weeks. 4D construction sequencing will optimize site logistics and MEP installation, preventing typical 3-4 week delays from coordination issues.

Quality & Risk: Virtual construction through BIM eliminates surprise clashes and buildability issues, reducing project risk and ensuring predictable delivery. Enhanced stakeholder visualization supports faster approvals and reduces design changes.

Operational Value: The digital twin will deliver 30% operational cost savings (£450K annually) through predictive maintenance, space optimization, and energy management. Structured asset data enables immediate FM system integration, avoiding 6-month manual data collection typically required.

Sustainability: Integrated energy and carbon analysis ensures net-zero operational carbon target is met, with embedded carbon reduced by 30% through informed material selection and lifecycle assessment.`,

      'Infrastructure': `Our BIM approach delivers value throughout the project lifecycle:

Risk Reduction: Clash-free design coordination between highway, structures, drainage, and utilities prevents costly on-site conflicts that typically delay infrastructure projects by 3-6 months. Early identification of constraints and clashes protects the £180M budget.

Programme Benefits: 4D construction sequencing optimizes traffic management and construction logistics, reducing overall programme by 6-8 weeks and minimizing disruption to the community. Faster delivery means earlier realization of scheme benefits.

Cost Certainty: Accurate quantity extraction from coordinated models provides reliable cost estimates (±3%), enabling better commercial management and value engineering. This typically saves 5-10% (£9-18M) through optimized design and construction planning.

Stakeholder Benefits: High-quality 3D visualization supports faster planning consent and public engagement, reducing approval timescales by 2-3 months. Environmental impact visualization demonstrates minimal impact on sensitive areas.

Asset Management: Complete as-built data in the client's asset management system enables predictive maintenance planning, lifecycle cost optimization, and integration with the strategic road network. This delivers 20-25% savings in whole-life maintenance costs.`
    },

    commonMistakes: [
      'Generic claims without quantified evidence or metrics',
      'Focusing only on BIM process rather than client value and benefits',
      'Not connecting value to client\'s specific pain points or objectives',
      'Missing long-term operational and asset management value',
      'No mention of risk reduction or quality improvements',
      'Vague savings claims like "significant cost savings" without numbers',
      'Ignoring sustainability and environmental benefits',
      'Not comparing to alternative (non-BIM) approach to show differential value'
    ],

    relatedFields: ['bimStrategy', 'bimGoals', 'keyCommitments']
  },

  // ====================================================================

  bimUses: {
    description: `Select the specific BIM uses that will be applied on this project. BIM uses are the specific ways that BIM processes and models will be utilized to deliver value.

Common BIM uses include:
• Design Authoring (3D modeling)
• Design Reviews and Coordination
• 3D Coordination / Clash Detection
• 4D Planning and Sequencing
• 5D Cost Estimation and Control
• Energy and Sustainability Analysis
• Structural Analysis
• MEP Analysis and Coordination
• Quantity Take-off
• Virtual Reality / Visualization
• Digital Fabrication
• Asset Management / Digital Twin
• Construction Sequencing
• Safety Planning

Select uses that are appropriate for your project type, complexity, and client requirements.`,

    iso19650: `ISO 19650-2:2018 Section 5.1.1 - Information Requirements

BIM uses should be selected based on the project's information requirements and the value they deliver. Each BIM use should support specific project objectives and information delivery milestones.`,

    bestPractices: [
      'Select BIM uses that directly support your stated BIM goals and objectives',
      'Prioritize uses that address project-specific challenges',
      'Consider the full project lifecycle - design, construction, and operations',
      'For each selected use, be prepared to explain the specific process and tools',
      'Infrastructure projects: focus on 3D coordination, 4D sequencing, quantities',
      'Commercial buildings: clash detection, 4D/5D, energy analysis, digital twin',
      'Healthcare: clash detection, MEP coordination, medical equipment planning',
      'Education: design reviews, sustainability analysis, stakeholder visualization',
      'Don\'t check every box - select 6-10 uses that truly add value',
      'Be realistic about team capability and project budget'
    ],

    examples: {
      'Commercial Building': `Selected BIM Uses for this project:

✓ Design Authoring - All disciplines (Architecture, Structure, MEP)
✓ Design Reviews - Weekly coordination meetings with federated models
✓ 3D Coordination - Bi-weekly clash detection achieving 95% clash-free
✓ Clash Detection - Using Navisworks with automated clash reports
✓ 4D Planning - Construction sequencing for MEP installation and core fit-out
✓ 5D Cost Management - Integrated cost model for value engineering
✓ Energy Analysis - IES-VE for net-zero carbon validation
✓ Quantity Take-off - For all major elements and cost planning
✓ Digital Twin - Complete asset data with IoT integration for FM
✓ Virtual Reality - Client design reviews and stakeholder engagement

These uses directly support our goals of design coordination, cost control, sustainability targets, and FM handover.`,

      'Infrastructure': `Selected BIM Uses for this bypass scheme:

✓ Design Authoring - Highway (12d), Structures (Tekla), Drainage (Civil 3D)
✓ 3D Coordination - Highway/structures/utilities clash detection
✓ Design Reviews - Federated model reviews with client and stakeholders
✓ 4D Planning - Construction sequencing and traffic management
✓ Quantity Take-off - Earthworks, structures, pavements, drainage
✓ Site Analysis - Environmental impact and constraints mapping
✓ Structural Analysis - Bridge design and analysis
✓ Construction Sequencing - Phasing plans and logistics
✓ Asset Management - As-built data for highway authority systems

Focus is on coordination, quantities, sequencing, and asset data delivery.`
    },

    commonMistakes: [
      'Selecting too many BIM uses without focus or justification',
      'Choosing uses that don\'t align with project goals or challenges',
      'Not considering team capability or software availability',
      'Missing critical uses for project type (e.g., clash detection for MEP-heavy projects)',
      'No explanation of how each use will be implemented',
      'Ignoring FM / asset management uses for projects requiring handover',
      'Selecting advanced uses (VR, digital fabrication) without proper justification',
      'Not prioritizing uses based on value delivery'
    ],

    relatedFields: ['bimGoals', 'primaryObjectives', 'bimStrategy']
  },

  // ====================================================================

  tenderApproach: {
    description: `Describe your proposed approach to delivering this project during the tender/pre-appointment phase. Explain your methodology, key strategies, and how you will meet the client's requirements.

Cover:
• Overall delivery philosophy and approach
• Key strategies (collaboration, early engagement, risk mitigation, etc.)
• How you will meet the client's specific requirements
• Phased delivery or staging approach if applicable
• Value engineering and optimization strategies
• Risk management approach
• Stakeholder engagement strategy
• Quality assurance processes

This is your chance to differentiate your approach from competitors.`,

    iso19650: `ISO 19650-2:2018 Section 5.1 - Appointment

The proposed approach should demonstrate understanding of the appointing party's requirements, project constraints, and information management expectations. It should show how the delivery team will establish capability and capacity.`,

    bestPractices: [
      'Lead with your core delivery philosophy (collaboration, innovation, quality, etc.)',
      'Emphasize early engagement and proactive coordination',
      'Highlight BIM as enabler for risk reduction and value delivery',
      'Mention phased approach aligned with RIBA stages or project phases',
      'Reference continuous value engineering throughout design',
      'Include stakeholder engagement and communication strategy',
      'Mention quality assurance and ISO 19650 compliance',
      'Address project-specific challenges or constraints',
      'Keep it client-focused - emphasize benefits to them'
    ],

    examples: {
      'Commercial Building': `Our approach emphasizes collaborative design coordination through advanced BIM workflows, early stakeholder engagement, and integrated sustainability analysis from the outset.

We propose a phased delivery strategy aligned with RIBA stages, with intensive coordination during Stage 3 to eliminate design conflicts before Stage 4 technical design. Continuous value engineering through 5D integration will identify cost savings opportunities while maintaining design quality and sustainability targets.

Our BIM-first approach enables early clash detection, reducing construction risks and ensuring predictable delivery. Weekly coordination meetings with all disciplines using federated models will maintain design quality and buildability. We will engage the contractor early (Stage 3) to validate constructability and optimize sequencing.

Risk management is embedded in our process through proactive clash detection, regular design reviews, and continuous stakeholder engagement. Quality is assured through ISO 19650-2:2018 compliance, automated model validation, and structured review gates at each RIBA stage milestone.`,

      'Infrastructure': `Our approach prioritizes early 3D design coordination to eliminate clashes between highway, structures, drainage, and utilities before detailed design, reducing construction risk and protecting the programme.

We will implement a staged delivery approach: Stage 1 - Options design with preliminary 3D models for stakeholder engagement; Stage 2 - Concept design with full 3D coordination; Stage 3 - Detailed design with clash-free construction documentation.

4D planning will be used from concept stage to optimize construction programme, traffic management, and stakeholder impacts. Early engagement with statutory undertakers will ensure utilities coordination is resolved during design, not on site.

Risk management focuses on early identification through 3D coordination, environmental analysis for sensitive areas, and stakeholder engagement for planning consent. Value engineering will be continuous through design development, supported by accurate quantity extraction from coordinated models.`
    },

    commonMistakes: [
      'Generic approach that could apply to any project',
      'Not addressing project-specific challenges or constraints',
      'Missing BIM integration in the delivery approach',
      'No mention of stakeholder engagement or communication',
      'Failing to connect approach to client\'s stated requirements',
      'Too much focus on process, not enough on value and outcomes',
      'Not explaining how risks will be managed',
      'Missing quality assurance and compliance commitments'
    ],

    relatedFields: ['bimStrategy', 'deliveryApproach', 'keyCommitments']
  },

  // ====================================================================

  deliveryApproach: {
    description: `Describe your confirmed approach to delivering this project during the post-appointment phase. Explain how you will execute the project, manage information, and deliver against the agreed commitments.

Cover:
• Delivery methodology and execution strategy
• Information management and CDE implementation
• Coordination processes and schedules
• Quality assurance and validation procedures
• Risk management and mitigation strategies
• Stakeholder engagement and communication protocols
• Phased delivery aligned with project programme
• Compliance with ISO 19650 and standards

Use confident, confirmed language - "We will..." not "We propose..."`,

    iso19650: `ISO 19650-2:2018 Section 5.2 - Mobilization

The delivery approach should confirm how information management capability will be mobilized, processes established, and information delivered throughout the appointment. It demonstrates readiness to execute against the agreed plan.`,

    bestPractices: [
      'Use confirmed language - "We will..." "Our team will..." "Implementation will..."',
      'Emphasize proven processes and established workflows',
      'Reference specific coordination schedules (weekly, bi-weekly)',
      'Mention CDE implementation and information management protocols',
      'Include quality gates and validation processes',
      'Address risk management and continuous improvement',
      'Reference ISO 19650 compliance and audit processes',
      'Explain phased delivery aligned with project programme',
      'Keep focus on execution, delivery, and outcomes'
    ],

    examples: {
      'Commercial Building': `Our delivery approach implements collaborative design coordination through established BIM workflows, stakeholder integration at defined milestones, and continuous value engineering throughout all project stages.

We will execute a phased delivery strategy aligned with RIBA stages and the agreed project programme. Stage 3 will focus on design development with bi-weekly coordination meetings, achieving 95% clash-free models before Stage 4. Stage 4 will deliver detailed technical design to LOD 350 with full MEP coordination and buildability validation.

BIM 360 CDE will be implemented within 2 weeks of appointment with role-based access, structured folders, and audit trails. Weekly clash detection will run throughout design with < 24hr resolution SLA for critical clashes. Monthly model quality validation using Solibri will ensure compliance with project standards.

Risk management is embedded through proactive clash detection, regular design reviews at each milestone, and continuous stakeholder engagement. Quality gates at Stage 3 and Stage 4 completion will ensure all deliverables meet ISO 19650-2:2018 requirements before progression.

Proactive risk management through early identification and mitigation will ensure on-time, on-budget completion. Our integrated sustainability analysis will validate net-zero carbon targets throughout design development.`,

      'Healthcare': `Our delivery approach for this hospital extension implements rigorous coordination processes tailored to the complexity of live hospital operations, stringent infection control requirements, and integration with existing building systems.

We will implement phased coordination models aligned with the construction programme, with room-by-room LOD 350 coordination for all clinical spaces. Weekly coordination meetings will focus on medical gases, critical ventilation, and ceiling void coordination, achieving zero clashes in critical clinical areas.

The CDE will be implemented using BIM 360 with the Trust's IT security requirements and NHS data standards. All clinical areas will undergo additional quality validation for HBN 04-01 compliance, including airflow analysis and room pressure validation.

4D models will be developed in coordination with the Trust's operational constraints, ensuring minimal disruption to clinical services. Construction sequencing will be validated with the Trust's clinical teams before any enabling works commence.

All information deliverables will be structured for integration with the Trust's CAFM system, with COBie data prepared progressively throughout design and validated before handover. O&M information will be linked to model components for immediate use by the FM team.`
    },

    commonMistakes: [
      'Using tentative language - "We propose..." instead of "We will..."',
      'Generic delivery approach not tailored to project specifics',
      'No mention of specific coordination schedules or frequencies',
      'Missing CDE implementation details',
      'No quality validation or compliance checking processes',
      'Failing to address project-specific constraints or challenges',
      'Not explaining phased delivery aligned with programme',
      'Missing risk management and continuous improvement processes'
    ],

    relatedFields: ['bimStrategy', 'tenderApproach', 'keyCommitments', 'informationManagementResponsibilities']
  }
};

export default HELP_CONTENT;
