// Project Information Help Content
export const projectInfoHelp = {
  projectName: {
    description: `The official name of the project as it will be referenced throughout all project documentation and correspondence.

Include:
• Clear, descriptive name that identifies the project
• Location if not obvious from the name
• Phase number if part of a larger development
• Avoid abbreviations that stakeholders may not understand

The project name should be consistent across all documentation including contracts, drawings, models, and correspondence.`,

    iso19650: `ISO 19650-2:2018 Section 5.1.2 - Project Information Requirements

The project name is a fundamental identifier that must be consistently used across all information containers and project deliverables. It forms part of the metadata structure that enables information management throughout the project lifecycle.`,

    bestPractices: [
      'Use a clear, descriptive name that stakeholders will immediately recognize',
      'Include location if managing multiple similar projects',
      'Add phase number if part of multi-phase development',
      'Avoid special characters that may cause issues in file systems',
      'Keep it concise but descriptive (typically 3-8 words)',
      'Establish naming convention early and document it',
      'Ensure consistency with contract documents',
      'Consider how the name will appear in file paths and databases'
    ],

    examples: {
      'Commercial': 'Manchester City Centre Office Tower Phase 2',
      'Residential': 'Greenwich Riverside Apartments Block A',
      'Infrastructure': 'M25 Junction 15 Improvement Scheme',
      'Healthcare': 'St Mary\'s Hospital Emergency Department Extension',
      'Education': 'Westfield Academy Sports Complex',
      'Mixed Use': 'Canary Wharf Station Quarter Development'
    },

    commonMistakes: [
      'Using project codes instead of meaningful names',
      'Including dates that will become outdated',
      'Too generic - "Office Building Project"',
      'Inconsistent naming across different documents',
      'Using internal codenames not understood by all stakeholders',
      'Too long or complex names that are difficult to reference'
    ],

    relatedFields: ['projectNumber', 'projectDescription', 'appointingParty']
  },

  projectNumber: {
    description: `A unique identifier or reference code assigned to the project for tracking and administrative purposes.

The project number should:
• Be unique within your organization's project portfolio
• Follow your organization's numbering convention
• Be easily referenced in databases and filing systems
• Remain consistent throughout the project lifecycle
• Be included in all official documentation

This number is used for financial tracking, document control, and project management systems.`,

    iso19650: `ISO 19650-2:2018 Section 5.3.1 - Information Container Identification

The project number forms part of the information container naming convention and is essential for organizing and retrieving project information. It should be incorporated into the file naming structure defined in the BEP.`,

    bestPractices: [
      'Follow your organization\'s established numbering system',
      'Include year if it helps with chronological organization',
      'Use leading zeros for numeric sequences (e.g., 001 instead of 1)',
      'Keep the format consistent across all projects',
      'Document the numbering convention in your BEP',
      'Ensure the number is assigned early and communicated to all parties',
      'Include the project number in all file naming conventions',
      'Register the project number in your PMO or central database'
    ],

    examples: {
      'Year-based': 'PRJ-2024-017',
      'Client-based': 'ABC-COM-024',
      'Location-based': 'LON-RES-15',
      'Sequential': '24-1547',
      'Department-based': 'ARCH-2024-32',
      'Combined': 'GLA-EDU-2024-05'
    },

    commonMistakes: [
      'Not assigning a project number early enough',
      'Using duplicate numbers across projects',
      'Changing the number mid-project',
      'Not communicating the number to all team members',
      'Using overly complex numbering that\'s hard to remember',
      'Not including the number in file naming conventions'
    ],

    relatedFields: ['projectName', 'fileNamingConvention']
  },

  projectType: {
    description: `Classification of the project by its primary use or sector. This helps set appropriate expectations for BIM requirements, standards, and complexity levels.

The project type influences:
• Applicable standards and regulations
• Level of detail requirements
• Typical BIM uses and applications
• Stakeholder expectations
• Delivery complexity
• Information requirements

Select the type that best represents the primary use or highest complexity aspect of the project.`,

    iso19650: `ISO 19650-2:2018 Section 5.1.2 - Appointing Party's Information Requirements

Project classification helps determine appropriate information requirements, level of information need, and delivery standards. Different project types may require specific industry standards (e.g., COBie for facilities management).`,

    bestPractices: [
      'Choose the primary project type if it\'s mixed-use',
      'Consider the dominant use by area or complexity',
      'Reference industry-specific standards for that project type',
      'Align LOIN requirements with typical sector expectations',
      'Consider regulatory requirements specific to the project type',
      'Use consistent classification across all project documentation',
      'Understand client\'s strategic objectives for that asset type',
      'Research benchmark BIM requirements for similar project types'
    ],

    examples: {
      'Commercial Building': 'Office towers, retail centers, hotels, warehouses',
      'Residential': 'Apartments, houses, student accommodation, senior living',
      'Infrastructure': 'Roads, bridges, tunnels, railways, utilities',
      'Healthcare': 'Hospitals, clinics, medical centers, care homes',
      'Education': 'Schools, universities, training centers, libraries',
      'Industrial': 'Factories, manufacturing plants, distribution centers',
      'Mixed Use': 'Combined residential/commercial, transit-oriented development',
      'Renovation/Retrofit': 'Building upgrades, heritage restoration, adaptive reuse'
    },

    commonMistakes: [
      'Selecting multiple types when one primary type should be chosen',
      'Not considering the complexity implications of the type',
      'Ignoring sector-specific BIM requirements',
      'Not aligning LOIN with industry expectations for that type',
      'Missing regulatory or compliance requirements specific to the type'
    ],

    relatedFields: ['projectDescription', 'bimUses', 'applicableStandards']
  },

  appointingParty: {
    description: `The organization or individual appointing the delivery team and receiving the information deliverables. This is typically the client or project owner who has issued the Exchange Information Requirements (EIR).

The appointing party:
• Sets the information requirements (EIR)
• Receives all project deliverables
• Makes key project decisions
• Provides access to project information
• Defines acceptance criteria
• Approves information delivery`,

    iso19650: `ISO 19650-2:2018 Section 3.2.1 - Appointing Party Definition

The appointing party is the receiver of information and is responsible for establishing the exchange information requirements (EIR), appointing the lead appointed party, and accepting information deliverables at project milestones.`,

    bestPractices: [
      'Use the full legal entity name as it appears in contracts',
      'Include department or division if applicable',
      'Confirm the name with contract documentation',
      'Identify the specific entity within larger organizations',
      'Document the appointing party\'s information manager contact',
      'Clarify decision-making authority levels',
      'Reference the EIR issued by this party',
      'Maintain consistency across all project documentation'
    ],

    examples: {
      'Private Developer': 'Greenfield Development Corporation Ltd',
      'Public Authority': 'Greater Manchester Combined Authority',
      'Government Department': 'Department for Education - Infrastructure Division',
      'Corporate Client': 'HSBC Bank PLC - UK Property Services',
      'Healthcare Trust': 'NHS Greater London Trust - Capital Projects',
      'University': 'University of Cambridge - Estates Management'
    },

    commonMistakes: [
      'Using informal names instead of legal entity names',
      'Not matching the name with contract documents',
      'Confusion between appointing party and end client',
      'Not identifying the specific division in large organizations',
      'Missing contact information for their information manager'
    ],

    relatedFields: ['projectName', 'leadAppointedParty', 'informationManager']
  },

  proposedTimeline: {
    description: `For Pre-Appointment BEP: Your proposed project schedule showing key phases, milestones, and anticipated information delivery dates.

The timeline should show:
• Major project phases (Design, Procurement, Construction, Handover)
• Key decision points and approvals
• Information delivery milestones
• BIM coordination events
• Critical path activities
• Review and approval periods`,

    iso19650: `ISO 19650-2:2018 Section 5.1.3 - Delivery Milestones

The timeline must align with information delivery milestones defined in the Master Information Delivery Plan (MIDP). It should demonstrate understanding of when information is needed to support project decisions.`,

    bestPractices: [
      'Use standard industry phase terminology (RIBA Plan of Work, etc.)',
      'Include buffer time for reviews and approvals',
      'Align with the client\'s strategic milestones',
      'Show coordination points between disciplines',
      'Include time for clash detection and resolution cycles',
      'Reference TIDP submission dates',
      'Consider procurement and long-lead items',
      'Build in contingency for complex coordination'
    ],

    examples: {
      'Office Building': 'Phase 1 Design (6 months), Phase 2 Tender (2 months), Phase 3 Construction (18 months), Phase 4 Handover (2 months)',
      'Infrastructure': 'Design Development (12 months), Procurement (4 months), Construction Phase 1 (24 months), Construction Phase 2 (18 months)',
      'Fast-track Project': 'Design-Build Overlap: Design (months 1-8), Early Works (months 4-6), Main Construction (months 6-20)'
    },

    commonMistakes: [
      'Overly optimistic timelines with no contingency',
      'Not aligning with client\'s strategic dates',
      'Missing key coordination milestones',
      'No time allocated for approval cycles',
      'Not considering design freeze dates',
      'Ignoring procurement lead times'
    ],

    relatedFields: ['confirmedTimeline', 'keyMilestones', 'informationDeliveryDates']
  },

  confirmedTimeline: {
    description: `For Post-Appointment BEP: The confirmed, agreed project schedule with actual dates for all phases, milestones, and information delivery requirements.

This timeline includes:
• Agreed start and completion dates
• Confirmed phase boundaries
• Contractual milestones and dates
• Information delivery schedule aligned with MIDP
• Coordination meeting schedule
• Review and approval periods with actual dates`,

    iso19650: `ISO 19650-2:2018 Section 5.4.2 - Delivery Team Mobilization Plan

The confirmed timeline forms part of the mobilization plan and establishes when the delivery team will produce information. It must align with the Master Information Delivery Plan and contract dates.`,

    bestPractices: [
      'Ensure alignment with contract documents',
      'Lock down key milestone dates as fixed',
      'Include actual calendar dates, not just durations',
      'Coordinate with all appointed parties on their submission dates',
      'Build in realistic review periods',
      'Allow float for complex coordination activities',
      'Reference specific TIDP submission dates',
      'Document assumptions and dependencies',
      'Plan for regular updates as project progresses'
    ],

    examples: {
      'Detailed Schedule': 'Stage 3 Complete: 15/03/2025; Stage 4 Complete: 30/06/2025; Construction Start: 15/08/2025; Practical Completion: 30/11/2026',
      'Phased Delivery': 'Phase 1A (Blocks 1-3): Complete Q2 2025; Phase 1B (Block 4-6): Complete Q4 2025; Phase 2: Complete Q3 2026'
    },

    commonMistakes: [
      'Not updating the timeline as the project progresses',
      'Dates that don\'t align with contract milestones',
      'Missing dependencies between tasks',
      'No contingency for delays or changes',
      'Information delivery dates that aren\'t realistic',
      'Not coordinating dates across all appointed parties'
    ],

    relatedFields: ['proposedTimeline', 'keyMilestones', 'midpDates', 'tidpDates']
  },

  estimatedBudget: {
    description: `For Pre-Appointment BEP: The anticipated or target project budget that will influence BIM implementation costs, resource allocation, and value engineering opportunities.

Consider including:
• Total project value or construction cost
• Budget range if exact figure is not available
• Currency and basis (e.g., Q1 2024 prices)
• Whether the budget includes professional fees
• BIM-specific budget allocation if known
• Technology investment budget`,

    iso19650: `ISO 19650-2:2018 Section 5.1.5 - Resource Allocation

Budget information helps demonstrate that appropriate resources will be allocated for information management. The BEP should show how BIM investment delivers value relative to project scale.`,

    bestPractices: [
      'Use ranges if exact figures are commercially sensitive',
      'Specify the currency and price base',
      'Clarify what\'s included (construction only vs. total project cost)',
      'Mention budget for BIM/technology if separately allocated',
      'Reference budget for specific BIM uses (clash detection, 4D, etc.)',
      'Show how BIM will support value engineering within budget',
      'Consider budget for staff training and upskilling',
      'Account for CDE and software licensing costs'
    ],

    examples: {
      'Range Format': '£45-50 million (Construction Cost, Q2 2024 prices, excluding VAT)',
      'Detailed': '£125 million total project value including £95M construction, £18M professional fees, £12M client costs',
      'With BIM Budget': '£67M project value with £350K allocated for BIM coordination and technology'
    },

    commonMistakes: [
      'Being too vague - "large budget"',
      'Not specifying currency or price base',
      'Unclear what\'s included in the figure',
      'No consideration of BIM-specific costs',
      'Not showing how BIM delivers value at this budget level',
      'Budget figures that don\'t align with project scope'
    ],

    relatedFields: ['confirmedBudget', 'projectDescription', 'bimUses', 'valueProposition']
  },

  confirmedBudget: {
    description: `For Post-Appointment BEP: The contractually agreed project budget with confirmed allocation for BIM activities, technology, and resources.

Document:
• Total confirmed project budget
• Budget allocated for information management
• Technology and software costs
• Training and competency development budget
• CDE and collaboration platform costs
• Contingency for BIM coordination activities
• Resource allocation for information management roles`,

    iso19650: `ISO 19650-2:2018 Section 5.4.3 - Mobilization of Resources and Systems

The confirmed budget must demonstrate adequate resource allocation for the information management function, including personnel, technology, and training required to meet the appointing party's requirements.`,

    bestPractices: [
      'Show clear breakdown of BIM-related costs',
      'Demonstrate value-for-money in BIM investment',
      'Include costs for all required software licenses',
      'Budget for information manager time allocation',
      'Include costs for coordination meetings and workshops',
      'Allow contingency for additional coordination needs',
      'Document cost-benefit analysis for major BIM uses',
      'Show how budget supports quality information delivery',
      'Include costs for model validation and checking tools'
    ],

    examples: {
      'Detailed Breakdown': '£82M Total Budget: £75M construction, £5.2M professional fees, £1.2M BIM coordination (software, CDE, training, IM time), £600K contingency',
      'BIM Investment': 'Information management budget: £450K (0.6% of project value) covering software (£120K), CDE (£80K), coordination (£150K), training (£50K), contingency (£50K)'
    },

    commonMistakes: [
      'No specific allocation shown for BIM activities',
      'Underestimating software and license costs',
      'Not budgeting for training and competency development',
      'Missing CDE subscription or hosting costs',
      'No contingency for coordination challenges',
      'Not showing cost-benefit justification for BIM investment'
    ],

    relatedFields: ['estimatedBudget', 'valueProposition', 'softwareTools', 'trainingRequirements']
  }
};
