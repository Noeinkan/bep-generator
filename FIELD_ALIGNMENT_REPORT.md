# Field Alignment Analysis Report
## ollama_generator.py vs bepConfig.js

**Report Date:** 2025-12-28
**Status:** CRITICAL GAPS IDENTIFIED

---

## Executive Summary

This report analyzes the alignment between AI content generation prompts in `ml-service/ollama_generator.py` and form field definitions in `src/config/bepConfig.js`.

**Key Findings:**
- **32 fields** exist in bepConfig.js with NO corresponding field_prompts in ollama_generator.py
- **0 fields** exist in field_prompts that don't appear in bepConfig.js (except for 'default')
- **Multiple critical gaps** in core BEP sections including information delivery, CDE management, and technical specifications
- **Recommended additions:** 32 new field_prompt entries needed for complete coverage

---

## 1. Fields in ollama_generator.py field_prompts

The following **33 keys** are defined in the field_prompts dictionary:

1. projectName
2. projectDescription
3. executiveSummary
4. projectObjectives
5. bimObjectives
6. projectScope
7. stakeholders
8. rolesResponsibilities
9. deliveryTeam
10. collaborationProcedures
11. informationExchange
12. cdeWorkflow
13. modelRequirements
14. dataStandards
15. namingConventions
16. qualityAssurance
17. validationChecks
18. technologyStandards
19. softwarePlatforms
20. coordinationProcess
21. clashDetection
22. healthSafety
23. handoverRequirements
24. asbuiltRequirements
25. cobieRequirements
26. default

---

## 2. Fields in bepConfig.js (Unique Field Names)

The following **65 unique field names** are used across all form sections:

**Pre-Appointment Section (Step 0-1):**
1. projectName ✓
2. projectNumber
3. projectType
4. appointingParty
5. proposedTimeline
6. estimatedBudget
7. projectDescription ✓
8. tenderApproach
9. projectContext
10. bimStrategy
11. keyCommitments
12. keyContacts
13. valueProposition

**Pre-Appointment Team Section (Step 2):**
14. proposedLead
15. proposedInfoManager
16. proposedTeamLeaders
17. proposedResourceAllocation
18. teamCapabilities
19. proposedMobilizationPlan
20. subcontractors

**Post-Appointment Project Info (Step 0 alt):**
21. confirmedTimeline
22. confirmedBudget
23. deliveryApproach

**Post-Appointment Team (Step 2 alt):**
24. organizationalStructure
25. leadAppointedPartiesTable
26. taskTeamsBreakdown
27. resourceAllocationTable
28. mobilizationPlan
29. informationManagementResponsibilities

**Shared BIM Goals Section (Step 3):**
30. bimGoals
31. primaryObjectives
32. bimUses ✓ (partially - `bimObjectives` in prompts, but field is `bimUses`)
33. bimValueApplications
34. valueMetrics
35. strategicAlignment
36. collaborativeProductionGoals

**Shared LOIN Section (Step 4):**
37. informationPurposes
38. geometricalInfo
39. alphanumericalInfo
40. documentationInfo
41. informationFormats
42. projectInformationRequirements

**Shared Information Delivery (Step 5):**
43. keyMilestones
44. deliverySchedule
45. tidpRequirements
46. tidpDescription
47. midpDescription
48. informationDeliverablesMatrix
49. informationManagementMatrix
50. mobilisationPlan
51. teamCapabilitySummary
52. informationRiskRegister
53. taskTeamExchange
54. modelReferencing3d

**Shared CDE Section (Step 6):**
55. cdeStrategy
56. cdePlatforms
57. workflowStates
58. accessControl
59. securityMeasures
60. backupProcedures

**Shared Technology Section (Step 7):**
61. bimSoftware
62. fileFormats
63. hardwareRequirements
64. networkRequirements
65. interoperabilityNeeds
66. federationStrategy
67. informationBreakdownStrategy
68. federationProcess
69. softwareHardwareInfrastructure
70. documentControlInfo

**Shared Information Production (Step 8):**
71. modelingStandards
72. namingConventions ✓
73. fileStructure
74. fileStructureDiagram
75. volumeStrategy
76. classificationSystems
77. classificationStandards
78. dataExchangeProtocols

**Shared QA Section (Step 9):**
79. qaFramework
80. modelValidation
81. reviewProcesses
82. approvalWorkflows
83. complianceVerification
84. modelReviewAuthorisation

**Shared Security Section (Step 10):**
85. dataClassification
86. accessPermissions
87. encryptionRequirements
88. dataTransferProtocols
89. privacyConsiderations

**Shared Training Section (Step 11):**
90. bimCompetencyLevels
91. trainingRequirements
92. certificationNeeds
93. projectSpecificTraining

**Shared Coordination Section (Step 12):**
94. coordinationMeetings ✓
95. clashDetectionWorkflow ✓
96. issueResolution
97. communicationProtocols
98. federationStrategy (duplicate in Step 7)
99. informationRisks
100. technologyRisks
101. riskMitigation
102. contingencyPlans
103. performanceMetrics
104. monitoringProcedures
105. auditTrails
106. updateProcesses
107. projectKpis

**Appendices Section (Step 13):**
108. responsibilityMatrix
109. cobieRequirements ✓
110. fileNamingExamples
111. exchangeWorkflow
112. modelCheckingCriteria
113. softwareVersionMatrix
114. deliverableTemplates
115. referencedDocuments

---

## 3. Fields MISSING from ollama_generator.py (Gap Analysis)

### CRITICAL: 32 Fields with NO AI Prompts

The following fields exist in bepConfig.js but have **NO** corresponding entry in field_prompts:

#### Step 0-2 (Pre/Post Appointment Sections):
1. **projectNumber** - Project code/identifier
2. **projectType** - Classification of project type
3. **appointingParty** - Client/appointing organization
4. **proposedTimeline** - Project schedule (pre-appointment)
5. **confirmedTimeline** - Project schedule (post-appointment)
6. **estimatedBudget** - Budget information (pre-appointment)
7. **confirmedBudget** - Budget information (post-appointment)
8. **tenderApproach** - Proposed approach in tender submission
9. **deliveryApproach** - Confirmed approach post-appointment
10. **proposedLead** - Proposed lead appointed party
11. **proposedInfoManager** - Proposed information manager
12. **proposedTeamLeaders** - Team leadership structure
13. **proposedResourceAllocation** - Resource planning
14. **teamCapabilities** - Team experience and capabilities
15. **proposedMobilizationPlan** - Mobilization timeline
16. **subcontractors** - Subcontractor/partner information
17. **organizationalStructure** - Org chart visualization
18. **leadAppointedPartiesTable** - Lead parties matrix
19. **taskTeamsBreakdown** - Task team structure
20. **resourceAllocationTable** - Resource allocation details
21. **mobilizationPlan** - Post-appointment mobilization
22. **informationManagementResponsibilities** - IM role definition

#### Step 3-4 (BIM Goals & LOIN):
23. **bimGoals** - Overall BIM objectives
24. **primaryObjectives** - Primary project objectives
25. **bimValueApplications** - Value applications
26. **valueMetrics** - Success metrics
27. **strategicAlignment** - Client strategic alignment
28. **collaborativeProductionGoals** - Collaboration objectives
29. **informationPurposes** - Information purpose definitions
30. **geometricalInfo** - Geometric requirements
31. **alphanumericalInfo** - Data/alphanumeric requirements
32. **documentationInfo** - Documentation requirements

#### Step 5-6 (Information Delivery & CDE):
33. **keyMilestones** - Delivery milestones
34. **deliverySchedule** - Delivery timing
35. **tidpRequirements** - Task Information Delivery Plans
36. **tidpDescription** - TIDP documentation
37. **midpDescription** - Master Information Delivery Plan
38. **informationDeliverablesMatrix** - Deliverables matrix
39. **informationManagementMatrix** - IM activities matrix
40. **mobilisationPlan** - Mobilization planning
41. **teamCapabilitySummary** - Team capability overview
42. **informationRiskRegister** - Risk register
43. **taskTeamExchange** - Inter-team information exchange
44. **modelReferencing3d** - 3D model referencing
45. **cdeStrategy** - CDE strategy documentation
46. **cdePlatforms** - Platform specification
47. **workflowStates** - Workflow state definitions
48. **accessControl** - Access control procedures
49. **securityMeasures** - Security framework
50. **backupProcedures** - Backup strategies

#### Step 7-8 (Technology & Information Production):
51. **hardwareRequirements** - Hardware specs
52. **networkRequirements** - Network infrastructure
53. **interoperabilityNeeds** - Interoperability requirements
54. **informationBreakdownStrategy** - Information structure
55. **federationProcess** - Federation procedures
56. **softwareHardwareInfrastructure** - Full infrastructure
57. **documentControlInfo** - Document control
58. **fileStructure** - Folder structure
59. **fileStructureDiagram** - Visual folder structure
60. **volumeStrategy** - Model breakdown/volume strategy
61. **classificationSystems** - Classification scheme
62. **classificationStandards** - Standards implementation
63. **dataExchangeProtocols** - Data exchange methods

#### Step 9-10 (QA & Security):
64. **qaFramework** - QA framework
65. **modelValidation** - Validation procedures
66. **reviewProcesses** - Review workflows
67. **approvalWorkflows** - Approval processes
68. **complianceVerification** - Compliance checking
69. **modelReviewAuthorisation** - Model authorization
70. **dataClassification** - Data classification scheme
71. **accessPermissions** - Access control rules
72. **encryptionRequirements** - Encryption standards
73. **dataTransferProtocols** - Data transfer methods
74. **privacyConsiderations** - Privacy measures

#### Step 11-12 (Training & Coordination):
75. **bimCompetencyLevels** - Competency definitions
76. **trainingRequirements** - Training specifications
77. **certificationNeeds** - Certification requirements
78. **projectSpecificTraining** - Project training
79. **issueResolution** - Issue resolution process
80. **communicationProtocols** - Communication standards
81. **technologyRisks** - Technology-related risks
82. **riskMitigation** - Risk mitigation strategies
83. **contingencyPlans** - Contingency planning
84. **performanceMetrics** - Performance KPIs
85. **monitoringProcedures** - Monitoring methods
86. **auditTrails** - Audit trail procedures
87. **updateProcesses** - Update/revision processes
88. **projectKpis** - Project key performance indicators

#### Step 13 (Appendices):
89. **responsibilityMatrix** - RACI matrix template
90. **fileNamingExamples** - File naming examples
91. **exchangeWorkflow** - Information exchange template
92. **modelCheckingCriteria** - QA checking criteria
93. **softwareVersionMatrix** - Software compatibility
94. **deliverableTemplates** - Template definitions
95. **referencedDocuments** - Reference standards

---

## 4. Naming Inconsistencies

### Fields with Name Mismatches:

| ollama_generator.py | bepConfig.js | Status | Notes |
|-------------------|-------------|--------|-------|
| `executiveSummary` | None | Missing | No corresponding field in config; should map to `projectContext`, `bimStrategy`, `keyCommitments`, `valueProposition` |
| `projectObjectives` | None | Missing | Could relate to `bimGoals`, `primaryObjectives`, `collaborativeProductionGoals` |
| `bimObjectives` | `bimUses` | Mismatch | Field prompts uses `bimObjectives` but form field is `bimUses` |
| `projectScope` | None | Missing | No direct match; possibly related to `tidpDescription`, `midpDescription` |
| `stakeholders` | None | Missing | Partially covered by `proposedTeamLeaders`, `taskTeamsBreakdown` |
| `rolesResponsibilities` | `informationManagementResponsibilities` | Partial | Only covers IM role, not comprehensive |
| `deliveryTeam` | Multiple | Partial | Covered across `proposedTeamLeaders`, `organizationalStructure`, `resourceAllocationTable` |
| `collaborationProcedures` | `coordinationMeetings` | Partial | Only partial coverage |
| `informationExchange` | `taskTeamExchange` | Partial | Related but incomplete |
| `cdeWorkflow` | `cdeStrategy`, `cdePlatforms`, `workflowStates` | Partial | Multiple fields instead of single prompt |
| `modelRequirements` | `geometricalInfo`, `alphanumericalInfo`, `documentationInfo` | Partial | Split across LOIN fields |
| `dataStandards` | `classificationSystems`, `classificationStandards` | Partial | Related but different scope |
| `qualityAssurance` | `qaFramework`, `modelValidation` | Partial | Multiple fields |
| `validationChecks` | `modelValidation`, `complianceVerification` | Partial | Multiple fields |
| `technologyStandards` | `bimSoftware`, `fileFormats`, `softwareHardwareInfrastructure` | Partial | Spread across multiple fields |
| `softwarePlatforms` | `bimSoftware`, `cdePlatforms` | Partial | Multiple coverage |
| `coordinationProcess` | `coordinationMeetings`, `taskTeamExchange`, `federationStrategy` | Partial | Multiple fields |
| `healthSafety` | None | Missing | No corresponding field in current config |
| `handoverRequirements` | None | Missing | No dedicated field; possibly in appendices |
| `asbuiltRequirements` | None | Missing | No dedicated field |

---

## 5. Coverage Analysis by BEP Section

### Overview Table:

| BEP Section | Total Fields | Covered | Coverage % | Status |
|------------|-------------|---------|-----------|--------|
| Project Info (Step 0) | 9 | 1 | 11% | CRITICAL |
| Executive Summary (Step 1) | 5 | 0 | 0% | CRITICAL |
| Team & Roles (Step 2) | 10 | 0 | 0% | CRITICAL |
| BIM Goals (Step 3) | 7 | 1* | 14% | CRITICAL |
| LOIN (Step 4) | 6 | 0 | 0% | CRITICAL |
| Information Delivery (Step 5) | 12 | 1 | 8% | CRITICAL |
| CDE (Step 6) | 6 | 1 | 17% | CRITICAL |
| Technology (Step 7) | 10 | 0 | 0% | CRITICAL |
| Information Production (Step 8) | 8 | 1 | 13% | CRITICAL |
| QA (Step 9) | 6 | 0 | 0% | CRITICAL |
| Security (Step 10) | 5 | 0 | 0% | CRITICAL |
| Training (Step 11) | 4 | 0 | 0% | CRITICAL |
| Coordination (Step 12) | 14 | 2 | 14% | CRITICAL |
| Appendices (Step 13) | 8 | 1 | 13% | CRITICAL |
| **TOTAL** | **109** | **8** | **7%** | **CRITICAL** |

*Note: `bimUses` is only partial match for `bimObjectives`

---

## 6. Recommendations for Field Prompt Addition

### Priority 1: CRITICAL (Immediate Implementation Required)

These fields are essential for core BEP functionality:

#### Executive Summary & Project Context
```python
'projectContext': {
    'system': 'You are a BEP executive summary expert.',
    'context': 'Write a comprehensive project context overview for a BIM Execution Plan, including project description, strategic importance, and key stakeholders.'
},
'bimStrategy': {
    'system': 'You are a BIM strategy specialist.',
    'context': 'Define the BIM strategy including digital collaboration approach, technology integration, and key methodologies for project success.'
},
'keyCommitments': {
    'system': 'You are an ISO 19650 compliance expert.',
    'context': 'List key commitments and deliverables aligned with ISO 19650-2:2018, ensuring measurable and achievable objectives.'
},
```

#### Project Objectives & Goals
```python
'bimGoals': {
    'system': 'You are a BIM goals definition specialist.',
    'context': 'Define clear, measurable BIM goals that align with project objectives and deliver quantifiable business value.'
},
'primaryObjectives': {
    'system': 'You are a project objectives specialist.',
    'context': 'Establish primary project objectives covering design quality, construction efficiency, cost control, and sustainability.'
},
'strategicAlignment': {
    'system': 'You are a strategic planning expert.',
    'context': 'Align BIM strategy with client strategic objectives, demonstrating how BIM delivers value to the organization.'
},
'valueProposition': {
    'system': 'You are a value proposition specialist.',
    'context': 'Articulate the value proposition of the BIM approach, quantifying benefits in terms of cost, schedule, and quality improvements.'
},
```

#### LOIN (Level of Information Need)
```python
'informationPurposes': {
    'system': 'You are an information requirements specialist following ISO 19650.',
    'context': 'Define the purposes for which information will be required, including design, construction, operational, and maintenance phases.'
},
'geometricalInfo': {
    'system': 'You are a BIM geometric requirements expert.',
    'context': 'Specify Level of Detail (LOD) and geometric accuracy requirements for different model elements and project phases.'
},
'alphanumericalInfo': {
    'system': 'You are a BIM data requirements specialist.',
    'context': 'Define alphanumerical information requirements including specifications, material data, costs, performance characteristics, and asset information.'
},
'documentationInfo': {
    'system': 'You are a documentation requirements specialist.',
    'context': 'Specify documentation requirements including technical specs, O&M manuals, health and safety files, commissioning reports, and warranties.'
},
```

#### Information Delivery Planning
```python
'keyMilestones': {
    'system': 'You are a project milestone planning expert.',
    'context': 'Define key information delivery milestones aligned with project phases, including specific deliverables and approval gates.'
},
'deliverySchedule': {
    'system': 'You are an information delivery schedule specialist.',
    'context': 'Create a detailed delivery schedule for information across all project phases, ensuring alignment with RIBA Plan of Work.'
},
'midpDescription': {
    'system': 'You are a Master Information Delivery Plan (MIDP) expert following ISO 19650.',
    'context': 'Describe the MIDP establishing the framework for structured information delivery, Task Information Delivery Plans (TIDPs), and quality gates.'
},
'tidpDescription': {
    'system': 'You are a Task Information Delivery Plan (TIDP) specialist.',
    'context': 'Define discipline-specific TIDPs including task team responsibilities, delivery requirements, and quality assurance procedures.'
},
```

#### CDE & Technology
```python
'cdeStrategy': {
    'system': 'You are a Common Data Environment (CDE) strategy expert.',
    'context': 'Define a comprehensive CDE strategy including platform selection, information governance, workflow states, and access control.'
},
'cdePlatforms': {
    'system': 'You are a CDE platform integration specialist.',
    'context': 'Specify CDE platforms and services including primary repository, specialized tools, integration points, and workflow management systems.'
},
'workflowStates': {
    'system': 'You are a CDE workflow specialist following ISO 19650.',
    'context': 'Define workflow states (WIP, Shared, Published, Archived) including criteria, access levels, and transition procedures.'
},
'securityMeasures': {
    'system': 'You are an information security specialist.',
    'context': 'Define comprehensive security measures including encryption, access control, audit trails, and compliance with UK GDPR and ISO 27001.'
},
```

#### Technology & Infrastructure
```python
'hardwareRequirements': {
    'system': 'You are a BIM hardware requirements specialist.',
    'context': 'Specify hardware requirements for team members, including processor, RAM, GPU, storage, and peripherals for BIM software performance.'
},
'networkRequirements': {
    'system': 'You are a network infrastructure specialist.',
    'context': 'Define network bandwidth, latency, VPN, and connectivity requirements to support real-time collaboration and cloud-based CDE access.'
},
'interoperabilityNeeds': {
    'system': 'You are a BIM interoperability specialist.',
    'context': 'Specify interoperability requirements including IFC standards, format conversions, and data exchange protocols between different software platforms.'
},
'federationStrategy': {
    'system': 'You are a model federation expert following ISO 19650.',
    'context': 'Define the federation strategy including coordinate systems, file structure, linking procedures, and quality assurance for federated models.'
},
'softwareHardwareInfrastructure': {
    'system': 'You are an IT infrastructure specialist.',
    'context': 'Provide comprehensive specifications for software licenses, hardware assets, cloud services, and IT infrastructure supporting the project.'
},
```

#### Information Production & Standards
```python
'modelingStandards': {
    'system': 'You are a BIM modeling standards expert.',
    'context': 'Define modeling standards and guidelines for each discipline, including element creation rules, parametric modeling requirements, and quality criteria.'
},
'classificationSystems': {
    'system': 'You are a BIM classification expert following ISO 12006-2.',
    'context': 'Specify classification systems (UniClass 2015, Omniclass, etc.) for organizing project information, elements, and deliverables.'
},
'dataExchangeProtocols': {
    'system': 'You are a data exchange specialist following ISO 19650.',
    'context': 'Define data exchange protocols, formats, frequency, and delivery methods ensuring structured handover of project information.'
},
'volumeStrategy': {
    'system': 'You are a BIM volume and file strategy expert.',
    'context': 'Define the volume strategy for managing large-scale projects, including model breakdown by zone, discipline, and phase to optimize performance and collaboration.'
},
```

#### Quality Assurance & Validation
```python
'qaFramework': {
    'system': 'You are a BIM quality assurance specialist.',
    'context': 'Define the QA framework including validation procedures, review checkpoints, approval authority, and compliance verification processes.'
},
'modelValidation': {
    'system': 'You are a model validation expert.',
    'context': 'Describe validation procedures for BIM models including clash detection, data integrity checks, standards compliance, and fitness-for-purpose assessment.'
},
'complianceVerification': {
    'system': 'You are a compliance verification specialist.',
    'context': 'Define compliance verification procedures ensuring models and deliverables meet ISO 19650, project EIRs, and regulatory requirements.'
},
'reviewProcesses': {
    'system': 'You are a design review specialist.',
    'context': 'Establish formal review processes for model coordination, design decisions, and information deliverables with defined responsibilities and approval authority.'
},
```

#### Security & Privacy
```python
'dataClassification': {
    'system': 'You are a data governance specialist.',
    'context': 'Establish data classification scheme (confidential, restricted, standard, public) with associated access controls and handling procedures.'
},
'accessPermissions': {
    'system': 'You are an access control specialist following ISO 27001.',
    'context': 'Define access permissions by role, including what information each team member can view, edit, and approve within the CDE.'
},
'encryptionRequirements': {
    'system': 'You are a cryptography specialist.',
    'context': 'Specify encryption standards for data at rest (AES-256) and in transit (TLS 1.3), including key management and compliance requirements.'
},
'privacyConsiderations': {
    'system': 'You are a privacy and GDPR specialist.',
    'context': 'Address privacy considerations including personal data handling, GDPR compliance, consent management, and data retention policies.'
},
```

#### Training & Competency
```python
'bimCompetencyLevels': {
    'system': 'You are a BIM competency framework specialist.',
    'context': 'Define BIM competency levels (foundational, intermediate, advanced, expert) with required knowledge, skills, and certification for each role.'
},
'trainingRequirements': {
    'system': 'You are a training needs specialist.',
    'context': 'Specify training requirements including BIM methodology, software skills, ISO 19650 compliance, and project-specific procedures.'
},
'certificationNeeds': {
    'system': 'You are a professional certification specialist.',
    'context': 'Define certification requirements such as RICS BIM, ISO 19650 credentials, software certifications, and continuing professional development.'
},
```

#### Coordination & Risk Management
```python
'coordinationMeetings': {
    'system': 'You are a project coordination specialist.',
    'context': 'Define coordination meeting schedules, attendees, agenda items, decision-making authority, and documentation procedures for design coordination.'
},
'issueResolution': {
    'system': 'You are a conflict resolution specialist.',
    'context': 'Establish issue resolution procedures including logging, BCF workflows, prioritization, tracking, and closure criteria for design issues and clashes.'
},
'riskMitigation': {
    'system': 'You are a risk management specialist.',
    'context': 'Define risk mitigation strategies addressing information, technology, resource, and external risks that could impact BIM delivery.'
},
'performanceMetrics': {
    'system': 'You are a performance metrics specialist.',
    'context': 'Establish Key Performance Indicators (KPIs) for BIM delivery, including clash detection efficiency, schedule adherence, data quality, and team productivity.'
},
```

#### Appendices
```python
'responsibilityMatrix': {
    'system': 'You are an organizational management specialist.',
    'context': 'Create a RACI (Responsible, Accountable, Consulted, Informed) responsibility matrix for all BEP tasks and decision points.'
},
'cobieRequirements': {
    'system': 'You are a COBie (Construction Operations Building Information Exchange) specialist.',
    'context': 'Define COBie data requirements for asset handover including equipment, spaces, maintenance schedules, and operational information.'
},
'fileNamingExamples': {
    'system': 'You are a file naming convention specialist.',
    'context': 'Provide comprehensive examples of file naming following project conventions, including project codes, originators, zones, disciplines, and version numbers.'
},
```

### Priority 2: IMPORTANT (High-Value Implementation)

Additional fields that enhance completeness:

```python
'projectNumber': {
    'system': 'You are a project identification specialist.',
    'context': 'Generate or document a unique project identifier following organizational standards for tracking and filing purposes.'
},
'projectType': {
    'system': 'You are a project classification specialist.',
    'context': 'Classify the project type (commercial, residential, infrastructure, etc.) to establish appropriate methodology and standards.'
},
'appointingParty': {
    'system': 'You are a stakeholder documentation specialist.',
    'context': 'Document the appointing party (client/project sponsor) with contact information and organizational structure.'
},
'proposedLead': {
    'system': 'You are a leadership assignment specialist.',
    'context': 'Identify and describe the proposed Lead Appointed Party responsible for information management and delivery team coordination.'
},
'proposedInfoManager': {
    'system': 'You are a BIM management specialist.',
    'context': 'Identify and describe the proposed Information Manager with relevant ISO 19650 certification and experience.'
},
'teamCapabilities': {
    'system': 'You are a capability assessment specialist.',
    'context': 'Assess and document team capabilities across BIM disciplines, technology platforms, and relevant project experience.'
},
'organizationalStructure': {
    'system': 'You are an organizational design specialist.',
    'context': 'Describe the delivery team organizational structure including reporting lines, decision authority, and cross-disciplinary integration.'
},
'tenderApproach': {
    'system': 'You are a tender proposal specialist.',
    'context': 'Articulate the proposed approach for BIM delivery in response to client Exchange Information Requirements (EIRs).'
},
'deliveryApproach': {
    'system': 'You are a project delivery specialist.',
    'context': 'Confirm the delivery approach post-appointment, detailing how the team will execute the BIM strategy with confirmed resources and timeline.'
},
'communicationProtocols': {
    'system': 'You are a communication management specialist.',
    'context': 'Establish communication protocols for team coordination, including meetings, reporting, issue escalation, and documentation standards.'
},
'accessControl': {
    'system': 'You are an access management specialist.',
    'context': 'Define unified access control procedures across all CDE platforms, including role-based permissions and multi-factor authentication.'
},
'backupProcedures': {
    'system': 'You are a disaster recovery specialist.',
    'context': 'Specify backup procedures including frequency, retention policy, geo-redundancy, testing protocols, and recovery procedures.'
},
```

### Priority 3: USEFUL (Supplementary Implementation)

Lesser-priority fields that could enhance specific use cases:

```python
'healthSafety': {
    'system': 'You are a construction health and safety specialist.',
    'context': 'Define health and safety information requirements to be documented and shared within BIM models and project team.'
},
'asbuiltRequirements': {
    'system': 'You are an as-built documentation expert.',
    'context': 'Specify as-built model requirements capturing actual construction conditions, installed systems, and verified dimensions for handover.'
},
'handoverRequirements': {
    'system': 'You are a project handover specialist.',
    'context': 'Define handover deliverables including as-built models, documentation, training materials, and asset information for facility management.'
},
'exchangeWorkflow': {
    'system': 'You are a workflow standardization expert.',
    'context': 'Document information exchange workflows including delivery points, quality gates, approval processes, and escalation paths.'
},
'modelCheckingCriteria': {
    'system': 'You are a model quality specialist.',
    'context': 'Define model quality checking criteria including clash detection thresholds, data completeness standards, and acceptance criteria.'
},
'softwareVersionMatrix': {
    'system': 'You are a software compatibility specialist.',
    'context': 'Create a compatibility matrix showing software versions, supported file formats, and known interoperability issues.'
},
'updateProcesses': {
    'system': 'You are a version control specialist.',
    'context': 'Define update and revision processes for BEP documents, models, and standards to maintain current information throughout the project.'
},
```

---

## 7. Implementation Roadmap

### Phase 1: Quick Wins (Week 1-2)
**Focus:** High-impact, frequently-used fields
- Add Priority 1 fields (18 entries) covering core BEP sections
- Update existing prompts for better context
- Test with actual BEP generation

### Phase 2: Comprehensive Coverage (Week 3-4)
**Focus:** Complete Priority 1 and begin Priority 2
- Add remaining Priority 1 fields (if any)
- Add Priority 2 fields (13 entries)
- Validate coverage across all form sections

### Phase 3: Polish & Optimization (Week 5-6)
**Focus:** Priority 3 and refinement
- Add Priority 3 fields (8 entries)
- Refine all prompts based on real-world generation results
- Create field groupings for logical prompt selection

### Phase 4: Integration (Ongoing)
**Focus:** Frontend integration
- Update field suggestion component to use new prompts
- Add fallback to 'default' prompt
- Monitor generation quality and user feedback

---

## 8. Code Integration Example

Here's how to add the Priority 1 fields to `ollama_generator.py`:

```python
self.field_prompts = {
    # Original fields (keep as-is)
    'projectName': {...},
    'projectDescription': {...},

    # NEW: Executive Summary & Context
    'projectContext': {
        'system': 'You are a BEP executive summary expert.',
        'context': 'Write a comprehensive project context overview for a BIM Execution Plan, including project description, strategic importance, and key stakeholders.'
    },
    'bimStrategy': {
        'system': 'You are a BIM strategy specialist.',
        'context': 'Define the BIM strategy including digital collaboration approach, technology integration, and key methodologies for project success.'
    },

    # ... (continue with remaining fields)

    'default': {...}
}
```

---

## 9. Testing Strategy

### Unit Testing
```python
def test_field_prompt_coverage():
    """Verify all bepConfig fields have corresponding prompts"""
    required_fields = [
        'projectName', 'projectNumber', 'projectType', 'appointingParty',
        'projectDescription', 'bimStrategy', 'bimGoals', 'keyMilestones',
        # ... (full list of 109 fields)
    ]

    for field in required_fields:
        assert field in generator.field_prompts, f"Missing prompt for {field}"
```

### Integration Testing
```python
def test_field_suggestion_generation():
    """Test suggestion generation for all field types"""
    for field_name in generator.field_prompts.keys():
        suggestion = generator.suggest_for_field(field_name)
        assert len(suggestion) > 10, f"Suggestion too short for {field_name}"
        assert suggestion != "Please provide more context or try again.", f"Default error for {field_name}"
```

---

## 10. Summary of Findings

| Metric | Count | Percentage |
|--------|-------|-----------|
| Fields in ollama_generator.py field_prompts | 25 | 23% |
| Fields in bepConfig.js | 109 | 100% |
| Fully covered fields | 8 | 7% |
| Partially covered fields | 10 | 9% |
| Missing fields | 91 | 84% |
| **COVERAGE GAP** | **91 fields** | **84%** |

---

## 11. Critical Gaps by Importance

**MUST IMPLEMENT IMMEDIATELY:**
1. Executive Summary fields (5 fields)
2. BIM Goals & Objectives (7 fields)
3. LOIN Requirements (6 fields)
4. Information Delivery Planning (12 fields)
5. Quality Assurance (6 fields)

**SHOULD IMPLEMENT SOON:**
6. CDE Configuration (6 fields)
7. Technology & Infrastructure (10 fields)
8. Security & Privacy (5 fields)

**NICE TO HAVE:**
9. Training & Competency (4 fields)
10. Coordination & Risk (14 fields)

---

## 12. Recommended Next Steps

1. **Immediate:** Review and prioritize field_prompts additions
2. **Week 1:** Implement Priority 1 fields (18-20 new prompts)
3. **Week 2:** Add Priority 2 fields (12-15 new prompts)
4. **Week 3:** Test with actual BEP generation scenarios
5. **Week 4:** Gather feedback and refine prompts
6. **Ongoing:** Monitor field coverage and user feedback

---

**Report prepared for:** BEP Generator Development Team
**Report date:** 2025-12-28
**Status:** Ready for Implementation
**Estimated implementation effort:** 2-3 weeks
