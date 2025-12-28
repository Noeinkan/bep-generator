# Fields Missing aiPrompt Properties

## Summary
- **Total fields in file:** 81
- **Fields with aiPrompt:** 21
- **Fields WITHOUT aiPrompt:** 60

This document lists all 60 fields in `src/data/helpContentData.js` that are missing the `aiPrompt` property configuration.

---

## Complete List of Fields Without aiPrompt

### Team & Organization Section

1. **keyCommitments** (line 146)
   - Description: List your firm commitments and key deliverables for this project. These are the specific, measurable outcomes you will deliver using BIM processes.

2. **proposedResourceAllocation** (line 312)
   - Description: Define your proposed resource allocation with detailed capability and capacity assessments for each role. This demonstrates your team's ability to meet the client's Exchange Information Requirements.

3. **proposedLead** (line 1021)
   - Description: Identify the proposed Lead Appointed Party - the organization that will take primary responsibility for managing information delivery.

4. **proposedInfoManager** (line 1057)
   - Description: Identify the proposed Information Manager - the individual responsible for managing information processes, CDE implementation, and ensuring compliance with ISO 19650.

5. **proposedTeamLeaders** (line 1094)
   - Description: List the proposed Task Team Leaders for each discipline in a structured table.

6. **subcontractors** (line 1135)
   - Description: List proposed subcontractors, specialist consultants, and key partners who will support project delivery.

7. **keyContacts** (line 893)
   - Description: List the key project contacts and their roles in a structured table format.

8. **leadAppointedParty** (line 1178)
   - Description: Confirm the appointed Lead Appointed Party - the organization taking primary responsibility for managing information delivery.

9. **informationManager** (line 1216)
   - Description: Confirm the appointed Information Manager - the named individual responsible for managing information processes and ISO 19650 compliance.

10. **assignedTeamLeaders** (line 1263)
    - Description: Confirm the assigned Task Team Leaders for each discipline. These are the appointed technical leads responsible for information production.

11. **resourceAllocationTable** (line 1307)
    - Description: Define the confirmed resource allocation with detailed capability and capacity assessments for each role.

12. **resourceAllocation** (line 1482)
    - Description: (Legacy field - use resourceAllocationTable and mobilizationPlan instead)

13. **organizationalStructure** (line 1616)
    - Description: This is an interactive organizational chart showing the delivery team's structure and composition.

14. **taskTeamsBreakdown** (line 1651)
    - Description: Provide a detailed breakdown of all task teams in a structured table format.

### BIM Uses & Value Section

15. **bimUses** (line 678)
    - Description: Select the specific BIM uses that will be applied on this project. BIM uses are the specific ways that BIM processes and models will be utilized.

16. **bimValueApplications** (line 1751)
    - Description: Explain how BIM will be applied to maximize project value across cost, time, quality, risk, and sustainability dimensions.

17. **valueMetrics** (line 1783)
    - Description: Define measurable success metrics and how BIM value will be tracked throughout the project.

18. **strategicAlignment** (line 1821)
    - Description: Explain how the BIM strategy aligns with and supports the client's strategic business objectives.

19. **collaborativeProductionGoals** (line 1851)
    - Description: Define goals for collaborative information production across the delivery team.

20. **alignmentStrategy** (line 1880)
    - Description: Define your comprehensive approach to facilitating information management goals and maintaining alignment throughout the project.

21. **referencedMaterial** (line 1699)
    - Description: List all referenced documents, standards, and materials that inform or govern this BEP.

### Information Requirements Section

22. **informationPurposes** (line 2013)
    - Description: Select the purposes for which information will be used throughout the project lifecycle.

23. **geometricalInfo** (line 2028)
    - Description: Define the geometrical information requirements - the level of detail, accuracy, and dimensional information needed in 3D models.

24. **alphanumericalInfo** (line 2055)
    - Description: Define the alphanumerical (non-graphical) information requirements - the properties, parameters, and data needed for model elements.

25. **documentationInfo** (line 2083)
    - Description: Define documentation requirements - the supporting documents, specifications, certificates, and manuals required alongside models.

26. **projectInformationRequirements** (line 2112)
    - Description: Define the Project Information Requirements (PIR) - the information needed to support asset management and operational objectives.

### Information Delivery Planning Section

27. **midpDescription** (line 2145)
    - Description: Describe the Master Information Delivery Plan (MIDP) - the high-level schedule that establishes when information will be delivered throughout the project.

28. **keyMilestones** (line 2172)
    - Description: Define key information delivery milestones in a structured table showing stage/phase, description, deliverables, and due dates.

29. **deliverySchedule** (line 2191)
    - Description: Provide a detailed information delivery schedule showing phased approach across the project timeline.

30. **tidpRequirements** (line 2210)
    - Description: Define Task Information Delivery Plan (TIDP) requirements - discipline-specific delivery plans that feed into the MIDP.

31. **responsibilityMatrix** (line 2237)
    - Description: Create a RACI (Responsible, Accountable, Consulted, Informed) matrix defining task/activity responsibilities across the team.

32. **milestoneInformation** (line 2256)
    - Description: Define specific information requirements at each milestone in a detailed table format.

33. **mobilisationPlan** (line 2275)
    - Description: Describe the project mobilisation plan - how the team, systems, and processes will be established at project start.

34. **teamCapabilitySummary** (line 2304)
    - Description: Summarize the delivery team's BIM capability and capacity to meet project information requirements.

35. **informationRiskRegister** (line 2324)
    - Description: Maintain a risk register specific to information delivery - identifying, assessing, and mitigating risks to information management.

36. **taskTeamExchange** (line 2344)
    - Description: Define protocols for information exchange between task teams - how disciplines will share, coordinate, and approve information.

### CDE & Infrastructure Section

37. **cdeStrategy** (line 2403)
    - Description: Describe the overall CDE strategy including platform selection, workflow implementation, and governance approach.

38. **cdePlatforms** (line 2422)
    - Description: List and describe CDE platforms in use, their purposes, information types managed, and workflow integration.

39. **workflowStates** (line 2441)
    - Description: Define the CDE workflow states (WIP, Shared, Published, Archived) and transition criteria between states.

40. **accessControl** (line 2460)
    - Description: Define access control policies including role-based permissions, authentication, and security protocols.

41. **securityMeasures** (line 2480)
    - Description: Define comprehensive security measures protecting information confidentiality, integrity, and availability.

42. **backupProcedures** (line 2500)
    - Description: Define backup and disaster recovery procedures ensuring information protection and business continuity.

43. **hardwareRequirements** (line 2524)
    - Description: Specify the hardware requirements necessary to support BIM activities throughout the project. This includes workstations, servers, mobile devices, and specialized equipment.

44. **networkRequirements** (line 2604)
    - Description: Define network infrastructure and connectivity requirements to support collaborative BIM workflows, model sharing, and CDE access.

45. **interoperabilityNeeds** (line 2684)
    - Description: Define interoperability requirements ensuring seamless data exchange between different software platforms, disciplines, and project stakeholders.

46. **softwareHardwareInfrastructure** (line 3066)
    - Description: Provide a comprehensive matrix of all software, hardware, and IT infrastructure components required for BIM delivery.

### Data Organization & Standards Section

47. **federationStrategy** (line 2766)
    - Description: Describe your strategy for federating discipline models into a coordinated whole-project model for clash detection, design coordination, and stakeholder visualization.

48. **informationBreakdownStrategy** (line 2855)
    - Description: Define how project information will be broken down and organized into manageable components, models, and deliverables.

49. **federationProcess** (line 2947)
    - Description: Define the detailed procedures and workflows for creating, validating, and distributing federated coordination models.

50. **documentControlInfo** (line 3159)
    - Description: Define document control procedures ensuring consistent identification, versioning, approval, and distribution of all project information.

51. **modelingStandards** (line 3282)
    - Description: Define the modeling standards and guidelines that all project team members must follow to ensure consistency, quality, and interoperability.

52. **namingConventions** (line 3691)
    - Description: Establish comprehensive naming conventions for all project files, models, drawings, views, families, and elements.

53. **fileStructure** (line 3794)
    - Description: Define the folder hierarchy and organization structure for the project CDE and local working environments.

54. **fileStructureDiagram** (line 3913)
    - Description: Create a visual diagram representing the project folder structure within the Common Data Environment (CDE).

55. **volumeStrategy** (line 4005)
    - Description: Define the volume strategy (model breakdown structure) showing how the project is divided into manageable information containers.

56. **classificationSystems** (line 4033)
    - Description: Define the classification systems and coding frameworks that will be used to organize and categorize project information, elements, spaces, and assets.

57. **classificationStandards** (line 4135)
    - Description: Provide detailed implementation guidelines for applying classification standards to specific element categories, spaces, and assets.

### Quality & Security Section

58. **dataExchangeProtocols** (line 4204)
    - Description: Define protocols and procedures for exchanging information between project team members, disciplines, and external stakeholders.

59. **qaFramework** (line 4282)
    - Description: Define the quality assurance framework including all QA activities, responsible parties, frequency, and tools/methods used to ensure information quality.

60. **dataClassification** (line 5039)
    - Description: Define data classification levels and corresponding security controls to protect sensitive project information.

---

## Quick Reference Table

| # | Field Name | Line | Category |
|---|---|---|---|
| 1 | keyCommitments | 146 | Team & Organization |
| 2 | proposedResourceAllocation | 312 | Team & Organization |
| 3 | bimUses | 678 | BIM Uses & Value |
| 4 | keyContacts | 893 | Team & Organization |
| 5 | proposedLead | 1021 | Team & Organization |
| 6 | proposedInfoManager | 1057 | Team & Organization |
| 7 | proposedTeamLeaders | 1094 | Team & Organization |
| 8 | subcontractors | 1135 | Team & Organization |
| 9 | leadAppointedParty | 1178 | Team & Organization |
| 10 | informationManager | 1216 | Team & Organization |
| 11 | assignedTeamLeaders | 1263 | Team & Organization |
| 12 | resourceAllocationTable | 1307 | Team & Organization |
| 13 | resourceAllocation | 1482 | Team & Organization |
| 14 | organizationalStructure | 1616 | Team & Organization |
| 15 | taskTeamsBreakdown | 1651 | Team & Organization |
| 16 | referencedMaterial | 1699 | BIM Uses & Value |
| 17 | bimValueApplications | 1751 | BIM Uses & Value |
| 18 | valueMetrics | 1783 | BIM Uses & Value |
| 19 | strategicAlignment | 1821 | BIM Uses & Value |
| 20 | collaborativeProductionGoals | 1851 | BIM Uses & Value |
| 21 | alignmentStrategy | 1880 | BIM Uses & Value |
| 22 | informationPurposes | 2013 | Information Requirements |
| 23 | geometricalInfo | 2028 | Information Requirements |
| 24 | alphanumericalInfo | 2055 | Information Requirements |
| 25 | documentationInfo | 2083 | Information Requirements |
| 26 | projectInformationRequirements | 2112 | Information Requirements |
| 27 | midpDescription | 2145 | Information Delivery Planning |
| 28 | keyMilestones | 2172 | Information Delivery Planning |
| 29 | deliverySchedule | 2191 | Information Delivery Planning |
| 30 | tidpRequirements | 2210 | Information Delivery Planning |
| 31 | responsibilityMatrix | 2237 | Information Delivery Planning |
| 32 | milestoneInformation | 2256 | Information Delivery Planning |
| 33 | mobilisationPlan | 2275 | Information Delivery Planning |
| 34 | teamCapabilitySummary | 2304 | Information Delivery Planning |
| 35 | informationRiskRegister | 2324 | Information Delivery Planning |
| 36 | taskTeamExchange | 2344 | Information Delivery Planning |
| 37 | cdeStrategy | 2403 | CDE & Infrastructure |
| 38 | cdePlatforms | 2422 | CDE & Infrastructure |
| 39 | workflowStates | 2441 | CDE & Infrastructure |
| 40 | accessControl | 2460 | CDE & Infrastructure |
| 41 | securityMeasures | 2480 | CDE & Infrastructure |
| 42 | backupProcedures | 2500 | CDE & Infrastructure |
| 43 | hardwareRequirements | 2524 | CDE & Infrastructure |
| 44 | networkRequirements | 2604 | CDE & Infrastructure |
| 45 | interoperabilityNeeds | 2684 | CDE & Infrastructure |
| 46 | federationStrategy | 2766 | Data Organization & Standards |
| 47 | informationBreakdownStrategy | 2855 | Data Organization & Standards |
| 48 | federationProcess | 2947 | Data Organization & Standards |
| 49 | softwareHardwareInfrastructure | 3066 | CDE & Infrastructure |
| 50 | documentControlInfo | 3159 | Data Organization & Standards |
| 51 | modelingStandards | 3282 | Data Organization & Standards |
| 52 | namingConventions | 3691 | Data Organization & Standards |
| 53 | fileStructure | 3794 | Data Organization & Standards |
| 54 | fileStructureDiagram | 3913 | Data Organization & Standards |
| 55 | volumeStrategy | 4005 | Data Organization & Standards |
| 56 | classificationSystems | 4033 | Data Organization & Standards |
| 57 | classificationStandards | 4135 | Data Organization & Standards |
| 58 | dataExchangeProtocols | 4204 | Quality & Security |
| 59 | qaFramework | 4282 | Quality & Security |
| 60 | dataClassification | 5039 | Quality & Security |

---

## Notes
- All line numbers reference the start of each field definition in the file
- Fields are grouped logically by category for easier navigation
- A sample aiPrompt structure looks like:
  ```javascript
  aiPrompt: {
    system: 'You are a BIM expert...',
    instructions: 'Generate content that...',
    style: 'descriptive, technical, formal'
  }
  ```
