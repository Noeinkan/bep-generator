"""ml-service/add_all_ai_prompts.py

Script to add aiPrompt fields to missing fields in src/data/helpContentData.js.

Note:
- The current single source of truth schema is aiPrompt: { system, instructions, style }
- The Python loader maps 'instructions' to 'context' for compatibility.
"""

import re
import json

# Field-specific aiPrompt configurations.
# This file historically stored 'context'; we keep it as the source text and
# write it out as 'instructions' in helpContentData.js.
FIELD_PROMPTS = {
    # Team & Organization
    'proposedResourceAllocation': {
        'system': 'You are a BIM resource planning expert specializing in capability and capacity assessment per ISO 19650.',
        'context': 'Generate a detailed resource allocation table for a BEP tender response. Include role, proposed personnel with qualifications, key competencies, weekly hours allocation, and software/hardware requirements. Focus on demonstrating capability (skills, certifications) and capacity (time, resources). Include ISO 19650 certifications and specific BIM tools. Maximum 150 words.'
    },
    'proposedLead': {
        'system': 'You are a BIM leadership consultant expert in ISO 19650 organizational structures.',
        'context': 'Generate a professional profile for a proposed lead appointed party or task team leader for a BEP tender. Include name, role, years of experience, ISO 19650 certifications, relevant project experience with measurable outcomes, leadership capabilities, and technical expertise. Maximum 100 words.'
    },
    'proposedInfoManager': {
        'system': 'You are an information management specialist with expertise in ISO 19650-2 information management roles.',
        'context': 'Generate a professional profile for a proposed Information Manager in a BEP tender. Highlight ISO 19650 Lead Assessor certification, CDE platform expertise (BIM 360, Aconex), years of experience managing information workflows, previous project successes, and information security capabilities. Maximum 100 words.'
    },
    'proposedTeamLeaders': {
        'system': 'You are a BIM team organization expert specializing in ISO 19650 task team structures.',
        'context': 'Generate profiles for proposed discipline/task team leaders in a BEP tender. For each leader include: discipline (Architecture, Structure, MEP), name with qualifications, years of experience, software expertise (Revit, Tekla, etc.), ISO 19650 training, and relevant project experience. Use bullet points for 3-5 leaders. Maximum 150 words.'
    },
    'subcontractors': {
        'system': 'You are a BIM supply chain expert specializing in ISO 19650 appointment and coordination.',
        'context': 'Generate a list of key proposed subcontractors/suppliers for a BEP with their BIM capabilities. For each include: company name, discipline/package, BIM capability level (ISO 19650 compliant, software used), and specific deliverables they will provide. Focus on information delivery readiness. Maximum 150 words.'
    },
    'leadAppointedParty': {
        'system': 'You are a BIM contract specialist with expertise in ISO 19650-2 appointment structures.',
        'context': 'Generate content describing the Lead Appointed Party for a BEP. Include company name, role in project delivery, overall BIM coordination responsibilities, ISO 19650 compliance oversight, CDE management, and federated model delivery. Reference Section 5.1 of ISO 19650-2. Maximum 100 words.'
    },
    'informationManager': {
        'system': 'You are an ISO 19650 information management authority.',
        'context': 'Generate content describing the appointed Information Manager role in a BEP. Include name, qualifications (ISO 19650 Lead Assessor), responsibilities (EIR compliance, CDE administration, information protocols, model validation oversight), and specific deliverables. Reference ISO 19650-2 Section 5.1.2. Maximum 100 words.'
    },
    'assignedTeamLeaders': {
        'system': 'You are a BIM organizational structure expert following ISO 19650 task team principles.',
        'context': 'Generate a list of assigned task team leaders with their responsibilities in a BEP. For each include: name, discipline, ISO 19650 role, model coordination responsibilities, clash detection ownership, and deliverable accountability. Use structured format. Maximum 150 words.'
    },
    'resourceAllocationTable': {
        'system': 'You are a BIM resource management expert specializing in ISO 19650 capacity planning.',
        'context': 'Generate a detailed resource allocation table for a BEP showing personnel, roles, time commitments, and competencies. Include columns for: Role, Personnel Name, Competencies/Certifications, Weekly Hours, Software Tools, and Project Phase allocation. Show 5-8 key resources with quantified commitments. Maximum 200 words.'
    },
    'resourceAllocation': {
        'system': 'You are a BIM project resourcing specialist following ISO 19650-2 capacity and capability principles.',
        'context': 'Generate comprehensive resource allocation content for a BEP. Describe how resources (people, software, hardware) are allocated across project phases. Include FTE counts, software licenses, hardware specs, training plans, and scalability approach. Reference ISO 19650-2 Section 5.1.3. Maximum 150 words.'
    },
    'organizationalStructure': {
        'system': 'You are a BIM organizational design expert specializing in ISO 19650 delivery team structures.',
        'context': 'Generate content describing the organizational structure for information management in a BEP. Include hierarchy from Appointing Party to Lead Appointed Party to Task Teams, information flow paths, reporting lines, and coordination interfaces. Reference ISO 19650-2 organizational requirements. Maximum 150 words.'
    },
    'taskTeamsBreakdown': {
        'system': 'You are a BIM task team coordination expert following ISO 19650 principles.',
        'context': 'Generate a breakdown of task teams for a BEP. For each team include: discipline name, team leader, team size, key deliverables, model responsibilities, software used, and coordination dependencies. Cover Architecture, Structure, MEP, and specialist teams. Maximum 200 words.'
    },

    # Information Delivery Planning
    'midpDescription': {
        'system': 'You are an ISO 19650 Master Information Delivery Plan (MIDP) specialist.',
        'context': 'Generate content describing the Master Information Delivery Plan (MIDP) for a BEP. Explain the MIDP purpose per ISO 19650-2 Section 5.6, its relationship to TIDPs, key milestones, information container delivery schedule, and how it aligns with the project programme. Maximum 150 words.'
    },
    'keyMilestones': {
        'system': 'You are a BIM project planning expert specializing in ISO 19650 information delivery milestones.',
        'context': 'Generate key project milestones for information delivery in a BEP. Include RIBA/project stages, milestone dates, information delivery requirements at each stage, model maturity levels (LOD/LOI), and review/approval gates. Align with ISO 19650-2 delivery cycle. Maximum 150 words.'
    },
    'deliverySchedule': {
        'system': 'You are a BIM scheduling specialist with expertise in ISO 19650 information delivery planning.',
        'context': 'Generate an information delivery schedule for a BEP. Create a table showing: milestone/stage, date, information containers to be delivered, maturity level (LOD), responsible party, and status. Cover design stages through handover. Maximum 200 words.'
    },
    'tidpRequirements': {
        'system': 'You are an ISO 19650 Task Information Delivery Plan (TIDP) expert.',
        'context': 'Generate content describing TIDP requirements for a BEP. Explain what each task team must include in their TIDP per ISO 19650-2 Section 5.6.2: information containers, delivery dates, formats, maturity levels, and coordination dependencies. Maximum 150 words.'
    },
    'responsibilityMatrix': {
        'system': 'You are a BIM responsibility assignment specialist following RACI principles and ISO 19650.',
        'context': 'Generate a responsibility matrix (RACI) for information management tasks in a BEP. Create a table showing tasks vs. roles with R (Responsible), A (Accountable), C (Consulted), I (Informed) assignments. Cover key BIM activities: model coordination, clash detection, validation, CDE management, approvals. Maximum 200 words.'
    },
    'milestoneInformation': {
        'system': 'You are an ISO 19650 information delivery milestone specialist.',
        'context': 'Generate detailed information about project milestones for a BEP. For each milestone include: name, date, information requirements, model maturity (LOD), deliverable formats (IFC, native), review criteria, and approval process. Align with ISO 19650-2 delivery phases. Maximum 150 words.'
    },
    'mobilisationPlan': {
        'system': 'You are a BIM mobilization planning expert following ISO 19650-2 appointment phase requirements.',
        'context': 'Generate a mobilization plan for a BEP describing how the team will be set up after appointment. Include: mobilization timeline, CDE setup, software deployment, team training, protocol establishment, and first information delivery. Reference ISO 19650-2 Section 5.4. Maximum 150 words.'
    },
    'teamCapabilitySummary': {
        'system': 'You are a BIM capability assessment expert specializing in ISO 19650 competency evaluation.',
        'context': 'Generate a team capability summary for a BEP. Describe collective team competencies: ISO 19650 certifications, software expertise, years of BIM experience, sector experience, successful project track record, and information security capabilities. Use quantified metrics where possible. Maximum 150 words.'
    },
    'informationRiskRegister': {
        'system': 'You are a BIM risk management specialist with expertise in ISO 19650 information risk assessment.',
        'context': 'Generate an information risk register for a BEP. Create a table with: risk description, likelihood, impact, mitigation strategy, and owner. Cover risks like: data loss, interoperability failures, security breaches, delayed deliveries, and insufficient model quality. Maximum 200 words.'
    },
    'taskTeamExchange': {
        'system': 'You are a BIM coordination specialist expert in ISO 19650 task team information exchange.',
        'context': 'Generate content describing information exchange between task teams in a BEP. Explain coordination workflows, model federation process, clash detection cycles, issue resolution procedures, and CDE-based collaboration. Reference ISO 19650-2 information exchange principles. Maximum 150 words.'
    },

    # Data & Standards
    'federationStrategy': {
        'system': 'You are a BIM federation specialist with expertise in ISO 19650 model coordination.',
        'context': 'Generate a federation strategy for a BEP. Describe how discipline models are combined into federated models, federation frequency (weekly/bi-weekly), software used (Navisworks/BIM 360), quality checks before federation, and clash detection protocols. Include LOD requirements for federation. Maximum 150 words.'
    },
    'informationBreakdownStrategy': {
        'system': 'You are a BIM information structuring expert following ISO 19650 information container principles.',
        'context': 'Generate content describing the information breakdown strategy for a BEP. Explain how project information is decomposed into manageable containers aligned with ISO 19650, including breakdown by discipline, building zone, system, or level. Describe naming and organization approach. Maximum 150 words.'
    },
    'federationProcess': {
        'system': 'You are a BIM model coordination expert specializing in ISO 19650 federation workflows.',
        'context': 'Generate a detailed federation process description for a BEP. Include step-by-step workflow: model submission to CDE, quality validation, federation in Navisworks/BIM 360, clash detection, issue distribution, and re-federation cycle. Specify roles, timings, and approval gates. Maximum 200 words.'
    },
    'documentControlInfo': {
        'system': 'You are a document management specialist with expertise in ISO 19650 information management protocols.',
        'context': 'Generate document control procedures for a BEP. Describe version control, revision tracking, approval workflows, document naming conventions, metadata requirements, and archival procedures. Align with ISO 19650 CDE workflow states (WIP, Shared, Published, Archived). Maximum 150 words.'
    },
    'modelingStandards': {
        'system': 'You are a BIM modeling standards expert following ISO 19650 and industry best practices.',
        'context': 'Generate BIM modeling standards for a BEP. Cover: LOD definitions for each project stage, geometric accuracy tolerances, object naming conventions, family/template requirements, coordinate systems, and software-specific standards (Revit, ArchiCAD, Tekla). Reference relevant standards (AEC UK BIM Protocol, NBS). Maximum 200 words.'
    },
    'namingConventions': {
        'system': 'You are a BIM information naming specialist following ISO 19650 and BS 1192 conventions.',
        'context': 'Generate file naming conventions for a BEP. Define the naming pattern structure following ISO 19650 principles: Project-Originator-Zone-Level-Type-Role-Classification-Number. Provide examples for models, drawings, and documents. Explain each field purpose. Maximum 150 words.'
    },
    'fileStructure': {
        'system': 'You are a BIM file organization expert specializing in ISO 19650 CDE structure.',
        'context': 'Generate a file structure description for a BEP CDE. Describe folder hierarchy: by discipline, project stage, information type, and status. Explain organization of models, drawings, specifications, and coordination files. Reference ISO 19650 container organization. Maximum 150 words.'
    },
    'fileStructureDiagram': {
        'system': 'You are a BIM CDE structure visualization expert.',
        'context': 'Generate a text-based file structure diagram for a BEP showing CDE folder organization. Use indentation and symbols to show hierarchy. Include top-level folders: Project Info, WIP, Shared, Published, Archived, and sub-folders by discipline and document type. Maximum 200 words.'
    },
    'volumeStrategy': {
        'system': 'You are a BIM spatial breakdown specialist with expertise in model organization.',
        'context': 'Generate a volume/zone breakdown strategy for a BEP. Explain how the project is divided spatially (by building, floor, wing, grid zones) to manage model complexity and coordinate multi-disciplinary work. Include rationale for the breakdown approach chosen. Maximum 100 words.'
    },
    'classificationSystems': {
        'system': 'You are a BIM classification specialist with expertise in industry classification standards.',
        'context': 'Generate content describing classification systems used in a BEP. Specify primary systems: Uniclass 2015, NRM, or other regional standards. Explain how classification codes are applied to model elements and information containers for consistent organization and asset management. Maximum 100 words.'
    },
    'classificationStandards': {
        'system': 'You are a BIM classification standards expert familiar with international classification systems.',
        'context': 'Generate detailed classification standards for a BEP. Describe which classification tables are used (Uniclass Entities, Activities, Products, etc.), how codes are structured, examples of classification application to building elements, and integration with asset management systems. Maximum 150 words.'
    },

    # CDE & Infrastructure
    'cdeStrategy': {
        'system': 'You are a Common Data Environment (CDE) strategy expert following ISO 19650 principles.',
        'context': 'Generate a CDE strategy for a BEP. Describe the overall approach to collaborative information management: platform choice rationale, workflow states (WIP, Shared, Published, Archived), access control philosophy, integration with project tools, and compliance with ISO 19650-1 Section 5.5. Maximum 150 words.'
    },
    'cdePlatforms': {
        'system': 'You are a BIM technology specialist with expertise in CDE platform selection and deployment.',
        'context': 'Generate content describing the CDE platform(s) used in a BEP. Specify platform name (BIM 360, Aconex, ProjectWise, SharePoint, etc.), version, key features utilized (document management, model coordination, workflows), integration capabilities, and user access approach. Maximum 100 words.'
    },
    'workflowStates': {
        'system': 'You are an ISO 19650 CDE workflow specialist.',
        'context': 'Generate content describing CDE workflow states for a BEP. Define the four states per ISO 19650: Work in Progress (WIP), Shared, Published (Information suitable for use), and Archived. Explain transition criteria between states, approval requirements, and access permissions for each state. Maximum 150 words.'
    },
    'accessControl': {
        'system': 'You are an information security and access management expert for BIM environments.',
        'context': 'Generate access control procedures for a BEP CDE. Describe role-based access control (RBAC), permission levels (view, comment, edit, approve), user onboarding/offboarding, guest access protocols, and audit trail requirements. Align with ISO 19650 security principles and GDPR. Maximum 150 words.'
    },
    'securityMeasures': {
        'system': 'You are a BIM cybersecurity specialist with expertise in ISO 19650 information security.',
        'context': 'Generate information security measures for a BEP. Cover: data encryption (at rest and in transit), authentication methods (MFA), network security, backup strategies, incident response procedures, and compliance with UK Cyber Essentials or equivalent standards. Maximum 150 words.'
    },
    'backupProcedures': {
        'system': 'You are a data protection and business continuity specialist for BIM projects.',
        'context': 'Generate backup and disaster recovery procedures for a BEP. Describe backup frequency (daily/weekly), retention periods, backup locations (on-site/cloud), recovery point objectives (RPO), recovery time objectives (RTO), and testing procedures. Include model and document backup strategies. Maximum 150 words.'
    },
    'hardwareRequirements': {
        'system': 'You are a BIM IT infrastructure specialist with expertise in workstation and server specifications.',
        'context': 'Generate hardware requirements for a BEP. Specify workstation specs for BIM users (CPU, RAM, GPU, storage), server requirements for CDE hosting, network infrastructure (bandwidth, switches), and any specialist hardware (rendering farms, VR equipment). Include minimum and recommended specifications. Maximum 150 words.'
    },
    'networkRequirements': {
        'system': 'You are a BIM network infrastructure expert.',
        'context': 'Generate network requirements for a BEP. Specify bandwidth needs for large model transfers, latency requirements for real-time collaboration, VPN requirements for remote access, network segmentation for security, and redundancy/failover provisions. Include local and cloud connectivity needs. Maximum 100 words.'
    },
    'interoperabilityNeeds': {
        'system': 'You are a BIM interoperability specialist with expertise in open standards and data exchange.',
        'context': 'Generate interoperability requirements for a BEP. Describe data exchange formats (IFC 2x3, IFC 4, BCF), software compatibility requirements, translation protocols, quality checking for exports/imports, and coordination between different authoring platforms (Revit, ArchiCAD, Tekla). Reference buildingSMART standards. Maximum 150 words.'
    },
    'softwareHardwareInfrastructure': {
        'system': 'You are a BIM technology infrastructure architect.',
        'context': 'Generate comprehensive software and hardware infrastructure description for a BEP. Include authoring software (Revit, Tekla, etc.) with versions, coordination platforms (Navisworks, Solibri), CDE platform, communication tools, hardware specs, and IT support structure. Show how components integrate. Maximum 200 words.'
    },

    # BIM Uses & Value
    'bimUses': {
        'system': 'You are a BIM value specialist following the Penn State BIM Uses framework.',
        'context': 'Generate BIM Uses for a BEP based on Penn State taxonomy. List 5-8 primary uses relevant to the project (e.g., Design Authoring, 3D Coordination, 4D Planning, 5D Cost Estimation, Energy Analysis, Asset Management). For each use, briefly describe application and value. Maximum 200 words.'
    },
    'referencedMaterial': {
        'system': 'You are a BIM documentation specialist with expertise in technical references and standards.',
        'context': 'Generate a list of referenced material and standards for a BEP. Include: ISO 19650 series, BS 1192, PAS 1192 (if applicable), Employer Information Requirements (EIR), project BIM Protocol, relevant British Standards, NBS specifications, and any sector-specific guidance. Use proper citation format. Maximum 150 words.'
    },
    'bimValueApplications': {
        'system': 'You are a BIM business value consultant specializing in measurable BIM outcomes.',
        'context': 'Generate BIM value applications for a BEP. Describe how BIM delivers tangible value: cost savings through clash reduction, time savings via coordination, quality improvements, risk mitigation, sustainability benefits, and improved asset management. Include quantified examples where possible (e.g., 30% RFI reduction). Maximum 150 words.'
    },
    'valueMetrics': {
        'system': 'You are a BIM performance measurement specialist.',
        'context': 'Generate value metrics for tracking BIM success in a BEP. Define KPIs: clash detection reduction (%), RFI reduction (%), cost savings (£ or %), schedule improvement (days/weeks), model quality scores, and asset data completeness (%). Specify measurement methods and reporting frequency. Maximum 150 words.'
    },
    'strategicAlignment': {
        'system': 'You are a strategic BIM consultant with expertise in aligning BIM with business objectives.',
        'context': 'Generate content showing how the BIM strategy aligns with project strategic goals in a BEP. Connect BIM processes to client objectives: sustainability targets, cost certainty, programme confidence, quality standards, and operational efficiency. Reference ISO 19650 purpose of information management. Maximum 150 words.'
    },
    'collaborativeProductionGoals': {
        'system': 'You are a collaborative working specialist following ISO 19650 collaborative production principles.',
        'context': 'Generate collaborative production goals for a BEP. Describe objectives for team collaboration: shared model environment, concurrent engineering, early problem detection, integrated design solutions, transparent communication, and collective decision-making. Align with ISO 19650-2 collaborative production requirements. Maximum 100 words.'
    },
    'alignmentStrategy': {
        'system': 'You are a BIM strategic alignment consultant.',
        'context': 'Generate an alignment strategy showing how BIM execution aligns with project delivery strategy in a BEP. Connect information management approach to project delivery method (Design & Build, Traditional, etc.), programme milestones, procurement strategy, and client requirements. Maximum 150 words.'
    },

    # Information Requirements
    'informationPurposes': {
        'system': 'You are an ISO 19650 information requirements specialist.',
        'context': 'Generate content describing the purposes of information in a BEP per ISO 19650. Explain how information serves different purposes: design development, coordination, construction, operation, and maintenance. Reference Employer Information Requirements (EIR) and Level of Information Need. Maximum 100 words.'
    },
    'geometricalInfo': {
        'system': 'You are a BIM geometric information specialist.',
        'context': 'Generate requirements for geometrical information in a BEP. Describe 3D model requirements: geometric accuracy tolerances, Level of Detail (LOD) by project stage, coordinate systems, origin points, and geometric representation methods. Reference ISO 19650 Level of Information Need. Maximum 100 words.'
    },
    'alphanumericalInfo': {
        'system': 'You are a BIM data and properties specialist.',
        'context': 'Generate requirements for alphanumerical information in a BEP. Describe required object properties: identification data, classification codes, technical specifications, performance data, maintenance information, and asset data. Specify property sets, naming conventions, and COBie requirements. Maximum 100 words.'
    },
    'documentationInfo': {
        'system': 'You are a BIM documentation requirements specialist.',
        'context': 'Generate requirements for documentation information in a BEP. Describe required documents: specifications, schedules, reports, certifications, O&M manuals, and as-built drawings. Specify formats (PDF, native CAD), metadata requirements, and linking to model elements. Maximum 100 words.'
    },
    'projectInformationRequirements': {
        'system': 'You are an ISO 19650 Project Information Requirements (PIR) specialist.',
        'context': 'Generate Project Information Requirements for a BEP. Describe information needed throughout project lifecycle: design, construction, and handover. Include geometric, alphanumeric, and documentation requirements aligned with ISO 19650-2. Reference the EIR and organizational information requirements (OIR) if applicable. Maximum 150 words.'
    },

    # Quality & Security
    'dataExchangeProtocols': {
        'system': 'You are a BIM data exchange specialist with expertise in interoperability protocols.',
        'context': 'Generate data exchange protocols for a BEP. Describe procedures for exchanging information between software platforms: export settings, IFC schemas, model validation, BCF for issue management, and quality checks. Include protocols for ensuring data integrity during exchange. Maximum 150 words.'
    },
    'qaFramework': {
        'system': 'You are a BIM quality assurance specialist following ISO 19650 quality principles.',
        'context': 'Generate a quality assurance framework for a BEP. Describe the QA approach: model validation tools (Solibri, BIM Collab), checking procedures, quality gates at milestones, issue resolution workflows, and continuous improvement processes. Reference ISO 19650 quality requirements. Maximum 150 words.'
    },
    'dataClassification': {
        'system': 'You are an information security classification specialist for BIM environments.',
        'context': 'Generate a data classification scheme for a BEP. Define classification levels: Public, Internal, Confidential, Restricted. Describe handling requirements for each level, labeling conventions, access restrictions, and retention policies. Align with government security classifications if applicable (OFFICIAL, SECRET). Maximum 150 words.'
    },
    'accessPermissions': {
        'system': 'You are a BIM access control specialist.',
        'context': 'Generate access permissions structure for a BEP CDE. Define permission levels: View Only, Contributor, Reviewer, Approver, Administrator. Describe what each level can do, how permissions are assigned by role, and procedures for permission changes. Include audit requirements. Maximum 150 words.'
    },
}

def find_field_end(lines, start_idx):
    """Find the end of a field definition (the closing brace with comma)"""
    brace_count = 0
    in_field = False

    for i in range(start_idx, len(lines)):
        line = lines[i].strip()

        # Count braces
        brace_count += line.count('{') - line.count('}')

        if '{' in line and not in_field:
            in_field = True

        # Field ends when we return to brace_count 0 and see '},\n'
        if in_field and brace_count == 0 and line.endswith('},'):
            return i

    return -1

def has_ai_prompt(lines, start_idx, end_idx):
    """Check if the field already has an aiPrompt"""
    for i in range(start_idx, end_idx + 1):
        if 'aiPrompt:' in lines[i]:
            return True
    return False

def find_insertion_point(lines, start_idx, end_idx):
    """Find where to insert aiPrompt (after commonMistakes or examples, before relatedFields)"""
    # Look for relatedFields - insert before it
    for i in range(end_idx, start_idx, -1):
        if 'relatedFields:' in lines[i]:
            # Insert before relatedFields, after the previous closing bracket/comma
            return i

    # If no relatedFields, insert before the closing brace
    return end_idx

def _default_style_for_field(field_name: str) -> str:
    """Provide a lightweight default style string when not explicitly defined."""
    table_like = {
        'proposedResourceAllocation', 'resourceAllocationTable', 'taskTeamsBreakdown',
        'deliverySchedule', 'responsibilityMatrix', 'informationRiskRegister',
        'valueMetrics', 'keyMilestones', 'milestoneInformation'
    }
    if field_name in table_like:
        return 'table-style, concise, role/metric specific, ISO 19650 terminology'
    if field_name in {'bimUses', 'referencedMaterial'}:
        return 'structured list, concise, project-specific, ISO 19650 aligned'
    return 'professional tone, ISO 19650 aligned, concise and actionable'


def create_ai_prompt_lines(field_name, indent='    '):
    """Create aiPrompt lines for a field (system/instructions/style)."""
    if field_name not in FIELD_PROMPTS:
        return None

    prompt = FIELD_PROMPTS[field_name]
    instructions = prompt.get('instructions') or prompt.get('context')
    if not instructions:
        return None
    style = prompt.get('style') or _default_style_for_field(field_name)

    # Escape single quotes for safe JS single-quoted strings
    system = prompt['system'].replace("'", "\\'")
    instructions = instructions.replace("'", "\\'")
    style = style.replace("'", "\\'")

    return [
        f"{indent}// AI Prompt Configuration for generating field content",
        f"{indent}aiPrompt: {{",
        f"{indent}  system: '{system}',",
        f"{indent}  instructions: '{instructions}',",
        f"{indent}  style: '{style}'",
        f"{indent}}},",
        "",
    ]

def process_file(input_file, output_file):
    """Process the helpContentData.js file and add missing aiPrompt fields"""

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')

    modified_lines = []
    i = 0
    fields_added = 0

    while i < len(lines):
        line = lines[i]
        modified_lines.append(line)

        # Check if this line starts a field definition
        # Pattern: "  fieldName: {"
        match = re.match(r'^  (\w+):\s*{', line)
        if match:
            field_name = match.group(1)
            start_idx = i
            end_idx = find_field_end(lines, i)

            if end_idx == -1:
                i += 1
                continue

            # Check if this field needs aiPrompt
            if field_name in FIELD_PROMPTS and not has_ai_prompt(lines, start_idx, end_idx):
                # Copy all lines of the field definition
                for j in range(i + 1, end_idx + 1):
                    modified_lines.append(lines[j])

                field_len = end_idx - start_idx + 1
                field_start_in_modified = len(modified_lines) - field_len
                field_end_in_modified = len(modified_lines) - 1

                # Find insertion point
                insertion_idx = find_insertion_point(
                    modified_lines,
                    field_start_in_modified,
                    field_end_in_modified,
                )

                # Insert aiPrompt before relatedFields or at the end
                ai_prompt_lines = create_ai_prompt_lines(field_name)
                if ai_prompt_lines:
                    # Insert right before the relatedFields line
                    modified_lines[insertion_idx:insertion_idx] = ai_prompt_lines
                    fields_added += 1
                    print(f"Added aiPrompt to: {field_name}")

                i = end_idx + 1
                continue

        i += 1

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(modified_lines))

    print(f"\nCompleted! Added aiPrompt to {fields_added} fields.")
    return fields_added

if __name__ == '__main__':
    input_file = r'c:\Users\andre\OneDrive\Documents\GitHub\bep-generator\src\data\helpContentData.js'
    output_file = r'c:\Users\andre\OneDrive\Documents\GitHub\bep-generator\src\data\helpContentData.js.new'

    fields_added = process_file(input_file, output_file)

    print(f"\nNew file created: {output_file}")
    print("Please review the changes and then replace the original file.")
