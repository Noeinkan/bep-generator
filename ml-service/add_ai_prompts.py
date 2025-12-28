"""
Script to automatically add aiPrompt configuration to all textarea fields in helpContentData.js

This script:
1. Reads helpContentData.js
2. Identifies all fields that need aiPrompt (69 textarea fields from bepConfig.js)
3. Generates appropriate aiPrompt based on existing field content (description, examples, bestPractices)
4. Inserts aiPrompt after commonMistakes and before relatedFields
"""

import re
import json

# List of 69 textarea fields from bepConfig.js that need aiPrompt
TEXTAREA_FIELDS = [
    # Section 1 - Pre/Post-Appointment
    'projectDescription', 'tenderApproach', 'deliveryApproach',

    # Section 2 - Executive Summary
    'projectContext', 'bimStrategy', 'valueProposition',

    # Section 3 - Team
    'teamCapabilities', 'proposedMobilizationPlan', 'mobilizationPlan',
    'informationManagementResponsibilities',

    # Section 4 - BIM Goals
    'bimGoals', 'primaryObjectives', 'bimValueApplications',
    'strategicAlignment', 'collaborativeProductionGoals',

    # Section 5 - LOIN
    'geometricalInfo', 'alphanumericalInfo', 'documentationInfo',
    'projectInformationRequirements',

    # Section 6 - Information Delivery
    'deliverySchedule', 'tidpDescription', 'midpDescription',
    'mobilisationPlan', 'teamCapabilitySummary', 'taskTeamExchange',
    'modelReferencing3d',

    # Section 7 - CDE
    'accessControl', 'securityMeasures', 'backupProcedures',

    # Section 8 - Technology
    'hardwareRequirements', 'networkRequirements', 'interoperabilityNeeds',
    'federationStrategy', 'informationBreakdownStrategy', 'federationProcess',
    'documentControlInfo',

    # Section 9 - Information Production
    'fileStructure',

    # Section 10 - Quality Assurance
    'modelValidation', 'reviewProcesses', 'approvalWorkflows',
    'complianceVerification', 'modelReviewAuthorisation',

    # Section 11 - Information Security
    'accessPermissions', 'encryptionRequirements', 'dataTransferProtocols',
    'privacyConsiderations',

    # Section 12 - Training
    'bimCompetencyLevels', 'trainingRequirements', 'certificationNeeds',
    'projectSpecificTraining',

    # Section 13 - Coordination & Risk
    'coordinationMeetings', 'clashDetectionWorkflow', 'issueResolution',
    'communicationProtocols', 'informationRisks', 'technologyRisks',
    'riskMitigation', 'contingencyPlans', 'performanceMetrics',
    'monitoringProcedures', 'auditTrails', 'updateProcesses',

    # Section 14 - Appendices
    'fileNamingExamples', 'deliverableTemplates'
]

# Mapping field names to expert roles and styles
FIELD_CONFIGS = {
    'projectDescription': {
        'role': 'BIM project documentation expert',
        'action': 'Write clear, comprehensive project descriptions with quantifiable details',
        'format': 'Include project type, scale (area, budget, timeline), sustainability targets (BREEAM/LEED), key challenges. Use structured paragraphs with specific metrics.',
        'style': 'detailed, quantified metrics, sustainability targets, structured'
    },
    'tenderApproach': {
        'role': 'BIM tender strategy expert',
        'action': 'Write compelling proposed approaches for tender submissions',
        'format': 'Cover collaborative coordination, stakeholder engagement, sustainability, phased delivery. Use persuasive tone with value emphasis.',
        'style': 'strategic, value-focused, collaborative approach'
    },
    'deliveryApproach': {
        'role': 'BIM delivery strategy expert',
        'action': 'Define confirmed delivery approaches for project execution',
        'format': 'Detail coordination workflow, integration methods, value engineering, delivery phases. Use structured approach.',
        'style': 'execution-focused, phased approach, structured'
    },
    'projectContext': {
        'role': 'BIM project context specialist',
        'action': 'Write comprehensive project overviews establishing strategic context',
        'format': 'Outline BIM approach, strategic objectives, project significance. Use contextual narrative.',
        'style': 'strategic, contextual, objective-driven'
    },
    'bimStrategy': {
        'role': 'BIM strategy expert',
        'action': 'Write concise, impactful BIM strategy summaries',
        'format': 'Cover clash detection, 4D/5D modeling, digital twin, federated models, collaboration platforms. Use bullet points with specific technologies.',
        'style': 'strategic, technology-specific, concise bullet points'
    },
    'valueProposition': {
        'role': 'BIM value proposition specialist',
        'action': 'Articulate quantifiable project benefits and cost savings',
        'format': 'Demonstrate cost reductions, schedule improvements, lifecycle benefits with percentages. Use metrics-driven approach.',
        'style': 'quantified benefits, percentage targets, ROI-focused'
    },
    'teamCapabilities': {
        'role': 'BIM team capability specialist',
        'action': 'Showcase team expertise, certifications, and proven track record',
        'format': 'Include ISO 19650 certifications, software expertise, project successes, specialized competencies. Use credential-focused format.',
        'style': 'credential-focused, experience-based, certification highlights'
    },
    'proposedMobilizationPlan': {
        'role': 'BIM project mobilization expert',
        'action': 'Design structured mobilization plans following ISO 19650-2',
        'format': 'Cover team onboarding, training, IT setup, CDE testing, risk mitigation with timeline. Use phased structure.',
        'style': 'phased timeline, ISO 19650-2 compliant, structured'
    },
    'mobilizationPlan': {
        'role': 'BIM mobilization planning expert following ISO 19650-2',
        'action': 'Create detailed phased mobilization timelines',
        'format': 'Detail weekly phases (Week 1-4) with specific activities and ISO 19650-2 clause 5.3.6 compliance. Use weekly breakdown.',
        'style': 'weekly phases, ISO compliant, detailed activities'
    },
    'informationManagementResponsibilities': {
        'role': 'information management specialist following ISO 19650-2',
        'action': 'Define clear Information Manager responsibilities',
        'format': 'Include CDE establishment, TIDP coordination, quality control, security, audits per ISO 19650-2:2018. Use responsibility list.',
        'style': 'role-based, ISO 19650-2 compliant, responsibility-focused'
    },
    'bimGoals': {
        'role': 'BIM goals specialist',
        'action': 'Define clear, measurable BIM objectives with quantifiable targets',
        'format': 'Cover clash detection, 4D sequencing, 5D costing, digital asset delivery with percentage targets. Use goal-oriented format.',
        'style': 'measurable targets, percentage-based, goal-oriented'
    },
    'primaryObjectives': {
        'role': 'BIM objectives expert',
        'action': 'Define primary project objectives aligned with ISO 19650',
        'format': 'Include conflict elimination, performance optimization, construction efficiency, sustainability, handover. Use objective list.',
        'style': 'objective-driven, ISO aligned, outcome-focused'
    },
    'bimValueApplications': {
        'role': 'BIM value applications specialist',
        'action': 'Articulate specific BIM applications maximizing project value',
        'format': 'Cover 4D scheduling, energy modeling, lifecycle costing, parametric design, prefabrication with quantified benefits. Use application-specific format.',
        'style': 'application-specific, quantified benefits, value-driven'
    },
    'strategicAlignment': {
        'role': 'strategic alignment expert',
        'action': 'Demonstrate how BIM strategy supports client strategic objectives',
        'format': 'Show delivery time reduction, carbon targets, performance enhancement, smart building integration with specific targets. Use alignment matrix.',
        'style': 'strategic alignment, target-based, client-focused'
    },
    'collaborativeProductionGoals': {
        'role': 'collaborative information production specialist following ISO 19650',
        'action': 'Define information management goals',
        'format': 'Cover unified standards, real-time coordination, version control, audit trails per ISO 19650. Use information management focus.',
        'style': 'ISO 19650 principles, collaborative workflow, information-centric'
    },
    'geometricalInfo': {
        'role': 'geometrical information requirements specialist',
        'action': 'Define precise LOD requirements',
        'format': 'Specify LOD 300/350/400 for disciplines, survey tolerances, connection details. Use LOD specification format.',
        'style': 'LOD-specific, tolerance-based, discipline-focused'
    },
    'alphanumericalInfo': {
        'role': 'alphanumerical information requirements expert',
        'action': 'Define comprehensive data requirements',
        'format': 'Include material specs, manufacturer data, cost linkage, maintenance schedules, COBie data. Use data attribute focus.',
        'style': 'data-centric, attribute-focused, COBie-ready'
    },
    'documentationInfo': {
        'role': 'documentation requirements specialist',
        'action': 'Define comprehensive handover documentation',
        'format': 'List O&M manuals, safety files, commissioning reports, warranties, certifications. Use document type listing.',
        'style': 'document-focused, handover-ready, compliance-oriented'
    },
    'projectInformationRequirements': {
        'role': 'Project Information Requirements (PIR) specialist following ISO 19650',
        'action': 'Define asset management deliverables',
        'format': 'Specify 3D models with data, IoT monitoring, maintenance scheduling, digital twin connectivity, CAFM integration. Use PIR structure.',
        'style': 'asset management focus, digital twin ready, CAFM integration'
    },
    'deliverySchedule': {
        'role': 'information delivery scheduling expert',
        'action': 'Create phased delivery timelines',
        'format': 'Detail monthly phases (Months 1-24) with stage gates and milestones. Use timeline format.',
        'style': 'phased timeline, stage gates, milestone-driven'
    },
    'tidpDescription': {
        'role': 'Task Information Delivery Plan (TIDP) specialist following ISO 19650',
        'action': 'Document TIDP coordination requirements',
        'format': 'Cover coordination protocols, delivery procedures, discipline integration. Use TIDP structure.',
        'style': 'ISO 19650 TIDP, coordination-focused, procedure-driven'
    },
    'midpDescription': {
        'role': 'Master Information Delivery Plan (MIDP) expert following ISO 19650',
        'action': 'Define structured delivery schedules',
        'format': 'Align with RIBA Plan of Work stages (3-5), quality gates, acceptance criteria. Use MIDP structure.',
        'style': 'RIBA aligned, quality gates, MIDP-structured'
    },
    'mobilisationPlan': {
        'role': 'project mobilisation specialist following ISO 19650',
        'action': 'Plan comprehensive 4-week mobilisation',
        'format': 'Detail Week 1-4 with CDE setup, training, testing, validation per ISO 19650. Use weekly breakdown.',
        'style': '4-week phased, ISO compliant, activity-specific'
    },
    'teamCapabilitySummary': {
        'role': 'delivery team capability assessor',
        'action': 'Summarize comprehensive BIM capabilities and capacity',
        'format': 'Include ISO 19650 professionals, modeling expertise, experience, clash detection record, capacity. Use capability summary.',
        'style': 'capability-focused, experience-based, capacity metrics'
    },
    'taskTeamExchange': {
        'role': 'information exchange protocol specialist following ISO 19650',
        'action': 'Define cross-disciplinary exchange procedures',
        'format': 'Detail weekly federation, coordination meetings, BCF workflows, CDE access, sign-off procedures. Use exchange protocol format.',
        'style': 'protocol-driven, BCF workflow, exchange-focused'
    },
    'modelReferencing3d': {
        'role': '3D model referencing specialist',
        'action': 'Define spatial coordination procedures',
        'format': 'Cover shared coordinate system, datums, reference linking, version control, clash detection. Use coordination protocol.',
        'style': 'spatial coordination, reference protocol, version controlled'
    },
    'accessControl': {
        'role': 'CDE access control specialist',
        'action': 'Define role-based access with SSO integration',
        'format': 'Specify role permissions, SSO, MFA, folder sync, guest access. Use permission matrix format.',
        'style': 'role-based, SSO integrated, permission-focused'
    },
    'securityMeasures': {
        'role': 'multi-platform CDE security expert following ISO 27001',
        'action': 'Define comprehensive security framework',
        'format': 'Include AES-256 encryption, SSL/TLS, ISO 27001 compliance, GDPR, security monitoring. Use security framework.',
        'style': 'ISO 27001 compliant, encryption-focused, GDPR aligned'
    },
    'backupProcedures': {
        'role': 'CDE backup and disaster recovery specialist',
        'action': 'Define comprehensive backup strategy',
        'format': 'Detail daily/weekly backups, retention periods, geo-redundancy, 99.9% uptime SLA, restoration procedures. Use backup schedule.',
        'style': 'SLA-driven, geo-redundant, disaster recovery focused'
    },
    'hardwareRequirements': {
        'role': 'BIM hardware requirements specialist',
        'action': 'Specify workstation and infrastructure needs',
        'format': 'Define workstation specs (CPU, RAM, GPU), storage, server infrastructure, mobile devices. Use specification format.',
        'style': 'specification-driven, hardware-focused, capacity-based'
    },
    'networkRequirements': {
        'role': 'network infrastructure specialist for BIM projects',
        'action': 'Define connectivity and bandwidth needs',
        'format': 'Specify bandwidth, internet connectivity, VPN, latency, redundancy, security. Use network specification.',
        'style': 'bandwidth-focused, redundancy-driven, connectivity specs'
    },
    'interoperabilityNeeds': {
        'role': 'interoperability specialist for BIM',
        'action': 'Define cross-platform data exchange requirements',
        'format': 'Cover IFC, BCF, open standards for seamless data exchange between tools. Use interoperability standards.',
        'style': 'standards-based, IFC/BCF focused, cross-platform'
    },
    'federationStrategy': {
        'role': 'BIM federation strategy expert',
        'action': 'Define model aggregation and coordination approach',
        'format': 'Include federation frequency, clash detection protocols, validation checks, coordination workflows. Use federation schedule.',
        'style': 'frequency-based, clash-focused, workflow-driven'
    },
    'informationBreakdownStrategy': {
        'role': 'information breakdown specialist following ISO 19650',
        'action': 'Define model subdivision approach',
        'format': 'Detail spatial breakdown, discipline breakdown, model file structure aligned with zones. Use breakdown structure.',
        'style': 'zone-based, discipline-focused, structured breakdown'
    },
    'federationProcess': {
        'role': 'federation process specialist',
        'action': 'Define step-by-step model coordination workflow',
        'format': 'Detail aggregation, clash detection, issue resolution, updates, re-federation cycles. Use workflow steps.',
        'style': 'process steps, cycle-based, coordination workflow'
    },
    'documentControlInfo': {
        'role': 'document control specialist following ISO 19650',
        'action': 'Define document management protocols',
        'format': 'Include document types, status codes (WIP/Shared/Published), revision tracking, approval workflows per ISO 19650. Use control protocol.',
        'style': 'ISO 19650 naming, status-based, revision controlled'
    },
    'fileStructure': {
        'role': 'folder structure specialist following ISO 19650',
        'action': 'Define logical file organization',
        'format': 'Detail project code hierarchy, discipline folders, status containers (WIP/Shared/Published/Archive). Use folder hierarchy.',
        'style': 'hierarchical, status-based, discipline-organized'
    },
    'modelValidation': {
        'role': 'BIM model validation expert',
        'action': 'Generate concise, practical validation procedures using checklist format',
        'format': 'Use checklist format (☑) with specific validation tools (e.g., Solibri Model Checker), quantifiable metrics (e.g., <50 clashes, ±5mm tolerance), and actionable items. Keep it practical and structured.',
        'style': 'checklist-based, specific tools mentioned, quantifiable metrics, structured categories'
    },
    'reviewProcesses': {
        'role': 'BIM review process specialist',
        'action': 'Define structured review workflows with clear milestones',
        'format': 'Include review stages, responsible parties, review criteria, documentation requirements. Use structured format with bullet points or numbered workflow.',
        'style': 'structured workflow, defined stages, clear responsibilities, milestone-based'
    },
    'approvalWorkflows': {
        'role': 'approval workflow specialist following ISO 19650',
        'action': 'Define authorization procedures',
        'format': 'Detail submission procedures, review cycles, authorization levels, status transitions (WIP→Shared→Published). Use workflow diagram.',
        'style': 'workflow-driven, status transitions, authorization levels'
    },
    'complianceVerification': {
        'role': 'compliance verification specialist',
        'action': 'Define standards conformance checking',
        'format': 'Cover ISO 19650, EIR compliance, naming conventions, classification systems through automated/manual checks. Use verification checklist.',
        'style': 'compliance-focused, automated checks, standards-based'
    },
    'modelReviewAuthorisation': {
        'role': 'information model review and authorization expert following ISO 19650',
        'action': 'Define model approval procedures',
        'format': 'Include review criteria, authorization roles, approval gates, suitability codes, sign-off documentation per ISO 19650-2. Use approval process.',
        'style': 'ISO 19650-2 compliant, approval gates, suitability codes'
    },
    'accessPermissions': {
        'role': 'access permissions specialist',
        'action': 'Define granular user access controls',
        'format': 'Detail role-based levels (admin/author/reviewer/viewer), folder permissions, read/write privileges. Use permission matrix.',
        'style': 'role-based, granular permissions, security-focused'
    },
    'encryptionRequirements': {
        'role': 'encryption specialist for BIM data',
        'action': 'Define data protection standards',
        'format': 'Specify AES-256, TLS/SSL, key management, GDPR compliance. Use encryption standards.',
        'style': 'encryption standards, GDPR compliant, security protocols'
    },
    'dataTransferProtocols': {
        'role': 'secure data transfer specialist',
        'action': 'Define protocols for information exchange',
        'format': 'Cover SFTP, HTTPS, encrypted email, CDE-based transfer, data integrity verification. Use transfer protocols.',
        'style': 'protocol-specific, secure transfer, integrity-verified'
    },
    'privacyConsiderations': {
        'role': 'privacy and GDPR compliance specialist for BIM projects',
        'action': 'Define privacy protection measures',
        'format': 'Include UK GDPR compliance, personal data handling, retention policies, right to erasure, privacy impact assessments. Use privacy framework.',
        'style': 'GDPR compliant, privacy-focused, data protection'
    },
    'bimCompetencyLevels': {
        'role': 'BIM competency framework specialist',
        'action': 'Define role-based competency levels',
        'format': 'Detail BIM Manager (Level 3), Coordinators (Level 2), Authors (Level 1) with ISO 19650 knowledge requirements. Use competency matrix.',
        'style': 'level-based, role-specific, competency framework'
    },
    'trainingRequirements': {
        'role': 'BIM training requirements specialist',
        'action': 'Define project-specific training needs',
        'format': 'Include ISO 19650 induction, software training (Revit, Navisworks), CDE platform, security training. Use training program.',
        'style': 'training-focused, software-specific, competency development'
    },
    'certificationNeeds': {
        'role': 'BIM certification specialist',
        'action': 'Define required professional certifications',
        'format': 'Specify ISO 19650 certification, BIM Manager (RICS/BRE), Autodesk Certified Professional, project-specific accreditation. Use certification list.',
        'style': 'certification-focused, professional accreditation, standards-based'
    },
    'projectSpecificTraining': {
        'role': 'project-specific BIM training coordinator',
        'action': 'Define tailored training programs',
        'format': 'Cover client EIR, naming conventions, classification, CDE workflows, coordination procedures with hands-on workshops. Use training schedule.',
        'style': 'project-tailored, hands-on, workflow-focused'
    },
    'coordinationMeetings': {
        'role': 'BIM coordination meeting specialist',
        'action': 'Define structured collaboration schedules',
        'format': 'Detail weekly/fortnightly/monthly meetings with clash detection reviews, agendas, documentation. Use meeting schedule.',
        'style': 'scheduled meetings, agenda-driven, documentation-focused'
    },
    'clashDetectionWorkflow': {
        'role': 'clash detection workflow expert',
        'action': 'Define systematic clash management procedures with specific software and metrics',
        'format': 'Include weekly cycles, clash categorization (hard/soft/workflow), specific tools (Navisworks Manage), BCF workflow, resolution tracking. Use numbered steps or bullet points.',
        'style': 'process-driven, weekly cycles, specific tools, clash categories, BCF-based'
    },
    'issueResolution': {
        'role': 'issue resolution process specialist using BCF workflows',
        'action': 'Define structured problem-solving procedures',
        'format': 'Detail BCF issue logging, assignment, resolution proposal, verification, closure with audit trail. Use process workflow.',
        'style': 'BCF-based, audit trail, resolution workflow'
    },
    'communicationProtocols': {
        'role': 'communication protocol specialist for BIM projects',
        'action': 'Define clear communication channels',
        'format': 'Specify channels (email, CDE, PM platform), escalation procedures, meeting schedules, stakeholder matrix. Use communication matrix.',
        'style': 'channel-specific, escalation-driven, stakeholder-focused'
    },
    'informationRisks': {
        'role': 'information-related risk specialist following ISO 19650',
        'action': 'Identify and assess information risks',
        'format': 'List data loss, version control errors, delays, corruption, non-compliance with impact/probability. Use risk register.',
        'style': 'risk-based, impact assessed, probability-rated'
    },
    'technologyRisks': {
        'role': 'technology risk specialist for BIM projects',
        'action': 'Identify technology and software risks',
        'format': 'Cover incompatibility, hardware failures, connectivity issues, platform downtime, licensing with mitigation. Use risk register.',
        'style': 'technology-focused, mitigation strategies, risk-based'
    },
    'riskMitigation': {
        'role': 'risk mitigation strategist for BIM projects',
        'action': 'Define proactive risk reduction measures',
        'format': 'Detail automated backups, redundant systems, updates, training, QA protocols, contingency allocation. Use mitigation plan.',
        'style': 'mitigation-focused, proactive measures, contingency-driven'
    },
    'contingencyPlans': {
        'role': 'contingency planning specialist for BIM projects',
        'action': 'Define backup plans for critical risks',
        'format': 'Cover CDE failure, personnel absence, software failure, data loss, network outage scenarios with alternatives. Use contingency matrix.',
        'style': 'scenario-based, alternative solutions, contingency-focused'
    },
    'performanceMetrics': {
        'role': 'BIM performance metrics specialist',
        'action': 'Define KPIs for project monitoring',
        'format': 'Specify model quality scores, clash reduction rates, timeliness, coordination effectiveness with target values. Use KPI dashboard.',
        'style': 'KPI-driven, target values, measurement methods'
    },
    'monitoringProcedures': {
        'role': 'monitoring procedures specialist for BIM projects',
        'action': 'Define performance tracking protocols',
        'format': 'Detail weekly KPI dashboards, monthly reports, audits, compliance reviews, trend analysis. Use monitoring schedule.',
        'style': 'tracking-focused, dashboard-based, trend analysis'
    },
    'auditTrails': {
        'role': 'audit trail specialist following ISO 19650',
        'action': 'Define information traceability requirements',
        'format': 'Include CDE activity logs, revision histories, approval documentation, change records per ISO 19650. Use traceability framework.',
        'style': 'ISO 19650 compliant, traceability-focused, audit-ready'
    },
    'updateProcesses': {
        'role': 'update process specialist for BIM execution plans',
        'action': 'Define BEP change management',
        'format': 'Detail change request procedures, impact assessment, version control, distribution, team notification. Use change control process.',
        'style': 'change management, version controlled, notification-driven'
    },
    'fileNamingExamples': {
        'role': 'file naming convention expert following ISO 19650',
        'action': 'Provide comprehensive naming examples',
        'format': 'Show examples (GF-SAA-L02-ARC-001) demonstrating project code, originator, zone, discipline, type per ISO 19650 BS 1192. Use example format.',
        'style': 'example-based, ISO 19650 BS 1192 compliant, structured naming'
    },
    'deliverableTemplates': {
        'role': 'information deliverable templates specialist',
        'action': 'Document standard template library',
        'format': 'List TIDP, Model Element Checklist, QA Report, Clash Report templates with CDE location. Use template catalog.',
        'style': 'template-focused, catalog-based, CDE-organized'
    }
}

def generate_ai_prompt(field_name):
    """Generate aiPrompt configuration for a field"""
    config = FIELD_CONFIGS.get(field_name)
    if not config:
        # Fallback for fields not in mapping
        return None

    ai_prompt = f"""
    // AI Prompt Configuration for generating field content
    aiPrompt: {{
      system: 'You are a {config['role']}. {config['action']}.',
      instructions: 'Generate content similar to the examples above. {config['format']} Keep it practical and actionable. Maximum 150 words.',
      style: '{config['style']}'
    }},"""

    return ai_prompt

def add_ai_prompts_to_helpContentData(input_file, output_file):
    """
    Read helpContentData.js, add aiPrompt to all textarea fields, write back
    """
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    fields_updated = 0
    fields_skipped = []

    for field_name in TEXTAREA_FIELDS:
        print(f"Processing field: {field_name}")

        # Check if aiPrompt already exists for this field
        if f"{field_name}: {{" in content:
            # Find the field block
            field_pattern = rf"({field_name}: \{{.*?)(relatedFields:)"

            # Check if aiPrompt already exists
            if re.search(rf"{field_name}: \{{.*?aiPrompt:", content, re.DOTALL):
                print(f"  SKIP: {field_name} - aiPrompt already exists")
                fields_skipped.append(field_name)
                continue

            # Generate aiPrompt
            ai_prompt = generate_ai_prompt(field_name)
            if not ai_prompt:
                print(f"  WARN: No config for {field_name}, skipping")
                fields_skipped.append(field_name)
                continue

            # Insert aiPrompt before relatedFields
            def replacer(match):
                before_related = match.group(1)
                related_fields = match.group(2)
                return f"{before_related}\n{ai_prompt}\n\n    {related_fields}"

            content = re.sub(field_pattern, replacer, content, flags=re.DOTALL)
            fields_updated += 1
            print(f"  OK: Added aiPrompt to {field_name}")
        else:
            print(f"  ERROR: Field {field_name} not found in helpContentData.js")
            fields_skipped.append(field_name)

    print(f"\nWriting updated content to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n{'='*60}")
    print(f"SUCCESS: Updated {fields_updated} fields")
    print(f"SKIPPED: {len(fields_skipped)} fields: {', '.join(fields_skipped[:5])}{'...' if len(fields_skipped) > 5 else ''}")
    print(f"{'='*60}")

if __name__ == "__main__":
    input_file = r"c:\Users\andre\OneDrive\Documents\GitHub\bep-generator\src\data\helpContentData.js"
    output_file = input_file  # Overwrite the same file

    add_ai_prompts_to_helpContentData(input_file, output_file)
    print("\nDONE! All aiPrompts have been added to helpContentData.js")
