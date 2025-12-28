"""
Script to create stub entries in helpContentData.js for missing fields

This ensures all 69 textarea fields have aiPrompt configuration in helpContentData.js
"""

# Fields that exist in bepConfig.js but not in helpContentData.js
MISSING_FIELDS = [
    'accessControl', 'accessPermissions', 'alphanumericalInfo', 'auditTrails',
    'backupProcedures', 'bimCompetencyLevels', 'bimValueApplications',
    'certificationNeeds', 'clashDetectionWorkflow', 'collaborativeProductionGoals',
    'communicationProtocols', 'contingencyPlans', 'coordinationMeetings',
    'deliverableTemplates', 'deliveryApproach', 'deliverySchedule',
    'documentControlInfo', 'documentationInfo', 'federationProcess',
    'federationStrategy', 'fileNamingExamples', 'fileStructure',
    'geometricalInfo', 'hardwareRequirements', 'informationBreakdownStrategy',
    'informationManagementResponsibilities', 'informationRisks',
    'interoperabilityNeeds', 'issueResolution', 'midpDescription',
    'mobilisationPlan', 'mobilizationPlan', 'modelReferencing3d',
    'modelValidation', 'monitoringProcedures', 'networkRequirements',
    'performanceMetrics', 'projectContext', 'projectInformationRequirements',
    'projectSpecificTraining', 'proposedMobilizationPlan', 'riskMitigation',
    'securityMeasures', 'strategicAlignment', 'taskTeamExchange',
    'teamCapabilities', 'teamCapabilitySummary', 'technologyRisks',
    'tenderApproach', 'tidpDescription', 'trainingRequirements', 'updateProcesses'
]

# Map from FIELD_CONFIGS in add_ai_prompts.py
FIELD_CONFIGS = {
    'tenderApproach': ('BIM tender strategy expert', 'Write compelling proposed approaches for tender submissions', 'Cover collaborative coordination, stakeholder engagement, sustainability, phased delivery. Use persuasive tone with value emphasis.', 'strategic, value-focused, collaborative approach'),
    'deliveryApproach': ('BIM delivery strategy expert', 'Define confirmed delivery approaches for project execution', 'Detail coordination workflow, integration methods, value engineering, delivery phases. Use structured approach.', 'execution-focused, phased approach, structured'),
    'projectContext': ('BIM project context specialist', 'Write comprehensive project overviews establishing strategic context', 'Outline BIM approach, strategic objectives, project significance. Use contextual narrative.', 'strategic, contextual, objective-driven'),
    'teamCapabilities': ('BIM team capability specialist', 'Showcase team expertise, certifications, and proven track record', 'Include ISO 19650 certifications, software expertise, project successes, specialized competencies. Use credential-focused format.', 'credential-focused, experience-based, certification highlights'),
    'proposedMobilizationPlan': ('BIM project mobilization expert', 'Design structured mobilization plans following ISO 19650-2', 'Cover team onboarding, training, IT setup, CDE testing, risk mitigation with timeline. Use phased structure.', 'phased timeline, ISO 19650-2 compliant, structured'),
    'mobilizationPlan': ('BIM mobilization planning expert following ISO 19650-2', 'Create detailed phased mobilization timelines', 'Detail weekly phases (Week 1-4) with specific activities and ISO 19650-2 clause 5.3.6 compliance. Use weekly breakdown.', 'weekly phases, ISO compliant, detailed activities'),
    'informationManagementResponsibilities': ('information management specialist following ISO 19650-2', 'Define clear Information Manager responsibilities', 'Include CDE establishment, TIDP coordination, quality control, security, audits per ISO 19650-2:2018. Use responsibility list.', 'role-based, ISO 19650-2 compliant, responsibility-focused'),
    'bimValueApplications': ('BIM value applications specialist', 'Articulate specific BIM applications maximizing project value', 'Cover 4D scheduling, energy modeling, lifecycle costing, parametric design, prefabrication with quantified benefits. Use application-specific format.', 'application-specific, quantified benefits, value-driven'),
    'strategicAlignment': ('strategic alignment expert', 'Demonstrate how BIM strategy supports client strategic objectives', 'Show delivery time reduction, carbon targets, performance enhancement, smart building integration with specific targets. Use alignment matrix.', 'strategic alignment, target-based, client-focused'),
    'collaborativeProductionGoals': ('collaborative information production specialist following ISO 19650', 'Define information management goals', 'Cover unified standards, real-time coordination, version control, audit trails per ISO 19650. Use information management focus.', 'ISO 19650 principles, collaborative workflow, information-centric'),
    'geometricalInfo': ('geometrical information requirements specialist', 'Define precise LOD requirements', 'Specify LOD 300/350/400 for disciplines, survey tolerances, connection details. Use LOD specification format.', 'LOD-specific, tolerance-based, discipline-focused'),
    'alphanumericalInfo': ('alphanumerical information requirements expert', 'Define comprehensive data requirements', 'Include material specs, manufacturer data, cost linkage, maintenance schedules, COBie data. Use data attribute focus.', 'data-centric, attribute-focused, COBie-ready'),
    'documentationInfo': ('documentation requirements specialist', 'Define comprehensive handover documentation', 'List O&M manuals, safety files, commissioning reports, warranties, certifications. Use document type listing.', 'document-focused, handover-ready, compliance-oriented'),
    'projectInformationRequirements': ('Project Information Requirements (PIR) specialist following ISO 19650', 'Define asset management deliverables', 'Specify 3D models with data, IoT monitoring, maintenance scheduling, digital twin connectivity, CAFM integration. Use PIR structure.', 'asset management focus, digital twin ready, CAFM integration'),
    'deliverySchedule': ('information delivery scheduling expert', 'Create phased delivery timelines', 'Detail monthly phases (Months 1-24) with stage gates and milestones. Use timeline format.', 'phased timeline, stage gates, milestone-driven'),
    'tidpDescription': ('Task Information Delivery Plan (TIDP) specialist following ISO 19650', 'Document TIDP coordination requirements', 'Cover coordination protocols, delivery procedures, discipline integration. Use TIDP structure.', 'ISO 19650 TIDP, coordination-focused, procedure-driven'),
    'midpDescription': ('Master Information Delivery Plan (MIDP) expert following ISO 19650', 'Define structured delivery schedules', 'Align with RIBA Plan of Work stages (3-5), quality gates, acceptance criteria. Use MIDP structure.', 'RIBA aligned, quality gates, MIDP-structured'),
    'mobilisationPlan': ('project mobilisation specialist following ISO 19650', 'Plan comprehensive 4-week mobilisation', 'Detail Week 1-4 with CDE setup, training, testing, validation per ISO 19650. Use weekly breakdown.', '4-week phased, ISO compliant, activity-specific'),
    'teamCapabilitySummary': ('delivery team capability assessor', 'Summarize comprehensive BIM capabilities and capacity', 'Include ISO 19650 professionals, modeling expertise, experience, clash detection record, capacity. Use capability summary.', 'capability-focused, experience-based, capacity metrics'),
    'taskTeamExchange': ('information exchange protocol specialist following ISO 19650', 'Define cross-disciplinary exchange procedures', 'Detail weekly federation, coordination meetings, BCF workflows, CDE access, sign-off procedures. Use exchange protocol format.', 'protocol-driven, BCF workflow, exchange-focused'),
    'modelReferencing3d': ('3D model referencing specialist', 'Define spatial coordination procedures', 'Cover shared coordinate system, datums, reference linking, version control, clash detection. Use coordination protocol.', 'spatial coordination, reference protocol, version controlled'),
    'accessControl': ('CDE access control specialist', 'Define role-based access with SSO integration', 'Specify role permissions, SSO, MFA, folder sync, guest access. Use permission matrix format.', 'role-based, SSO integrated, permission-focused'),
    'securityMeasures': ('multi-platform CDE security expert following ISO 27001', 'Define comprehensive security framework', 'Include AES-256 encryption, SSL/TLS, ISO 27001 compliance, GDPR, security monitoring. Use security framework.', 'ISO 27001 compliant, encryption-focused, GDPR aligned'),
    'backupProcedures': ('CDE backup and disaster recovery specialist', 'Define comprehensive backup strategy', 'Detail daily/weekly backups, retention periods, geo-redundancy, 99.9% uptime SLA, restoration procedures. Use backup schedule.', 'SLA-driven, geo-redundant, disaster recovery focused'),
    'hardwareRequirements': ('BIM hardware requirements specialist', 'Specify workstation and infrastructure needs', 'Define workstation specs (CPU, RAM, GPU), storage, server infrastructure, mobile devices. Use specification format.', 'specification-driven, hardware-focused, capacity-based'),
    'networkRequirements': ('network infrastructure specialist for BIM projects', 'Define connectivity and bandwidth needs', 'Specify bandwidth, internet connectivity, VPN, latency, redundancy, security. Use network specification.', 'bandwidth-focused, redundancy-driven, connectivity specs'),
    'interoperabilityNeeds': ('interoperability specialist for BIM', 'Define cross-platform data exchange requirements', 'Cover IFC, BCF, open standards for seamless data exchange between tools. Use interoperability standards.', 'standards-based, IFC/BCF focused, cross-platform'),
    'federationStrategy': ('BIM federation strategy expert', 'Define model aggregation and coordination approach', 'Include federation frequency, clash detection protocols, validation checks, coordination workflows. Use federation schedule.', 'frequency-based, clash-focused, workflow-driven'),
    'informationBreakdownStrategy': ('information breakdown specialist following ISO 19650', 'Define model subdivision approach', 'Detail spatial breakdown, discipline breakdown, model file structure aligned with zones. Use breakdown structure.', 'zone-based, discipline-focused, structured breakdown'),
    'federationProcess': ('federation process specialist', 'Define step-by-step model coordination workflow', 'Detail aggregation, clash detection, issue resolution, updates, re-federation cycles. Use workflow steps.', 'process steps, cycle-based, coordination workflow'),
    'documentControlInfo': ('document control specialist following ISO 19650', 'Define document management protocols', 'Include document types, status codes (WIP/Shared/Published), revision tracking, approval workflows per ISO 19650. Use control protocol.', 'ISO 19650 naming, status-based, revision controlled'),
    'fileStructure': ('folder structure specialist following ISO 19650', 'Define logical file organization', 'Detail project code hierarchy, discipline folders, status containers (WIP/Shared/Published/Archive). Use folder hierarchy.', 'hierarchical, status-based, discipline-organized'),
    'modelValidation': ('BIM model validation expert', 'Generate concise, practical validation procedures using checklist format', 'Use checklist format (☑) with specific validation tools (e.g., Solibri Model Checker), quantifiable metrics (e.g., <50 clashes, ±5mm tolerance), and actionable items. Keep it practical and structured.', 'checklist-based, specific tools mentioned, quantifiable metrics, structured categories'),
    'accessPermissions': ('access permissions specialist', 'Define granular user access controls', 'Detail role-based levels (admin/author/reviewer/viewer), folder permissions, read/write privileges. Use permission matrix.', 'role-based, granular permissions, security-focused'),
    'bimCompetencyLevels': ('BIM competency framework specialist', 'Define role-based competency levels', 'Detail BIM Manager (Level 3), Coordinators (Level 2), Authors (Level 1) with ISO 19650 knowledge requirements. Use competency matrix.', 'level-based, role-specific, competency framework'),
    'trainingRequirements': ('BIM training requirements specialist', 'Define project-specific training needs', 'Include ISO 19650 induction, software training (Revit, Navisworks), CDE platform, security training. Use training program.', 'training-focused, software-specific, competency development'),
    'certificationNeeds': ('BIM certification specialist', 'Define required professional certifications', 'Specify ISO 19650 certification, BIM Manager (RICS/BRE), Autodesk Certified Professional, project-specific accreditation. Use certification list.', 'certification-focused, professional accreditation, standards-based'),
    'projectSpecificTraining': ('project-specific BIM training coordinator', 'Define tailored training programs', 'Cover client EIR, naming conventions, classification, CDE workflows, coordination procedures with hands-on workshops. Use training schedule.', 'project-tailored, hands-on, workflow-focused'),
    'coordinationMeetings': ('BIM coordination meeting specialist', 'Define structured collaboration schedules', 'Detail weekly/fortnightly/monthly meetings with clash detection reviews, agendas, documentation. Use meeting schedule.', 'scheduled meetings, agenda-driven, documentation-focused'),
    'clashDetectionWorkflow': ('clash detection workflow expert', 'Define systematic clash management procedures with specific software and metrics', 'Include weekly cycles, clash categorization (hard/soft/workflow), specific tools (Navisworks Manage), BCF workflow, resolution tracking. Use numbered steps or bullet points.', 'process-driven, weekly cycles, specific tools, clash categories, BCF-based'),
    'issueResolution': ('issue resolution process specialist using BCF workflows', 'Define structured problem-solving procedures', 'Detail BCF issue logging, assignment, resolution proposal, verification, closure with audit trail. Use process workflow.', 'BCF-based, audit trail, resolution workflow'),
    'communicationProtocols': ('communication protocol specialist for BIM projects', 'Define clear communication channels', 'Specify channels (email, CDE, PM platform), escalation procedures, meeting schedules, stakeholder matrix. Use communication matrix.', 'channel-specific, escalation-driven, stakeholder-focused'),
    'informationRisks': ('information-related risk specialist following ISO 19650', 'Identify and assess information risks', 'List data loss, version control errors, delays, corruption, non-compliance with impact/probability. Use risk register.', 'risk-based, impact assessed, probability-rated'),
    'technologyRisks': ('technology risk specialist for BIM projects', 'Identify technology and software risks', 'Cover incompatibility, hardware failures, connectivity issues, platform downtime, licensing with mitigation. Use risk register.', 'technology-focused, mitigation strategies, risk-based'),
    'riskMitigation': ('risk mitigation strategist for BIM projects', 'Define proactive risk reduction measures', 'Detail automated backups, redundant systems, updates, training, QA protocols, contingency allocation. Use mitigation plan.', 'mitigation-focused, proactive measures, contingency-driven'),
    'contingencyPlans': ('contingency planning specialist for BIM projects', 'Define backup plans for critical risks', 'Cover CDE failure, personnel absence, software failure, data loss, network outage scenarios with alternatives. Use contingency matrix.', 'scenario-based, alternative solutions, contingency-focused'),
    'performanceMetrics': ('BIM performance metrics specialist', 'Define KPIs for project monitoring', 'Specify model quality scores, clash reduction rates, timeliness, coordination effectiveness with target values. Use KPI dashboard.', 'KPI-driven, target values, measurement methods'),
    'monitoringProcedures': ('monitoring procedures specialist for BIM projects', 'Define performance tracking protocols', 'Detail weekly KPI dashboards, monthly reports, audits, compliance reviews, trend analysis. Use monitoring schedule.', 'tracking-focused, dashboard-based, trend analysis'),
    'auditTrails': ('audit trail specialist following ISO 19650', 'Define information traceability requirements', 'Include CDE activity logs, revision histories, approval documentation, change records per ISO 19650. Use traceability framework.', 'ISO 19650 compliant, traceability-focused, audit-ready'),
    'updateProcesses': ('update process specialist for BIM execution plans', 'Define BEP change management', 'Detail change request procedures, impact assessment, version control, distribution, team notification. Use change control process.', 'change management, version controlled, notification-driven'),
    'fileNamingExamples': ('file naming convention expert following ISO 19650', 'Provide comprehensive naming examples', 'Show examples (GF-SAA-L02-ARC-001) demonstrating project code, originator, zone, discipline, type per ISO 19650 BS 1192. Use example format.', 'example-based, ISO 19650 BS 1192 compliant, structured naming'),
    'deliverableTemplates': ('information deliverable templates specialist', 'Document standard template library', 'List TIDP, Model Element Checklist, QA Report, Clash Report templates with CDE location. Use template catalog.', 'template-focused, catalog-based, CDE-organized')
}

def generate_field_stub(field_name):
    """Generate a complete field stub for helpContentData.js"""
    if field_name not in FIELD_CONFIGS:
        return None

    role, action, format_desc, style = FIELD_CONFIGS[field_name]

    stub = f'''
  {field_name}: {{
    description: `[Auto-generated] {action}`,

    iso19650: `ISO 19650 standards apply to this field.`,

    bestPractices: [
      'Follow ISO 19650 guidelines',
      'Use clear, professional language',
      'Include specific examples and metrics where applicable'
    ],

    examples: {{
      'Default': `Example content for {field_name} will be added here.`
    }},

    commonMistakes: [
      'Being too generic without specific details',
      'Missing quantifiable metrics or targets',
      'Not following ISO 19650 terminology'
    ],

    // AI Prompt Configuration for generating field content
    aiPrompt: {{
      system: 'You are a {role}. {action}.',
      instructions: 'Generate content similar to the examples above. {format_desc} Keep it practical and actionable. Maximum 150 words.',
      style: '{style}'
    }},

    relatedFields: []
  }},
'''
    return stub

if __name__ == "__main__":
    print("Generated stub entries for missing fields:\n")
    print("="*70)

    for field_name in sorted(MISSING_FIELDS):
        stub = generate_field_stub(field_name)
        if stub:
            print(stub)

    print("="*70)
    print(f"\nGenerated {len(MISSING_FIELDS)} field stubs")
    print("\nCopy these entries and paste them into helpContentData.js")
    print("in the appropriate sections!")
