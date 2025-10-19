/**
 * ISO 19650-2 Annex A - Information Management Activities Template
 * Pre-defined activities based on ISO 19650-2 standard
 */

export const ISO_ACTIVITY_PHASES = {
  APPOINTMENT: 'Appointment',
  MOBILIZATION: 'Mobilization',
  COLLABORATIVE_PRODUCTION: 'Collaborative Production',
  INFORMATION_EXCHANGE: 'Information Exchange',
  PROJECT_CLOSEOUT: 'Project Close-out'
};

export const RACI_ROLES = {
  RESPONSIBLE: 'R',
  ACCOUNTABLE: 'A',
  CONSULTED: 'C',
  INFORMED: 'I',
  NOT_APPLICABLE: 'N/A'
};

export const RACI_ROLE_DESCRIPTIONS = {
  R: 'Responsible - Performs the work',
  A: 'Accountable - Approves/has final authority',
  C: 'Consulted - Provides input',
  I: 'Informed - Kept updated',
  'N/A': 'Not Applicable'
};

/**
 * ISO 19650-2 Pre-defined Information Management Activities
 * Each activity includes:
 * - name: Activity name
 * - description: Detailed description
 * - phase: Activity phase (from ISO_ACTIVITY_PHASES)
 * - isoReference: ISO 19650-2 reference clause
 * - defaultRoles: Suggested RACI assignments
 */
export const ISO_19650_ACTIVITIES = [
  // APPOINTMENT PHASE ACTIVITIES (5)
  {
    name: 'Establish information requirements (EIR)',
    description: 'Define and document the employer\'s information requirements for the project',
    phase: ISO_ACTIVITY_PHASES.APPOINTMENT,
    isoReference: 'ISO 19650-2:2018, 5.1.2',
    defaultRoles: {
      appointingParty: 'A',
      leadAppointedParty: 'C',
      appointedParties: 'I',
      thirdParties: 'N/A'
    }
  },
  {
    name: 'Assess appointed party\'s capability',
    description: 'Evaluate the capability and capacity of potential appointed parties',
    phase: ISO_ACTIVITY_PHASES.APPOINTMENT,
    isoReference: 'ISO 19650-2:2018, 5.1.3',
    defaultRoles: {
      appointingParty: 'A',
      leadAppointedParty: 'R',
      appointedParties: 'C',
      thirdParties: 'N/A'
    }
  },
  {
    name: 'Establish assessment criteria for appointments',
    description: 'Define criteria for evaluating and selecting appointed parties',
    phase: ISO_ACTIVITY_PHASES.APPOINTMENT,
    isoReference: 'ISO 19650-2:2018, 5.1.3',
    defaultRoles: {
      appointingParty: 'A',
      leadAppointedParty: 'C',
      appointedParties: 'I',
      thirdParties: 'N/A'
    }
  },
  {
    name: 'Define information delivery milestones',
    description: 'Establish key dates and milestones for information delivery throughout the project',
    phase: ISO_ACTIVITY_PHASES.APPOINTMENT,
    isoReference: 'ISO 19650-2:2018, 5.1.4',
    defaultRoles: {
      appointingParty: 'A',
      leadAppointedParty: 'R',
      appointedParties: 'C',
      thirdParties: 'I'
    }
  },
  {
    name: 'Establish acceptance criteria for information',
    description: 'Define the criteria that information must meet to be accepted at each milestone',
    phase: ISO_ACTIVITY_PHASES.APPOINTMENT,
    isoReference: 'ISO 19650-2:2018, 5.1.5',
    defaultRoles: {
      appointingParty: 'A',
      leadAppointedParty: 'R',
      appointedParties: 'C',
      thirdParties: 'N/A'
    }
  },

  // MOBILIZATION PHASE ACTIVITIES (8)
  {
    name: 'Establish information standard',
    description: 'Define and agree on information standards to be used throughout the project',
    phase: ISO_ACTIVITY_PHASES.MOBILIZATION,
    isoReference: 'ISO 19650-2:2018, 5.2.1',
    defaultRoles: {
      appointingParty: 'C',
      leadAppointedParty: 'A',
      appointedParties: 'R',
      thirdParties: 'I'
    }
  },
  {
    name: 'Establish CDE and workflows',
    description: 'Setup the Common Data Environment and define information workflows',
    phase: ISO_ACTIVITY_PHASES.MOBILIZATION,
    isoReference: 'ISO 19650-2:2018, 5.2.2',
    defaultRoles: {
      appointingParty: 'I',
      leadAppointedParty: 'A',
      appointedParties: 'R',
      thirdParties: 'I'
    }
  },
  {
    name: 'Setup information production methods',
    description: 'Establish methods and processes for producing information',
    phase: ISO_ACTIVITY_PHASES.MOBILIZATION,
    isoReference: 'ISO 19650-2:2018, 5.2.3',
    defaultRoles: {
      appointingParty: 'I',
      leadAppointedParty: 'A',
      appointedParties: 'R',
      thirdParties: 'C'
    }
  },
  {
    name: 'Define reference information',
    description: 'Identify and make available reference information for the project',
    phase: ISO_ACTIVITY_PHASES.MOBILIZATION,
    isoReference: 'ISO 19650-2:2018, 5.2.4',
    defaultRoles: {
      appointingParty: 'R',
      leadAppointedParty: 'A',
      appointedParties: 'C',
      thirdParties: 'I'
    }
  },
  {
    name: 'Establish shared resources',
    description: 'Setup shared resources including libraries, templates, and reference data',
    phase: ISO_ACTIVITY_PHASES.MOBILIZATION,
    isoReference: 'ISO 19650-2:2018, 5.2.5',
    defaultRoles: {
      appointingParty: 'C',
      leadAppointedParty: 'A',
      appointedParties: 'R',
      thirdParties: 'I'
    }
  },
  {
    name: 'Setup collaborative working procedures',
    description: 'Define procedures for collaborative working and coordination',
    phase: ISO_ACTIVITY_PHASES.MOBILIZATION,
    isoReference: 'ISO 19650-2:2018, 5.2.6',
    defaultRoles: {
      appointingParty: 'I',
      leadAppointedParty: 'A',
      appointedParties: 'R',
      thirdParties: 'C'
    }
  },
  {
    name: 'Establish BIM Execution Plan',
    description: 'Develop and agree on the BIM Execution Plan (BEP)',
    phase: ISO_ACTIVITY_PHASES.MOBILIZATION,
    isoReference: 'ISO 19650-2:2018, 5.2.7',
    defaultRoles: {
      appointingParty: 'A',
      leadAppointedParty: 'R',
      appointedParties: 'C',
      thirdParties: 'I'
    }
  },
  {
    name: 'Establish Task Information Delivery Plans',
    description: 'Create TIDPs for each appointed party defining their information deliverables',
    phase: ISO_ACTIVITY_PHASES.MOBILIZATION,
    isoReference: 'ISO 19650-2:2018, 5.2.8',
    defaultRoles: {
      appointingParty: 'C',
      leadAppointedParty: 'A',
      appointedParties: 'R',
      thirdParties: 'I'
    }
  },

  // COLLABORATIVE PRODUCTION PHASE ACTIVITIES (7)
  {
    name: 'Generate information',
    description: 'Produce information in accordance with the TIDP and agreed standards',
    phase: ISO_ACTIVITY_PHASES.COLLABORATIVE_PRODUCTION,
    isoReference: 'ISO 19650-2:2018, 5.3.1',
    defaultRoles: {
      appointingParty: 'I',
      leadAppointedParty: 'A',
      appointedParties: 'R',
      thirdParties: 'R'
    }
  },
  {
    name: 'Undertake quality assurance checks',
    description: 'Perform quality checks on information before sharing',
    phase: ISO_ACTIVITY_PHASES.COLLABORATIVE_PRODUCTION,
    isoReference: 'ISO 19650-2:2018, 5.3.2',
    defaultRoles: {
      appointingParty: 'I',
      leadAppointedParty: 'A',
      appointedParties: 'R',
      thirdParties: 'R'
    }
  },
  {
    name: 'Review and authorize information',
    description: 'Review information for suitability and authorize for sharing',
    phase: ISO_ACTIVITY_PHASES.COLLABORATIVE_PRODUCTION,
    isoReference: 'ISO 19650-2:2018, 5.3.3',
    defaultRoles: {
      appointingParty: 'I',
      leadAppointedParty: 'A',
      appointedParties: 'R',
      thirdParties: 'C'
    }
  },
  {
    name: 'Model federation and coordination',
    description: 'Federate models and coordinate information across disciplines',
    phase: ISO_ACTIVITY_PHASES.COLLABORATIVE_PRODUCTION,
    isoReference: 'ISO 19650-2:2018, 5.3.4',
    defaultRoles: {
      appointingParty: 'I',
      leadAppointedParty: 'A',
      appointedParty: 'R',
      thirdParties: 'C'
    }
  },
  {
    name: 'Undertake information model uses',
    description: 'Use information models for intended purposes (e.g., clash detection, analysis)',
    phase: ISO_ACTIVITY_PHASES.COLLABORATIVE_PRODUCTION,
    isoReference: 'ISO 19650-2:2018, 5.3.5',
    defaultRoles: {
      appointingParty: 'C',
      leadAppointedParty: 'A',
      appointedParties: 'R',
      thirdParties: 'R'
    }
  },
  {
    name: 'Information container breakdown',
    description: 'Break down information into appropriate containers for delivery',
    phase: ISO_ACTIVITY_PHASES.COLLABORATIVE_PRODUCTION,
    isoReference: 'ISO 19650-2:2018, 5.3.6',
    defaultRoles: {
      appointingParty: 'I',
      leadAppointedParty: 'A',
      appointedParties: 'R',
      thirdParties: 'C'
    }
  },
  {
    name: 'Manage information changes',
    description: 'Control and coordinate changes to information throughout production',
    phase: ISO_ACTIVITY_PHASES.COLLABORATIVE_PRODUCTION,
    isoReference: 'ISO 19650-2:2018, 5.3.7',
    defaultRoles: {
      appointingParty: 'I',
      leadAppointedParty: 'A',
      appointedParties: 'R',
      thirdParties: 'C'
    }
  },

  // INFORMATION EXCHANGE PHASE ACTIVITIES (3)
  {
    name: 'Submit information for review',
    description: 'Submit information deliverables at key decision points for review',
    phase: ISO_ACTIVITY_PHASES.INFORMATION_EXCHANGE,
    isoReference: 'ISO 19650-2:2018, 5.4.1',
    defaultRoles: {
      appointingParty: 'I',
      leadAppointedParty: 'A',
      appointedParties: 'R',
      thirdParties: 'R'
    }
  },
  {
    name: 'Carry out information review',
    description: 'Review submitted information against acceptance criteria',
    phase: ISO_ACTIVITY_PHASES.INFORMATION_EXCHANGE,
    isoReference: 'ISO 19650-2:2018, 5.4.2',
    defaultRoles: {
      appointingParty: 'A',
      leadAppointedParty: 'R',
      appointedParties: 'C',
      thirdParties: 'I'
    }
  },
  {
    name: 'Accept/approve information for next stage',
    description: 'Formally accept or approve information deliverables to proceed to next stage',
    phase: ISO_ACTIVITY_PHASES.INFORMATION_EXCHANGE,
    isoReference: 'ISO 19650-2:2018, 5.4.3',
    defaultRoles: {
      appointingParty: 'A',
      leadAppointedParty: 'C',
      appointedParties: 'I',
      thirdParties: 'I'
    }
  },

  // PROJECT CLOSE-OUT PHASE ACTIVITIES (2)
  {
    name: 'Archive project information',
    description: 'Archive all project information in accordance with agreed protocols',
    phase: ISO_ACTIVITY_PHASES.PROJECT_CLOSEOUT,
    isoReference: 'ISO 19650-2:2018, 5.5.1',
    defaultRoles: {
      appointingParty: 'A',
      leadAppointedParty: 'R',
      appointedParties: 'R',
      thirdParties: 'C'
    }
  },
  {
    name: 'Handover asset information for operational use',
    description: 'Transfer asset information to operations team in appropriate format',
    phase: ISO_ACTIVITY_PHASES.PROJECT_CLOSEOUT,
    isoReference: 'ISO 19650-2:2018, 5.5.2',
    defaultRoles: {
      appointingParty: 'A',
      leadAppointedParty: 'R',
      appointedParties: 'R',
      thirdParties: 'I'
    }
  }
];

/**
 * Get activities by phase
 */
export const getActivitiesByPhase = (phase) => {
  return ISO_19650_ACTIVITIES.filter(activity => activity.phase === phase);
};

/**
 * Get all unique phases
 */
export const getAllPhases = () => {
  return Object.values(ISO_ACTIVITY_PHASES);
};

/**
 * Create default activity set for a project
 */
export const createDefaultActivitiesForProject = (projectId) => {
  return ISO_19650_ACTIVITIES.map((activity, index) => ({
    id: `${projectId}-activity-${index}`,
    projectId,
    activityName: activity.name,
    activityDescription: activity.description,
    activityPhase: activity.phase,
    isoReference: activity.isoReference,
    appointingPartyRole: activity.defaultRoles.appointingParty,
    leadAppointedPartyRole: activity.defaultRoles.leadAppointedParty,
    appointedPartiesRole: activity.defaultRoles.appointedParties,
    thirdPartiesRole: activity.defaultRoles.thirdParties,
    notes: '',
    displayOrder: index,
    isCustom: false,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  }));
};

export default {
  ISO_ACTIVITY_PHASES,
  RACI_ROLES,
  RACI_ROLE_DESCRIPTIONS,
  ISO_19650_ACTIVITIES,
  getActivitiesByPhase,
  getAllPhases,
  createDefaultActivitiesForProject
};
