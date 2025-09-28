const Joi = require('joi');

// KPI schema
const kpiSchema = Joi.object({
  'KPI Name': Joi.string().required().min(1).max(100),
  'Target': Joi.string().required(),
  'Current Value': Joi.string().optional().allow(''),
  'Measurement Method': Joi.string().required().min(1).max(200)
});

// Risk schema
const riskSchema = Joi.object({
  id: Joi.string().optional(),
  'Risk Description': Joi.string().required().min(1).max(500),
  'Impact': Joi.string().valid('Low', 'Medium', 'High', 'Critical').required(),
  'Probability': Joi.string().valid('Low', 'Medium', 'High').required(),
  'Mitigation Strategy': Joi.string().required().min(1).max(500),
  'Contingency Plan': Joi.string().optional().allow(''),
  'Responsible Party': Joi.string().required().min(1).max(100),
  'Status': Joi.string().valid('Open', 'In Progress', 'Mitigated', 'Closed').optional()
});

// Milestone schema
const milestoneSchema = Joi.object({
  'Stage': Joi.string().required(),
  'Milestone Name': Joi.string().required().min(1).max(200),
  'Description': Joi.string().required().min(1).max(500),
  'Due Date': Joi.date().iso().required(),
  'Required Information': Joi.string().required(),
  'Responsible Teams': Joi.string().required(),
  'Review Duration': Joi.string().required(),
  'Status': Joi.string().valid('Upcoming', 'In Progress', 'Under Review', 'Completed', 'Delayed').required()
});

// Main MIDP schema
const midpSchema = Joi.object({
  id: Joi.string().optional(),

  // Project Overview
  projectName: Joi.string().required().min(1).max(200),
  leadAppointedParty: Joi.string().required().min(1).max(200),
  informationManager: Joi.string().required().min(1).max(100),
  baselineDate: Joi.date().iso().required(),
  version: Joi.string().required(),

  // Delivery Milestones
  milestones: Joi.array().items(milestoneSchema).min(1).required(),

  // Risk Management
  riskRegister: Joi.array().items(riskSchema).optional(),

  // Progress Tracking
  kpis: Joi.array().items(kpiSchema).optional(),
  reportingSchedule: Joi.string().max(1000).allow('').optional(),

  // Metadata
  projectId: Joi.string().optional(),
  status: Joi.string().valid('Draft', 'Under Review', 'Approved', 'Active', 'Completed').optional(),
  createdBy: Joi.string().optional(),
  createdDate: Joi.date().iso().optional(),
  lastModified: Joi.date().iso().optional()
});

// MIDP creation from TIDPs schema
const midpFromTidpsSchema = Joi.object({
  midpData: Joi.object({
    projectName: Joi.string().required().min(1).max(200),
    leadAppointedParty: Joi.string().required().min(1).max(200),
    informationManager: Joi.string().required().min(1).max(100),
    baselineDate: Joi.date().iso().required(),
    version: Joi.string().optional().default('1.0'),
    projectId: Joi.string().optional(),
    status: Joi.string().valid('Draft', 'Under Review', 'Approved', 'Active', 'Completed').optional().default('Draft')
  }).required(),
  tidpIds: Joi.array().items(Joi.string()).min(1).max(50).required()
});

// Update schema (all fields optional except constraints)
const midpUpdateSchema = Joi.object({
  projectName: Joi.string().min(1).max(200).optional(),
  leadAppointedParty: Joi.string().min(1).max(200).optional(),
  informationManager: Joi.string().min(1).max(100).optional(),
  baselineDate: Joi.date().iso().optional(),
  version: Joi.string().optional(),
  milestones: Joi.array().items(milestoneSchema).optional(),
  riskRegister: Joi.array().items(riskSchema).optional(),
  kpis: Joi.array().items(kpiSchema).optional(),
  reportingSchedule: Joi.string().max(1000).allow('').optional(),
  projectId: Joi.string().optional(),
  status: Joi.string().valid('Draft', 'Under Review', 'Approved', 'Active', 'Completed').optional(),
  lastModified: Joi.date().iso().optional()
});

// Validation middleware functions
const validateMIDP = (req, res, next) => {
  const { error, value } = midpFromTidpsSchema.validate(req.body, {
    abortEarly: false,
    stripUnknown: true
  });

  if (error) {
    return res.status(400).json({
      success: false,
      error: 'MIDP validation failed',
      details: error.details.map(detail => ({
        field: detail.path.join('.'),
        message: detail.message,
        value: detail.context.value
      }))
    });
  }

  req.body = value;
  next();
};

const validateMIDPUpdate = (req, res, next) => {
  const { error, value } = midpUpdateSchema.validate(req.body, {
    abortEarly: false,
    stripUnknown: true
  });

  if (error) {
    return res.status(400).json({
      success: false,
      error: 'MIDP update validation failed',
      details: error.details.map(detail => ({
        field: detail.path.join('.'),
        message: detail.message,
        value: detail.context.value
      }))
    });
  }

  req.body = value;
  next();
};

const validateMIDPStandalone = (req, res, next) => {
  const { error, value } = midpSchema.validate(req.body, {
    abortEarly: false,
    stripUnknown: true
  });

  if (error) {
    return res.status(400).json({
      success: false,
      error: 'MIDP validation failed',
      details: error.details.map(detail => ({
        field: detail.path.join('.'),
        message: detail.message,
        value: detail.context.value
      }))
    });
  }

  req.body = value;
  next();
};

// Custom validation functions
const validateMilestoneSequence = (milestones) => {
  const issues = [];

  // Sort milestones by due date
  const sortedMilestones = milestones
    .filter(m => m['Due Date'])
    .sort((a, b) => new Date(a['Due Date']) - new Date(b['Due Date']));

  // Check for logical sequence
  for (let i = 1; i < sortedMilestones.length; i++) {
    const current = sortedMilestones[i];
    const previous = sortedMilestones[i - 1];

    const currentDate = new Date(current['Due Date']);
    const previousDate = new Date(previous['Due Date']);
    const daysDiff = (currentDate - previousDate) / (1000 * 60 * 60 * 24);

    // Check for milestones too close together
    if (daysDiff < 7) {
      issues.push({
        milestone: current['Milestone Name'],
        issue: `Only ${Math.ceil(daysDiff)} days after previous milestone`,
        severity: 'warning',
        suggestion: 'Consider extending timeline or combining milestones'
      });
    }

    // Check for unrealistic review durations
    const reviewDays = parseInt(current['Review Duration']) || 7;
    if (reviewDays > daysDiff / 2) {
      issues.push({
        milestone: current['Milestone Name'],
        issue: 'Review duration is too long relative to milestone spacing',
        severity: 'warning'
      });
    }
  }

  return issues;
};

const validateRiskConsistency = (risks) => {
  const issues = [];

  risks.forEach((risk, index) => {
    // Check impact vs probability consistency
    if (risk.Impact === 'Low' && risk.Probability === 'High') {
      issues.push({
        risk: risk['Risk Description'],
        issue: 'Low impact with high probability may need reassessment',
        severity: 'info'
      });
    }

    if (risk.Impact === 'High' && !risk['Contingency Plan']) {
      issues.push({
        risk: risk['Risk Description'],
        issue: 'High impact risks should have contingency plans',
        severity: 'warning'
      });
    }

    // Check for duplicate or similar risks
    for (let j = index + 1; j < risks.length; j++) {
      const otherRisk = risks[j];
      const similarity = calculateStringSimilarity(
        risk['Risk Description'].toLowerCase(),
        otherRisk['Risk Description'].toLowerCase()
      );

      if (similarity > 0.8) {
        issues.push({
          risk: risk['Risk Description'],
          issue: `Possible duplicate risk: "${otherRisk['Risk Description']}"`,
          severity: 'info'
        });
      }
    }
  });

  return issues;
};

const validateResourceAllocation = (aggregatedData) => {
  const issues = [];

  if (!aggregatedData) return issues;

  // Check for uneven workload distribution
  if (aggregatedData.estimatedHoursByDiscipline) {
    const disciplineHours = Object.values(aggregatedData.estimatedHoursByDiscipline);
    const maxHours = Math.max(...disciplineHours);
    const minHours = Math.min(...disciplineHours);

    if (maxHours > minHours * 4) {
      issues.push({
        issue: 'Significant workload imbalance between disciplines',
        severity: 'warning',
        details: `Max: ${maxHours}h, Min: ${minHours}h`
      });
    }
  }

  // Check total hours against typical project sizes
  if (aggregatedData.totalEstimatedHours) {
    const totalHours = aggregatedData.totalEstimatedHours;
    const containerCount = aggregatedData.totalContainers || 0;
    const avgHoursPerContainer = containerCount > 0 ? totalHours / containerCount : 0;

    if (avgHoursPerContainer > 100) {
      issues.push({
        issue: 'High average hours per container may indicate overestimation',
        severity: 'info',
        details: `${avgHoursPerContainer.toFixed(1)} hours per container`
      });
    }

    if (avgHoursPerContainer < 20 && containerCount > 10) {
      issues.push({
        issue: 'Low average hours per container may indicate underestimation',
        severity: 'warning',
        details: `${avgHoursPerContainer.toFixed(1)} hours per container`
      });
    }
  }

  return issues;
};

const validateTIDPIntegration = (midp, tidps) => {
  const issues = [];

  if (!tidps || tidps.length === 0) {
    return [{
      issue: 'No TIDPs available for validation',
      severity: 'error'
    }];
  }

  // Check if all referenced TIDPs exist
  midp.includedTIDPs?.forEach(tidpId => {
    const tidpExists = tidps.some(t => t.id === tidpId);
    if (!tidpExists) {
      issues.push({
        issue: `Referenced TIDP not found: ${tidpId}`,
        severity: 'error'
      });
    }
  });

  // Check for TIDP consistency
  const disciplines = new Set();
  tidps.forEach(tidp => {
    if (midp.includedTIDPs?.includes(tidp.id)) {
      disciplines.add(tidp.discipline);

      // Check TIDP completeness
      if (!tidp.containers || tidp.containers.length === 0) {
        issues.push({
          issue: `TIDP "${tidp.teamName}" has no information containers`,
          severity: 'warning'
        });
      }

      // Check for missing critical information
      if (!tidp.qualityChecks && !tidp.reviewProcess) {
        issues.push({
          issue: `TIDP "${tidp.teamName}" lacks quality requirements`,
          severity: 'info'
        });
      }
    }
  });

  // Check discipline coverage
  const criticalDisciplines = ['Architecture', 'Structural Engineering', 'MEP Engineering'];
  const missingCritical = criticalDisciplines.filter(d => !disciplines.has(d));

  if (missingCritical.length > 0) {
    issues.push({
      issue: `Missing critical disciplines: ${missingCritical.join(', ')}`,
      severity: 'warning'
    });
  }

  return issues;
};

// Helper function for string similarity
function calculateStringSimilarity(str1, str2) {
  const longer = str1.length > str2.length ? str1 : str2;
  const shorter = str1.length > str2.length ? str2 : str1;

  if (longer.length === 0) return 1.0;

  const editDistance = levenshteinDistance(longer, shorter);
  return (longer.length - editDistance) / longer.length;
}

function levenshteinDistance(str1, str2) {
  const matrix = [];

  for (let i = 0; i <= str2.length; i++) {
    matrix[i] = [i];
  }

  for (let j = 0; j <= str1.length; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= str2.length; i++) {
    for (let j = 1; j <= str1.length; j++) {
      if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }

  return matrix[str2.length][str1.length];
}

module.exports = {
  validateMIDP,
  validateMIDPUpdate,
  validateMIDPStandalone,
  validateMilestoneSequence,
  validateRiskConsistency,
  validateResourceAllocation,
  validateTIDPIntegration,
  midpSchema,
  midpFromTidpsSchema,
  midpUpdateSchema,
  kpiSchema,
  riskSchema,
  milestoneSchema
};