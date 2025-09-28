const express = require('express');
const router = express.Router();
const tidpService = require('../services/tidpService');
const midpService = require('../services/midpService');
const {
  validateContainerDependencies,
  validateMilestoneDates
} = require('../validators/tidpValidator');
const {
  validateMilestoneSequence,
  validateRiskConsistency,
  validateResourceAllocation,
  validateTIDPIntegration
} = require('../validators/midpValidator');

/**
 * POST /api/validation/tidp/:id
 * Comprehensive TIDP validation
 */
router.post('/tidp/:id', async (req, res, next) => {
  try {
    const { id } = req.params;
    const tidp = tidpService.getTIDP(id);

    const validation = {
      isValid: true,
      issues: [],
      warnings: [],
      recommendations: [],
      summary: {
        totalIssues: 0,
        criticalIssues: 0,
        warnings: 0,
        completeness: 0
      }
    };

    // 1. Container Dependencies Validation
    if (tidp.containers) {
      const dependencyIssues = validateContainerDependencies(tidp.containers);
      dependencyIssues.forEach(issue => {
        if (issue.severity === 'error') {
          validation.issues.push(issue);
          validation.isValid = false;
        } else {
          validation.warnings.push(issue);
        }
      });
    }

    // 2. Milestone Date Validation
    if (tidp.containers) {
      const milestoneIssues = validateMilestoneDates(tidp.containers);
      validation.warnings.push(...milestoneIssues);
    }

    // 3. Completeness Check
    const completeness = calculateTIDPCompleteness(tidp);
    validation.summary.completeness = completeness.percentage;
    validation.recommendations.push(...completeness.recommendations);

    // 4. Quality Requirements Check
    if (!tidp.qualityChecks && !tidp.reviewProcess) {
      validation.warnings.push({
        category: 'Quality',
        issue: 'No quality checking procedures or review process defined',
        severity: 'warning',
        recommendation: 'Add quality assurance procedures to ensure deliverable standards'
      });
    }

    // 5. Team Information Validation
    if (!tidp.teamMembers || tidp.teamMembers.length === 0) {
      validation.warnings.push({
        category: 'Team',
        issue: 'No team members defined',
        severity: 'info',
        recommendation: 'Consider adding team member details for better coordination'
      });
    }

    // 6. Resource Estimation Validation
    if (tidp.containers) {
      const resourceIssues = validateResourceEstimation(tidp.containers);
      validation.warnings.push(...resourceIssues);
    }

    // 7. Timeline Validation
    if (tidp.containers) {
      const timelineIssues = validateTimeline(tidp.containers);
      validation.warnings.push(...timelineIssues);
    }

    // Update summary
    validation.summary.totalIssues = validation.issues.length + validation.warnings.length;
    validation.summary.criticalIssues = validation.issues.length;
    validation.summary.warnings = validation.warnings.length;

    res.json({
      success: true,
      data: validation
    });
  } catch (error) {
    if (error.message.includes('not found')) {
      return res.status(404).json({
        success: false,
        error: error.message
      });
    }
    next(error);
  }
});

/**
 * POST /api/validation/midp/:id
 * Comprehensive MIDP validation
 */
router.post('/midp/:id', async (req, res, next) => {
  try {
    const { id } = req.params;
    const midp = midpService.getMIDP(id);

    const validation = {
      isValid: true,
      issues: [],
      warnings: [],
      recommendations: [],
      summary: {
        totalIssues: 0,
        criticalIssues: 0,
        warnings: 0,
        integrationScore: 0
      }
    };

    // 1. Milestone Sequence Validation
    if (midp.aggregatedData && midp.aggregatedData.milestones) {
      const milestoneIssues = validateMilestoneSequence(midp.aggregatedData.milestones);
      validation.warnings.push(...milestoneIssues);
    }

    // 2. Risk Register Validation
    if (midp.riskRegister && midp.riskRegister.risks) {
      const riskIssues = validateRiskConsistency(midp.riskRegister.risks);
      validation.warnings.push(...riskIssues);
    }

    // 3. Resource Allocation Validation
    if (midp.aggregatedData) {
      const resourceIssues = validateResourceAllocation(midp.aggregatedData);
      validation.warnings.push(...resourceIssues);
    }

    // 4. TIDP Integration Validation
    if (midp.includedTIDPs) {
      const tidps = midp.includedTIDPs.map(id => {
        try {
          return tidpService.getTIDP(id);
        } catch (error) {
          return null;
        }
      }).filter(Boolean);

      const integrationIssues = validateTIDPIntegration(midp, tidps);
      integrationIssues.forEach(issue => {
        if (issue.severity === 'error') {
          validation.issues.push(issue);
          validation.isValid = false;
        } else {
          validation.warnings.push(issue);
        }
      });

      // Calculate integration score
      validation.summary.integrationScore = calculateIntegrationScore(midp, tidps);
    }

    // 5. Delivery Schedule Validation
    if (midp.deliverySchedule) {
      const scheduleIssues = validateDeliverySchedule(midp.deliverySchedule);
      validation.warnings.push(...scheduleIssues);
    }

    // 6. Quality Gates Validation
    if (midp.qualityGates) {
      const qualityIssues = validateQualityGates(midp.qualityGates, midp.aggregatedData);
      validation.recommendations.push(...qualityIssues);
    }

    // 7. Critical Path Analysis
    if (midp.dependencyMatrix) {
      const criticalPathIssues = analyzeCriticalPath(midp.dependencyMatrix);
      validation.recommendations.push(...criticalPathIssues);
    }

    // Update summary
    validation.summary.totalIssues = validation.issues.length + validation.warnings.length;
    validation.summary.criticalIssues = validation.issues.length;
    validation.summary.warnings = validation.warnings.length;

    res.json({
      success: true,
      data: validation
    });
  } catch (error) {
    if (error.message.includes('not found')) {
      return res.status(404).json({
        success: false,
        error: error.message
      });
    }
    next(error);
  }
});

/**
 * POST /api/validation/project/:projectId/comprehensive
 * Comprehensive project validation (all TIDPs + MIDP)
 */
router.post('/project/:projectId/comprehensive', async (req, res, next) => {
  try {
    const { projectId } = req.params;
    const { midpId } = req.body;

    if (!midpId) {
      return res.status(400).json({
        success: false,
        error: 'MIDP ID is required for comprehensive validation'
      });
    }

    const tidps = tidpService.getTIDPsByProject(projectId);
    const midp = midpService.getMIDP(midpId);

    const validation = {
      projectId,
      isValid: true,
      overallScore: 0,
      tidpValidations: [],
      midpValidation: null,
      projectLevelIssues: [],
      recommendations: [],
      summary: {
        totalTIDPs: tidps.length,
        validTIDPs: 0,
        totalIssues: 0,
        criticalIssues: 0,
        readinessScore: 0
      }
    };

    // 1. Validate each TIDP
    for (const tidp of tidps) {
      const tidpValidation = await validateSingleTIDP(tidp);
      validation.tidpValidations.push({
        tidpId: tidp.id,
        teamName: tidp.teamName,
        discipline: tidp.discipline,
        ...tidpValidation
      });

      if (tidpValidation.isValid) {
        validation.summary.validTIDPs++;
      }
    }

    // 2. Validate MIDP
    validation.midpValidation = await validateSingleMIDP(midp, tidps);

    // 3. Project-level cross-validation
    const crossValidation = validateProjectIntegration(tidps, midp);
    validation.projectLevelIssues = crossValidation.issues;
    validation.recommendations.push(...crossValidation.recommendations);

    // 4. Calculate overall scores
    validation.summary.totalIssues = validation.tidpValidations.reduce((sum, tv) => sum + tv.summary.totalIssues, 0) +
                                   validation.midpValidation.summary.totalIssues +
                                   validation.projectLevelIssues.length;

    validation.summary.criticalIssues = validation.tidpValidations.reduce((sum, tv) => sum + tv.summary.criticalIssues, 0) +
                                      validation.midpValidation.summary.criticalIssues +
                                      validation.projectLevelIssues.filter(i => i.severity === 'error').length;

    validation.summary.readinessScore = calculateProjectReadinessScore(validation);
    validation.overallScore = calculateOverallScore(validation);

    // Determine if project is valid
    validation.isValid = validation.summary.criticalIssues === 0 && validation.summary.readinessScore >= 70;

    res.json({
      success: true,
      data: validation
    });
  } catch (error) {
    next(error);
  }
});

/**
 * GET /api/validation/standards/iso19650
 * Get ISO 19650 compliance checklist
 */
router.get('/standards/iso19650', (req, res) => {
  const checklist = {
    tidpRequirements: [
      {
        requirement: 'Task team identification and responsibilities',
        description: 'Clear identification of task team leader, members, and responsibilities',
        section: 'Team Information',
        mandatory: true
      },
      {
        requirement: 'Information container identification',
        description: 'All information containers must be identified with appropriate level of information need',
        section: 'Information Containers',
        mandatory: true
      },
      {
        requirement: 'Delivery schedule alignment',
        description: 'TIDP delivery schedule must align with project milestones',
        section: 'Schedule',
        mandatory: true
      },
      {
        requirement: 'Quality assurance procedures',
        description: 'Define quality checking and validation procedures',
        section: 'Quality',
        mandatory: true
      },
      {
        requirement: 'Information dependencies',
        description: 'Identify and document information dependencies',
        section: 'Dependencies',
        mandatory: false
      }
    ],
    midpRequirements: [
      {
        requirement: 'TIDP aggregation',
        description: 'All task team TIDPs must be aggregated into the MIDP',
        section: 'Aggregation',
        mandatory: true
      },
      {
        requirement: 'Delivery milestone definition',
        description: 'Clear definition of information delivery milestones',
        section: 'Milestones',
        mandatory: true
      },
      {
        requirement: 'Risk register maintenance',
        description: 'Consolidated risk register with mitigation strategies',
        section: 'Risk Management',
        mandatory: true
      },
      {
        requirement: 'Resource allocation planning',
        description: 'Resource planning across all task teams',
        section: 'Resources',
        mandatory: false
      },
      {
        requirement: 'Progress monitoring procedures',
        description: 'Procedures for monitoring and reporting progress',
        section: 'Monitoring',
        mandatory: true
      }
    ],
    complianceScoring: {
      mandatory: 100, // All mandatory requirements must be met
      recommended: 70, // 70% of recommended requirements should be met
      overall: 85 // Overall compliance score should be >= 85%
    }
  };

  res.json({
    success: true,
    data: checklist
  });
});

// Helper functions
async function validateSingleTIDP(tidp) {
  // Simplified version of TIDP validation
  const validation = {
    isValid: true,
    issues: [],
    warnings: [],
    summary: { totalIssues: 0, criticalIssues: 0, completeness: 0 }
  };

  // Basic validation logic
  if (!tidp.containers || tidp.containers.length === 0) {
    validation.issues.push({
      category: 'Content',
      issue: 'No information containers defined',
      severity: 'error'
    });
    validation.isValid = false;
  }

  const completeness = calculateTIDPCompleteness(tidp);
  validation.summary.completeness = completeness.percentage;

  validation.summary.totalIssues = validation.issues.length + validation.warnings.length;
  validation.summary.criticalIssues = validation.issues.length;

  return validation;
}

async function validateSingleMIDP(midp, tidps) {
  // Simplified version of MIDP validation
  const validation = {
    isValid: true,
    issues: [],
    warnings: [],
    summary: { totalIssues: 0, criticalIssues: 0, integrationScore: 0 }
  };

  // Basic validation logic
  if (!midp.includedTIDPs || midp.includedTIDPs.length === 0) {
    validation.issues.push({
      category: 'Integration',
      issue: 'No TIDPs included in MIDP',
      severity: 'error'
    });
    validation.isValid = false;
  }

  validation.summary.integrationScore = calculateIntegrationScore(midp, tidps);
  validation.summary.totalIssues = validation.issues.length + validation.warnings.length;
  validation.summary.criticalIssues = validation.issues.length;

  return validation;
}

function calculateTIDPCompleteness(tidp) {
  const requiredFields = [
    'teamName', 'discipline', 'leader', 'company', 'responsibilities', 'containers'
  ];
  const optionalFields = [
    'teamMembers', 'predecessors', 'qualityChecks', 'reviewProcess'
  ];

  let completedRequired = 0;
  let completedOptional = 0;
  const recommendations = [];

  requiredFields.forEach(field => {
    if (tidp[field]) {
      if (Array.isArray(tidp[field]) ? tidp[field].length > 0 : tidp[field].trim()) {
        completedRequired++;
      }
    }
  });

  optionalFields.forEach(field => {
    if (tidp[field]) {
      if (Array.isArray(tidp[field]) ? tidp[field].length > 0 : tidp[field].trim()) {
        completedOptional++;
      } else {
        recommendations.push({
          category: 'Completeness',
          recommendation: `Consider adding ${field} for better coordination`
        });
      }
    }
  });

  const percentage = ((completedRequired / requiredFields.length) * 70) +
                    ((completedOptional / optionalFields.length) * 30);

  return {
    percentage: Math.round(percentage),
    recommendations
  };
}

function calculateIntegrationScore(midp, tidps) {
  if (!tidps || tidps.length === 0) return 0;

  let score = 0;
  let maxScore = 0;

  // Check TIDP inclusion
  const includedCount = midp.includedTIDPs ? midp.includedTIDPs.length : 0;
  score += (includedCount / tidps.length) * 30;
  maxScore += 30;

  // Check aggregated data quality
  if (midp.aggregatedData) {
    if (midp.aggregatedData.totalContainers > 0) score += 20;
    if (midp.aggregatedData.milestones && midp.aggregatedData.milestones.length > 0) score += 20;
    if (midp.dependencyMatrix) score += 15;
    if (midp.resourcePlan) score += 15;
    maxScore += 70;
  }

  return Math.round((score / maxScore) * 100);
}

function validateProjectIntegration(tidps, midp) {
  const issues = [];
  const recommendations = [];

  // Check discipline coverage
  const disciplines = new Set(tidps.map(t => t.discipline));
  const criticalDisciplines = ['Architecture', 'Structural Engineering', 'MEP Engineering'];
  const missing = criticalDisciplines.filter(d => !disciplines.has(d));

  if (missing.length > 0) {
    issues.push({
      category: 'Coverage',
      issue: `Missing critical disciplines: ${missing.join(', ')}`,
      severity: 'warning'
    });
  }

  // Check container dependencies across TIDPs
  const allContainers = tidps.flatMap(t => t.containers || []);
  const dependencyIssues = validateContainerDependencies(allContainers, tidps);
  issues.push(...dependencyIssues.filter(i => i.severity === 'error'));

  // Check milestone alignment
  const milestoneMap = new Map();
  allContainers.forEach(container => {
    const milestone = container.Milestone || container.deliveryMilestone;
    if (milestone) {
      if (!milestoneMap.has(milestone)) {
        milestoneMap.set(milestone, []);
      }
      milestoneMap.get(milestone).push(container);
    }
  });

  if (milestoneMap.size === 0) {
    issues.push({
      category: 'Schedule',
      issue: 'No delivery milestones defined across TIDPs',
      severity: 'error'
    });
  }

  return { issues, recommendations };
}

function calculateProjectReadinessScore(validation) {
  let score = 0;

  // TIDP completeness (40%)
  const avgTidpCompleteness = validation.tidpValidations.reduce((sum, tv) =>
    sum + (tv.summary?.completeness || 0), 0) / validation.tidpValidations.length;
  score += (avgTidpCompleteness * 0.4);

  // MIDP integration (30%)
  const midpIntegration = validation.midpValidation?.summary?.integrationScore || 0;
  score += (midpIntegration * 0.3);

  // Critical issues penalty (30%)
  const criticalPenalty = Math.min(validation.summary.criticalIssues * 10, 30);
  score += Math.max(0, 30 - criticalPenalty);

  return Math.round(score);
}

function calculateOverallScore(validation) {
  const readinessScore = validation.summary.readinessScore;
  const issueRatio = validation.summary.totalIssues / Math.max(validation.summary.totalTIDPs, 1);
  const issuePenalty = Math.min(issueRatio * 10, 20);

  return Math.max(0, Math.round(readinessScore - issuePenalty));
}

function validateResourceEstimation(containers) {
  const issues = [];

  containers.forEach(container => {
    const estimatedTime = container['Est. Time'] || container.estimatedProductionTime;
    if (!estimatedTime) {
      issues.push({
        category: 'Resources',
        issue: `No time estimation for container: ${container['Container Name'] || container.name}`,
        severity: 'warning'
      });
    }
  });

  return issues;
}

function validateTimeline(containers) {
  const issues = [];
  const now = new Date();

  containers.forEach(container => {
    const dueDate = container['Due Date'] || container.dueDate;
    if (dueDate) {
      const due = new Date(dueDate);
      if (due < now) {
        issues.push({
          category: 'Schedule',
          issue: `Past due date for container: ${container['Container Name'] || container.name}`,
          severity: 'warning'
        });
      }
    }
  });

  return issues;
}

function validateDeliverySchedule(schedule) {
  const issues = [];

  if (!schedule.phases || schedule.phases.length === 0) {
    issues.push({
      category: 'Schedule',
      issue: 'No delivery phases defined',
      severity: 'warning'
    });
  }

  return issues;
}

function validateQualityGates(qualityGates, aggregatedData) {
  const recommendations = [];

  if (!qualityGates || qualityGates.length === 0) {
    recommendations.push({
      category: 'Quality',
      recommendation: 'Consider defining quality gates for milestone reviews'
    });
  }

  return recommendations;
}

function analyzeCriticalPath(dependencyMatrix) {
  const recommendations = [];

  if (!dependencyMatrix.summary) {
    return recommendations;
  }

  if (dependencyMatrix.summary.criticalDependencies > 0) {
    recommendations.push({
      category: 'Schedule',
      recommendation: 'Monitor critical path dependencies closely'
    });
  }

  return recommendations;
}

module.exports = router;