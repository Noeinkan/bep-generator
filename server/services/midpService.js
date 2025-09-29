const { v4: uuidv4 } = require('uuid');
const { format, addDays, parseISO, differenceInDays } = require('date-fns');
const _ = require('lodash');
const tidpService = require('./tidpService');

class MIDPService {
  constructor() {
    this.midps = new Map();
  }

  /**
   * Create MIDP by aggregating TIDPs
   * @param {Object} midpData - Basic MIDP information
   * @param {Array} tidpIds - Array of TIDP IDs to include
   * @returns {Object} Generated MIDP
   */
  createMIDPFromTIDPs(midpData, tidpIds) {
    const tidps = tidpIds.map(id => tidpService.getTIDP(id));

    const midp = {
      id: uuidv4(),
      ...midpData,
      includedTIDPs: tidpIds,
      aggregatedData: this.aggregateTIDPs(tidps),
      deliverySchedule: this.generateDeliverySchedule(tidps),
      riskRegister: this.consolidateRisks(tidps),
      dependencyMatrix: this.generateDependencyMatrix(tidps),
      resourcePlan: this.generateResourcePlan(tidps),
      qualityGates: this.defineQualityGates(tidps),
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      version: '1.0',
      status: 'Active'
    };

    this.midps.set(midp.id, midp);
    return midp;
  }

  /**
   * Update MIDP with new TIDP data
   * @param {string} midpId - MIDP ID
   * @param {Array} updatedTidpIds - Updated TIDP IDs
   * @returns {Object} Updated MIDP
   */
  updateMIDPFromTIDPs(midpId, updatedTidpIds) {
    const existingMidp = this.midps.get(midpId);
    if (!existingMidp) {
      throw new Error(`MIDP with ID ${midpId} not found`);
    }

    const tidps = updatedTidpIds.map(id => tidpService.getTIDP(id));

    const updatedMidp = {
      ...existingMidp,
      includedTIDPs: updatedTidpIds,
      aggregatedData: this.aggregateTIDPs(tidps),
      deliverySchedule: this.generateDeliverySchedule(tidps),
      riskRegister: this.consolidateRisks(tidps),
      dependencyMatrix: this.generateDependencyMatrix(tidps),
      resourcePlan: this.generateResourcePlan(tidps),
      updatedAt: new Date().toISOString(),
      version: this.incrementVersion(existingMidp.version)
    };

    this.midps.set(midpId, updatedMidp);
    return updatedMidp;
  }

  /**
   * Aggregate TIDP data into consolidated format
   * @param {Array} tidps - Array of TIDP objects
   * @returns {Object} Aggregated data
   */
  aggregateTIDPs(tidps) {
    const containers = [];
    const milestones = new Map();
    const disciplines = new Set();
    const totalEstimatedHours = new Map();

    tidps.forEach(tidp => {
      disciplines.add(tidp.discipline);

      if (tidp.containers) {
        tidp.containers.forEach(container => {
          const aggregatedContainer = {
            id: container.id,
            name: container['Container Name'] || container.name,
            type: container.Type || container.type,
            format: container.Format || container.format,
            loiLevel: container['LOI Level'] || container.levelOfInformation,
            author: container.Author || container.author,
            tidpSource: {
              id: tidp.id,
              teamName: tidp.teamName,
              discipline: tidp.discipline
            },
            estimatedTime: container['Est. Time'] || container.estimatedProductionTime,
            milestone: container.Milestone || container.deliveryMilestone,
            dueDate: container['Due Date'] || container.dueDate,
            status: container.Status || container.status,
            dependencies: this.processDependencies(container.dependencies || [], tidps)
          };

          containers.push(aggregatedContainer);

          // Group by milestone
          const milestone = aggregatedContainer.milestone;
          if (milestone) {
            if (!milestones.has(milestone)) {
              milestones.set(milestone, {
                name: milestone,
                containers: [],
                teams: new Set(),
                earliestDate: aggregatedContainer.dueDate,
                latestDate: aggregatedContainer.dueDate,
                totalEstimatedHours: 0
              });
            }

            const ms = milestones.get(milestone);
            ms.containers.push(aggregatedContainer);
            ms.teams.add(tidp.teamName);

            // Update date range
            if (aggregatedContainer.dueDate) {
              if (!ms.earliestDate || aggregatedContainer.dueDate < ms.earliestDate) {
                ms.earliestDate = aggregatedContainer.dueDate;
              }
              if (!ms.latestDate || aggregatedContainer.dueDate > ms.latestDate) {
                ms.latestDate = aggregatedContainer.dueDate;
              }
            }

            // Add estimated hours
            const hours = this.parseTimeToHours(aggregatedContainer.estimatedTime);
            ms.totalEstimatedHours += hours;

            // Track by discipline
            const discipline = tidp.discipline;
            if (!totalEstimatedHours.has(discipline)) {
              totalEstimatedHours.set(discipline, 0);
            }
            totalEstimatedHours.set(discipline, totalEstimatedHours.get(discipline) + hours);
          }
        });
      }
    });

    return {
      containers,
      milestones: Array.from(milestones.values()).map(ms => ({
        ...ms,
        teams: Array.from(ms.teams),
        reviewDuration: this.calculateReviewDuration(ms.containers.length),
        riskLevel: this.assessMilestoneRisk(ms)
      })),
      disciplines: Array.from(disciplines),
      totalContainers: containers.length,
      totalEstimatedHours: Array.from(totalEstimatedHours.values()).reduce((sum, hours) => sum + hours, 0),
      estimatedHoursByDiscipline: Object.fromEntries(totalEstimatedHours)
    };
  }

  /**
   * Generate delivery schedule with critical path analysis
   * @param {Array} tidps - Array of TIDP objects
   * @returns {Object} Delivery schedule
   */
  generateDeliverySchedule(tidps) {
    const schedule = {
      phases: [],
      criticalPath: [],
      bufferAnalysis: {},
      recommendedSequence: []
    };

    // Extract all containers with dates
    const allContainers = [];
    tidps.forEach(tidp => {
      if (tidp.containers) {
        tidp.containers.forEach(container => {
          if (container['Due Date'] || container.dueDate) {
            allContainers.push({
              ...container,
              tidpId: tidp.id,
              teamName: tidp.teamName,
              discipline: tidp.discipline,
              dueDate: container['Due Date'] || container.dueDate
            });
          }
        });
      }
    });

    // Sort by due date
    allContainers.sort((a, b) => new Date(a.dueDate) - new Date(b.dueDate));

    // Group into phases (monthly)
    const phases = _.groupBy(allContainers, container => {
      const date = new Date(container.dueDate);
      return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
    });

    schedule.phases = Object.entries(phases).map(([period, containers]) => ({
      period,
      containerCount: containers.length,
      disciplines: _.uniq(containers.map(c => c.discipline)),
      totalEstimatedHours: containers.reduce((sum, c) =>
        sum + this.parseTimeToHours(c['Est. Time'] || c.estimatedProductionTime || '0'), 0
      ),
      containers: containers.map(c => ({
        id: c.id,
        name: c['Container Name'] || c.name,
        team: c.teamName,
        dueDate: c.dueDate
      }))
    }));

    // Identify critical path (simplified)
    schedule.criticalPath = this.identifyCriticalPath(allContainers);

    // Buffer analysis
    schedule.bufferAnalysis = this.analyzeBuffers(allContainers);

    return schedule;
  }

  /**
   * Consolidate risks from all TIDPs
   * @param {Array} tidps - Array of TIDP objects
   * @returns {Array} Consolidated risk register
   */
  consolidateRisks(tidps) {
    const risks = [];
    const riskCategories = new Map();

    tidps.forEach(tidp => {
      // Add TIDP-specific risks
      if (tidp.risks) {
        tidp.risks.forEach(risk => {
          risks.push({
            id: uuidv4(),
            ...risk,
            source: `TIDP: ${tidp.teamName}`,
            category: risk.category || 'Information Delivery'
          });
        });
      }

      // Analyze containers for potential risks
      if (tidp.containers) {
        tidp.containers.forEach(container => {
          // Dependency risks
          if (container.dependencies && container.dependencies.length > 2) {
            risks.push({
              id: uuidv4(),
              description: `High dependency count for ${container['Container Name'] || container.name}`,
              impact: 'Medium',
              probability: 'Medium',
              category: 'Dependency',
              mitigation: 'Review and simplify dependencies where possible',
              source: `Auto-detected from ${tidp.teamName}`,
              affectedContainer: container.id
            });
          }

          // Timeline risks
          if (container['Est. Time'] || container.estimatedProductionTime) {
            const hours = this.parseTimeToHours(container['Est. Time'] || container.estimatedProductionTime);
            if (hours > 200) { // More than 5 weeks
              risks.push({
                id: uuidv4(),
                description: `Extended timeline for ${container['Container Name'] || container.name}`,
                impact: 'High',
                probability: 'Medium',
                category: 'Schedule',
                mitigation: 'Consider breaking into smaller deliverables',
                source: `Auto-detected from ${tidp.teamName}`,
                affectedContainer: container.id
              });
            }
          }
        });
      }
    });

    // Categorize and prioritize risks
    risks.forEach(risk => {
      const category = risk.category || 'General';
      if (!riskCategories.has(category)) {
        riskCategories.set(category, []);
      }
      riskCategories.get(category).push(risk);
    });

    return {
      risks: risks.sort((a, b) => this.getRiskScore(b) - this.getRiskScore(a)),
      categories: Object.fromEntries(riskCategories),
      summary: {
        total: risks.length,
        high: risks.filter(r => r.impact === 'High').length,
        medium: risks.filter(r => r.impact === 'Medium').length,
        low: risks.filter(r => r.impact === 'Low').length
      }
    };
  }

  /**
   * Generate resource allocation plan
   * @param {Array} tidps - Array of TIDP objects
   * @returns {Object} Resource plan
   */
  generateResourcePlan(tidps) {
    const plan = {
      byDiscipline: {},
      byTimePeriod: {},
      peakUtilization: {},
      recommendations: []
    };

    tidps.forEach(tidp => {
      const discipline = tidp.discipline;

      if (!plan.byDiscipline[discipline]) {
        plan.byDiscipline[discipline] = {
          teams: 0,
          totalHours: 0,
          containers: 0,
          peakMonth: null,
          peakHours: 0
        };
      }

      plan.byDiscipline[discipline].teams += 1;

      if (tidp.containers) {
        plan.byDiscipline[discipline].containers += tidp.containers.length;

        tidp.containers.forEach(container => {
          const hours = this.parseTimeToHours(container['Est. Time'] || container.estimatedProductionTime || '0');
          plan.byDiscipline[discipline].totalHours += hours;

          // Track by time period
          const dueDate = container['Due Date'] || container.dueDate;
          if (dueDate) {
            const period = format(new Date(dueDate), 'yyyy-MM');

            if (!plan.byTimePeriod[period]) {
              plan.byTimePeriod[period] = {
                totalHours: 0,
                disciplines: new Set(),
                containers: 0
              };
            }

            plan.byTimePeriod[period].totalHours += hours;
            plan.byTimePeriod[period].disciplines.add(discipline);
            plan.byTimePeriod[period].containers += 1;
          }
        });
      }
    });

    // Convert sets to arrays
    Object.keys(plan.byTimePeriod).forEach(period => {
      plan.byTimePeriod[period].disciplines = Array.from(plan.byTimePeriod[period].disciplines);
    });

    // Identify peak utilization
    const periods = Object.entries(plan.byTimePeriod);
    if (periods.length > 0) {
      const peakPeriod = periods.reduce((max, [period, data]) =>
        data.totalHours > max[1].totalHours ? [period, data] : max
      );

      plan.peakUtilization = {
        period: peakPeriod[0],
        hours: peakPeriod[1].totalHours,
        disciplines: peakPeriod[1].disciplines.length
      };
    }

    // Generate recommendations
    plan.recommendations = this.generateResourceRecommendations(plan);

    return plan;
  }

  /**
   * Define quality gates based on aggregated information
   * @param {Array} tidps - Array of TIDP objects
   * @returns {Array} Quality gates
   */
  defineQualityGates(tidps) {
    const gates = [];
    const milestones = new Set();

    tidps.forEach(tidp => {
      if (tidp.containers) {
        tidp.containers.forEach(container => {
          const milestone = container.Milestone || container.deliveryMilestone;
          if (milestone) {
            milestones.add(milestone);
          }
        });
      }
    });

    milestones.forEach(milestone => {
      gates.push({
        id: uuidv4(),
        name: `Quality Gate: ${milestone}`,
        milestone,
        criteria: [
          'All information containers completed and validated',
          'Clash detection performed and conflicts resolved',
          'Model federation successful',
          'Quality assurance checks passed',
          'Client review and approval obtained'
        ],
        approvers: ['Information Manager', 'Project Director', 'Client Representative'],
        estimatedDuration: '1 week',
        dependencies: [`Completion of all ${milestone} deliverables`]
      });
    });

    return gates.sort((a, b) => a.milestone.localeCompare(b.milestone));
  }

  // Helper methods
  parseTimeToHours(timeString) {
    if (!timeString) return 0;

    const timeStr = timeString.toString().toLowerCase();

    if (timeStr.includes('week')) {
      const weeks = parseFloat(timeStr);
      return weeks * 40;
    } else if (timeStr.includes('day')) {
      const days = parseFloat(timeStr);
      return days * 8;
    } else if (timeStr.includes('hour')) {
      return parseFloat(timeStr);
    }

    return 0;
  }

  processDependencies(dependencies, allTidps) {
    return dependencies.map(depId => {
      const sourceTidp = allTidps.find(tidp =>
        tidp.containers && tidp.containers.some(c => c.id === depId)
      );

      if (sourceTidp) {
        const sourceContainer = sourceTidp.containers.find(c => c.id === depId);
        return {
          dependencyId: depId,
          sourceTeam: sourceTidp.teamName,
          sourceDiscipline: sourceTidp.discipline,
          sourceContainer: sourceContainer ? sourceContainer['Container Name'] || sourceContainer.name : 'Unknown',
          type: 'information'
        };
      }

      return {
        dependencyId: depId,
        type: 'external'
      };
    });
  }

  calculateReviewDuration(containerCount) {
    // Base 1 week + 1 day per 3 containers
    const baseDays = 5; // 1 week
    const additionalDays = Math.ceil(containerCount / 3);
    const totalDays = baseDays + additionalDays;

    return `${Math.ceil(totalDays / 5)} week${totalDays > 5 ? 's' : ''}`;
  }

  assessMilestoneRisk(milestone) {
    let riskLevel = 'Low';

    if (milestone.containers.length > 10) riskLevel = 'Medium';
    if (milestone.containers.length > 20) riskLevel = 'High';
    if (milestone.teams.length > 5) riskLevel = 'High';
    if (milestone.totalEstimatedHours > 500) riskLevel = 'High';

    return riskLevel;
  }

  identifyCriticalPath(containers) {
    // Simplified critical path identification
    // In production, implement proper CPM algorithm
    return containers
      .filter(c => c.dependencies && c.dependencies.length > 0)
      .map(c => c.id);
  }

  analyzeBuffers(containers) {
    // Simplified buffer analysis
    return {
      recommendedBuffer: '10%',
      criticalContainers: containers.filter(c => c.dependencies && c.dependencies.length > 2).length,
      averageLeadTime: '2 weeks'
    };
  }

  getRiskScore(risk) {
    const impactScore = { 'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4 };
    const probabilityScore = { 'Low': 1, 'Medium': 2, 'High': 3 };

    return (impactScore[risk.impact] || 1) * (probabilityScore[risk.probability] || 1);
  }

  generateResourceRecommendations(plan) {
    const recommendations = [];

    // Check for peak utilization
    if (plan.peakUtilization.hours > 400) {
      recommendations.push({
        type: 'workload',
        priority: 'High',
        message: `Peak utilization in ${plan.peakUtilization.period} requires resource balancing`,
        action: 'Consider redistributing workload or adding temporary resources'
      });
    }

    // Check discipline balance
    const disciplineHours = Object.values(plan.byDiscipline).map(d => d.totalHours);
    const maxHours = Math.max(...disciplineHours);
    const minHours = Math.min(...disciplineHours);

    if (maxHours > minHours * 3) {
      recommendations.push({
        type: 'balance',
        priority: 'Medium',
        message: 'Uneven workload distribution across disciplines',
        action: 'Review scope distribution and consider load balancing'
      });
    }

    return recommendations;
  }

  incrementVersion(currentVersion) {
    const parts = currentVersion.split('.');
    const minor = parseInt(parts[1] || 0) + 1;
    return `${parts[0]}.${minor}.0`;
  }

  getMIDP(id) {
    const midp = this.midps.get(id);
    if (!midp) {
      throw new Error(`MIDP with ID ${id} not found`);
    }
    return midp;
  }

  getAllMIDPs() {
    return Array.from(this.midps.values());
  }

  deleteMIDP(id) {
    if (!this.midps.has(id)) {
      throw new Error(`MIDP with ID ${id} not found`);
    }
    return this.midps.delete(id);
  }
}

module.exports = new MIDPService();