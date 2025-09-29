const { v4: uuidv4 } = require('uuid');
const { format, addDays, parseISO } = require('date-fns');
const _ = require('lodash');

class TIDPService {
  constructor() {
    // In-memory storage for development. Replace with database in production.
    this.tidps = new Map();
  }

  /**
   * Create a new TIDP
   * @param {Object} tidpData - The TIDP data
   * @returns {Object} Created TIDP with generated ID
   */
  createTIDP(tidpData) {
    const tidp = {
      id: uuidv4(),
      ...tidpData,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      version: '1.0',
      status: 'Draft'
    };

    // Validate and process information containers
    if (tidp.containers) {
      tidp.containers = tidp.containers.map(container => ({
        id: container.id || uuidv4(),
        ...container,
        createdAt: new Date().toISOString()
      }));
    }

    // Process dependencies
    tidp.dependencies = this.processDependencies(tidp.dependencies || []);

    this.tidps.set(tidp.id, tidp);
    return tidp;
  }

  /**
   * Update an existing TIDP
   * @param {string} id - TIDP ID
   * @param {Object} updateData - Data to update
   * @returns {Object} Updated TIDP
   */
  updateTIDP(id, updateData) {
    const existingTidp = this.tidps.get(id);
    if (!existingTidp) {
      throw new Error(`TIDP with ID ${id} not found`);
    }

    const updatedTidp = {
      ...existingTidp,
      ...updateData,
      id, // Ensure ID doesn't change
      updatedAt: new Date().toISOString(),
      version: this.incrementVersion(existingTidp.version)
    };

    this.tidps.set(id, updatedTidp);
    return updatedTidp;
  }

  /**
   * Get TIDP by ID
   * @param {string} id - TIDP ID
   * @returns {Object} TIDP data
   */
  getTIDP(id) {
    const tidp = this.tidps.get(id);
    if (!tidp) {
      throw new Error(`TIDP with ID ${id} not found`);
    }
    return tidp;
  }

  /**
   * Get all TIDPs
   * @returns {Array} Array of all TIDPs
   */
  getAllTIDPs() {
    return Array.from(this.tidps.values());
  }

  /**
   * Get TIDPs by project
   * @param {string} projectId - Project ID
   * @returns {Array} Array of TIDPs for the project
   */
  getTIDPsByProject(projectId) {
    return Array.from(this.tidps.values())
      .filter(tidp => tidp.projectId === projectId);
  }

  /**
   * Delete TIDP
   * @param {string} id - TIDP ID
   * @returns {boolean} Success status
   */
  deleteTIDP(id) {
    if (!this.tidps.has(id)) {
      throw new Error(`TIDP with ID ${id} not found`);
    }
    return this.tidps.delete(id);
  }

  /**
   * Validate TIDP dependencies
   * @param {Array} dependencies - Array of dependencies
   * @returns {Object} Validation result
   */
  validateDependencies(dependencies) {
    const issues = [];
    const warnings = [];

    dependencies.forEach((dep, index) => {
      // Check for circular dependencies
      if (this.hasCircularDependency(dep)) {
        issues.push(`Circular dependency detected in dependency ${index + 1}`);
      }

      // Check if predecessor exists
      if (dep.predecessorId && !this.tidps.has(dep.predecessorId)) {
        warnings.push(`Predecessor TIDP ${dep.predecessorId} not found for dependency ${index + 1}`);
      }

      // Validate dates
      if (dep.requiredDate && dep.availableDate) {
        const required = parseISO(dep.requiredDate);
        const available = parseISO(dep.availableDate);

        if (available > required) {
          issues.push(`Dependency ${index + 1}: Available date is after required date`);
        }
      }
    });

    return {
      isValid: issues.length === 0,
      issues,
      warnings
    };
  }

  /**
   * Generate dependency matrix for a project
   * @param {string} projectId - Project ID
   * @returns {Object} Dependency matrix and analysis
   */
  generateDependencyMatrix(projectId) {
    const projectTidps = this.getTIDPsByProject(projectId);
    const matrix = [];
    const criticalPath = [];

    projectTidps.forEach(tidp => {
      if (tidp.containers) {
        tidp.containers.forEach(container => {
          const dependencies = container.dependencies || [];

          dependencies.forEach(depId => {
            const dependencyTidp = projectTidps.find(t =>
              t.containers && t.containers.some(c => c.id === depId)
            );

            if (dependencyTidp) {
              matrix.push({
                from: {
                  tidpId: dependencyTidp.id,
                  tidpName: dependencyTidp.teamName,
                  containerId: depId
                },
                to: {
                  tidpId: tidp.id,
                  tidpName: tidp.teamName,
                  containerId: container.id,
                  containerName: container['Container Name'] || container.name
                },
                type: 'information_dependency',
                criticalPath: this.isOnCriticalPath(dependencyTidp.id, tidp.id, projectTidps)
              });
            }
          });
        });
      }
    });

    return {
      matrix,
      criticalPath: this.calculateCriticalPath(projectTidps),
      summary: {
        totalDependencies: matrix.length,
        criticalDependencies: matrix.filter(m => m.criticalPath).length,
        teamsInvolved: _.uniq(matrix.flatMap(m => [m.from.tidpName, m.to.tidpName])).length
      }
    };
  }

  /**
   * Calculate resource allocation across TIDPs
   * @param {Array} tidps - Array of TIDPs
   * @returns {Object} Resource allocation summary
   */
  calculateResourceAllocation(tidps) {
    const allocation = {
      byDiscipline: {},
      byPeriod: {},
      totalResources: 0,
      peakUtilization: { period: null, resources: 0 }
    };

    tidps.forEach(tidp => {
      // Group by discipline
      const discipline = tidp.discipline;
      if (!allocation.byDiscipline[discipline]) {
        allocation.byDiscipline[discipline] = {
          teams: 0,
          containers: 0,
          estimatedHours: 0
        };
      }

      allocation.byDiscipline[discipline].teams += 1;
      allocation.byDiscipline[discipline].containers += tidp.containers?.length || 0;

      // Calculate estimated hours from containers
      if (tidp.containers) {
        tidp.containers.forEach(container => {
          const timeStr = container['Est. Time'] || container.estimatedProductionTime || '0';
          const hours = this.parseTimeToHours(timeStr);
          allocation.byDiscipline[discipline].estimatedHours += hours;
          allocation.totalResources += hours;
        });
      }
    });

    return allocation;
  }

  // Helper methods
  processDependencies(dependencies) {
    return dependencies.map(dep => ({
      id: dep.id || uuidv4(),
      ...dep,
      processedAt: new Date().toISOString()
    }));
  }

  incrementVersion(currentVersion) {
    const parts = currentVersion.split('.');
    const patch = parseInt(parts[2] || 0) + 1;
    return `${parts[0]}.${parts[1]}.${patch}`;
  }

  hasCircularDependency(dependency) {
    // Simplified circular dependency check
    // In a real implementation, this would be more sophisticated
    return false;
  }

  isOnCriticalPath(fromTidpId, toTidpId, allTidps) {
    // Simplified critical path calculation
    // In a real implementation, this would use proper CPM algorithms
    return false;
  }

  calculateCriticalPath(tidps) {
    // Simplified critical path calculation
    // Return array of TIDP IDs on critical path
    return [];
  }

  parseTimeToHours(timeString) {
    // Parse time strings like "2 weeks", "3 days", "40 hours"
    const timeStr = timeString.toLowerCase();

    if (timeStr.includes('week')) {
      const weeks = parseInt(timeStr);
      return weeks * 40; // Assuming 40 hours per week
    } else if (timeStr.includes('day')) {
      const days = parseInt(timeStr);
      return days * 8; // Assuming 8 hours per day
    } else if (timeStr.includes('hour')) {
      return parseInt(timeStr);
    }

    return 0;
  }

  /**
   * Export TIDP summary for reporting
   * @param {string} tidpId - TIDP ID
   * @returns {Object} TIDP summary for export
   */
  exportTIDPSummary(tidpId) {
    const tidp = this.getTIDP(tidpId);

    return {
      id: tidp.id,
      teamName: tidp.teamName,
      discipline: tidp.discipline,
      leader: tidp.leader,
      company: tidp.company,
      containerCount: tidp.containers?.length || 0,
      totalEstimatedHours: tidp.containers?.reduce((total, container) => {
        return total + this.parseTimeToHours(container['Est. Time'] || container.estimatedProductionTime || '0');
      }, 0) || 0,
      milestones: _.uniq(tidp.containers?.map(c => c.Milestone || c.deliveryMilestone) || []),
      lastUpdated: tidp.updatedAt,
      version: tidp.version,
      status: tidp.status
    };
  }
}

module.exports = new TIDPService();