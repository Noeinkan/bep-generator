import axios from 'axios';

// Default API URL: use 3001 for the local backend (can be overridden with REACT_APP_API_URL)
const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001';

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    // Add request timestamp for debugging
    config.metadata = { startTime: new Date() };
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);

    return config;
  },
  (error) => {
    console.error('❌ Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    const duration = new Date() - response.config.metadata.startTime;
    console.log(`✅ API Response: ${response.config.method?.toUpperCase()} ${response.config.url} (${duration}ms)`);
    return response;
  },
  (error) => {
    const duration = error.config?.metadata ? new Date() - error.config.metadata.startTime : 0;
    console.error(`❌ API Error: ${error.config?.method?.toUpperCase()} ${error.config?.url} (${duration}ms)`, error.response?.data);

    // Handle common HTTP errors
    if (error.response?.status === 401) {
      // Unauthorized - clear auth and redirect
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }

    return Promise.reject(error);
  }
);

class ApiService {
  // ======================
  // TIDP Services
  // ======================

  async getAllTIDPs(projectId = null) {
    try {
      const params = projectId ? { projectId } : {};
      const response = await apiClient.get('/tidp', { params });
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to fetch TIDPs');
    }
  }

  async getTIDP(id) {
    try {
      const response = await apiClient.get(`/tidp/${id}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to fetch TIDP ${id}`);
    }
  }

  async createTIDP(tidpData) {
    try {
      const response = await apiClient.post('/tidp', tidpData);
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to create TIDP');
    }
  }

  // New TIDP import methods
  async importTIDPsFromExcel(excelData, projectId) {
    try {
      const response = await apiClient.post('/tidp/import/excel', { data: excelData, projectId });
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to import TIDPs from Excel');
    }
  }

  async importTIDPsFromCSV(csvData, projectId) {
    try {
      const response = await apiClient.post('/tidp/import/csv', { data: csvData, projectId });
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to import TIDPs from CSV');
    }
  }

  async getTIDPImportTemplate() {
    try {
      const response = await apiClient.get('/tidp/template/excel');
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to fetch TIDP import template');
    }
  }

  async updateTIDP(id, updateData) {
    try {
      const response = await apiClient.put(`/tidp/${id}`, updateData);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to update TIDP ${id}`);
    }
  }

  async deleteTIDP(id) {
    try {
      const response = await apiClient.delete(`/tidp/${id}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to delete TIDP ${id}`);
    }
  }

  async validateTIDPDependencies(id) {
    try {
      const response = await apiClient.post(`/tidp/${id}/validate-dependencies`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to validate TIDP dependencies for ${id}`);
    }
  }

  async getTIDPSummary(id) {
    try {
      const response = await apiClient.get(`/tidp/${id}/summary`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to get TIDP summary for ${id}`);
    }
  }

  async generateDependencyMatrix(projectId) {
    try {
      const response = await apiClient.get(`/tidp/project/${projectId}/dependency-matrix`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to generate dependency matrix for project ${projectId}`);
    }
  }

  async getResourceAllocation(projectId) {
    try {
      const response = await apiClient.get(`/tidp/project/${projectId}/resource-allocation`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to get resource allocation for project ${projectId}`);
    }
  }

  async createTIDPBatch(tidps) {
    try {
      const response = await apiClient.post('/tidp/batch', { tidps });
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to create TIDPs in batch');
    }
  }

  async updateTIDPBatch(updates) {
    try {
      const response = await apiClient.put('/tidp/batch', { updates });
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to update TIDPs in batch');
    }
  }

  // ======================
  // MIDP Services
  // ======================

  async getAllMIDPs() {
    try {
      const response = await apiClient.get('/midp');
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to fetch MIDPs');
    }
  }

  async getMIDP(id) {
    try {
      const response = await apiClient.get(`/midp/${id}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to fetch MIDP ${id}`);
    }
  }

  async createMIDPFromTIDPs(midpData, tidpIds) {
    try {
      const response = await apiClient.post('/midp/from-tidps', { midpData, tidpIds });
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to create MIDP from TIDPs');
    }
  }

  // New MIDP methods
  async autoGenerateMIDP(projectId, midpData = {}) {
    try {
      const response = await apiClient.post(`/midp/auto-generate/${projectId}`, midpData);
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to auto-generate MIDP');
    }
  }

  async getMIDPEvolution(id) {
    try {
      const response = await apiClient.get(`/midp/${id}/evolution`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to fetch MIDP evolution');
    }
  }

  async getMIDPDeliverablesDashboard(id) {
    try {
      const response = await apiClient.get(`/midp/${id}/deliverables-dashboard`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to fetch MIDP deliverables dashboard');
    }
  }

  async updateMIDPFromTIDPs(id, tidpIds) {
    try {
      const response = await apiClient.put(`/midp/${id}/update-from-tidps`, { tidpIds });
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to update MIDP ${id} from TIDPs`);
    }
  }

  async deleteMIDP(id) {
    try {
      const response = await apiClient.delete(`/midp/${id}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to delete MIDP ${id}`);
    }
  }

  async getMIDPDeliverySchedule(id) {
    try {
      const response = await apiClient.get(`/midp/${id}/delivery-schedule`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to get delivery schedule for MIDP ${id}`);
    }
  }

  async getMIDPRiskRegister(id) {
    try {
      const response = await apiClient.get(`/midp/${id}/risk-register`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to get risk register for MIDP ${id}`);
    }
  }

  async getMIDPDependencyMatrix(id) {
    try {
      const response = await apiClient.get(`/midp/${id}/dependency-matrix`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to get dependency matrix for MIDP ${id}`);
    }
  }

  async getMIDPResourcePlan(id) {
    try {
      const response = await apiClient.get(`/midp/${id}/resource-plan`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to get resource plan for MIDP ${id}`);
    }
  }

  async getMIDPAggregatedData(id) {
    try {
      const response = await apiClient.get(`/midp/${id}/aggregated-data`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to get aggregated data for MIDP ${id}`);
    }
  }

  async getMIDPQualityGates(id) {
    try {
      const response = await apiClient.get(`/midp/${id}/quality-gates`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to get quality gates for MIDP ${id}`);
    }
  }

  async getMIDPMilestones(id) {
    try {
      const response = await apiClient.get(`/midp/${id}/milestones`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to get milestones for MIDP ${id}`);
    }
  }

  async refreshMIDP(id) {
    try {
      const response = await apiClient.post(`/midp/${id}/refresh`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to refresh MIDP ${id}`);
    }
  }

  async getMIDPDashboard(id) {
    try {
      const response = await apiClient.get(`/midp/${id}/dashboard`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to get dashboard for MIDP ${id}`);
    }
  }

  // ======================
  // Export Services
  // ======================

  async exportTIDPToExcel(id, template = null) {
    try {
      const body = template ? { template } : {};
      const response = await apiClient.post(`/export/tidp/${id}/excel`, body, {
        responseType: 'blob'
      });
      return this.downloadFile(response, `TIDP_${id}.xlsx`);
    } catch (error) {
      throw this.handleError(error, `Failed to export TIDP ${id} to Excel`);
    }
  }

  async exportTIDPToPDF(id, template = null) {
    try {
      const body = template ? { template } : {};
      const response = await apiClient.post(`/export/tidp/${id}/pdf`, body, {
        responseType: 'blob'
      });
      return this.downloadFile(response, `TIDP_${id}.pdf`);
    } catch (error) {
      throw this.handleError(error, `Failed to export TIDP ${id} to PDF`);
    }
  }

  async exportMIDPToExcel(id, template = null) {
    try {
      const body = template ? { template } : {};
      const response = await apiClient.post(`/export/midp/${id}/excel`, body, {
        responseType: 'blob'
      });
      return this.downloadFile(response, `MIDP_${id}.xlsx`);
    } catch (error) {
      throw this.handleError(error, `Failed to export MIDP ${id} to Excel`);
    }
  }

  async exportMIDPToPDF(id, template = null) {
    try {
      const body = template ? { template } : {};
      const response = await apiClient.post(`/export/midp/${id}/pdf`, body, {
        responseType: 'blob'
      });
      return this.downloadFile(response, `MIDP_${id}.pdf`);
    } catch (error) {
      throw this.handleError(error, `Failed to export MIDP ${id} to PDF`);
    }
  }

  async exportConsolidatedProject(projectId, midpId) {
    try {
      const response = await apiClient.post(`/export/project/${projectId}/consolidated-excel`,
        { midpId },
        { responseType: 'blob' }
      );
      return this.downloadFile(response, `Project_${projectId}_Consolidated.xlsx`);
    } catch (error) {
      throw this.handleError(error, `Failed to export consolidated project ${projectId}`);
    }
  }

  async getExportFormats() {
    try {
      const response = await apiClient.get('/export/formats');
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to get export formats');
    }
  }

  async getExportTemplates() {
    try {
      const response = await apiClient.get('/export/templates');
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to get export templates');
    }
  }

  async getTIDPExportPreview(id, format, template = null) {
    try {
      const body = template ? { format, template } : { format };
      const response = await apiClient.post(`/export/preview/tidp/${id}`, body);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to get TIDP ${id} export preview`);
    }
  }

  async getMIDPExportPreview(id, format, template = null) {
    try {
      const body = template ? { format, template } : { format };
      const response = await apiClient.post(`/export/preview/midp/${id}`, body);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to get MIDP ${id} export preview`);
    }
  }

  // ======================
  // Validation Services
  // ======================

  async validateTIDP(id) {
    try {
      const response = await apiClient.post(`/validation/tidp/${id}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to validate TIDP ${id}`);
    }
  }

  async validateMIDP(id) {
    try {
      const response = await apiClient.post(`/validation/midp/${id}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to validate MIDP ${id}`);
    }
  }

  async validateProjectComprehensive(projectId, midpId) {
    try {
      const response = await apiClient.post(`/validation/project/${projectId}/comprehensive`, { midpId });
      return response.data;
    } catch (error) {
      throw this.handleError(error, `Failed to validate project ${projectId} comprehensively`);
    }
  }

  async getISO19650Standards() {
    try {
      const response = await apiClient.get('/validation/standards/iso19650');
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to get ISO 19650 standards');
    }
  }

  // ======================
  // Utility Methods
  // ======================

  downloadFile(response, defaultFilename) {
    try {
      // Get filename from Content-Disposition header if available
      const contentDisposition = response.headers['content-disposition'];
      let filename = defaultFilename;

      if (contentDisposition) {
        const fileNameMatch = contentDisposition.match(/filename="?([^"]*)"?/);
        if (fileNameMatch && fileNameMatch[1]) {
          filename = fileNameMatch[1];
        }
      }

      // Create blob and download
      const blob = new Blob([response.data], {
        type: response.headers['content-type'] || 'application/octet-stream'
      });

      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      return { success: true, filename };
    } catch (error) {
      console.error('Failed to download file:', error);
      throw new Error('Failed to download file');
    }
  }

  handleError(error, defaultMessage) {
    if (error.response?.data?.error) {
      // API returned structured error
      return new Error(error.response.data.error);
    } else if (error.response?.data?.details) {
      // Validation error with details
      const details = error.response.data.details;
      const detailMessages = Array.isArray(details)
        ? details.map(d => d.message || d).join(', ')
        : details;
      return new Error(`${defaultMessage}: ${detailMessages}`);
    } else if (error.message) {
      // Network or other error
      return new Error(`${defaultMessage}: ${error.message}`);
    } else {
      // Fallback
      return new Error(defaultMessage);
    }
  }

  // ======================
  // Health Check
  // ======================

  async healthCheck() {
    try {
      const response = await apiClient.get('/health', { timeout: 5000 });
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Health check failed');
    }
  }

  // ======================
  // Batch Operations
  // ======================

  async batchOperation(operations) {
    try {
      const promises = operations.map(async (operation) => {
        try {
          const result = await this[operation.method](...operation.args);
          return { success: true, operation: operation.id, result };
        } catch (error) {
          return { success: false, operation: operation.id, error: error.message };
        }
      });

      const results = await Promise.allSettled(promises);
      return results.map(result => result.value || { success: false, error: result.reason });
    } catch (error) {
      throw this.handleError(error, 'Batch operation failed');
    }
  }

  // ======================
  // Cache Management
  // ======================

  clearCache() {
    // Clear any cached data if implementing caching
    console.log('🗑️ API cache cleared');
  }

  // ======================
  // Configuration
  // ======================

  getConfig() {
    return {
      baseURL: BASE_URL,
      timeout: apiClient.defaults.timeout,
      headers: apiClient.defaults.headers
    };
  }

  updateConfig(config) {
    Object.assign(apiClient.defaults, config);
  }
}

const apiServiceInstance = new ApiService();

export default apiServiceInstance;