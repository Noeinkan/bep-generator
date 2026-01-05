// Since bepConfig.js is an ES6 module, we'll use dynamic import or hardcode essential data
// For now, we'll define the essential structure here
const CONFIG = {
  bepTypeDefinitions: {
    'pre-appointment': {
      title: 'Pre-Appointment BEP',
      subtitle: 'Tender Phase Document',
      description: 'A document outlining the prospective delivery team\'s proposed approach, capability, and capacity.'
    },
    'post-appointment': {
      title: 'Post-Appointment BEP',
      subtitle: 'Project Execution Document',
      description: 'Confirms the delivery team\'s information management approach and includes detailed planning and schedules.'
    }
  },
  steps: [],
  formFields: {}
};

// Try to load CONFIG dynamically (this will work after build)
try {
  const configPath = require.resolve('../../src/config/bepConfig.js');
  delete require.cache[configPath];
  const loadedConfig = require('../../src/config/bepConfig');
  if (loadedConfig.default) {
    Object.assign(CONFIG, loadedConfig.default);
  } else if (loadedConfig.CONFIG) {
    Object.assign(CONFIG, loadedConfig.CONFIG);
  } else {
    Object.assign(CONFIG, loadedConfig);
  }
} catch (error) {
  console.warn('⚠️  Could not load bepConfig.js, using default structure:', error.message);
}

/**
 * HTML Template Service
 * Generates complete HTML for BEP PDF generation
 * Replicates the structure of BepPreviewRenderer.js with inline CSS
 */
class HtmlTemplateService {
  /**
   * Generate complete BEP HTML document
   * @param {Object} formData - Form data
   * @param {string} bepType - BEP type (pre-appointment/post-appointment)
   * @param {Array} tidpData - TIDP data
   * @param {Array} midpData - MIDP data
   * @param {Object} componentImages - Map of fieldName -> base64 image
   * @returns {Promise<string>} Complete HTML document
   */
  async generateBEPHTML(formData, bepType, tidpData = [], midpData = [], componentImages = {}) {
    const css = this.getInlineCSS();
    const bodyContent = this.renderBEPContent(formData, bepType, tidpData, midpData, componentImages);

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BIM Execution Plan - ${formData.projectName || 'BEP'}</title>
  <style>${css}</style>
</head>
<body>
  ${bodyContent}
</body>
</html>`;
  }

  /**
   * Render BEP content body
   */
  renderBEPContent(formData, bepType, tidpData, midpData, componentImages) {
    const bepConfig = CONFIG.bepTypeDefinitions?.[bepType] || CONFIG.bepTypeDefinitions['pre-appointment'];

    let html = '<div class="container">';

    // Cover page
    html += this.renderCoverPage(formData, bepType, bepConfig);

    // Content sections
    html += this.renderContentSections(formData, bepType, componentImages);

    // TIDP/MIDP sections
    if (tidpData.length > 0 || midpData.length > 0) {
      html += this.renderTIDPMIDPSections(tidpData, midpData);
    }

    html += '</div>';

    return html;
  }

  /**
   * Render cover page
   */
  renderCoverPage(formData, bepType, bepConfig) {
    const projectName = this.escapeHtml(formData.projectName || 'Not specified');
    const projectNumber = this.escapeHtml(formData.projectNumber || 'Not specified');
    const bepTitle = this.escapeHtml(bepConfig.title || bepType);
    const now = new Date();
    const dateStr = now.toLocaleDateString();
    const timeStr = now.toLocaleTimeString();

    return `
      <div class="cover-page">
        <h1 class="cover-title">BIM EXECUTION PLAN</h1>
        <p class="cover-subtitle">${bepTitle}</p>
        <p class="cover-iso">ISO 19650-2:2018 Compliant</p>
        <div class="cover-info">
          <p class="cover-info-item"><strong>Project:</strong> ${projectName}</p>
          <p class="cover-info-item"><strong>Project Number:</strong> ${projectNumber}</p>
          <p class="cover-info-date">Generated: ${dateStr} ${timeStr}</p>
        </div>
      </div>
    `;
  }

  /**
   * Render content sections
   */
  renderContentSections(formData, bepType, componentImages) {
    let html = '';
    const steps = CONFIG.steps || [];

    steps.forEach((step, stepIndex) => {
      const stepConfig = this.getFormFields(bepType, stepIndex);
      if (!stepConfig || !stepConfig.fields) return;

      // Section header
      html += `
        <div class="section">
          <div class="section-header">
            <h2 class="section-title">${stepConfig.number}. ${this.escapeHtml(stepConfig.title)}</h2>
          </div>
          <div class="section-content">
      `;

      // Section fields
      stepConfig.fields.forEach(field => {
        if (field.type === 'section-header') return;

        const value = formData[field.name];
        if (!value) return;

        html += `
          <div class="field-group">
            <h3 class="field-label">${field.number ? field.number + ' ' : ''}${this.escapeHtml(field.label)}</h3>
            ${this.renderFieldValue(field, value, componentImages)}
          </div>
        `;
      });

      html += `
          </div>
        </div>
      `;
    });

    return html;
  }

  /**
   * Render field value based on type
   */
  renderFieldValue(field, value, componentImages) {
    // Custom visual components - use embedded screenshots
    const visualComponentTypes = ['orgchart', 'cdeDiagram', 'mindmap', 'fileStructure', 'naming-conventions', 'federation-strategy'];

    if (visualComponentTypes.includes(field.type)) {
      return this.renderComponentImage(field.name, componentImages);
    }

    // Standard field types
    switch (field.type) {
      case 'table':
        return this.renderTable(field, value);

      case 'checkbox':
        return this.renderCheckboxList(value);

      case 'textarea':
        return this.renderTextarea(value);

      case 'introTable':
        return this.renderIntroTable(field, value);

      default:
        return this.renderSimpleField(value);
    }
  }

  /**
   * Render component screenshot image
   */
  renderComponentImage(fieldName, componentImages) {
    const imageData = componentImages[fieldName];
    if (!imageData) {
      return `<div class="component-placeholder">Visual component not available</div>`;
    }

    return `
      <div class="component-image">
        <img src="${imageData}" alt="${this.escapeHtml(fieldName)}" />
      </div>
    `;
  }

  /**
   * Render table
   */
  renderTable(field, rows) {
    if (!Array.isArray(rows) || rows.length === 0) return '';

    const columns = field.columns || [];

    let html = '<div class="table-wrapper"><table class="data-table"><thead><tr>';

    columns.forEach(col => {
      html += `<th>${this.escapeHtml(col)}</th>`;
    });

    html += '</tr></thead><tbody>';

    rows.forEach(row => {
      html += '<tr>';
      columns.forEach(col => {
        html += `<td>${this.escapeHtml(row[col] || '-')}</td>`;
      });
      html += '</tr>';
    });

    html += '</tbody></table></div>';

    return html;
  }

  /**
   * Render checkbox list
   */
  renderCheckboxList(items) {
    if (!Array.isArray(items) || items.length === 0) return '';

    let html = '<ul class="checkbox-list">';

    items.forEach(item => {
      html += `<li class="checkbox-item"><span class="checkmark">✓</span> ${this.escapeHtml(item)}</li>`;
    });

    html += '</ul>';

    return html;
  }

  /**
   * Render textarea
   */
  renderTextarea(value) {
    return `<p class="textarea-content">${this.escapeHtml(value).replace(/\n/g, '<br>')}</p>`;
  }

  /**
   * Render introTable
   */
  renderIntroTable(field, value) {
    let html = '';

    if (value.intro) {
      html += `<p class="intro-text">${this.escapeHtml(value.intro).replace(/\n/g, '<br>')}</p>`;
    }

    if (value.rows && Array.isArray(value.rows) && value.rows.length > 0) {
      const columns = field.tableColumns || [];

      html += '<div class="table-wrapper"><table class="data-table"><thead><tr>';

      columns.forEach(col => {
        html += `<th>${this.escapeHtml(col)}</th>`;
      });

      html += '</tr></thead><tbody>';

      value.rows.forEach(row => {
        html += '<tr>';
        columns.forEach(col => {
          html += `<td>${this.escapeHtml(row[col] || '-')}</td>`;
        });
        html += '</tr>';
      });

      html += '</tbody></table></div>';
    }

    return html;
  }

  /**
   * Render simple field
   */
  renderSimpleField(value) {
    return `<p class="field-value">${this.escapeHtml(String(value))}</p>`;
  }

  /**
   * Render TIDP/MIDP sections
   */
  renderTIDPMIDPSections(tidpData, midpData) {
    let html = `
      <div class="section">
        <div class="section-header tidp-header">
          <h2 class="section-title">Information Delivery Planning</h2>
        </div>
        <div class="section-content">
    `;

    // TIDP table
    if (tidpData.length > 0) {
      html += `
        <h3 class="subsection-title">Task Information Delivery Plans (TIDPs)</h3>
        <div class="table-wrapper">
          <table class="data-table tidp-table">
            <thead>
              <tr>
                <th>Task Team</th>
                <th>Discipline</th>
                <th>Team Leader</th>
                <th>Reference</th>
              </tr>
            </thead>
            <tbody>
      `;

      tidpData.forEach((tidp, idx) => {
        const teamName = this.escapeHtml(tidp.teamName || tidp.taskTeam || `Task Team ${idx + 1}`);
        const discipline = this.escapeHtml(tidp.discipline || 'N/A');
        const leader = this.escapeHtml(tidp.leader || tidp.teamLeader || 'TBD');
        const ref = `TIDP-${String(idx + 1).padStart(2, '0')}`;

        html += `
          <tr>
            <td>${teamName}</td>
            <td>${discipline}</td>
            <td>${leader}</td>
            <td class="monospace">${ref}</td>
          </tr>
        `;
      });

      html += '</tbody></table></div>';
    }

    // MIDP table
    if (midpData.length > 0) {
      html += `
        <h3 class="subsection-title">Master Information Delivery Plan (MIDP)</h3>
        <div class="table-wrapper">
          <table class="data-table midp-table">
            <thead>
              <tr>
                <th>MIDP Reference</th>
                <th>Version</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
      `;

      midpData.forEach((midp, idx) => {
        const ref = `MIDP-${String(idx + 1).padStart(2, '0')}`;
        const version = this.escapeHtml(midp.version || '1.0');
        const status = this.escapeHtml(midp.status || 'Active');

        html += `
          <tr>
            <td class="monospace">${ref}</td>
            <td>${version}</td>
            <td>${status}</td>
          </tr>
        `;
      });

      html += '</tbody></table></div>';
    }

    html += '</div></div>';

    return html;
  }

  /**
   * Get form fields for step (mirrors frontend CONFIG.getFormFields)
   */
  getFormFields(bepType, stepIndex) {
    try {
      // Try to access formFields from CONFIG
      const formFields = CONFIG.formFields || {};
      const typeFields = formFields[bepType] || {};
      return typeFields[stepIndex];
    } catch (error) {
      console.error('Error getting form fields:', error);
      return null;
    }
  }

  /**
   * Escape HTML to prevent XSS
   */
  escapeHtml(text) {
    if (text === null || text === undefined) return '';

    const str = String(text);
    const map = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    };

    return str.replace(/[&<>"']/g, m => map[m]);
  }

  /**
   * Get inline CSS (Tailwind-like styles + print CSS)
   */
  getInlineCSS() {
    return `
      * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
      }

      body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        font-size: 14px;
        line-height: 1.6;
        color: #1f2937;
        background-color: #ffffff;
      }

      .container {
        max-width: 1200px;
        margin: 0 auto;
        background-color: #ffffff;
      }

      /* Cover Page */
      .cover-page {
        margin-bottom: 60px;
        padding: 60px 40px;
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        color: #ffffff;
        border-radius: 12px;
        page-break-after: always;
      }

      .cover-title {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 20px;
      }

      .cover-subtitle {
        font-size: 28px;
        margin-bottom: 10px;
      }

      .cover-iso {
        font-size: 16px;
        font-style: italic;
        opacity: 0.9;
        margin-bottom: 40px;
      }

      .cover-info {
        padding-top: 30px;
        border-top: 1px solid rgba(255, 255, 255, 0.3);
      }

      .cover-info-item {
        font-size: 20px;
        margin-bottom: 8px;
      }

      .cover-info-date {
        font-size: 14px;
        margin-top: 20px;
        opacity: 0.75;
      }

      /* Sections */
      .section {
        margin-bottom: 50px;
        page-break-inside: avoid;
      }

      .section-header {
        margin-bottom: 30px;
        padding-bottom: 10px;
        border-bottom: 3px solid #2563eb;
      }

      .section-header.tidp-header {
        border-bottom-color: #d97706;
      }

      .section-title {
        font-size: 32px;
        font-weight: 700;
        color: #111827;
      }

      .section-content {
        padding-left: 20px;
      }

      /* Fields */
      .field-group {
        margin-bottom: 30px;
        page-break-inside: avoid;
      }

      .field-label {
        font-size: 20px;
        font-weight: 600;
        color: #374151;
        margin-bottom: 10px;
      }

      .field-value {
        margin: 10px 0;
        color: #4b5563;
      }

      .subsection-title {
        font-size: 20px;
        font-weight: 600;
        color: #374151;
        margin: 25px 0 15px 0;
      }

      /* Tables */
      .table-wrapper {
        margin: 20px 0;
        overflow-x: auto;
      }

      .data-table {
        width: 100%;
        border-collapse: collapse;
        border: 1px solid #d1d5db;
      }

      .data-table thead {
        background-color: #f3f4f6;
      }

      .data-table th {
        padding: 12px 16px;
        text-align: left;
        font-size: 12px;
        font-weight: 600;
        color: #374151;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid #d1d5db;
      }

      .data-table td {
        padding: 12px 16px;
        font-size: 14px;
        color: #111827;
        border-bottom: 1px solid #e5e7eb;
      }

      .data-table tbody tr:hover {
        background-color: #f9fafb;
      }

      .data-table tbody tr:nth-child(even) {
        background-color: #fafafa;
      }

      .tidp-table thead {
        background-color: #fef3c7;
      }

      .midp-table thead {
        background-color: #fef3c7;
      }

      .monospace {
        font-family: 'Courier New', Courier, monospace;
      }

      /* Checkbox list */
      .checkbox-list {
        list-style: none;
        margin: 10px 0;
      }

      .checkbox-item {
        display: flex;
        align-items: center;
        color: #374151;
        margin-bottom: 5px;
      }

      .checkmark {
        color: #059669;
        margin-right: 8px;
        font-weight: bold;
      }

      /* Textarea */
      .textarea-content {
        margin: 10px 0;
        color: #4b5563;
        white-space: pre-wrap;
      }

      .intro-text {
        margin-bottom: 20px;
        color: #4b5563;
        white-space: pre-wrap;
      }

      /* Component images */
      .component-image {
        margin: 30px 0;
        text-align: center;
        page-break-inside: avoid;
      }

      .component-image img {
        max-width: 100%;
        height: auto;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
      }

      .component-placeholder {
        padding: 60px 20px;
        background-color: #f3f4f6;
        border: 2px dashed #d1d5db;
        border-radius: 8px;
        text-align: center;
        color: #9ca3af;
        font-style: italic;
      }

      /* Print-specific styles */
      @page {
        size: A4;
        margin: 25mm 20mm;
      }

      @media print {
        body {
          -webkit-print-color-adjust: exact;
          print-color-adjust: exact;
        }

        .page-break {
          page-break-before: always;
        }

        .no-break {
          page-break-inside: avoid;
        }
      }
    `;
  }
}

module.exports = new HtmlTemplateService();
