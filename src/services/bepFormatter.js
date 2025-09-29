import DOMPurify from 'dompurify';
import CONFIG from '../config/bepConfig';

export const generateBEPContent = (formData, bepType) => {
  const currentDate = new Date();
  const formattedDate = currentDate.toLocaleDateString();
  const formattedTime = currentDate.toLocaleTimeString();

  // Group steps by category
  const groupedSteps = CONFIG.steps.reduce((acc, step, index) => {
    const cat = step.category;
    if (!acc[cat]) acc[cat] = [];
    const stepConfig = CONFIG.getFormFields(bepType, index);
    if (stepConfig) {
      acc[cat].push({ index, title: `${acc[cat].length + 1}. ${stepConfig.title}`, fields: stepConfig.fields });
    }
    return acc;
  }, {});

  const renderField = (field) => {
    let value = formData[field.name];
    if (!value) return '';

    if (field.type === 'checkbox' && Array.isArray(value)) {
      return `
        <div class="field-container">
          <h4 class="field-title">${field.label}</h4>
          <ul class="checkbox-list">
            ${value.map(item => `<li class="checkbox-item">${DOMPurify.sanitize(item)}</li>`).join('')}
          </ul>
        </div>
      `;
    }

    if (field.type === 'table' && Array.isArray(value)) {
      if (value.length === 0) return '';

      const columns = field.columns || ['Column 1', 'Column 2', 'Column 3'];
      let tableHtml = `
        <div class="field-container">
          <h4 class="field-title">${field.label}</h4>
          <div class="table-container">
            <table class="data-table">
              <thead>
                <tr>
                  ${columns.map(col => `<th>${col}</th>`).join('')}
                </tr>
              </thead>
              <tbody>
                ${value.map((row, index) => `
                  <tr class="${index % 2 === 0 ? 'even-row' : 'odd-row'}">
                    ${columns.map(col => `<td>${DOMPurify.sanitize(row[col] || '')}</td>`).join('')}
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>
      `;
      return tableHtml;
    }

    if (field.type === 'textarea') {
      return `
        <div class="field-container">
          <h4 class="field-title">${field.label}</h4>
          <div class="textarea-content">${DOMPurify.sanitize(value)}</div>
        </div>
      `;
    }

    return `
      <div class="field-pair">
        <span class="field-label">${field.label}:</span>
        <span class="field-value">${DOMPurify.sanitize(value)}</span>
      </div>
    `;
  };

  const sectionsHtml = Object.entries(groupedSteps).map(([cat, items]) => `
    <section class="category-section">
      <div class="category-header">
        <div class="category-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
          </svg>
        </div>
        <h2 class="category-title">${CONFIG.categories[cat].name}</h2>
      </div>

      ${items.map(item => {
        const fields = item.fields;
        const tableFields = fields.filter(f => f.type !== 'textarea' && f.type !== 'checkbox');
        const otherFields = fields.filter(f => f.type === 'textarea' || f.type === 'checkbox');

        return `
          <div class="content-section">
            <h3 class="section-title">${item.title}</h3>
            <div class="section-content">
              ${tableFields.length > 0 ? `
                <div class="info-table">
                  ${tableFields.map(renderField).join('')}
                </div>
              ` : ''}
              ${otherFields.map(renderField).join('')}
            </div>
          </div>
        `;
      }).join('')}
    </section>
  `).join('');

  return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>BIM Execution Plan - ${formData.projectName || 'Unnamed Project'}</title>
      <style>
        /* Reset and base styles */
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        body {
          font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif;
          line-height: 1.6;
          color: #1f2937;
          background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
          min-height: 100vh;
        }

        .container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 20px;
        }

        /* Header styles */
        .document-header {
          background: linear-gradient(135deg, #1e40af 0%, #2563eb 50%, #3b82f6 100%);
          color: white;
          padding: 40px 30px;
          border-radius: 16px;
          margin-bottom: 30px;
          box-shadow: 0 10px 25px rgba(30, 64, 175, 0.2);
          text-align: center;
          position: relative;
          overflow: hidden;
        }

        .document-header::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
          opacity: 0.1;
        }

        .document-title {
          font-size: 2.5rem;
          font-weight: 700;
          margin-bottom: 10px;
          text-shadow: 0 2px 4px rgba(0,0,0,0.1);
          position: relative;
          z-index: 1;
        }

        .document-subtitle {
          font-size: 1.25rem;
          font-weight: 600;
          margin-bottom: 20px;
          opacity: 0.95;
          position: relative;
          z-index: 1;
        }

        .bep-type-badge {
          display: inline-block;
          background: rgba(255, 255, 255, 0.2);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.3);
          padding: 12px 24px;
          border-radius: 25px;
          font-weight: 600;
          font-size: 1.1rem;
          margin: 15px 0;
          position: relative;
          z-index: 1;
        }

        .bep-description {
          background: rgba(255, 255, 255, 0.15);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.2);
          padding: 20px;
          border-radius: 12px;
          font-style: italic;
          max-width: 800px;
          margin: 20px auto 0;
          position: relative;
          z-index: 1;
        }

        /* Document info box */
        .document-info {
          background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
          border: 2px solid #10b981;
          border-radius: 16px;
          padding: 30px;
          margin-bottom: 30px;
          box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
        }

        .document-info h3 {
          color: #065f46;
          font-size: 1.5rem;
          margin-bottom: 20px;
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .document-info h3::before {
          content: '📋';
          font-size: 1.2rem;
        }

        .info-table {
          display: grid;
          gap: 12px;
        }

        .field-pair {
          display: flex;
          align-items: flex-start;
          padding: 12px 16px;
          background: rgba(255, 255, 255, 0.7);
          border-radius: 8px;
          border-left: 4px solid #10b981;
        }

        .field-label {
          font-weight: 600;
          color: #374151;
          min-width: 200px;
          flex-shrink: 0;
        }

        .field-value {
          color: #1f2937;
          flex: 1;
        }

        /* Category sections */
        .category-section {
          margin-bottom: 40px;
        }

        .category-header {
          background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
          color: white;
          padding: 20px 25px;
          border-radius: 12px;
          margin-bottom: 25px;
          display: flex;
          align-items: center;
          gap: 15px;
          box-shadow: 0 6px 20px rgba(30, 64, 175, 0.15);
        }

        .category-icon {
          background: rgba(255, 255, 255, 0.2);
          padding: 8px;
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .category-title {
          font-size: 1.5rem;
          font-weight: 700;
          margin: 0;
        }

        /* Content sections */
        .content-section {
          background: white;
          border-radius: 12px;
          padding: 25px;
          margin-bottom: 20px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
          border: 1px solid #e5e7eb;
        }

        .section-title {
          color: #1e40af;
          font-size: 1.25rem;
          font-weight: 600;
          margin-bottom: 20px;
          padding-bottom: 10px;
          border-bottom: 2px solid #dbeafe;
        }

        .section-content {
          color: #374151;
        }

        /* Field containers */
        .field-container {
          margin-bottom: 25px;
        }

        .field-title {
          color: #1f2937;
          font-size: 1.1rem;
          font-weight: 600;
          margin-bottom: 12px;
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .field-title::before {
          content: '📝';
          font-size: 0.9rem;
        }

        .textarea-content {
          background: #f8fafc;
          padding: 16px;
          border-radius: 8px;
          border-left: 4px solid #3b82f6;
          line-height: 1.7;
          white-space: pre-wrap;
        }

        /* Checkbox lists */
        .checkbox-list {
          background: #f8fafc;
          border-radius: 8px;
          padding: 16px;
          border-left: 4px solid #8b5cf6;
        }

        .checkbox-item {
          margin-bottom: 8px;
          padding-left: 8px;
          position: relative;
        }

        .checkbox-item::before {
          content: '✓';
          color: #8b5cf6;
          font-weight: bold;
          position: absolute;
          left: -8px;
          top: 0;
        }

        /* Tables */
        .table-container {
          border-radius: 8px;
          overflow: hidden;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
          margin-top: 12px;
        }

        .data-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.9rem;
        }

        .data-table thead {
          background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
          color: white;
        }

        .data-table th {
          padding: 16px 12px;
          text-align: left;
          font-weight: 600;
          font-size: 0.85rem;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .data-table td {
          padding: 14px 12px;
          border-bottom: 1px solid #e5e7eb;
        }

        .data-table .even-row {
          background: #ffffff;
        }

        .data-table .odd-row {
          background: #f8fafc;
        }

        .data-table tbody tr:hover {
          background: #dbeafe;
          transition: background-color 0.2s ease;
        }

        /* Footer */
        .document-footer {
          background: linear-gradient(135deg, #374151 0%, #1f2937 100%);
          color: white;
          padding: 30px;
          border-radius: 16px;
          margin-top: 40px;
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        }

        .document-footer h3 {
          font-size: 1.5rem;
          margin-bottom: 20px;
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .document-footer h3::before {
          content: '🏢';
          font-size: 1.2rem;
        }

        .footer-table {
          display: grid;
          gap: 12px;
        }

        .footer-table .field-pair {
          background: rgba(255, 255, 255, 0.1);
          border-left-color: #60a5fa;
        }

        /* Print styles */
        @media print {
          body {
            background: white !important;
            font-size: 12px;
          }

          .container {
            max-width: none;
            padding: 0;
          }

          .document-header,
          .category-header,
          .content-section,
          .document-info,
          .document-footer {
            box-shadow: none !important;
            break-inside: avoid;
          }

          .data-table {
            page-break-inside: avoid;
          }
        }

        /* Responsive design */
        @media (max-width: 768px) {
          .container {
            padding: 10px;
          }

          .document-title {
            font-size: 2rem;
          }

          .document-header,
          .category-header,
          .content-section,
          .document-info,
          .document-footer {
            padding: 20px;
          }

          .field-pair {
            flex-direction: column;
            gap: 4px;
          }

          .field-label {
            min-width: auto;
            font-weight: 700;
          }

          .data-table {
            font-size: 0.8rem;
          }

          .data-table th,
          .data-table td {
            padding: 8px 6px;
          }
        }
      </style>
    </head>
    <body>
      <div class="container">
        <header class="document-header">
          <h1 class="document-title">BIM EXECUTION PLAN (BEP)</h1>
          <div class="document-subtitle">ISO 19650-2:2018 Compliant</div>
          <div class="bep-type-badge">${CONFIG.bepTypeDefinitions[bepType].title}</div>
          <div class="bep-description">
            ${CONFIG.bepTypeDefinitions[bepType].description}
          </div>
        </header>

        <section class="document-info">
          <h3>Document Information</h3>
          <div class="info-table">
            <div class="field-pair">
              <span class="field-label">Document Type:</span>
              <span class="field-value">${CONFIG.bepTypeDefinitions[bepType].title}</span>
            </div>
            <div class="field-pair">
              <span class="field-label">Document Purpose:</span>
              <span class="field-value">${CONFIG.bepTypeDefinitions[bepType].purpose}</span>
            </div>
            <div class="field-pair">
              <span class="field-label">Project Name:</span>
              <span class="field-value">${formData.projectName || 'Not specified'}</span>
            </div>
            <div class="field-pair">
              <span class="field-label">Project Number:</span>
              <span class="field-value">${formData.projectNumber || 'Not specified'}</span>
            </div>
            <div class="field-pair">
              <span class="field-label">Generated Date:</span>
              <span class="field-value">${formattedDate} at ${formattedTime}</span>
            </div>
            <div class="field-pair">
              <span class="field-label">Status:</span>
              <span class="field-value">${bepType === 'pre-appointment' ? 'Tender Submission' : 'Working Document'}</span>
            </div>
          </div>
        </section>

        ${sectionsHtml}

        <footer class="document-footer">
          <h3>Document Control Information</h3>
          <div class="footer-table">
            <div class="field-pair">
              <span class="field-label">Document Type:</span>
              <span class="field-value">BIM Execution Plan (BEP)</span>
            </div>
            <div class="field-pair">
              <span class="field-label">ISO Standard:</span>
              <span class="field-value">ISO 19650-2:2018</span>
            </div>
            <div class="field-pair">
              <span class="field-label">Document Status:</span>
              <span class="field-value">Work in Progress</span>
            </div>
            <div class="field-pair">
              <span class="field-label">Generated By:</span>
              <span class="field-value">Professional BEP Generator Tool</span>
            </div>
            <div class="field-pair">
              <span class="field-label">Generated Date:</span>
              <span class="field-value">${formattedDate}</span>
            </div>
            <div class="field-pair">
              <span class="field-label">Generated Time:</span>
              <span class="field-value">${formattedTime}</span>
            </div>
          </div>
        </footer>
      </div>
    </body>
    </html>
  `;
};