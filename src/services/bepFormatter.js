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
      acc[cat].push({ index, title: `${acc[cat].length + 1}. ${stepConfig.title.toUpperCase()}`, fields: stepConfig.fields });
    }
    return acc;
  }, {});

  const renderField = (field) => {
    let value = formData[field.name];
    if (!value) return '';

    if (field.type === 'checkbox' && Array.isArray(value)) {
      return `<h3>${field.label}</h3><ul>${value.map(item => `<li>${DOMPurify.sanitize(item)}</li>`).join('')}</ul>`;
    }

    if (field.type === 'table' && Array.isArray(value)) {
      if (value.length === 0) return '';

      const columns = field.columns || ['Role/Discipline', 'Name/Company', 'Experience/Notes'];
      let tableHtml = `<h3>${field.label}</h3>`;
      tableHtml += '<table class="table-data" style="width: 100%; border-collapse: collapse; margin: 10px 0;">';

      // Header
      tableHtml += '<thead><tr style="background: #f3f4f6;">';
      columns.forEach(col => {
        tableHtml += `<th style="border: 1px solid #d1d5db; padding: 8px; text-align: left; font-weight: bold;">${col}</th>`;
      });
      tableHtml += '</tr></thead>';

      // Rows
      tableHtml += '<tbody>';
      value.forEach((row, index) => {
        tableHtml += `<tr style="background: ${index % 2 === 0 ? '#ffffff' : '#f9fafb'};">`;
        columns.forEach(col => {
          const cellValue = DOMPurify.sanitize(row[col] || '');
          tableHtml += `<td style="border: 1px solid #d1d5db; padding: 8px;">${cellValue}</td>`;
        });
        tableHtml += '</tr>';
      });
      tableHtml += '</tbody></table>';

      return tableHtml;
    }

    if (field.type === 'textarea') {
      return `<h3>${field.label}</h3><p>${DOMPurify.sanitize(value)}</p>`;
    }

    return `<tr><td class="label">${field.label}:</td><td>${DOMPurify.sanitize(value)}</td></tr>`;
  };

  const sectionsHtml = Object.entries(groupedSteps).map(([cat, items]) => [
    `<div class="category-header">${CONFIG.categories[cat].name}</div>`,
    items.map(item => {
      const fields = item.fields;
      const tableFields = fields.filter(f => f.type !== 'textarea' && f.type !== 'checkbox');
      const otherFields = fields.filter(f => f.type === 'textarea' || f.type === 'checkbox');
      return `
        <div class="section">
          <h2>${item.title}</h2>
          <div class="info-box">
            ${tableFields.length > 0 ? `<table>${tableFields.map(renderField).join('')}</table>` : ''}
            ${otherFields.map(renderField).join('')}
          </div>
        </div>
      `;
    }).join('')
  ]).flat().join('');

  return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>BIM Execution Plan - ${formData.projectName}</title>
      <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }
        .header { text-align: center; border-bottom: 3px solid #2563eb; padding-bottom: 20px; margin-bottom: 30px; }
        h1 { color: #1e40af; font-size: 28px; margin-bottom: 10px; }
        .subtitle { color: #059669; font-size: 18px; font-weight: bold; margin-bottom: 5px; }
        .bep-type { background: #dbeafe; padding: 10px; border-radius: 8px; display: inline-block; margin: 10px 0; font-weight: bold; }
        h2 { color: #1e40af; margin-top: 35px; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; font-size: 20px; }
        h3 { color: #374151; margin-top: 25px; font-size: 16px; border-left: 4px solid #2563eb; padding-left: 12px; }
        .section { margin: 25px 0; }
        .info-box { background-color: #f8fafc; padding: 20px; border-left: 4px solid #2563eb; margin: 20px 0; border-radius: 0 8px 8px 0; }
        .category-header { background: linear-gradient(135deg, #2563eb, #1e40af); color: white; padding: 15px; margin: 30px 0 20px 0; border-radius: 8px; font-weight: bold; font-size: 18px; }
        ul { margin: 10px 0; padding-left: 25px; }
        li { margin: 8px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; border-bottom: 1px solid #e5e7eb; text-align: left; }
        th { background-color: #f8fafc; font-weight: bold; color: #374151; }
        .label { font-weight: bold; width: 250px; color: #4b5563; }
        .compliance-box { background: #ecfdf5; border: 2px solid #10b981; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .footer { margin-top: 50px; padding-top: 25px; border-top: 2px solid #e5e7eb; background: #f9fafb; padding: 25px; border-radius: 8px; }
      </style>
    </head>
    <body>
      <div class="header">
        <h1>BIM EXECUTION PLAN (BEP)</h1>
        <div class="subtitle">ISO 19650-2 Compliant</div>
        <div class="bep-type">${CONFIG.bepTypeDefinitions[bepType].title}</div>
        <div class="bep-purpose" style="background: #f3f4f6; padding: 10px; border-radius: 8px; margin: 10px 0; font-style: italic; color: #6b7280;">
          ${CONFIG.bepTypeDefinitions[bepType].description}
        </div>
      </div>

      <div class="compliance-box">
        <h3>Document Information</h3>
        <table>
          <tr><td class="label">Document Type:</td><td>${CONFIG.bepTypeDefinitions[bepType].title}</td></tr>
          <tr><td class="label">Document Purpose:</td><td>${CONFIG.bepTypeDefinitions[bepType].purpose}</td></tr>
          <tr><td class="label">Project Name:</td><td>${formData.projectName || 'Not specified'}</td></tr>
          <tr><td class="label">Project Number:</td><td>${formData.projectNumber || 'Not specified'}</td></tr>
          <tr><td class="label">Generated Date:</td><td>${formattedDate} at ${formattedTime}</td></tr>
          <tr><td class="label">Status:</td><td>${bepType === 'pre-appointment' ? 'Tender Submission' : 'Working Document'}</td></tr>
        </table>
      </div>

      ${sectionsHtml}

      <div class="footer">
        <h3>Document Control Information</h3>
        <table>
          <tr><td class="label">Document Type:</td><td>BIM Execution Plan (BEP)</td></tr>
          <tr><td class="label">ISO Standard:</td><td>ISO 19650-2:2018</td></tr>
          <tr><td class="label">Document Status:</td><td>Work in Progress</td></tr>
          <tr><td class="label">Generated By:</td><td>Professional BEP Generator Tool</td></tr>
          <tr><td class="label">Generated Date:</td><td>${formattedDate}</td></tr>
          <tr><td class="label">Generated Time:</td><td>${formattedTime}</td></tr>
        </table>
      </div>
    </body>
    </html>
  `;
};