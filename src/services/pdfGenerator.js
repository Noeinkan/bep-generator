import jsPDF from 'jspdf';
import CONFIG from '../config/bepConfig';

export const generatePDF = (formData, bepType) => {
  const pdf = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4'
  });

  // Configuration
  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();
  const margin = 20;
  const contentWidth = pageWidth - 2 * margin;
  let y = margin + 15; // Start below header
  let pageNum = 1;

  // Colors - Enhanced palette matching HTML design
  const colors = {
    primary: [30, 64, 175],     // Blue 600
    primaryLight: [59, 130, 246], // Blue 500
    primaryDark: [17, 24, 39],    // Gray 900
    secondary: [75, 85, 99],    // Gray 600
    accent: [139, 92, 246],     // Violet 500
    success: [16, 185, 129],    // Emerald 500
    text: [31, 41, 55],         // Gray 800
    textLight: [107, 114, 128], // Gray 500
    lightGray: [249, 250, 251], // Gray 50
    border: [209, 213, 219],    // Gray 300
    shadow: [0, 0, 0, 0.1]      // Shadow color
  };

  // Helper functions - Enhanced with modern design elements
  const addHeader = () => {
    // Modern gradient header
    const gradientSteps = 20;
    for (let i = 0; i < gradientSteps; i++) {
      const ratio = i / gradientSteps;
      const r = Math.round(colors.primary[0] + (colors.primaryLight[0] - colors.primary[0]) * ratio);
      const g = Math.round(colors.primary[1] + (colors.primaryLight[1] - colors.primary[1]) * ratio);
      const b = Math.round(colors.primary[2] + (colors.primaryLight[2] - colors.primary[2]) * ratio);
      pdf.setFillColor(r, g, b);
      pdf.rect(0, i * (15 / gradientSteps), pageWidth, 15 / gradientSteps + 0.1, 'F');
    }

    // Add subtle pattern overlay
    pdf.setFillColor(255, 255, 255);
    pdf.setGState(new pdf.GState({ opacity: 0.05 }));
    for (let x = 0; x < pageWidth; x += 10) {
      for (let y_pos = 0; y_pos < 15; y_pos += 10) {
        pdf.circle(x + 5, y_pos + 5, 1, 'F');
      }
    }
    pdf.setGState(new pdf.GState({ opacity: 1 }));

    // Header text with shadow effect
    pdf.setTextColor(255, 255, 255);
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(14);
    pdf.text('🏗️ BIM EXECUTION PLAN (BEP)', margin, 9);

    // Page number in styled badge
    pdf.setFillColor(255, 255, 255);
    pdf.setGState(new pdf.GState({ opacity: 0.9 }));
    const pageBadgeWidth = 25;
    const pageBadgeX = pageWidth - margin - pageBadgeWidth;
    pdf.roundedRect(pageBadgeX, 3, pageBadgeWidth, 9, 4, 4, 'F');
    pdf.setGState(new pdf.GState({ opacity: 1 }));

    pdf.setTextColor(...colors.primary);
    pdf.setFontSize(8);
    pdf.text(`Page ${pageNum}`, pageBadgeX + 5, 9);
  };

  const addFooter = () => {
    const footerY = pageHeight - 15;

    // Modern footer with subtle background
    pdf.setFillColor(...colors.lightGray);
    pdf.rect(0, footerY, pageWidth, 15, 'F');

    // Footer line with gradient
    pdf.setDrawColor(...colors.border);
    pdf.setLineWidth(0.5);
    pdf.line(margin, footerY, pageWidth - margin, footerY);

    // Footer content
    pdf.setTextColor(...colors.secondary);
    pdf.setFont('helvetica', 'normal');
    pdf.setFontSize(7);
    pdf.text('📋 ISO 19650-2:2018 Compliant | 🏢 Professional BEP Generator Tool', margin, footerY + 5);

    pdf.setFontSize(6);
    pdf.setTextColor(...colors.textLight);
    pdf.text(`Generated: ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString()}`, pageWidth - margin - 60, footerY + 10);
  };

  const checkPageBreak = (requiredHeight = 20) => {
    if (y + requiredHeight > pageHeight - 25) { // Leave space for footer
      addFooter();
      pdf.addPage();
      pageNum++;
      addHeader();
      y = margin + 20;
      return true;
    }
    return false;
  };

  // Initialize first page
  addHeader();

  // Modern title section with enhanced styling
  checkPageBreak(40);

  // Main title with shadow effect
  pdf.setTextColor(...colors.primary);
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(24);
  pdf.text('BIM EXECUTION PLAN (BEP)', margin, y);
  y += 8;

  // Subtitle with accent color
  pdf.setTextColor(...colors.accent);
  pdf.setFontSize(16);
  pdf.setFont('helvetica', 'bold');
  pdf.text('ISO 19650-2:2018 Compliant', margin, y);
  y += 12;

  // BEP Type badge - modern design
  const badgeText = CONFIG.bepTypeDefinitions[bepType].title;
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(12);
  const badgeWidth = pdf.getTextWidth(badgeText) + 20;
  const badgeHeight = 12;
  const badgeX = margin;
  const badgeY = y - 8;

  // Badge background with gradient
  pdf.setFillColor(...colors.accent);
  pdf.roundedRect(badgeX, badgeY, badgeWidth, badgeHeight, 6, 6, 'F');

  // Badge text
  pdf.setTextColor(255, 255, 255);
  pdf.text(badgeText, badgeX + 10, badgeY + 8);

  y += 8;

  // BEP description in styled box
  const description = CONFIG.bepTypeDefinitions[bepType].description;
  pdf.setTextColor(...colors.text);
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(10);

  // Description background
  const descBoxHeight = 15;
  pdf.setFillColor(248, 250, 252);
  pdf.roundedRect(margin, y, contentWidth, descBoxHeight, 4, 4, 'F');

  // Description border
  pdf.setDrawColor(...colors.success);
  pdf.setLineWidth(1);
  pdf.roundedRect(margin, y, contentWidth, descBoxHeight, 4, 4, 'S');

  // Description text
  const descLines = pdf.splitTextToSize(description, contentWidth - 8);
  pdf.text(descLines[0], margin + 4, y + 6);

  y += descBoxHeight + 8;

  // Process form sections
  const groupedSteps = CONFIG.steps.reduce((acc, step, index) => {
    const cat = step.category;
    if (!acc[cat]) acc[cat] = [];
    const stepConfig = CONFIG.getFormFields(bepType, index);
    if (stepConfig) {
      acc[cat].push({ index, title: stepConfig.title, fields: stepConfig.fields });
    }
    return acc;
  }, {});

  Object.entries(groupedSteps).forEach(([cat, items]) => {
    checkPageBreak(30);

    // Category header with modern styling
    pdf.setFillColor(...colors.primary);
    pdf.rect(margin, y, contentWidth, 12, 'F');

    // Category title with icon
    pdf.setTextColor(255, 255, 255);
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(12);
    const categoryIcon = '📋'; // Default icon
    pdf.text(`${categoryIcon} ${CONFIG.categories[cat].name}`, margin + 4, y + 8);

    y += 16;

    items.forEach(item => {
      checkPageBreak(30);

      // Section header with modern styling
      pdf.setFillColor(...colors.secondary);
      pdf.rect(margin, y, contentWidth, 10, 'F');

      // Section title
      pdf.setTextColor(255, 255, 255);
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(10);
      pdf.text(item.title, margin + 4, y + 7);

      y += 14;

      // Process section fields
      item.fields.forEach(field => {
        const value = formData[field.name] || '';

        if (!value || value === '') return; // Skip empty fields

        checkPageBreak(25);

        // Field card with modern design
        const fieldCardHeight = 20;
        pdf.setFillColor(248, 250, 252);
        pdf.roundedRect(margin, y, contentWidth, fieldCardHeight, 4, 4, 'F');

        // Field border
        pdf.setDrawColor(...colors.border);
        pdf.setLineWidth(0.5);
        pdf.roundedRect(margin, y, contentWidth, fieldCardHeight, 4, 4, 'S');

        // Field label
        pdf.setTextColor(...colors.primary);
        pdf.setFont('helvetica', 'bold');
        pdf.setFontSize(9);
        pdf.text(field.label + ':', margin + 4, y + 7);

        // Field value
        pdf.setFont('helvetica', 'normal');
        pdf.setTextColor(...colors.text);
        const labelWidth = pdf.getTextWidth(field.label + ': ');
        const maxValueWidth = contentWidth - labelWidth - 8;
        let valueText = '';

        if (field.type === 'checkbox' && Array.isArray(value)) {
          valueText = value.join(', ');
        } else {
          valueText = String(value);
        }

        const valueLines = pdf.splitTextToSize(valueText, maxValueWidth);

        if (valueLines.length === 1) {
          pdf.text(valueLines[0], margin + 4 + labelWidth, y + 7);
        } else {
          // Multi-line value
          for (let i = 0; i < Math.min(valueLines.length, 2); i++) {
            pdf.text(valueLines[i], margin + 4 + labelWidth, y + 5 + (i * 4));
          }
          if (valueLines.length > 2) {
            pdf.text('...', margin + 4 + labelWidth + pdf.getTextWidth(valueLines.slice(0, 2).join(' ')) + 2, y + 9);
          }
        }

        y += fieldCardHeight + 3;
      });

      y += 5; // Extra space between sections
    });

    y += 5; // Extra space between categories
  });

  // Add footer to last page
  addFooter();

  return pdf;
};