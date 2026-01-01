import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import CONFIG from '../config/bepConfig';

/**
 * Professional PDF Generator for BIM Execution Plans
 * Uses jsPDF with autotable plugin for structured, readable output
 */

// Color palette matching the HTML preview
const COLORS = {
  primary: [30, 64, 175],       // Blue
  secondary: [37, 99, 235],     // Lighter blue
  success: [22, 163, 74],       // Green
  warning: [217, 119, 6],       // Amber
  dark: [31, 41, 55],           // Dark gray
  light: [248, 250, 252],       // Light background
  text: [31, 41, 55],           // Text color
  white: [255, 255, 255],
  border: [209, 213, 219]       // Border color
};

// Professional document layout constants
const LAYOUT = {
  marginLeft: 20,
  marginRight: 20,
  marginTop: 25,
  marginBottom: 25,
  contentWidth: 170,            // A4 width (210) - margins (40)
  lineHeight: 5,
  paragraphSpacing: 8,
  sectionSpacing: 12
};

export const generatePDF = async (formData, bepType, options = {}) => {
  const {
    orientation = 'portrait',
    format = 'a4',
    filename = `BEP_${bepType}_${new Date().toISOString().split('T')[0]}.pdf`,
    tidpData = [],
    midpData = [],
    componentScreenshots = {} // Map of fieldName -> base64 image data
  } = options;

  try {
    console.log('Starting professional PDF generation...');

    // Create PDF document with professional settings
    const doc = new jsPDF({
      orientation,
      unit: 'mm',
      format,
      compress: true
    });

    // Set document metadata
    doc.setProperties({
      title: `BIM Execution Plan - ${bepType} - ISO 19650-2:2018 Compliant`,
      subject: 'BIM Execution Plan Document - Information Management using Building Information Modelling',
      author: formData.informationManager || formData.proposedInfoManager || 'BEP Generator',
      creator: 'Professional BEP Generator Tool',
      keywords: 'BIM, BEP, ISO 19650, ISO 19650-2:2018, Information Management, TIDP, MIDP',
      producer: 'BEP Generator v1.0 - ISO 19650 Compliant'
    });

    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();

    // === COVER PAGE ===
    addCoverPage(doc, formData, bepType, pageWidth, pageHeight);

    // === DOCUMENT INFORMATION ===
    doc.addPage();
    let yPos = LAYOUT.marginTop;
    yPos = addSectionHeader(doc, 'Document Information', yPos, COLORS.primary);
    yPos = addDocumentInfo(doc, formData, bepType, yPos);

    // === ISO 19650 COMPLIANCE SECTION ===
    yPos = addSectionHeader(doc, 'ISO 19650-2:2018 Compliance Statement', yPos + LAYOUT.sectionSpacing, COLORS.success);
    yPos = addISOComplianceSection(doc, yPos);

    // === TIDP/MIDP SECTION ===
    if (tidpData.length > 0 || midpData.length > 0) {
      yPos = addSectionHeader(doc, 'Information Delivery Plan', yPos + LAYOUT.sectionSpacing, COLORS.warning);
      yPos = addTIDPMIDPSection(doc, tidpData, midpData, yPos);
    }

    // === BEP CONTENT SECTIONS ===
    yPos = addBEPContentSections(doc, formData, bepType, yPos, componentScreenshots);

    // === FOOTER ON ALL PAGES ===
    addPageNumbers(doc);

    // Save the PDF
    doc.save(filename);
    console.log('PDF generation completed successfully');

    const pageCount = doc.internal.getNumberOfPages();
    return {
      success: true,
      filename,
      pages: pageCount,
      size: doc.output('arraybuffer').byteLength
    };

  } catch (error) {
    console.error('PDF generation failed:', error);
    throw new Error(`PDF generation failed: ${error.message}`);
  }
};

// === HELPER FUNCTIONS ===

function addCoverPage(doc, formData, bepType, pageWidth, pageHeight) {
  // Professional header band
  doc.setFillColor(...COLORS.primary);
  doc.rect(0, 0, pageWidth, 80, 'F');

  // Secondary accent band
  doc.setFillColor(...COLORS.secondary);
  doc.rect(0, 80, pageWidth, 8, 'F');

  // Main title - centered properly
  doc.setTextColor(...COLORS.white);
  doc.setFontSize(28);
  doc.setFont('helvetica', 'bold');
  doc.text('BIM EXECUTION PLAN', pageWidth / 2, 35, { align: 'center' });

  // ISO Compliance subtitle
  doc.setFontSize(12);
  doc.setFont('helvetica', 'normal');
  doc.text('ISO 19650-2:2018 Compliant', pageWidth / 2, 48, { align: 'center' });

  // BEP Type badge
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  const bepTypeLabel = CONFIG.bepTypeDefinitions[bepType]?.title || bepType;
  doc.text(bepTypeLabel, pageWidth / 2, 65, { align: 'center' });

  // Project information card - centered with proper margins
  const cardMargin = 25;
  const cardWidth = pageWidth - (cardMargin * 2);
  const cardX = cardMargin;
  const cardY = 100;
  const cardHeight = 70;

  // Card background with border
  doc.setFillColor(...COLORS.light);
  doc.setDrawColor(...COLORS.border);
  doc.setLineWidth(0.5);
  doc.roundedRect(cardX, cardY, cardWidth, cardHeight, 4, 4, 'FD');

  // Card header
  doc.setFillColor(...COLORS.primary);
  doc.roundedRect(cardX, cardY, cardWidth, 12, 4, 4, 'F');
  doc.rect(cardX, cardY + 8, cardWidth, 4, 'F'); // Fill bottom corners

  doc.setTextColor(...COLORS.white);
  doc.setFontSize(12);
  doc.setFont('helvetica', 'bold');
  doc.text('PROJECT INFORMATION', pageWidth / 2, cardY + 8, { align: 'center' });

  // Project details - left aligned within card
  doc.setTextColor(...COLORS.text);
  const detailsX = cardX + 15;
  let detailsY = cardY + 25;

  doc.setFontSize(10);
  doc.setFont('helvetica', 'bold');
  doc.text('Project Name:', detailsX, detailsY);
  doc.setFont('helvetica', 'normal');
  doc.text(formData.projectName || 'Not specified', detailsX + 35, detailsY);

  detailsY += 10;
  doc.setFont('helvetica', 'bold');
  doc.text('Project Number:', detailsX, detailsY);
  doc.setFont('helvetica', 'normal');
  doc.text(formData.projectNumber || 'Not specified', detailsX + 35, detailsY);

  detailsY += 10;
  doc.setFont('helvetica', 'bold');
  doc.text('Client:', detailsX, detailsY);
  doc.setFont('helvetica', 'normal');
  doc.text(formData.client || formData.appointingParty || 'Not specified', detailsX + 35, detailsY);

  detailsY += 10;
  doc.setFont('helvetica', 'bold');
  doc.text('Generated:', detailsX, detailsY);
  doc.setFont('helvetica', 'normal');
  doc.text(new Date().toLocaleDateString('en-GB', { day: '2-digit', month: 'long', year: 'numeric' }), detailsX + 35, detailsY);

  // Description box - below project info
  const description = CONFIG.bepTypeDefinitions[bepType]?.description || '';
  if (description) {
    const descY = cardY + cardHeight + 15;
    doc.setFillColor(245, 247, 250);
    doc.setDrawColor(...COLORS.border);
    doc.roundedRect(cardX, descY, cardWidth, 35, 3, 3, 'FD');

    doc.setTextColor(...COLORS.text);
    doc.setFontSize(9);
    doc.setFont('helvetica', 'italic');
    const descLines = doc.splitTextToSize(description, cardWidth - 20);
    doc.text(descLines, cardX + 10, descY + 12);
  }

  // Compliance badge at bottom
  const badgeY = pageHeight - 50;
  const badgeHeight = 14;
  doc.setFillColor(...COLORS.success);
  doc.roundedRect(cardX, badgeY, cardWidth, badgeHeight, 3, 3, 'F');

  doc.setTextColor(...COLORS.white);
  doc.setFontSize(10);
  doc.setFont('helvetica', 'bold');
  doc.text('Generated in compliance with ISO 19650-2:2018', pageWidth / 2, badgeY + 9, { align: 'center' });

  // Footer with version
  doc.setTextColor(150, 150, 150);
  doc.setFontSize(8);
  doc.setFont('helvetica', 'normal');
  doc.text('BEP Generator v1.0', pageWidth / 2, pageHeight - 20, { align: 'center' });
}

function addSectionHeader(doc, title, yPos, color = COLORS.primary) {
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();

  // Check for page break
  if (yPos > pageHeight - 50) {
    doc.addPage();
    yPos = LAYOUT.marginTop;
  }

  // Section header bar with proper margins
  const headerHeight = 10;
  doc.setFillColor(...color);
  doc.roundedRect(LAYOUT.marginLeft, yPos, LAYOUT.contentWidth, headerHeight, 2, 2, 'F');

  // Header text - properly centered vertically
  doc.setTextColor(...COLORS.white);
  doc.setFontSize(11);
  doc.setFont('helvetica', 'bold');
  doc.text(title, LAYOUT.marginLeft + 5, yPos + 7);

  doc.setTextColor(...COLORS.text);
  return yPos + headerHeight + 8;
}

function addDocumentInfo(doc, formData, bepType, yPos) {
  const docInfoData = [
    ['Document Type', CONFIG.bepTypeDefinitions[bepType]?.title || bepType],
    ['Document Purpose', CONFIG.bepTypeDefinitions[bepType]?.purpose || 'N/A'],
    ['Project Name', formData.projectName || 'Not specified'],
    ['Project Number', formData.projectNumber || 'Not specified'],
    ['Client / Appointing Party', formData.client || formData.appointingParty || 'Not specified'],
    ['Generated Date', new Date().toLocaleDateString('en-GB')],
    ['Status', bepType === 'pre-appointment' ? 'Tender Submission' : 'Working Document'],
    ['Document Version', formData.documentVersion || '1.0']
  ];

  autoTable(doc, {
    startY: yPos,
    head: [['Field', 'Value']],
    body: docInfoData,
    theme: 'grid',
    headStyles: {
      fillColor: COLORS.primary,
      textColor: COLORS.white,
      fontStyle: 'bold',
      fontSize: 10,
      cellPadding: 4
    },
    bodyStyles: {
      fontSize: 9,
      textColor: COLORS.text,
      cellPadding: 4
    },
    alternateRowStyles: {
      fillColor: [250, 250, 252]
    },
    columnStyles: {
      0: { fontStyle: 'bold', cellWidth: 55 },
      1: { cellWidth: 'auto' }
    },
    margin: { left: LAYOUT.marginLeft, right: LAYOUT.marginRight },
    tableWidth: LAYOUT.contentWidth
  });

  return doc.lastAutoTable.finalY + 8;
}

function addISOComplianceSection(doc, yPos) {
  const pageHeight = doc.internal.pageSize.getHeight();

  if (yPos > pageHeight - 100) {
    doc.addPage();
    yPos = LAYOUT.marginTop;
  }

  // Declaration text with proper margins
  doc.setFontSize(9);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(...COLORS.text);
  const declarationText = 'This BIM Execution Plan (BEP) has been prepared in accordance with ISO 19650-2:2018 "Organization and digitization of information about buildings and civil engineering works, including building information modelling (BIM) — Information management using building information modelling — Part 2: Delivery phase of the assets."';
  const lines = doc.splitTextToSize(declarationText, LAYOUT.contentWidth);
  doc.text(lines, LAYOUT.marginLeft, yPos);
  yPos += lines.length * 4 + 10;

  // Requirements coverage table
  const coverageData = [
    ['5.1', 'Information Management', 'Information management function and responsibilities'],
    ['5.2', 'Planning Approach', 'Master Information Delivery Plan (MIDP) and TIDPs'],
    ['5.3', 'Information Requirements', 'Exchange information requirements and LOD/LOI'],
    ['5.4', 'Collaborative Production', 'CDE workflows and federation strategy'],
    ['5.5', 'Quality Assurance', 'Validation, review and approval processes'],
    ['5.6', 'Information Security', 'Security protocols and access control']
  ];

  autoTable(doc, {
    startY: yPos,
    head: [['Ref', 'ISO 19650-2 Requirement', 'Coverage']],
    body: coverageData,
    theme: 'striped',
    headStyles: {
      fillColor: COLORS.success,
      textColor: COLORS.white,
      fontStyle: 'bold',
      fontSize: 9,
      cellPadding: 3
    },
    bodyStyles: {
      fontSize: 8,
      textColor: COLORS.text,
      cellPadding: 3
    },
    columnStyles: {
      0: { cellWidth: 15, fontStyle: 'bold', halign: 'center' },
      1: { cellWidth: 50, fontStyle: 'bold' },
      2: { cellWidth: 'auto' }
    },
    margin: { left: LAYOUT.marginLeft, right: LAYOUT.marginRight },
    tableWidth: LAYOUT.contentWidth
  });

  return doc.lastAutoTable.finalY + LAYOUT.sectionSpacing;
}

function addTIDPMIDPSection(doc, tidpData, midpData, yPos) {
  const pageHeight = doc.internal.pageSize.getHeight();

  // TIDP Section
  if (tidpData.length > 0) {
    if (yPos > pageHeight - 70) {
      doc.addPage();
      yPos = LAYOUT.marginTop;
    }

    doc.setFontSize(11);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.warning);
    doc.text('Task Information Delivery Plans (TIDPs)', LAYOUT.marginLeft, yPos);
    yPos += 6;

    doc.setFontSize(9);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...COLORS.text);
    doc.text('The following TIDPs have been established for this project:', LAYOUT.marginLeft, yPos);
    yPos += 8;

    const tidpTableData = tidpData.map((tidp, index) => [
      tidp.teamName || tidp.taskTeam || `Task Team ${index + 1}`,
      tidp.discipline || 'N/A',
      tidp.leader || tidp.teamLeader || 'TBD',
      `${tidp.containers?.length || 0} containers`,
      `TIDP-${String(index + 1).padStart(2, '0')}`
    ]);

    autoTable(doc, {
      startY: yPos,
      head: [['Task Team', 'Discipline', 'Team Leader', 'Deliverables', 'Reference']],
      body: tidpTableData,
      theme: 'grid',
      headStyles: {
        fillColor: COLORS.secondary,
        textColor: COLORS.white,
        fontStyle: 'bold',
        fontSize: 8,
        cellPadding: 3
      },
      bodyStyles: {
        fontSize: 8,
        textColor: COLORS.text,
        cellPadding: 3
      },
      alternateRowStyles: {
        fillColor: [255, 251, 235]
      },
      margin: { left: LAYOUT.marginLeft, right: LAYOUT.marginRight },
      tableWidth: LAYOUT.contentWidth
    });

    yPos = doc.lastAutoTable.finalY + 10;
  }

  // MIDP Section
  if (midpData.length > 0) {
    if (yPos > pageHeight - 70) {
      doc.addPage();
      yPos = LAYOUT.marginTop;
    }

    doc.setFontSize(11);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.warning);
    doc.text('Master Information Delivery Plan (MIDP)', LAYOUT.marginLeft, yPos);
    yPos += 8;

    const midpTableData = midpData.map((midp, index) => [
      `MIDP-${String(index + 1).padStart(2, '0')}`,
      midp.version || '1.0',
      `${midp.aggregatedTidps?.length || tidpData.length} TIDPs`,
      midp.totalDeliverables || '-',
      midp.status || 'Active'
    ]);

    autoTable(doc, {
      startY: yPos,
      head: [['MIDP Reference', 'Version', 'Aggregated TIDPs', 'Total Deliverables', 'Status']],
      body: midpTableData,
      theme: 'grid',
      headStyles: {
        fillColor: COLORS.secondary,
        textColor: COLORS.white,
        fontStyle: 'bold',
        fontSize: 8,
        cellPadding: 3
      },
      bodyStyles: {
        fontSize: 8,
        textColor: COLORS.text,
        cellPadding: 3
      },
      alternateRowStyles: {
        fillColor: [255, 251, 235]
      },
      margin: { left: LAYOUT.marginLeft, right: LAYOUT.marginRight },
      tableWidth: LAYOUT.contentWidth
    });

    yPos = doc.lastAutoTable.finalY + 8;

    if (midpData[0]?.description) {
      doc.setFontSize(8);
      doc.setFont('helvetica', 'italic');
      doc.setTextColor(...COLORS.text);
      const descLines = doc.splitTextToSize(`Description: ${midpData[0].description}`, LAYOUT.contentWidth);
      doc.text(descLines, LAYOUT.marginLeft, yPos);
      yPos += descLines.length * 4 + 10;
    }
  }

  return yPos;
}

function addBEPContentSections(doc, formData, bepType, yPos, componentScreenshots = {}) {
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();

  let currentCategory = null;

  // Process steps in sequential order
  CONFIG.steps.forEach((step, stepIndex) => {
    const stepConfig = CONFIG.getFormFields(bepType, stepIndex);
    if (!stepConfig) return;

    // Add category header when category changes
    if (currentCategory !== step.category) {
      currentCategory = step.category;
      const categoryName = CONFIG.categories[step.category]?.name || step.category;
      
      // Always start category on new page for cleaner layout
      if (yPos > LAYOUT.marginTop + 20) {
        doc.addPage();
        yPos = LAYOUT.marginTop;
      }
      yPos = addSectionHeader(doc, categoryName, yPos, COLORS.primary);
    }

    // Check if we need a new page
    if (yPos > pageHeight - 50) {
      doc.addPage();
      yPos = LAYOUT.marginTop;
    }

    // Section title with number
    doc.setFontSize(11);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.secondary);
    doc.text(`${stepConfig.number}. ${stepConfig.title}`, LAYOUT.marginLeft, yPos);
    yPos += 8;

    // Process fields
    stepConfig.fields.forEach((field) => {
      const value = formData[field.name];

      // Skip section headers and fields without values
      if (field.type === 'section-header') return;
      if (!value) return;

      if (yPos > pageHeight - 40) {
        doc.addPage();
        yPos = LAYOUT.marginTop;
      }

      // Handle custom visual components
      if (['orgchart', 'cdeDiagram', 'naming-conventions', 'federation-strategy', 'mindmap', 'fileStructure'].includes(field.type)) {
        yPos = addVisualComponentPlaceholder(doc, field, value, yPos, componentScreenshots);
        return;
      }

      // Handle standard field types
      if (field.type === 'table' && Array.isArray(value) && value.length > 0) {
        yPos = renderTableField(doc, field, value, yPos);
      } else if (field.type === 'checkbox' && Array.isArray(value)) {
        yPos = renderCheckboxField(doc, field, value, yPos, pageHeight);
      } else if (field.type === 'textarea') {
        yPos = renderTextareaField(doc, field, value, yPos, pageWidth, pageHeight);
      } else if (field.type === 'introTable' && typeof value === 'object') {
        yPos = renderIntroTableField(doc, field, value, yPos);
      } else {
        yPos = renderSimpleField(doc, field, value, yPos);
      }
    });

    yPos += LAYOUT.paragraphSpacing;
  });

  return yPos;
}

// Helper function to render table fields
function renderTableField(doc, field, value, yPos) {
  const columns = field.columns || ['Column 1', 'Column 2', 'Column 3'];
  const tableData = value.map(row => columns.map(col => row[col] || ''));

  doc.setFontSize(9);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...COLORS.text);
  doc.text(`${field.number ? field.number + ' ' : ''}${field.label}`, LAYOUT.marginLeft, yPos);
  yPos += 5;

  autoTable(doc, {
    startY: yPos,
    head: [columns],
    body: tableData,
    theme: 'striped',
    headStyles: {
      fillColor: COLORS.primary,
      textColor: COLORS.white,
      fontSize: 8,
      fontStyle: 'bold',
      cellPadding: 3
    },
    bodyStyles: {
      fontSize: 8,
      textColor: COLORS.text,
      cellPadding: 3
    },
    alternateRowStyles: {
      fillColor: [250, 250, 252]
    },
    margin: { left: LAYOUT.marginLeft, right: LAYOUT.marginRight },
    tableWidth: LAYOUT.contentWidth
  });

  return doc.lastAutoTable.finalY + LAYOUT.paragraphSpacing;
}

// Helper function to render checkbox fields
function renderCheckboxField(doc, field, value, yPos, pageHeight) {
  doc.setFontSize(9);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...COLORS.text);
  doc.text(`${field.number ? field.number + ' ' : ''}${field.label}`, LAYOUT.marginLeft, yPos);
  yPos += 6;

  doc.setFontSize(8);
  doc.setFont('helvetica', 'normal');
  
  // Create two-column layout for checkboxes
  const colWidth = LAYOUT.contentWidth / 2;
  let col = 0;
  let startY = yPos;

  value.forEach((item, index) => {
    if (yPos > pageHeight - 20) {
      doc.addPage();
      yPos = LAYOUT.marginTop;
      startY = yPos;
    }
    
    const xPos = LAYOUT.marginLeft + (col * colWidth);
    doc.setTextColor(...COLORS.success);
    doc.text('✓', xPos, yPos);
    doc.setTextColor(...COLORS.text);
    doc.text(item, xPos + 5, yPos);
    
    col++;
    if (col >= 2) {
      col = 0;
      yPos += 5;
    }
  });

  // If we ended on first column, move to next line
  if (col !== 0) {
    yPos += 5;
  }

  return yPos + 4;
}

// Helper function to render textarea fields
function renderTextareaField(doc, field, value, yPos, pageWidth, pageHeight) {
  doc.setFontSize(9);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...COLORS.text);
  doc.text(`${field.number ? field.number + ' ' : ''}${field.label}`, LAYOUT.marginLeft, yPos);
  yPos += 6;

  // Add subtle background box for text content
  doc.setFontSize(8);
  doc.setFont('helvetica', 'normal');
  const textLines = doc.splitTextToSize(value, LAYOUT.contentWidth - 10);
  
  const boxHeight = Math.min(textLines.length * 4 + 6, 60);
  doc.setFillColor(250, 250, 252);
  doc.setDrawColor(...COLORS.border);
  doc.roundedRect(LAYOUT.marginLeft, yPos - 2, LAYOUT.contentWidth, boxHeight, 2, 2, 'FD');

  let lineY = yPos + 3;
  textLines.forEach((line) => {
    if (lineY > pageHeight - 20) {
      doc.addPage();
      lineY = LAYOUT.marginTop;
    }
    doc.text(line, LAYOUT.marginLeft + 4, lineY);
    lineY += 4;
  });

  return yPos + boxHeight + 4;
}

// Helper function to render introTable fields
function renderIntroTableField(doc, field, value, yPos) {
  // Render intro text if present
  if (value.intro) {
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.text);
    doc.text(`${field.number ? field.number + ' ' : ''}${field.label}`, LAYOUT.marginLeft, yPos);
    yPos += 6;

    doc.setFontSize(8);
    doc.setFont('helvetica', 'normal');
    const introLines = doc.splitTextToSize(value.intro, LAYOUT.contentWidth);
    introLines.forEach((line) => {
      doc.text(line, LAYOUT.marginLeft, yPos);
      yPos += 4;
    });
    yPos += 4;
  }

  // Render table if present
  if (value.rows && Array.isArray(value.rows) && value.rows.length > 0) {
    const columns = field.tableColumns || ['Column 1', 'Column 2', 'Column 3'];
    const tableData = value.rows.map(row => columns.map(col => row[col] || ''));

    autoTable(doc, {
      startY: yPos,
      head: [columns],
      body: tableData,
      theme: 'striped',
      headStyles: {
        fillColor: COLORS.primary,
        textColor: COLORS.white,
        fontSize: 8,
        fontStyle: 'bold',
        cellPadding: 3
      },
      bodyStyles: {
        fontSize: 8,
        textColor: COLORS.text,
        cellPadding: 3
      },
      alternateRowStyles: {
        fillColor: [250, 250, 252]
      },
      margin: { left: LAYOUT.marginLeft, right: LAYOUT.marginRight },
      tableWidth: LAYOUT.contentWidth
    });

    yPos = doc.lastAutoTable.finalY + LAYOUT.paragraphSpacing;
  }

  return yPos;
}

// Helper function to render simple fields
function renderSimpleField(doc, field, value, yPos) {
  const pageHeight = doc.internal.pageSize.getHeight();
  
  if (yPos > pageHeight - 25) {
    doc.addPage();
    yPos = LAYOUT.marginTop;
  }

  doc.setFontSize(9);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...COLORS.text);
  
  const labelText = `${field.number ? field.number + ' ' : ''}${field.label}:`;
  doc.text(labelText, LAYOUT.marginLeft, yPos);

  doc.setFont('helvetica', 'normal');
  const valueStr = String(value);
  const valueLines = doc.splitTextToSize(valueStr, LAYOUT.contentWidth - 70);
  
  // Position value after label
  const labelWidth = doc.getTextWidth(labelText);
  const valueX = Math.min(LAYOUT.marginLeft + labelWidth + 3, LAYOUT.marginLeft + 65);
  
  doc.text(valueLines, valueX, yPos);

  return yPos + Math.max(6, valueLines.length * 4 + 2);
}

// Helper function to add placeholder for visual components
function addVisualComponentPlaceholder(doc, field, value, yPos, componentScreenshots = {}) {
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();

  console.log(`📄 PDF: Processing visual component "${field.name}" (${field.type})`);
  console.log(`Available screenshots:`, Object.keys(componentScreenshots));

  if (yPos > pageHeight - 90) {
    doc.addPage();
    yPos = LAYOUT.marginTop;
  }

  // Field label
  doc.setFontSize(9);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...COLORS.secondary);
  doc.text(`${field.number ? field.number + ' ' : ''}${field.label}`, LAYOUT.marginLeft, yPos);
  yPos += 6;

  // Check if we have a screenshot for this component
  const screenshot = componentScreenshots[field.name];
  console.log(`Screenshot for "${field.name}":`, screenshot ? `Found (${screenshot.substring(0, 50)}...)` : 'NOT FOUND');

  if (screenshot) {
    // Render the screenshot image
    yPos = renderComponentScreenshot(doc, screenshot, yPos, pageWidth, pageHeight);
  } else {
    // Fallback: Try to render structured data or show placeholder
    if (field.type === 'naming-conventions' && value) {
      yPos = renderNamingConventionsData(doc, value, yPos);
    } else if (field.type === 'federation-strategy' && value) {
      yPos = renderFederationStrategyData(doc, value, yPos);
    } else if (field.type === 'orgchart' && value) {
      yPos = renderOrgChartData(doc, value, yPos);
    } else {
      // Visual placeholder box for components without structured data or screenshot
      const boxHeight = 50;
      doc.setDrawColor(...COLORS.border);
      doc.setFillColor(250, 250, 252);
      doc.setLineWidth(0.5);
      doc.roundedRect(LAYOUT.marginLeft, yPos, LAYOUT.contentWidth, boxHeight, 3, 3, 'FD');

      // Icon and text - centered properly
      doc.setFontSize(9);
      doc.setFont('helvetica', 'italic');
      doc.setTextColor(120, 120, 120);

      const componentTypeLabels = {
        'orgchart': 'Organizational Structure Chart',
        'cdeDiagram': 'CDE Workflow Diagram',
        'naming-conventions': 'Naming Convention Rules',
        'federation-strategy': 'Federation Strategy & Clash Matrix',
        'mindmap': 'Volume Strategy Mind Map',
        'fileStructure': 'Folder Structure Diagram'
      };

      const label = componentTypeLabels[field.type] || 'Visual Component';
      const centerX = LAYOUT.marginLeft + (LAYOUT.contentWidth / 2);
      doc.text(`[ ${label} ]`, centerX, yPos + (boxHeight / 2) - 4, { align: 'center' });
      doc.setFontSize(8);
      doc.text('Interactive component - see online BEP for full details', centerX, yPos + (boxHeight / 2) + 6, { align: 'center' });

      yPos += boxHeight + LAYOUT.paragraphSpacing;
    }
  }

  doc.setTextColor(...COLORS.text);
  return yPos;
}

// Helper function to render component screenshot
function renderComponentScreenshot(doc, screenshotBase64, yPos, pageWidth, pageHeight) {
  try {
    // Calculate dimensions - fit to content width with proper margins
    const maxWidth = LAYOUT.contentWidth;
    const maxHeight = 100; // Maximum height in mm

    // Add the image
    const imgProps = doc.getImageProperties(screenshotBase64);
    const imgAspectRatio = imgProps.width / imgProps.height;

    let imgWidth = maxWidth;
    let imgHeight = imgWidth / imgAspectRatio;

    // If image is too tall, scale down to max height
    if (imgHeight > maxHeight) {
      imgHeight = maxHeight;
      imgWidth = imgHeight * imgAspectRatio;
    }

    // Check if we need a new page
    if (yPos + imgHeight > pageHeight - LAYOUT.marginBottom) {
      doc.addPage();
      yPos = LAYOUT.marginTop;
    }

    // Center the image within content area
    const xPos = LAYOUT.marginLeft + (LAYOUT.contentWidth - imgWidth) / 2;

    // Add subtle border around image
    doc.setDrawColor(...COLORS.border);
    doc.setLineWidth(0.3);
    doc.rect(xPos - 1, yPos - 1, imgWidth + 2, imgHeight + 2);

    // Add image to PDF
    doc.addImage(screenshotBase64, 'PNG', xPos, yPos, imgWidth, imgHeight);

    yPos += imgHeight + LAYOUT.paragraphSpacing;
  } catch (error) {
    console.error('Error rendering component screenshot:', error);
    // If image rendering fails, show a warning box
    doc.setFontSize(8);
    doc.setTextColor(200, 50, 50);
    doc.text('Failed to render component image', LAYOUT.marginLeft, yPos);
    yPos += 8;
    doc.setTextColor(...COLORS.text);
  }

  return yPos;
}

// Helper function to render naming conventions data
function renderNamingConventionsData(doc, value, yPos) {
  if (value.pattern) {
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.text);
    doc.text('Naming Pattern:', LAYOUT.marginLeft, yPos);
    yPos += 5;

    // Pattern in monospace-style box
    doc.setFillColor(245, 245, 250);
    doc.setDrawColor(...COLORS.border);
    doc.roundedRect(LAYOUT.marginLeft, yPos - 2, LAYOUT.contentWidth, 10, 2, 2, 'FD');
    doc.setFont('courier', 'normal');
    doc.setFontSize(9);
    doc.text(value.pattern, LAYOUT.marginLeft + 4, yPos + 5);
    yPos += 14;
  }

  if (value.fields && Array.isArray(value.fields) && value.fields.length > 0) {
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.text);
    doc.text('Field Definitions:', LAYOUT.marginLeft, yPos);
    yPos += 5;

    const tableData = value.fields.map(f => [
      f.code || '',
      f.description || '',
      f.example || ''
    ]);

    autoTable(doc, {
      startY: yPos,
      head: [['Code', 'Description', 'Example']],
      body: tableData,
      theme: 'striped',
      headStyles: {
        fillColor: COLORS.warning,
        textColor: COLORS.white,
        fontSize: 8,
        fontStyle: 'bold',
        cellPadding: 3
      },
      bodyStyles: {
        fontSize: 7,
        textColor: COLORS.text,
        cellPadding: 2
      },
      columnStyles: {
        0: { cellWidth: 30, fontStyle: 'bold' },
        1: { cellWidth: 'auto' },
        2: { cellWidth: 40, fontStyle: 'italic' }
      },
      margin: { left: LAYOUT.marginLeft, right: LAYOUT.marginRight },
      tableWidth: LAYOUT.contentWidth
    });

    yPos = doc.lastAutoTable.finalY + LAYOUT.paragraphSpacing;
  }

  return yPos;
}

// Helper function to render federation strategy data
function renderFederationStrategyData(doc, value, yPos) {
  if (value.approach) {
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.text);
    doc.text('Federation Approach:', LAYOUT.marginLeft, yPos);
    yPos += 5;

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8);
    const lines = doc.splitTextToSize(value.approach, LAYOUT.contentWidth);
    lines.forEach(line => {
      doc.text(line, LAYOUT.marginLeft, yPos);
      yPos += 4;
    });
    yPos += 4;
  }

  if (value.clashMatrix && Array.isArray(value.clashMatrix) && value.clashMatrix.length > 0) {
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.text);
    doc.text('Clash Detection Matrix:', LAYOUT.marginLeft, yPos);
    yPos += 5;

    const tableData = value.clashMatrix.map(row => [
      row.discipline1 || '',
      row.discipline2 || '',
      row.priority || '',
      row.frequency || ''
    ]);

    autoTable(doc, {
      startY: yPos,
      head: [['Discipline A', 'Discipline B', 'Priority', 'Check Frequency']],
      body: tableData,
      theme: 'grid',
      headStyles: {
        fillColor: COLORS.warning,
        textColor: COLORS.white,
        fontSize: 8,
        fontStyle: 'bold',
        cellPadding: 3
      },
      bodyStyles: {
        fontSize: 7,
        textColor: COLORS.text,
        cellPadding: 2
      },
      margin: { left: LAYOUT.marginLeft, right: LAYOUT.marginRight },
      tableWidth: LAYOUT.contentWidth
    });

    yPos = doc.lastAutoTable.finalY + LAYOUT.paragraphSpacing;
  }

  return yPos;
}

// Helper function to render org chart data
function renderOrgChartData(doc, value, yPos) {
  if (value.nodes && Array.isArray(value.nodes) && value.nodes.length > 0) {
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.text);
    doc.text('Organizational Roles:', LAYOUT.marginLeft, yPos);
    yPos += 5;

    const tableData = value.nodes.map(node => [
      node.role || node.title || '',
      node.name || '',
      node.company || '',
      node.contact || ''
    ]);

    autoTable(doc, {
      startY: yPos,
      head: [['Role', 'Name', 'Company', 'Contact']],
      body: tableData,
      theme: 'striped',
      headStyles: {
        fillColor: COLORS.success,
        textColor: COLORS.white,
        fontSize: 8,
        fontStyle: 'bold',
        cellPadding: 3
      },
      bodyStyles: {
        fontSize: 7,
        textColor: COLORS.text,
        cellPadding: 2
      },
      margin: { left: LAYOUT.marginLeft, right: LAYOUT.marginRight },
      tableWidth: LAYOUT.contentWidth
    });

    yPos = doc.lastAutoTable.finalY + 4;
  }

  // Add note about visual representation
  doc.setFontSize(7);
  doc.setFont('helvetica', 'italic');
  doc.setTextColor(120, 120, 120);
  doc.text('Note: See online BEP for interactive organizational chart diagram', LAYOUT.marginLeft, yPos);
  doc.setTextColor(...COLORS.text);
  yPos += LAYOUT.paragraphSpacing;

  return yPos;
}

function addPageNumbers(doc) {
  const pageCount = doc.internal.getNumberOfPages();
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();

  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    
    // Skip footer on cover page (page 1)
    if (i === 1) continue;

    // Footer separator line
    doc.setDrawColor(...COLORS.border);
    doc.setLineWidth(0.3);
    doc.line(LAYOUT.marginLeft, pageHeight - 18, pageWidth - LAYOUT.marginRight, pageHeight - 18);

    // Page number - right aligned
    doc.setFontSize(8);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(100, 100, 100);
    doc.text(
      `Page ${i} of ${pageCount}`,
      pageWidth - LAYOUT.marginRight,
      pageHeight - 12,
      { align: 'right' }
    );

    // Footer text - left aligned
    doc.setFontSize(7);
    doc.text(
      'ISO 19650-2:2018 Compliant BIM Execution Plan',
      LAYOUT.marginLeft,
      pageHeight - 12
    );

    // Document identifier - center
    doc.text(
      'BEP Generator',
      pageWidth / 2,
      pageHeight - 12,
      { align: 'center' }
    );
  }
}
