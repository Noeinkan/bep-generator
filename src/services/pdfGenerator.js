import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import CONFIG from '../config/bepConfig';

/**
 * Professional PDF Generator for BIM Execution Plans
 * Uses jsPDF with autotable plugin for structured, readable output
 */

// Color palette matching the HTML preview
const COLORS = {
  primary: [30, 64, 175],      // Blue
  secondary: [37, 99, 235],     // Lighter blue
  success: [16, 185, 129],      // Green
  warning: [245, 158, 11],      // Amber
  dark: [31, 41, 55],           // Dark gray
  light: [248, 250, 252],       // Light background
  text: [31, 41, 55],           // Text color
  white: [255, 255, 255]
};

export const generatePDF = async (formData, bepType, options = {}) => {
  const {
    orientation = 'portrait',
    format = 'a4',
    filename = `BEP_${bepType}_${new Date().toISOString().split('T')[0]}.pdf`,
    tidpData = [],
    midpData = []
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
    let yPos = 20;

    // === COVER PAGE ===
    addCoverPage(doc, formData, bepType, pageWidth);

    // === DOCUMENT INFORMATION ===
    doc.addPage();
    yPos = 20;
    yPos = addSectionHeader(doc, 'Document Information', yPos, COLORS.success);
    yPos = addDocumentInfo(doc, formData, bepType, yPos);

    // === ISO 19650 COMPLIANCE SECTION ===
    yPos = addSectionHeader(doc, 'ISO 19650-2:2018 Compliance Statement', yPos + 10, COLORS.success);
    yPos = addISOComplianceSection(doc, yPos);

    // === TIDP/MIDP SECTION ===
    if (tidpData.length > 0 || midpData.length > 0) {
      yPos = addSectionHeader(doc, 'Information Delivery Plan', yPos + 10, COLORS.warning);
      yPos = addTIDPMIDPSection(doc, tidpData, midpData, yPos);
    }

    // === BEP CONTENT SECTIONS ===
    yPos = addBEPContentSections(doc, formData, bepType, yPos);

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

function addCoverPage(doc, formData, bepType, pageWidth) {
  // Background gradient effect (simulated with rectangles)
  doc.setFillColor(...COLORS.primary);
  doc.rect(0, 0, pageWidth, 100, 'F');

  doc.setFillColor(...COLORS.secondary);
  doc.rect(0, 100, pageWidth, 50, 'F');

  // Main title
  doc.setTextColor(...COLORS.white);
  doc.setFontSize(32);
  doc.setFont('helvetica', 'bold');
  doc.text('BIM EXECUTION PLAN', pageWidth / 2, 40, { align: 'center' });

  // ISO Compliance badge
  doc.setFontSize(14);
  doc.setFont('helvetica', 'italic');
  doc.text('ISO 19650-2:2018 Compliant', pageWidth / 2, 55, { align: 'center' });

  // BEP Type badge
  doc.setFontSize(16);
  doc.setFont('helvetica', 'bold');
  const bepTypeLabel = CONFIG.bepTypeDefinitions[bepType]?.title || bepType;
  doc.text(bepTypeLabel, pageWidth / 2, 75, { align: 'center' });

  // Description box
  doc.setFillColor(255, 255, 255, 0.2);
  const descY = 95;
  doc.roundedRect(20, descY, pageWidth - 40, 40, 3, 3, 'F');

  doc.setTextColor(...COLORS.white);
  doc.setFontSize(11);
  doc.setFont('helvetica', 'italic');
  const description = CONFIG.bepTypeDefinitions[bepType]?.description || '';
  const descLines = doc.splitTextToSize(description, pageWidth - 50);
  doc.text(descLines, pageWidth / 2, descY + 10, { align: 'center', maxWidth: pageWidth - 50 });

  // Project information box
  doc.setTextColor(...COLORS.dark);
  doc.setFillColor(...COLORS.light);
  doc.roundedRect(20, 160, pageWidth - 40, 60, 3, 3, 'F');

  doc.setFontSize(18);
  doc.setFont('helvetica', 'bold');
  doc.text('Project Information', pageWidth / 2, 172, { align: 'center' });

  doc.setFontSize(12);
  doc.setFont('helvetica', 'normal');
  doc.text(`Project: ${formData.projectName || 'Not specified'}`, pageWidth / 2, 188, { align: 'center' });
  doc.text(`Project Number: ${formData.projectNumber || 'Not specified'}`, pageWidth / 2, 198, { align: 'center' });

  doc.setFontSize(10);
  doc.setFont('helvetica', 'italic');
  const dateStr = `Generated: ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString()}`;
  doc.text(dateStr, pageWidth / 2, 210, { align: 'center' });

  // Footer compliance badge
  doc.setFillColor(...COLORS.success);
  doc.roundedRect(40, 240, pageWidth - 80, 15, 3, 3, 'F');
  doc.setTextColor(...COLORS.white);
  doc.setFontSize(11);
  doc.setFont('helvetica', 'bold');
  doc.text('✓ Generated in compliance with ISO 19650-2:2018', pageWidth / 2, 249, { align: 'center' });
}

function addSectionHeader(doc, title, yPos, color = COLORS.primary) {
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();

  if (yPos > pageHeight - 40) {
    doc.addPage();
    yPos = 20;
  }

  doc.setFillColor(...color);
  doc.roundedRect(15, yPos, pageWidth - 30, 12, 2, 2, 'F');

  doc.setTextColor(...COLORS.white);
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text(title, 20, yPos + 8);

  doc.setTextColor(...COLORS.text);
  return yPos + 18;
}

function addDocumentInfo(doc, formData, bepType, yPos) {
  const docInfoData = [
    ['Document Type', CONFIG.bepTypeDefinitions[bepType]?.title || bepType],
    ['Document Purpose', CONFIG.bepTypeDefinitions[bepType]?.purpose || 'N/A'],
    ['Project Name', formData.projectName || 'Not specified'],
    ['Project Number', formData.projectNumber || 'Not specified'],
    ['Generated Date', new Date().toLocaleDateString()],
    ['Generated Time', new Date().toLocaleTimeString()],
    ['Status', bepType === 'pre-appointment' ? 'Tender Submission' : 'Working Document'],
    ['Document Version', '1.0']
  ];

  autoTable(doc, {
    startY: yPos,
    head: [['Field', 'Value']],
    body: docInfoData,
    theme: 'grid',
    headStyles: {
      fillColor: COLORS.success,
      textColor: COLORS.white,
      fontStyle: 'bold',
      fontSize: 11
    },
    bodyStyles: {
      fontSize: 10,
      textColor: COLORS.text
    },
    alternateRowStyles: {
      fillColor: COLORS.light
    },
    columnStyles: {
      0: { fontStyle: 'bold', cellWidth: 60 },
      1: { cellWidth: 'auto' }
    },
    margin: { left: 15, right: 15 }
  });

  return doc.lastAutoTable.finalY + 5;
}

function addISOComplianceSection(doc, yPos) {
  const pageHeight = doc.internal.pageSize.getHeight();

  if (yPos > pageHeight - 80) {
    doc.addPage();
    yPos = 20;
  }

  // Declaration text
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  const declarationText = 'This BIM Execution Plan (BEP) has been prepared in accordance with ISO 19650-2:2018 "Organization and digitization of information about buildings and civil engineering works, including building information modelling (BIM) — Information management using building information modelling — Part 2: Delivery phase of the assets."';
  const lines = doc.splitTextToSize(declarationText, 180);
  doc.text(lines, 15, yPos);
  yPos += lines.length * 5 + 8;

  // Requirements coverage table
  const coverageData = [
    ['5.1', 'Information Management', 'Information management function and responsibilities defined'],
    ['5.2', 'Planning Approach', 'Master Information Delivery Plan (MIDP) and Task Information Delivery Plans (TIDPs)'],
    ['5.3', 'Information Requirements', 'Exchange information requirements and level of information need defined'],
    ['5.4', 'Collaborative Production', 'Common Data Environment (CDE) workflows and federation strategy'],
    ['5.5', 'Quality Assurance', 'Information validation, review and approval processes'],
    ['5.6', 'Information Security', 'Information security protocols and access control procedures']
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
      fontSize: 10
    },
    bodyStyles: {
      fontSize: 9,
      textColor: COLORS.text
    },
    columnStyles: {
      0: { cellWidth: 15, fontStyle: 'bold', halign: 'center' },
      1: { cellWidth: 50, fontStyle: 'bold' },
      2: { cellWidth: 'auto' }
    },
    margin: { left: 15, right: 15 }
  });

  return doc.lastAutoTable.finalY + 10;
}

function addTIDPMIDPSection(doc, tidpData, midpData, yPos) {
  const pageHeight = doc.internal.pageSize.getHeight();

  // TIDP Section
  if (tidpData.length > 0) {
    if (yPos > pageHeight - 60) {
      doc.addPage();
      yPos = 20;
    }

    doc.setFontSize(12);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.warning);
    doc.text('Task Information Delivery Plans (TIDPs)', 15, yPos);
    yPos += 8;

    doc.setFontSize(9);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...COLORS.text);
    doc.text('The following TIDPs have been established for this project:', 15, yPos);
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
        fontSize: 9
      },
      bodyStyles: {
        fontSize: 8,
        textColor: COLORS.text
      },
      alternateRowStyles: {
        fillColor: [255, 251, 235]
      },
      margin: { left: 15, right: 15 }
    });

    yPos = doc.lastAutoTable.finalY + 10;
  }

  // MIDP Section
  if (midpData.length > 0) {
    if (yPos > pageHeight - 60) {
      doc.addPage();
      yPos = 20;
    }

    doc.setFontSize(12);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.warning);
    doc.text('Master Information Delivery Plan (MIDP)', 15, yPos);
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
        fontSize: 9
      },
      bodyStyles: {
        fontSize: 8,
        textColor: COLORS.text
      },
      alternateRowStyles: {
        fillColor: [255, 251, 235]
      },
      margin: { left: 15, right: 15 }
    });

    yPos = doc.lastAutoTable.finalY + 5;

    if (midpData[0]?.description) {
      doc.setFontSize(9);
      doc.setFont('helvetica', 'italic');
      const descLines = doc.splitTextToSize(`Description: ${midpData[0].description}`, 180);
      doc.text(descLines, 15, yPos + 5);
      yPos += descLines.length * 4 + 10;
    }
  }

  return yPos;
}

function addBEPContentSections(doc, formData, bepType, yPos) {
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();

  // Group steps by category
  const groupedSteps = CONFIG.steps.reduce((acc, step, index) => {
    const cat = step.category;
    if (!acc[cat]) acc[cat] = [];
    const stepConfig = CONFIG.getFormFields(bepType, index);
    if (stepConfig) {
      acc[cat].push({ index, title: stepConfig.title, fields: stepConfig.fields });
    }
    return acc;
  }, {});

  // Process each category
  Object.entries(groupedSteps).forEach(([catKey, items]) => {
    const categoryName = CONFIG.categories[catKey]?.name || catKey;

    // Add category header
    yPos = addSectionHeader(doc, categoryName, yPos, COLORS.primary);

    items.forEach((item) => {
      if (yPos > pageHeight - 40) {
        doc.addPage();
        yPos = 20;
      }

      // Section title
      doc.setFontSize(12);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(...COLORS.secondary);
      doc.text(item.title, 15, yPos);
      yPos += 8;

      // Process fields
      item.fields.forEach((field) => {
        const value = formData[field.name];
        if (!value) return;

        if (yPos > pageHeight - 30) {
          doc.addPage();
          yPos = 20;
        }

        if (field.type === 'table' && Array.isArray(value) && value.length > 0) {
          // Render as table
          const columns = field.columns || ['Column 1', 'Column 2', 'Column 3'];
          const tableData = value.map(row => columns.map(col => row[col] || ''));

          doc.setFontSize(10);
          doc.setFont('helvetica', 'bold');
          doc.text(`${field.number ? field.number + ' ' : ''}${field.label}`, 15, yPos);
          yPos += 6;

          autoTable(doc, {
            startY: yPos,
            head: [columns],
            body: tableData,
            theme: 'striped',
            headStyles: {
              fillColor: COLORS.primary,
              textColor: COLORS.white,
              fontSize: 9,
              fontStyle: 'bold'
            },
            bodyStyles: {
              fontSize: 8,
              textColor: COLORS.text
            },
            alternateRowStyles: {
              fillColor: COLORS.light
            },
            margin: { left: 15, right: 15 }
          });

          yPos = doc.lastAutoTable.finalY + 8;

        } else if (field.type === 'checkbox' && Array.isArray(value)) {
          // Render checkbox list
          doc.setFontSize(10);
          doc.setFont('helvetica', 'bold');
          doc.text(`${field.number ? field.number + ' ' : ''}${field.label}`, 15, yPos);
          yPos += 6;

          doc.setFontSize(9);
          doc.setFont('helvetica', 'normal');
          value.forEach((item) => {
            if (yPos > pageHeight - 15) {
              doc.addPage();
              yPos = 20;
            }
            doc.text(`✓ ${item}`, 20, yPos);
            yPos += 5;
          });
          yPos += 3;

        } else if (field.type === 'textarea') {
          // Render textarea
          doc.setFontSize(10);
          doc.setFont('helvetica', 'bold');
          doc.text(`${field.number ? field.number + ' ' : ''}${field.label}`, 15, yPos);
          yPos += 6;

          doc.setFontSize(9);
          doc.setFont('helvetica', 'normal');
          const textLines = doc.splitTextToSize(value, pageWidth - 30);
          textLines.forEach((line) => {
            if (yPos > pageHeight - 15) {
              doc.addPage();
              yPos = 20;
            }
            doc.text(line, 15, yPos);
            yPos += 5;
          });
          yPos += 3;

        } else {
          // Render as simple field pair
          doc.setFontSize(9);
          doc.setFont('helvetica', 'bold');
          doc.text(`${field.number ? field.number + ' ' : ''}${field.label}:`, 15, yPos);

          doc.setFont('helvetica', 'normal');
          const valueLines = doc.splitTextToSize(String(value), pageWidth - 80);
          doc.text(valueLines, 80, yPos);
          yPos += Math.max(5, valueLines.length * 5);
        }
      });

      yPos += 5;
    });

    yPos += 5;
  });

  return yPos;
}

function addPageNumbers(doc) {
  const pageCount = doc.internal.getNumberOfPages();
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();

  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(9);
    doc.setFont('helvetica', 'italic');
    doc.setTextColor(128, 128, 128);

    // Page number
    doc.text(
      `Page ${i} of ${pageCount}`,
      pageWidth / 2,
      pageHeight - 10,
      { align: 'center' }
    );

    // Footer text
    doc.setFontSize(8);
    doc.text(
      'ISO 19650-2:2018 Compliant BIM Execution Plan',
      pageWidth / 2,
      pageHeight - 6,
      { align: 'center' }
    );
  }
}
