const ExcelJS = require('exceljs');
const PDFDocument = require('pdfkit');
const fs = require('fs');
const path = require('path');
const { format } = require('date-fns');

class ExportService {
  constructor() {
    this.tempDir = path.join(__dirname, '../temp');
    this.ensureTempDir();
  }

  ensureTempDir() {
    if (!fs.existsSync(this.tempDir)) {
      fs.mkdirSync(this.tempDir, { recursive: true });
    }
  }

  /**
   * Export TIDP to Excel
   * @param {Object} tidp - TIDP data
   * @returns {Promise<string>} File path
   */
  async exportTIDPToExcel(tidp) {
    const workbook = new ExcelJS.Workbook();

    // Set workbook properties
    workbook.creator = 'BEP Generator';
    workbook.lastModifiedBy = 'BEP Generator';
    workbook.created = new Date();
    workbook.modified = new Date();
    workbook.lastPrinted = new Date();

    // Summary Sheet
    const summarySheet = workbook.addWorksheet('TIDP Summary');
    this.createTIDPSummarySheet(summarySheet, tidp);

    // Information Containers Sheet
    const containersSheet = workbook.addWorksheet('Information Containers');
    this.createContainersSheet(containersSheet, tidp.containers || []);

    // Dependencies Sheet
    if (tidp.predecessors && tidp.predecessors.length > 0) {
      const depsSheet = workbook.addWorksheet('Dependencies');
      this.createDependenciesSheet(depsSheet, tidp.predecessors);
    }

    // Quality Requirements Sheet
    const qualitySheet = workbook.addWorksheet('Quality Requirements');
    this.createQualitySheet(qualitySheet, tidp);

    const filename = `TIDP_${tidp.teamName.replace(/\s+/g, '_')}_${format(new Date(), 'yyyyMMdd_HHmmss')}.xlsx`;
    const filepath = path.join(this.tempDir, filename);

    await workbook.xlsx.writeFile(filepath);
    return filepath;
  }

  /**
   * Export MIDP to Excel
   * @param {Object} midp - MIDP data
   * @returns {Promise<string>} File path
   */
  async exportMIDPToExcel(midp) {
    const workbook = new ExcelJS.Workbook();

    workbook.creator = 'BEP Generator';
    workbook.created = new Date();

    // Executive Summary
    const summarySheet = workbook.addWorksheet('MIDP Summary');
    this.createMIDPSummarySheet(summarySheet, midp);

    // Delivery Schedule
    const scheduleSheet = workbook.addWorksheet('Delivery Schedule');
    this.createDeliveryScheduleSheet(scheduleSheet, midp.deliverySchedule);

    // All Containers Consolidated
    const containersSheet = workbook.addWorksheet('All Information Containers');
    this.createAllContainersSheet(containersSheet, midp.aggregatedData.containers);

    // Milestones
    const milestonesSheet = workbook.addWorksheet('Milestones');
    this.createMilestonesSheet(milestonesSheet, midp.aggregatedData.milestones);

    // Dependency Matrix
    const depsSheet = workbook.addWorksheet('Dependency Matrix');
    this.createDependencyMatrixSheet(depsSheet, midp.dependencyMatrix);

    // Risk Register
    const riskSheet = workbook.addWorksheet('Risk Register');
    this.createRiskRegisterSheet(riskSheet, midp.riskRegister);

    // Resource Plan
    const resourceSheet = workbook.addWorksheet('Resource Plan');
    this.createResourcePlanSheet(resourceSheet, midp.resourcePlan);

    const filename = `MIDP_${midp.projectName.replace(/\s+/g, '_')}_${format(new Date(), 'yyyyMMdd_HHmmss')}.xlsx`;
    const filepath = path.join(this.tempDir, filename);

    await workbook.xlsx.writeFile(filepath);
    return filepath;
  }

  /**
   * Export TIDP to PDF
   * @param {Object} tidp - TIDP data
   * @returns {Promise<string>} File path
   */
  async exportTIDPToPDF(tidp) {
    const filename = `TIDP_${tidp.teamName.replace(/\s+/g, '_')}_${format(new Date(), 'yyyyMMdd_HHmmss')}.pdf`;
    const filepath = path.join(this.tempDir, filename);

    const doc = new PDFDocument({ margin: 50 });
    doc.pipe(fs.createWriteStream(filepath));

    // Header
    doc.fontSize(20).text('Task Information Delivery Plan (TIDP)', { align: 'center' });
    doc.moveDown();

    // Basic Information
    doc.fontSize(16).text('Task Team Information', { underline: true });
    doc.moveDown(0.5);
    doc.fontSize(12)
      .text(`Team Name: ${tidp.teamName}`)
      .text(`Discipline: ${tidp.discipline}`)
      .text(`Leader: ${tidp.leader}`)
      .text(`Company: ${tidp.company}`)
      .text(`Created: ${format(new Date(tidp.createdAt), 'PPP')}`)
      .text(`Version: ${tidp.version}`);

    doc.moveDown();

    // Responsibilities
    if (tidp.responsibilities) {
      doc.fontSize(14).text('Responsibilities:', { underline: true });
      doc.moveDown(0.3);
      doc.fontSize(11).text(tidp.responsibilities, { align: 'justify' });
      doc.moveDown();
    }

    // Information Containers
    if (tidp.containers && tidp.containers.length > 0) {
      doc.addPage();
      doc.fontSize(16).text('Information Containers', { underline: true });
      doc.moveDown();

      tidp.containers.forEach((container, index) => {
        if (index > 0) doc.moveDown();

        doc.fontSize(12)
          .text(`${index + 1}. ${container['Container Name'] || container.name}`, { underline: true })
          .text(`   Type: ${container.Type || container.type}`)
          .text(`   Format: ${container.Format || container.format}`)
          .text(`   LOI Level: ${container['LOI Level'] || container.levelOfInformation}`)
          .text(`   Author: ${container.Author || container.author}`)
          .text(`   Estimated Time: ${container['Est. Time'] || container.estimatedProductionTime}`)
          .text(`   Milestone: ${container.Milestone || container.deliveryMilestone}`)
          .text(`   Due Date: ${container['Due Date'] || container.dueDate}`)
          .text(`   Status: ${container.Status || container.status}`);
      });
    }

    // Quality Requirements
    if (tidp.qualityChecks || tidp.reviewProcess) {
      doc.addPage();
      doc.fontSize(16).text('Quality Requirements', { underline: true });
      doc.moveDown();

      if (tidp.qualityChecks) {
        doc.fontSize(14).text('Quality Checking Procedures:');
        doc.fontSize(11).text(tidp.qualityChecks, { align: 'justify' });
        doc.moveDown();
      }

      if (tidp.reviewProcess) {
        doc.fontSize(14).text('Review and Approval Process:');
        doc.fontSize(11).text(tidp.reviewProcess, { align: 'justify' });
      }
    }

    doc.end();

    return new Promise((resolve, reject) => {
      doc.on('end', () => resolve(filepath));
      doc.on('error', reject);
    });
  }

  /**
   * Export MIDP to PDF
   * @param {Object} midp - MIDP data
   * @returns {Promise<string>} File path
   */
  async exportMIDPToPDF(midp) {
    const filename = `MIDP_${midp.projectName.replace(/\s+/g, '_')}_${format(new Date(), 'yyyyMMdd_HHmmss')}.pdf`;
    const filepath = path.join(this.tempDir, filename);

    const doc = new PDFDocument({ margin: 50 });
    doc.pipe(fs.createWriteStream(filepath));

    // Title Page
    doc.fontSize(24).text('Master Information Delivery Plan', { align: 'center' });
    doc.moveDown();
    doc.fontSize(20).text(midp.projectName, { align: 'center' });
    doc.moveDown(2);

    // Project Information
    doc.fontSize(16).text('Project Information', { underline: true });
    doc.moveDown(0.5);
    doc.fontSize(12)
      .text(`Lead Appointed Party: ${midp.leadAppointedParty}`)
      .text(`Information Manager: ${midp.informationManager}`)
      .text(`Baseline Date: ${format(new Date(midp.baselineDate), 'PPP')}`)
      .text(`Version: ${midp.version}`)
      .text(`Status: ${midp.status}`);

    doc.moveDown();

    // Included TIDPs
    doc.fontSize(14).text('Included Task Teams:', { underline: true });
    doc.moveDown(0.3);
    midp.aggregatedData.disciplines.forEach(discipline => {
      doc.fontSize(11).text(`• ${discipline}`);
    });

    // Summary Statistics
    doc.moveDown();
    doc.fontSize(14).text('Summary Statistics:', { underline: true });
    doc.moveDown(0.3);
    doc.fontSize(11)
      .text(`Total Information Containers: ${midp.aggregatedData.totalContainers}`)
      .text(`Total Estimated Hours: ${midp.aggregatedData.totalEstimatedHours}`)
      .text(`Number of Milestones: ${midp.aggregatedData.milestones.length}`)
      .text(`Disciplines Involved: ${midp.aggregatedData.disciplines.length}`);

    // Milestones
    if (midp.aggregatedData.milestones.length > 0) {
      doc.addPage();
      doc.fontSize(16).text('Delivery Milestones', { underline: true });
      doc.moveDown();

      midp.aggregatedData.milestones.forEach((milestone, index) => {
        if (index > 0) doc.moveDown();

        doc.fontSize(12)
          .text(`${index + 1}. ${milestone.name}`, { underline: true })
          .text(`   Containers: ${milestone.containers.length}`)
          .text(`   Teams: ${milestone.teams.join(', ')}`)
          .text(`   Earliest Date: ${milestone.earliestDate}`)
          .text(`   Latest Date: ${milestone.latestDate}`)
          .text(`   Estimated Hours: ${milestone.totalEstimatedHours}`)
          .text(`   Review Duration: ${milestone.reviewDuration}`)
          .text(`   Risk Level: ${milestone.riskLevel}`);
      });
    }

    // Risk Summary
    if (midp.riskRegister && midp.riskRegister.risks.length > 0) {
      doc.addPage();
      doc.fontSize(16).text('Risk Register Summary', { underline: true });
      doc.moveDown();

      doc.fontSize(12)
        .text(`Total Risks: ${midp.riskRegister.summary.total}`)
        .text(`High Impact: ${midp.riskRegister.summary.high}`)
        .text(`Medium Impact: ${midp.riskRegister.summary.medium}`)
        .text(`Low Impact: ${midp.riskRegister.summary.low}`);

      doc.moveDown();

      // Top 5 risks
      const topRisks = midp.riskRegister.risks.slice(0, 5);
      doc.fontSize(14).text('Top Priority Risks:', { underline: true });
      doc.moveDown(0.3);

      topRisks.forEach((risk, index) => {
        doc.fontSize(11)
          .text(`${index + 1}. ${risk.description}`)
          .text(`   Impact: ${risk.impact} | Probability: ${risk.probability}`)
          .text(`   Mitigation: ${risk.mitigation}`)
          .moveDown(0.3);
      });
    }

    doc.end();

    return new Promise((resolve, reject) => {
      doc.on('end', () => resolve(filepath));
      doc.on('error', reject);
    });
  }

  // Helper methods for Excel sheet creation
  createTIDPSummarySheet(sheet, tidp) {
    // Title
    sheet.mergeCells('A1:D1');
    sheet.getCell('A1').value = 'Task Information Delivery Plan (TIDP)';
    sheet.getCell('A1').font = { size: 16, bold: true };
    sheet.getCell('A1').alignment = { horizontal: 'center' };

    // Basic Information
    sheet.getCell('A3').value = 'Team Information';
    sheet.getCell('A3').font = { bold: true };

    sheet.getCell('A4').value = 'Team Name:';
    sheet.getCell('B4').value = tidp.teamName;
    sheet.getCell('A5').value = 'Discipline:';
    sheet.getCell('B5').value = tidp.discipline;
    sheet.getCell('A6').value = 'Leader:';
    sheet.getCell('B6').value = tidp.leader;
    sheet.getCell('A7').value = 'Company:';
    sheet.getCell('B7').value = tidp.company;
    sheet.getCell('A8').value = 'Version:';
    sheet.getCell('B8').value = tidp.version;
    sheet.getCell('A9').value = 'Status:';
    sheet.getCell('B9').value = tidp.status;

    // Responsibilities
    if (tidp.responsibilities) {
      sheet.getCell('A11').value = 'Responsibilities:';
      sheet.getCell('A11').font = { bold: true };
      sheet.mergeCells('A12:D15');
      sheet.getCell('A12').value = tidp.responsibilities;
      sheet.getCell('A12').alignment = { wrapText: true, vertical: 'top' };
    }

    // Auto-fit columns
    sheet.columns = [
      { width: 20 },
      { width: 30 },
      { width: 20 },
      { width: 30 }
    ];
  }

  createContainersSheet(sheet, containers) {
    // Headers
    const headers = [
      'Container Name', 'Type', 'Format', 'LOI Level', 'Author',
      'Dependencies', 'Est. Time', 'Milestone', 'Due Date', 'Status'
    ];

    headers.forEach((header, index) => {
      const cell = sheet.getCell(1, index + 1);
      cell.value = header;
      cell.font = { bold: true };
      cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE6E6FA' } };
    });

    // Data
    containers.forEach((container, rowIndex) => {
      const row = rowIndex + 2;
      sheet.getCell(row, 1).value = container['Container Name'] || container.name;
      sheet.getCell(row, 2).value = container.Type || container.type;
      sheet.getCell(row, 3).value = container.Format || container.format;
      sheet.getCell(row, 4).value = container['LOI Level'] || container.levelOfInformation;
      sheet.getCell(row, 5).value = container.Author || container.author;
      sheet.getCell(row, 6).value = Array.isArray(container.dependencies)
        ? container.dependencies.join(', ')
        : container.dependencies || '';
      sheet.getCell(row, 7).value = container['Est. Time'] || container.estimatedProductionTime;
      sheet.getCell(row, 8).value = container.Milestone || container.deliveryMilestone;
      sheet.getCell(row, 9).value = container['Due Date'] || container.dueDate;
      sheet.getCell(row, 10).value = container.Status || container.status;
    });

    // Auto-fit columns
    sheet.columns.forEach(column => {
      column.width = 15;
    });
  }

  createMIDPSummarySheet(sheet, midp) {
    // Title
    sheet.mergeCells('A1:E1');
    sheet.getCell('A1').value = 'Master Information Delivery Plan (MIDP)';
    sheet.getCell('A1').font = { size: 16, bold: true };
    sheet.getCell('A1').alignment = { horizontal: 'center' };

    // Project Information
    sheet.getCell('A3').value = 'Project Information';
    sheet.getCell('A3').font = { bold: true };

    sheet.getCell('A4').value = 'Project Name:';
    sheet.getCell('B4').value = midp.projectName;
    sheet.getCell('A5').value = 'Lead Appointed Party:';
    sheet.getCell('B5').value = midp.leadAppointedParty;
    sheet.getCell('A6').value = 'Information Manager:';
    sheet.getCell('B6').value = midp.informationManager;
    sheet.getCell('A7').value = 'Version:';
    sheet.getCell('B7').value = midp.version;
    sheet.getCell('A8').value = 'Status:';
    sheet.getCell('B8').value = midp.status;

    // Summary Statistics
    sheet.getCell('A10').value = 'Summary Statistics';
    sheet.getCell('A10').font = { bold: true };

    sheet.getCell('A11').value = 'Total Containers:';
    sheet.getCell('B11').value = midp.aggregatedData.totalContainers;
    sheet.getCell('A12').value = 'Total Estimated Hours:';
    sheet.getCell('B12').value = midp.aggregatedData.totalEstimatedHours;
    sheet.getCell('A13').value = 'Number of Milestones:';
    sheet.getCell('B13').value = midp.aggregatedData.milestones.length;
    sheet.getCell('A14').value = 'Disciplines Involved:';
    sheet.getCell('B14').value = midp.aggregatedData.disciplines.length;

    // Disciplines List
    sheet.getCell('D10').value = 'Involved Disciplines';
    sheet.getCell('D10').font = { bold: true };

    midp.aggregatedData.disciplines.forEach((discipline, index) => {
      sheet.getCell(11 + index, 4).value = discipline;
    });

    sheet.columns = [
      { width: 25 },
      { width: 30 },
      { width: 5 },
      { width: 25 },
      { width: 20 }
    ];
  }

  createMilestonesSheet(sheet, milestones) {
    const headers = [
      'Milestone Name', 'Container Count', 'Teams Involved', 'Earliest Date',
      'Latest Date', 'Estimated Hours', 'Review Duration', 'Risk Level'
    ];

    headers.forEach((header, index) => {
      const cell = sheet.getCell(1, index + 1);
      cell.value = header;
      cell.font = { bold: true };
      cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE6E6FA' } };
    });

    milestones.forEach((milestone, rowIndex) => {
      const row = rowIndex + 2;
      sheet.getCell(row, 1).value = milestone.name;
      sheet.getCell(row, 2).value = milestone.containers.length;
      sheet.getCell(row, 3).value = milestone.teams.join(', ');
      sheet.getCell(row, 4).value = milestone.earliestDate;
      sheet.getCell(row, 5).value = milestone.latestDate;
      sheet.getCell(row, 6).value = milestone.totalEstimatedHours;
      sheet.getCell(row, 7).value = milestone.reviewDuration;
      sheet.getCell(row, 8).value = milestone.riskLevel;

      // Color code risk levels
      const riskCell = sheet.getCell(row, 8);
      switch (milestone.riskLevel) {
        case 'High':
          riskCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFF9999' } };
          break;
        case 'Medium':
          riskCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFFFF99' } };
          break;
        case 'Low':
          riskCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF99FF99' } };
          break;
      }
    });

    sheet.columns.forEach(column => {
      column.width = 18;
    });
  }

  createAllContainersSheet(sheet, containers) {
    const headers = [
      'Container Name', 'Type', 'Format', 'LOI Level', 'Source Team',
      'Discipline', 'Author', 'Estimated Time', 'Milestone', 'Due Date', 'Status'
    ];

    headers.forEach((header, index) => {
      const cell = sheet.getCell(1, index + 1);
      cell.value = header;
      cell.font = { bold: true };
      cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE6E6FA' } };
    });

    containers.forEach((container, rowIndex) => {
      const row = rowIndex + 2;
      sheet.getCell(row, 1).value = container.name;
      sheet.getCell(row, 2).value = container.type;
      sheet.getCell(row, 3).value = container.format;
      sheet.getCell(row, 4).value = container.loiLevel;
      sheet.getCell(row, 5).value = container.tidpSource.teamName;
      sheet.getCell(row, 6).value = container.tidpSource.discipline;
      sheet.getCell(row, 7).value = container.author;
      sheet.getCell(row, 8).value = container.estimatedTime;
      sheet.getCell(row, 9).value = container.milestone;
      sheet.getCell(row, 10).value = container.dueDate;
      sheet.getCell(row, 11).value = container.status;
    });

    sheet.columns.forEach(column => {
      column.width = 16;
    });
  }

  createDeliveryScheduleSheet(sheet, schedule) {
    sheet.getCell('A1').value = 'Delivery Schedule by Phase';
    sheet.getCell('A1').font = { size: 14, bold: true };

    const headers = ['Period', 'Container Count', 'Disciplines', 'Total Hours'];

    headers.forEach((header, index) => {
      const cell = sheet.getCell(3, index + 1);
      cell.value = header;
      cell.font = { bold: true };
      cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE6E6FA' } };
    });

    schedule.phases.forEach((phase, rowIndex) => {
      const row = rowIndex + 4;
      sheet.getCell(row, 1).value = phase.period;
      sheet.getCell(row, 2).value = phase.containerCount;
      sheet.getCell(row, 3).value = phase.disciplines.join(', ');
      sheet.getCell(row, 4).value = phase.totalEstimatedHours;
    });

    sheet.columns = [
      { width: 15 },
      { width: 15 },
      { width: 30 },
      { width: 15 }
    ];
  }

  createDependencyMatrixSheet(sheet, dependencyMatrix) {
    sheet.getCell('A1').value = 'Dependency Matrix';
    sheet.getCell('A1').font = { size: 14, bold: true };

    if (!dependencyMatrix || !dependencyMatrix.matrix) {
      sheet.getCell('A3').value = 'No dependency data available';
      return;
    }

    const headers = ['From Team', 'From Container', 'To Team', 'To Container', 'Type', 'Critical Path'];

    headers.forEach((header, index) => {
      const cell = sheet.getCell(3, index + 1);
      cell.value = header;
      cell.font = { bold: true };
      cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE6E6FA' } };
    });

    dependencyMatrix.matrix.forEach((dep, rowIndex) => {
      const row = rowIndex + 4;
      sheet.getCell(row, 1).value = dep.from.tidpName;
      sheet.getCell(row, 2).value = dep.from.containerName || 'N/A';
      sheet.getCell(row, 3).value = dep.to.tidpName;
      sheet.getCell(row, 4).value = dep.to.containerName;
      sheet.getCell(row, 5).value = dep.type;
      sheet.getCell(row, 6).value = dep.criticalPath ? 'Yes' : 'No';

      if (dep.criticalPath) {
        for (let col = 1; col <= 6; col++) {
          sheet.getCell(row, col).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFFCC99' } };
        }
      }
    });

    sheet.columns.forEach(column => {
      column.width = 20;
    });
  }

  createRiskRegisterSheet(sheet, riskRegister) {
    sheet.getCell('A1').value = 'Risk Register';
    sheet.getCell('A1').font = { size: 14, bold: true };

    if (!riskRegister || !riskRegister.risks) {
      sheet.getCell('A3').value = 'No risk data available';
      return;
    }

    const headers = ['Risk Description', 'Impact', 'Probability', 'Category', 'Source', 'Mitigation'];

    headers.forEach((header, index) => {
      const cell = sheet.getCell(3, index + 1);
      cell.value = header;
      cell.font = { bold: true };
      cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE6E6FA' } };
    });

    riskRegister.risks.forEach((risk, rowIndex) => {
      const row = rowIndex + 4;
      sheet.getCell(row, 1).value = risk.description;
      sheet.getCell(row, 2).value = risk.impact;
      sheet.getCell(row, 3).value = risk.probability;
      sheet.getCell(row, 4).value = risk.category;
      sheet.getCell(row, 5).value = risk.source;
      sheet.getCell(row, 6).value = risk.mitigation;

      // Color code by impact
      const impactCell = sheet.getCell(row, 2);
      switch (risk.impact) {
        case 'High':
          impactCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFF9999' } };
          break;
        case 'Medium':
          impactCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFFFF99' } };
          break;
        case 'Low':
          impactCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF99FF99' } };
          break;
      }
    });

    sheet.columns = [
      { width: 40 },
      { width: 12 },
      { width: 12 },
      { width: 15 },
      { width: 20 },
      { width: 40 }
    ];
  }

  createResourcePlanSheet(sheet, resourcePlan) {
    sheet.getCell('A1').value = 'Resource Planning Summary';
    sheet.getCell('A1').font = { size: 14, bold: true };

    // By Discipline
    sheet.getCell('A3').value = 'Resource Allocation by Discipline';
    sheet.getCell('A3').font = { bold: true };

    const disciplineHeaders = ['Discipline', 'Teams', 'Containers', 'Total Hours'];
    disciplineHeaders.forEach((header, index) => {
      const cell = sheet.getCell(4, index + 1);
      cell.value = header;
      cell.font = { bold: true };
      cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE6E6FA' } };
    });

    let row = 5;
    Object.entries(resourcePlan.byDiscipline).forEach(([discipline, data]) => {
      sheet.getCell(row, 1).value = discipline;
      sheet.getCell(row, 2).value = data.teams;
      sheet.getCell(row, 3).value = data.containers;
      sheet.getCell(row, 4).value = data.totalHours;
      row++;
    });

    // Peak Utilization
    row += 2;
    sheet.getCell(`A${row}`).value = 'Peak Utilization';
    sheet.getCell(`A${row}`).font = { bold: true };
    row++;

    if (resourcePlan.peakUtilization) {
      sheet.getCell(`A${row}`).value = 'Period:';
      sheet.getCell(`B${row}`).value = resourcePlan.peakUtilization.period;
      row++;
      sheet.getCell(`A${row}`).value = 'Hours:';
      sheet.getCell(`B${row}`).value = resourcePlan.peakUtilization.hours;
      row++;
      sheet.getCell(`A${row}`).value = 'Disciplines:';
      sheet.getCell(`B${row}`).value = resourcePlan.peakUtilization.disciplines;
    }

    sheet.columns = [
      { width: 20 },
      { width: 15 },
      { width: 15 },
      { width: 15 }
    ];
  }

  createDependenciesSheet(sheet, dependencies) {
    const headers = ['Required Information', 'Source Team', 'Format', 'Required Date'];

    headers.forEach((header, index) => {
      const cell = sheet.getCell(1, index + 1);
      cell.value = header;
      cell.font = { bold: true };
      cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE6E6FA' } };
    });

    dependencies.forEach((dep, rowIndex) => {
      const row = rowIndex + 2;
      sheet.getCell(row, 1).value = dep['Required Information'];
      sheet.getCell(row, 2).value = dep['Source Team'];
      sheet.getCell(row, 3).value = dep.Format;
      sheet.getCell(row, 4).value = dep['Required Date'];
    });

    sheet.columns.forEach(column => {
      column.width = 20;
    });
  }

  createQualitySheet(sheet, tidp) {
    sheet.getCell('A1').value = 'Quality Requirements';
    sheet.getCell('A1').font = { size: 14, bold: true };

    if (tidp.qualityChecks) {
      sheet.getCell('A3').value = 'Quality Checking Procedures:';
      sheet.getCell('A3').font = { bold: true };
      sheet.mergeCells('A4:D8');
      sheet.getCell('A4').value = tidp.qualityChecks;
      sheet.getCell('A4').alignment = { wrapText: true, vertical: 'top' };
    }

    if (tidp.reviewProcess) {
      sheet.getCell('A10').value = 'Review and Approval Process:';
      sheet.getCell('A10').font = { bold: true };
      sheet.mergeCells('A11:D15');
      sheet.getCell('A11').value = tidp.reviewProcess;
      sheet.getCell('A11').alignment = { wrapText: true, vertical: 'top' };
    }

    sheet.columns = [
      { width: 20 },
      { width: 25 },
      { width: 25 },
      { width: 25 }
    ];
  }

  /**
   * Clean up temporary files
   * @param {string} filepath - File path to clean up
   */
  cleanupFile(filepath) {
    try {
      if (fs.existsSync(filepath)) {
        fs.unlinkSync(filepath);
      }
    } catch (error) {
      console.error('Error cleaning up file:', error);
    }
  }
}

module.exports = new ExportService();