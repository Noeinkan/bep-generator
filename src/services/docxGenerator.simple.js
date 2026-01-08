import {
  Document,
  Paragraph,
  TextRun,
  HeadingLevel,
  Table,
  TableRow,
  TableCell,
  WidthType,
  AlignmentType,
  Packer,
  BorderStyle,
  ShadingType,
  convertInchesToTwip
} from 'docx';
import CONFIG from '../config/bepConfig';

// Enhanced version with styling
export const generateDocxSimple = async (formData, bepType, options = {}) => {
  const { tidpData = [], midpData = [] } = options;
  const currentDate = new Date();
  const formattedDate = currentDate.toLocaleDateString();
  const formattedTime = currentDate.toLocaleTimeString();

  const sections = [];

  // Helper function to create bordered cells
  const createBorderedCell = (content, isBold = false) => {
    return new TableCell({
      children: [new Paragraph({
        children: [new TextRun({ text: String(content), bold: isBold })]
      })],
      borders: {
        top: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
        bottom: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
        left: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
        right: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" }
      }
    });
  };

  // Cover page with styled text
  sections.push(
    new Paragraph({
      children: [new TextRun({
        text: "BIM EXECUTION PLAN (BEP)",
        bold: true,
        size: 48  // 24pt
      })],
      heading: HeadingLevel.HEADING_1,
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 }
    }),
    new Paragraph({
      children: [new TextRun({
        text: "ISO 19650-2:2018 Compliant",
        bold: true,
        size: 32  // 16pt
      })],
      heading: HeadingLevel.HEADING_2,
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 }
    }),
    new Paragraph({
      children: [new TextRun({
        text: CONFIG.bepTypeDefinitions[bepType].title,
        size: 28  // 14pt
      })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 }
    }),
    new Paragraph({
      children: [new TextRun({
        text: CONFIG.bepTypeDefinitions[bepType].description,
        size: 24,  // 12pt
        italics: true
      })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 }
    })
  );

  // ISO Compliance Section (new page)
  sections.push(
    new Paragraph({
      children: [new TextRun({
        text: "ISO 19650 COMPLIANCE STATEMENT",
        bold: true,
        size: 32
      })],
      heading: HeadingLevel.HEADING_2,
      spacing: { before: 400, after: 200 },
      pageBreakBefore: true
    }),
    new Paragraph({
      children: [
        new TextRun({ text: "✓ ", size: 28 }),
        new TextRun({ text: "Formal Declaration of Conformity", bold: true, size: 24 })
      ],
      spacing: { after: 200 }
    }),
    new Paragraph({
      text: 'This BIM Execution Plan (BEP) has been prepared in accordance with ISO 19650-2:2018 "Organization and digitization of information about buildings and civil engineering works, including building information modelling (BIM) — Information management using building information modelling — Part 2: Delivery phase of the assets."',
      spacing: { after: 200 }
    }),
    new Paragraph({
      children: [new TextRun({ text: "Key Compliance Areas:", bold: true, size: 22 })],
      spacing: { before: 200, after: 100 }
    }),
    new Paragraph({ text: "✓ Information Management Strategy", bullet: { level: 0 } }),
    new Paragraph({ text: "✓ Information Delivery Planning (TIDP/MIDP)", bullet: { level: 0 } }),
    new Paragraph({ text: "✓ Common Data Environment (CDE) Workflow", bullet: { level: 0 } }),
    new Paragraph({ text: "✓ Information Security and Classification", bullet: { level: 0 } }),
    new Paragraph({ text: "✓ Quality Assurance and Review Procedures", bullet: { level: 0 }, spacing: { after: 400 } })
  );

  // Document Information Table
  sections.push(
    new Paragraph({
      children: [new TextRun({ text: "DOCUMENT INFORMATION", bold: true, size: 28 })],
      heading: HeadingLevel.HEADING_3,
      spacing: { before: 400, after: 200 }
    }),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      rows: [
        new TableRow({
          children: [
            createBorderedCell("Document Type:", true),
            createBorderedCell(CONFIG.bepTypeDefinitions[bepType].title)
          ]
        }),
        new TableRow({
          children: [
            createBorderedCell("Document Purpose:", true),
            createBorderedCell(CONFIG.bepTypeDefinitions[bepType].purpose)
          ]
        }),
        new TableRow({
          children: [
            createBorderedCell("Project Name:", true),
            createBorderedCell(formData.projectName || 'Not specified')
          ]
        }),
        new TableRow({
          children: [
            createBorderedCell("Project Number:", true),
            createBorderedCell(formData.projectNumber || 'Not specified')
          ]
        }),
        new TableRow({
          children: [
            createBorderedCell("Generated Date:", true),
            createBorderedCell(`${formattedDate} at ${formattedTime}`)
          ]
        }),
        new TableRow({
          children: [
            createBorderedCell("Status:", true),
            createBorderedCell(bepType === 'pre-appointment' ? 'Tender Submission' : 'Working Document')
          ]
        })
      ]
    })
  );

  // Group steps by category and add content
  const groupedSteps = CONFIG.steps.reduce((acc, step, index) => {
    const cat = step.category;
    if (!acc[cat]) acc[cat] = [];
    const stepConfig = CONFIG.getFormFields(bepType, index);
    if (stepConfig) {
      acc[cat].push({ index, title: `${acc[cat].length + 1}. ${stepConfig.title.toUpperCase()}`, fields: stepConfig.fields });
    }
    return acc;
  }, {});

  Object.entries(groupedSteps).forEach(([cat, items], catIndex) => {
    sections.push(
      new Paragraph({
        children: [new TextRun({ text: CONFIG.categories[cat].name, bold: true, size: 36 })],
        heading: HeadingLevel.HEADING_1,
        pageBreakBefore: catIndex > 0,
        spacing: { after: 300 }
      })
    );

    items.forEach((item) => {
      sections.push(
        new Paragraph({
          children: [new TextRun({ text: item.title, bold: true, size: 28 })],
          heading: HeadingLevel.HEADING_2,
          spacing: { before: 200, after: 200 }
        })
      );

      const fields = item.fields;
      const tableFields = fields.filter(f => f.type !== 'textarea' && f.type !== 'checkbox' && f.type !== 'custom');
      const otherFields = fields.filter(f => f.type === 'textarea' || f.type === 'checkbox' || f.type === 'custom');

      if (tableFields.length > 0) {
        const tableRows = tableFields.map(field =>
          new TableRow({
            children: [
              createBorderedCell(field.label + ":", true),
              createBorderedCell(formData[field.name] || '')
            ]
          })
        );

        sections.push(new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, rows: tableRows }));
      }

      otherFields.forEach(field => {
        sections.push(
          new Paragraph({
            children: [new TextRun({ text: field.label, bold: true, size: 22 })],
            heading: HeadingLevel.HEADING_3,
            spacing: { before: 200, after: 100 }
          })
        );

        const value = formData[field.name];
        if (field.type === 'checkbox' && Array.isArray(value)) {
          value.forEach(item => {
            sections.push(new Paragraph({ text: `✓ ${item}`, bullet: { level: 0 }, spacing: { after: 50 } }));
          });
        } else if (field.type === 'textarea' && value) {
          const lines = value.split('\n');
          lines.forEach(line => {
            sections.push(new Paragraph({ text: line || '', spacing: { after: 100 } }));
          });
        }
      });
    });
  });

  // Footer
  sections.push(
    new Paragraph({
      children: [new TextRun({ text: "DOCUMENT CONTROL INFORMATION", bold: true, size: 28 })],
      heading: HeadingLevel.HEADING_3,
      pageBreakBefore: true,
      spacing: { after: 200 }
    }),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      rows: [
        new TableRow({ children: [createBorderedCell("Document Type:", true), createBorderedCell("BIM Execution Plan (BEP)")] }),
        new TableRow({ children: [createBorderedCell("ISO Standard:", true), createBorderedCell("ISO 19650-2:2018")] }),
        new TableRow({ children: [createBorderedCell("Document Status:", true), createBorderedCell("Work in Progress")] }),
        new TableRow({ children: [createBorderedCell("Generated By:", true), createBorderedCell("Professional BEP Generator Tool")] }),
        new TableRow({ children: [createBorderedCell("Generated Date:", true), createBorderedCell(formattedDate)] }),
        new TableRow({ children: [createBorderedCell("Generated Time:", true), createBorderedCell(formattedTime)] })
      ]
    })
  );

  const doc = new Document({
    styles: {
      default: {
        document: {
          run: { font: "Calibri", size: 22 },
          paragraph: { spacing: { line: 276, before: 100, after: 100 } }
        }
      }
    },
    sections: [{
      properties: {
        page: {
          margin: {
            top: convertInchesToTwip(1),
            right: convertInchesToTwip(1),
            bottom: convertInchesToTwip(1),
            left: convertInchesToTwip(1)
          }
        }
      },
      children: sections,
    }],
  });

  return Packer.toBlob(doc);
};
