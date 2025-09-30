import { Document, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, AlignmentType, Packer } from 'docx';
import CONFIG from '../config/bepConfig';

export const generateDocx = async (formData, bepType, options = {}) => {
  const { tidpData = [], midpData = [] } = options;
  const currentDate = new Date();
  const formattedDate = currentDate.toLocaleDateString();
  const formattedTime = currentDate.toLocaleTimeString();

  const sections = [];

  // Header
  sections.push(
    new Paragraph({
      text: "BIM EXECUTION PLAN (BEP)",
      heading: HeadingLevel.HEADING_1,
      alignment: AlignmentType.CENTER
    }),
    new Paragraph({
      text: "ISO 19650-2 Compliant",
      heading: HeadingLevel.HEADING_2,
      alignment: AlignmentType.CENTER
    }),
    new Paragraph({
      text: CONFIG.bepTypeDefinitions[bepType].title,
      alignment: AlignmentType.CENTER
    }),
    new Paragraph({
      text: CONFIG.bepTypeDefinitions[bepType].description,
      alignment: AlignmentType.CENTER
    })
  );

  // Document Information Table
  sections.push(
    new Paragraph({
      text: "Document Information",
      heading: HeadingLevel.HEADING_3
    }),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      rows: [
        new TableRow({
          children: [
            new TableCell({ children: [new Paragraph("Document Type:")], width: { size: 50, type: WidthType.PERCENTAGE } }),
            new TableCell({ children: [new Paragraph(CONFIG.bepTypeDefinitions[bepType].title)] })
          ]
        }),
        new TableRow({
          children: [
            new TableCell({ children: [new Paragraph("Document Purpose:")] }),
            new TableCell({ children: [new Paragraph(CONFIG.bepTypeDefinitions[bepType].purpose)] })
          ]
        }),
        new TableRow({
          children: [
            new TableCell({ children: [new Paragraph("Project Name:")] }),
            new TableCell({ children: [new Paragraph(formData.projectName || 'Not specified')] })
          ]
        }),
        new TableRow({
          children: [
            new TableCell({ children: [new Paragraph("Project Number:")] }),
            new TableCell({ children: [new Paragraph(formData.projectNumber || 'Not specified')] })
          ]
        }),
        new TableRow({
          children: [
            new TableCell({ children: [new Paragraph("Generated Date:")] }),
            new TableCell({ children: [new Paragraph(`${formattedDate} at ${formattedTime}`)] })
          ]
        }),
        new TableRow({
          children: [
            new TableCell({ children: [new Paragraph("Status:")] }),
            new TableCell({ children: [new Paragraph(bepType === 'pre-appointment' ? 'Tender Submission' : 'Working Document')] })
          ]
        })
      ]
    })
  );

  // Information Delivery Plan Section
  if (tidpData.length > 0 || midpData.length > 0) {
    sections.push(
      new Paragraph({
        text: "INFORMATION DELIVERY PLAN",
        heading: HeadingLevel.HEADING_2,
        pageBreakBefore: true
      })
    );

    if (tidpData.length > 0) {
      sections.push(
        new Paragraph({
          text: "Task Information Delivery Plans (TIDPs)",
          heading: HeadingLevel.HEADING_3
        }),
        new Paragraph({
          text: "The following TIDPs have been created for this project, defining specific information delivery requirements for each task team:"
        })
      );

      tidpData.forEach((tidp, index) => {
        sections.push(
          new Paragraph({
            text: `${tidp.teamName || tidp.taskTeam || `Task Team ${index + 1}`}`,
            heading: HeadingLevel.HEADING_4
          }),
          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: [
              new TableRow({
                children: [
                  new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: "Team Leader:", bold: true })] })] }),
                  new TableCell({ children: [new Paragraph(tidp.leader || tidp.teamLeader || 'TBD')] })
                ]
              }),
              new TableRow({
                children: [
                  new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: "Responsibilities:", bold: true })] })] }),
                  new TableCell({ children: [new Paragraph(tidp.responsibilities || tidp.description || 'TBD')] })
                ]
              })
            ]
          })
        );

        if (tidp.containers && tidp.containers.length > 0) {
          sections.push(
            new Paragraph({
              text: "Information Containers:",
              bold: true
            })
          );
          tidp.containers.forEach(container => {
            sections.push(
              new Paragraph({
                text: `• ${container.name || container}`,
                indent: { left: 720 } // 0.5 inch indent
              })
            );
          });
        }
      });
    }

    if (midpData.length > 0) {
      sections.push(
        new Paragraph({
          text: "Master Information Delivery Plan (MIDP)",
          heading: HeadingLevel.HEADING_3
        }),
        new Paragraph({
          text: "The consolidated MIDP provides a project-wide view of all information delivery milestones:"
        })
      );

      midpData.forEach((midp, index) => {
        sections.push(
          new Paragraph({
            text: `${midp.name || `MIDP ${index + 1}`}`,
            heading: HeadingLevel.HEADING_4
          }),
          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: [
              new TableRow({
                children: [
                  new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: "Description:", bold: true })] })] }),
                  new TableCell({ children: [new Paragraph(midp.description || 'Consolidated information delivery plan')] })
                ]
              }),
              new TableRow({
                children: [
                  new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: "Status:", bold: true })] })] }),
                  new TableCell({ children: [new Paragraph(midp.status || 'Active')] })
                ]
              })
            ]
          })
        );

        if (midp.milestones && midp.milestones.length > 0) {
          sections.push(
            new Paragraph({
              text: "Key Milestones:",
              bold: true
            })
          );
          midp.milestones.slice(0, 5).forEach(milestone => {
            sections.push(
              new Paragraph({
                text: `• ${milestone.name || milestone.title || milestone} - ${milestone.date || 'TBD'}`,
                indent: { left: 720 }
              })
            );
          });
          if (midp.milestones.length > 5) {
            sections.push(
              new Paragraph({
                text: `... and ${midp.milestones.length - 5} more milestones`,
                indent: { left: 720 },
                italics: true
              })
            );
          }
        }
      });
    }

    sections.push(
      new Paragraph({
        text: "Integration with BEP",
        heading: HeadingLevel.HEADING_4
      }),
      new Paragraph({
        text: "The TIDPs and MIDP defined above are integral components of this BIM Execution Plan, providing the detailed information delivery framework required by ISO 19650-2:2018. The BEP establishes the overarching information management strategy, while the TIDPs and MIDP provide the specific implementation details for each task team and the project as a whole."
      })
    );
  }

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

  Object.entries(groupedSteps).forEach(([cat, items]) => {
    sections.push(
      new Paragraph({
        text: CONFIG.categories[cat].name,
        heading: HeadingLevel.HEADING_1
      })
    );

    items.forEach(item => {
      sections.push(
        new Paragraph({
          text: item.title,
          heading: HeadingLevel.HEADING_2
        })
      );

      const fields = item.fields;
      const tableFields = fields.filter(f => f.type !== 'textarea' && f.type !== 'checkbox');
      const otherFields = fields.filter(f => f.type === 'textarea' || f.type === 'checkbox');

      if (tableFields.length > 0) {
        const tableRows = tableFields.map(field =>
          new TableRow({
            children: [
              new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: field.label + ":", bold: true }) ] })] }),
              new TableCell({ children: [new Paragraph(formData[field.name] || '')] })
            ]
          })
        );

        sections.push(
          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: tableRows
          })
        );
      }

      otherFields.forEach(field => {
        sections.push(
          new Paragraph({
            text: field.label,
            heading: HeadingLevel.HEADING_3
          })
        );

        const value = formData[field.name];
        if (field.type === 'checkbox' && Array.isArray(value)) {
          value.forEach(item => {
            sections.push(
              new Paragraph({
                text: item,
                bullet: { level: 0 }
              })
            );
          });
        } else if (field.type === 'textarea') {
          sections.push(
            new Paragraph(value || '')
          );
        }
      });
    });
  });

  // Footer
  sections.push(
    new Paragraph({
      text: "Document Control Information",
      heading: HeadingLevel.HEADING_3
    }),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      rows: [
        new TableRow({
          children: [
            new TableCell({ children: [new Paragraph("Document Type:")] }),
            new TableCell({ children: [new Paragraph("BIM Execution Plan (BEP)")] })
          ]
        }),
        new TableRow({
          children: [
            new TableCell({ children: [new Paragraph("ISO Standard:")] }),
            new TableCell({ children: [new Paragraph("ISO 19650-2:2018")] })
          ]
        }),
        new TableRow({
          children: [
            new TableCell({ children: [new Paragraph("Document Status:")] }),
            new TableCell({ children: [new Paragraph("Work in Progress")] })
          ]
        }),
        new TableRow({
          children: [
            new TableCell({ children: [new Paragraph("Generated By:")] }),
            new TableCell({ children: [new Paragraph("Professional BEP Generator Tool")] })
          ]
        }),
        new TableRow({
          children: [
            new TableCell({ children: [new Paragraph("Generated Date:")] }),
            new TableCell({ children: [new Paragraph(formattedDate)] })
          ]
        }),
        new TableRow({
          children: [
            new TableCell({ children: [new Paragraph("Generated Time:")] }),
            new TableCell({ children: [new Paragraph(formattedTime)] })
          ]
        })
      ]
    })
  );

  const doc = new Document({
    sections: [{
      properties: {},
      children: sections,
    }],
  });

  return Packer.toBlob(doc);
};