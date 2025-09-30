import { generateBEPContent } from './bepFormatter';

export const generatePDF = async (formData, bepType, options = {}) => {
  const {
    orientation = 'portrait',
    format = 'a4',
    margin = [20, 20, 20, 20], // [top, right, bottom, left] in mm
    filename = `BEP_${bepType}_${new Date().toISOString().split('T')[0]}.pdf`,
    tidpData = [],
    midpData = []
  } = options;

  try {
    console.log('Starting PDF generation with options:', {
      orientation, format, margin, filename
    });

    // Generate the HTML content using the same formatter as the HTML export
    const htmlContent = generateBEPContent(formData, bepType, { tidpData, midpData });
    console.log('HTML content generated, length:', htmlContent.length);

    // Validate that we have content
    if (!htmlContent || htmlContent.trim().length === 0) {
      throw new Error('No HTML content generated for PDF');
    }

    // Simple and reliable approach: extract text and create PDF with jsPDF directly
    const { jsPDF } = await import('jspdf');

    // Create PDF document
    const doc = new jsPDF({
      orientation: orientation,
      unit: 'mm',
      format: format,
      compress: true
    });

    // Extract text content from HTML
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = htmlContent;
    let textContent = tempDiv.textContent || tempDiv.innerText || 'No content available';

    // Clean up the text
    textContent = textContent.replace(/\s+/g, ' ').trim();

    console.log('Extracted text length:', textContent.length);

    // Set up document properties with ISO 19650 metadata
    doc.setProperties({
      title: `BIM Execution Plan - ${bepType} - ISO 19650-2:2018 Compliant`,
      subject: 'BIM Execution Plan Document - Information Management using Building Information Modelling',
      author: formData.informationManager || formData.proposedInfoManager || 'BEP Generator',
      creator: 'Professional BEP Generator Tool',
      keywords: 'BIM, BEP, ISO 19650, ISO 19650-2:2018, Information Management, TIDP, MIDP, Building Information Modelling',
      producer: `BEP Generator v1.0 - ISO 19650 Compliant`,
      description: `${bepType === 'pre-appointment' ? 'Pre-Appointment' : 'Post-Appointment'} BIM Execution Plan prepared in accordance with ISO 19650-2:2018 for ${formData.projectName || 'project'}`
    });

    // Set font and size
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(11);

    // Calculate page dimensions
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const marginLeft = margin[3];
    const marginRight = margin[1];
    const marginTop = margin[0];
    const marginBottom = margin[2];
    const contentWidth = pageWidth - marginLeft - marginRight;
    const contentHeight = pageHeight - marginTop - marginBottom;

    // Split text into lines that fit the page width
    const lines = doc.splitTextToSize(textContent, contentWidth);

    console.log('Total lines to render:', lines.length);

    let y = marginTop;
    let pageCount = 1;

    // Add title with ISO compliance
    doc.setFontSize(16);
    doc.setFont('helvetica', 'bold');
    const title = `BIM EXECUTION PLAN`;
    const titleLines = doc.splitTextToSize(title, contentWidth);
    titleLines.forEach(line => {
      if (y + 10 > pageHeight - marginBottom) {
        doc.addPage();
        y = marginTop;
        pageCount++;
      }
      doc.text(line, marginLeft, y);
      y += 8;
    });

    // Add ISO compliance badge
    doc.setFontSize(11);
    doc.setFont('helvetica', 'italic');
    doc.setTextColor(16, 185, 129); // Green color
    doc.text('ISO 19650-2:2018 Compliant', marginLeft, y);
    doc.setTextColor(0, 0, 0); // Reset to black
    y += 7;

    // Add BEP type
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(12);
    doc.text(`${bepType === 'pre-appointment' ? 'Pre-Appointment BEP' : 'Post-Appointment BEP'}`, marginLeft, y);
    y += 7;

    // Add date
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    const dateStr = `Generated on: ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString()}`;
    if (y + 6 > pageHeight - marginBottom) {
      doc.addPage();
      y = marginTop;
      pageCount++;
    }
    doc.text(dateStr, marginLeft, y);
    y += 10;

    // Add separator line
    doc.setLineWidth(0.5);
    doc.line(marginLeft, y, pageWidth - marginRight, y);
    y += 8;

    // Add content
    doc.setFontSize(11);
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // Check if we need a new page
      if (y + 6 > pageHeight - marginBottom) {
        doc.addPage();
        y = marginTop;
        pageCount++;

        // Re-add header on new pages
        doc.setFontSize(10);
        doc.setFont('helvetica', 'italic');
        doc.text(`Page ${pageCount} - Continued`, marginLeft, y);
        y += 8;
        doc.setFontSize(11);
        doc.setFont('helvetica', 'normal');
      }

      doc.text(line, marginLeft, y);
      y += 6; // Line height
    }

    console.log(`PDF created with ${pageCount} pages, ${lines.length} lines`);

    // Save the PDF
    doc.save(filename);

    console.log('PDF download completed successfully');

    // Return success indicator
    return { success: true, filename, pages: pageCount, lines: lines.length };

  } catch (error) {
    console.error('PDF generation failed:', error);

    // Fallback: try to create a minimal PDF
    try {
      console.log('Attempting fallback PDF creation...');
      const { jsPDF } = await import('jspdf');
      const doc = new jsPDF();

      doc.setFontSize(20);
      doc.text('BIM Execution Plan', 20, 30);
      doc.setFontSize(12);
      doc.text('Content generation failed, but PDF was created successfully.', 20, 50);
      doc.text(`Error: ${error.message}`, 20, 70);
      doc.text(`BEP Type: ${bepType}`, 20, 90);
      doc.text(`Timestamp: ${new Date().toISOString()}`, 20, 110);

      doc.save(`BEP_Error_${bepType}_${Date.now()}.pdf`);
      console.log('Fallback PDF created');

    } catch (fallbackError) {
      console.error('Even fallback PDF failed:', fallbackError);
      alert('PDF generation failed completely: ' + error.message);
    }

    throw new Error(`PDF generation failed: ${error.message}`);
  }
};
