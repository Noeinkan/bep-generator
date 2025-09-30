import { generateBEPContent } from './bepFormatter';

export const generatePDF = async (formData, bepType, options = {}) => {
  const {
    orientation = 'portrait',
    format = 'a4',
    margin = [20, 20, 20, 20], // [top, right, bottom, left] in mm
    filename = `BEP_${bepType}_${new Date().toISOString().split('T')[0]}.pdf`
  } = options;

  try {
    console.log('Starting PDF generation with options:', {
      orientation, format, margin, filename
    });

    // Generate the HTML content using the same formatter as the HTML export
    const htmlContent = generateBEPContent(formData, bepType);
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

    // Set up document properties
    doc.setProperties({
      title: `BIM Execution Plan - ${bepType}`,
      subject: 'BIM Execution Plan Document',
      author: 'BEP Generator',
      creator: 'BEP Generator'
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

    // Add title
    doc.setFontSize(16);
    doc.setFont('helvetica', 'bold');
    const title = `BIM Execution Plan - ${bepType.toUpperCase()}`;
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
