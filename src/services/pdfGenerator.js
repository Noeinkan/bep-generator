import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { generateBEPContent } from './bepFormatter';

export const generatePDF = async (formData, bepType) => {
  // Generate the HTML content using the same formatter as the HTML export
  const htmlContent = generateBEPContent(formData, bepType);

  // Create a temporary container to render the HTML
  const container = document.createElement('div');
  container.innerHTML = htmlContent;
  container.style.position = 'absolute';
  container.style.left = '-9999px';
  container.style.top = '-9999px';
  container.style.width = '210mm'; // A4 width
  container.style.backgroundColor = 'white';
  container.style.fontFamily = 'Arial, sans-serif';
  container.style.lineHeight = '1.4';
  container.style.padding = '20px';
  document.body.appendChild(container);

  try {
    // Use html2canvas to capture the HTML as an image
    const canvas = await html2canvas(container, {
      scale: 2, // Higher resolution
      useCORS: true,
      allowTaint: true,
      backgroundColor: '#ffffff',
      width: 794, // A4 width in pixels at 96 DPI
      height: container.scrollHeight, // Use actual content height
      scrollX: 0,
      scrollY: 0,
    });

    // Remove the temporary container
    document.body.removeChild(container);

    // Create PDF from canvas
    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4'
    });

    const imgWidth = 210; // A4 width in mm
    const pageHeight = 297; // A4 height in mm
    const imgHeight = (canvas.height * imgWidth) / canvas.width;
    let heightLeft = imgHeight;

    let position = 0;

    // Add the first page
    pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
    heightLeft -= pageHeight;

    // Add additional pages if needed
    while (heightLeft > 0) {
      position = heightLeft - imgHeight;
      pdf.addPage();
      pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
      heightLeft -= pageHeight;
    }

    return pdf;
  } catch (error) {
    // Remove the temporary container in case of error
    if (document.body.contains(container)) {
      document.body.removeChild(container);
    }
    throw error;
  }
};
