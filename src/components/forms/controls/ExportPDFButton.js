import React, { useCallback } from 'react';
import { FileDown } from 'lucide-react';
import jsPDF from 'jspdf';

const ExportPDFButton = ({ editor, className = '' }) => {
  const exportToPDF = useCallback(() => {
    if (!editor) return;

    // Get HTML content
    const content = editor.getHTML();

    // Create a temporary div to render HTML
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = content;
    tempDiv.style.position = 'absolute';
    tempDiv.style.left = '-9999px';
    tempDiv.style.width = '210mm'; // A4 width
    tempDiv.style.padding = '20px';
    tempDiv.style.fontFamily = 'Arial, sans-serif';
    tempDiv.style.fontSize = '12pt';
    tempDiv.style.lineHeight = '1.5';
    document.body.appendChild(tempDiv);

    try {
      // Create PDF
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4',
      });

      // Get text content for simple export
      const text = editor.getText();
      const lines = pdf.splitTextToSize(text, 170); // 170mm width with margins

      let y = 20; // Start position
      const lineHeight = 7;
      const pageHeight = 280; // A4 height in mm

      lines.forEach((line, index) => {
        if (y > pageHeight - 20) {
          pdf.addPage();
          y = 20;
        }
        pdf.text(line, 20, y);
        y += lineHeight;
      });

      // Download PDF
      const fileName = `document-${new Date().toISOString().split('T')[0]}.pdf`;
      pdf.save(fileName);
    } catch (error) {
      console.error('Error exporting PDF:', error);
      alert('Error exporting to PDF. Please try again.');
    } finally {
      document.body.removeChild(tempDiv);
    }
  }, [editor]);

  return (
    <button
      onClick={exportToPDF}
      className={`px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2 ${className}`}
      title="Export to PDF"
      type="button"
    >
      <FileDown size={18} />
      <span className="text-sm font-medium">Export PDF</span>
    </button>
  );
};

export default ExportPDFButton;
