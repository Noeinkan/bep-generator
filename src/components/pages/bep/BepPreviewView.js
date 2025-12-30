import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useBepForm } from '../../../contexts/BepFormContext';
import { useTidpData } from '../../../hooks/useTidpData';
import { useMidpData } from '../../../hooks/useMidpData';
import PreviewExportPage from '../PreviewExportPage';
import { generateBEPContent } from '../../../services/bepFormatter';
import { generatePDF } from '../../../services/pdfGenerator';
import { generateDocx } from '../../../services/docxGenerator';

/**
 * View component for BEP preview and export
 */
const BepPreviewView = () => {
  const navigate = useNavigate();
  const { bepType, getFormData } = useBepForm();
  const { tidps } = useTidpData();
  const { midps } = useMidpData();

  const [isGenerating, setIsGenerating] = useState(false);
  const [exportFormat, setExportFormat] = useState('pdf');

  const formData = getFormData();

  const handlePreviewBEP = useCallback(() => {
    const content = generateBEPContent(formData, bepType, { tidpData: tidps, midpData: midps });
    const newWindow = window.open('', '_blank');
    if (newWindow) {
      newWindow.document.write(content);
      newWindow.document.close();
    }
  }, [formData, bepType, tidps, midps]);

  const handleExport = useCallback(async () => {
    try {
      setIsGenerating(true);

      if (exportFormat === 'pdf') {
        try {
          const result = await generatePDF(formData, bepType, { tidpData: tidps, midpData: midps });
          if (result.success) {
            console.log(`PDF generated successfully: ${result.filename} (${result.size} bytes)`);
          }
        } catch (error) {
          console.error('PDF generation failed:', error);
          alert('PDF generation failed: ' + error.message);
        }
      } else if (exportFormat === 'word') {
        const docxBlob = await generateDocx(formData, bepType, { tidpData: tidps, midpData: midps });
        const url = URL.createObjectURL(docxBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `BEP_${bepType}_${new Date().toISOString().split('T')[0]}.docx`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } else if (exportFormat === 'html') {
        const content = await generateBEPContent(formData, bepType, { tidpData: tidps, midpData: midps });
        const blob = new Blob([content], { type: 'text/html;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `BEP_${bepType}_${new Date().toISOString().split('T')[0]}.html`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed: ' + error.message);
    } finally {
      setIsGenerating(false);
    }
  }, [exportFormat, formData, bepType, tidps, midps]);

  const handleBack = useCallback(() => {
    navigate(-1); // Go back to previous step
  }, [navigate]);

  if (!bepType) {
    navigate('/bep-generator');
    return null;
  }

  return (
    <PreviewExportPage
      formData={formData}
      bepType={bepType}
      onBack={handleBack}
      onExport={handleExport}
      isGenerating={isGenerating}
      exportFormat={exportFormat}
      setExportFormat={setExportFormat}
      previewBEP={handlePreviewBEP}
      downloadBEP={handleExport}
      isExporting={isGenerating}
      tidpData={tidps}
      midpData={midps}
    />
  );
};

export default BepPreviewView;
