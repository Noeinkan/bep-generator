import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useBepForm } from '../../../contexts/BepFormContext';
import { useTidpData } from '../../../hooks/useTidpData';
import { useMidpData } from '../../../hooks/useMidpData';
import PreviewExportPage from '../PreviewExportPage';
import { generateBEPContent } from '../../../services/bepFormatter';
import { generatePDF } from '../../../services/pdfGenerator';
import { generateDocx } from '../../../services/docxGenerator';
import { captureCustomComponentScreenshots } from '../../../services/componentScreenshotCapture';
import HiddenComponentsRenderer from '../../export/HiddenComponentsRenderer';
import '../../../utils/debugScreenshots'; // Load debug tools

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
  const [error, setError] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');

  const formData = getFormData();

  const downloadBlob = useCallback((blob, filename) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, []);

  const handlePreviewBEP = useCallback(() => {
    const content = generateBEPContent(formData, bepType, { tidpData: tidps, midpData: midps });
    const newWindow = window.open('', '_blank');
    if (newWindow) {
      newWindow.document.write(content);
      newWindow.document.close();
      newWindow.focus();
    } else {
      setError('Preview popup was blocked. Please allow popups and try again.');
    }
  }, [formData, bepType, tidps, midps]);

  const handleExport = useCallback(async () => {
    try {
      setError(null);
      setIsGenerating(true);
      setStatusMessage(`Generating ${exportFormat.toUpperCase()} document, please wait...`);

      if (exportFormat === 'pdf') {
        // Wait for components to render
        setStatusMessage('Waiting for components to render...');
        console.log('🎬 Starting PDF export...');
        console.log('FormData available:', Object.keys(formData));
        console.log('FormData:', formData);

        // Give components time to fully render (especially SVG/Canvas)
        await new Promise(resolve => setTimeout(resolve, 2500));

        // Capture screenshots from hidden components
        setStatusMessage('Capturing component diagrams...');
        console.log('⏳ About to call captureCustomComponentScreenshots...');

        let componentScreenshots = {};
        try {
          componentScreenshots = await captureCustomComponentScreenshots(formData);
          console.log('✅ Screenshots captured:', Object.keys(componentScreenshots));
          console.log('📊 Screenshot details:', componentScreenshots);
        } catch (error) {
          console.error('❌ Error capturing screenshots:', error);
        }

        // Generate PDF with screenshots
        setStatusMessage('Generating PDF document...');
        const result = await generatePDF(formData, bepType, {
          tidpData: tidps,
          midpData: midps,
          componentScreenshots
        });

        if (result.success) {
          console.log(`✓ PDF generated (${result.pages} pages, ${(result.size / 1024).toFixed(1)} KB)`);
          setStatusMessage(`Document exported successfully as ${exportFormat.toUpperCase()}.`);
        }
      } else if (exportFormat === 'word') {
        const docxBlob = await generateDocx(formData, bepType, { tidpData: tidps, midpData: midps });
        downloadBlob(docxBlob, `BEP_${bepType}_${new Date().toISOString().split('T')[0]}.docx`);
        setStatusMessage(`Document exported successfully as ${exportFormat.toUpperCase()}.`);
      } else if (exportFormat === 'html') {
        const content = generateBEPContent(formData, bepType, { tidpData: tidps, midpData: midps });
        const blob = new Blob([content], { type: 'text/html;charset=utf-8' });
        downloadBlob(blob, `BEP_${bepType}_${new Date().toISOString().split('T')[0]}.html`);
        setStatusMessage(`Document exported successfully as ${exportFormat.toUpperCase()}.`);
      }
    } catch (error) {
      setStatusMessage(`Export failed: ${error.message}`);
    } finally {
      setIsGenerating(false);
      if (statusMessage.includes('successfully') || statusMessage.includes('failed')) {
        setTimeout(() => setStatusMessage(''), 5000);
      }
      setStatusMessage('');
    }
  }, [exportFormat, formData, bepType, tidps, midps, downloadBlob, statusMessage]);

  const handleBack = useCallback(() => {
    navigate(-1); // Go back to previous step
  }, [navigate]);

  if (!bepType) {
    navigate('/bep-generator');
    return null;
  }

  return (
    <>
      <div
        aria-live="polite"
        aria-atomic="true"
        role="status"
        style={{ position: 'absolute', left: '-9999px', width: '1px', height: '1px', overflow: 'hidden' }}
      >
        {statusMessage}
      </div>

      {/* Hidden components for screenshot capture */}
      <HiddenComponentsRenderer formData={formData} bepType={bepType} />

      <main aria-busy={isGenerating}>
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
          error={error}
        />
      </main>
    </>
  );
};

export default BepPreviewView;
