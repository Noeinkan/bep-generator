import { useState, useEffect } from 'react';
import ApiService from '../services/apiService';

export const useExport = () => {
  const [templates, setTemplates] = useState([]);
  const [selectedTemplate, setSelectedTemplate] = useState(null);
  const [exportLoading, setExportLoading] = useState({});
  const [bulkExportRunning, setBulkExportRunning] = useState(false);
  const [bulkProgress, setBulkProgress] = useState({ done: 0, total: 0 });

  useEffect(() => {
    loadTemplates();
  }, []);

  const loadTemplates = async () => {
    try {
      const resp = await ApiService.getExportTemplates();
      if (resp && Array.isArray(resp)) {
        setTemplates(resp);
      } else if (resp && resp.templates && Array.isArray(resp.templates)) {
        setTemplates(resp.templates);
      } else if (resp && resp.data && typeof resp.data === 'object') {
        const normalized = Object.keys(resp.data).map(key => ({
          id: key,
          name: key.toUpperCase(),
          ...resp.data[key]
        }));
        setTemplates(normalized);
      } else if (resp) {
        try {
          const coerced = Array.isArray(resp) ? resp : [resp];
          setTemplates(coerced);
        } catch (e) {
          setTemplates([]);
        }
      } else {
        setTemplates([]);
      }
    } catch (err) {
      console.warn('Failed to load export templates', err);
    }
  };

  const exportTidpExcel = async (id) => {
    setExportLoading(prev => ({ ...prev, [id]: true }));
    try {
      await ApiService.exportTIDPToExcel(id, selectedTemplate);
      return { success: true };
    } catch (err) {
      console.error(err);
      throw err;
    } finally {
      setExportLoading(prev => ({ ...prev, [id]: false }));
    }
  };

  const exportTidpPdf = async (id, { silent = false } = {}) => {
    setExportLoading(prev => ({ ...prev, [id]: true }));
    try {
      await ApiService.exportTIDPToPDF(id, selectedTemplate);
      return { success: true };
    } catch (err) {
      console.error(err);
      return { success: false, error: err };
    } finally {
      setExportLoading(prev => ({ ...prev, [id]: false }));
    }
  };

  const exportMidpExcel = async (id) => {
    setExportLoading(prev => ({ ...prev, [id]: true }));
    try {
      await ApiService.exportMIDPToExcel(id, selectedTemplate);
      return { success: true };
    } catch (err) {
      console.error(err);
      throw err;
    } finally {
      setExportLoading(prev => ({ ...prev, [id]: false }));
    }
  };

  const exportMidpPdf = async (id) => {
    setExportLoading(prev => ({ ...prev, [id]: true }));
    try {
      await ApiService.exportMIDPToPDF(id, selectedTemplate);
      return { success: true };
    } catch (err) {
      console.error(err);
      throw err;
    } finally {
      setExportLoading(prev => ({ ...prev, [id]: false }));
    }
  };

  const exportAllTidpPdfs = async (tidps, concurrency = 3, onProgress) => {
    if (bulkExportRunning) return;
    if (tidps.length === 0) {
      throw new Error('No TIDPs to export');
    }

    setBulkExportRunning(true);
    setBulkProgress({ done: 0, total: tidps.length });

    try {
      let i = 0;
      let done = 0;
      const errors = [];

      const runners = Array.from({ length: Math.min(concurrency, tidps.length) }).map(async () => {
        while (i < tidps.length) {
          const idx = i++;
          try {
            await exportTidpPdf(tidps[idx].id, { silent: true });
          } catch (e) {
            console.error('Worker error', e);
            errors.push({ item: tidps[idx], error: e });
          }
          done++;
          setBulkProgress({ done, total: tidps.length });
          if (onProgress) onProgress(done, tidps.length);
        }
      });

      await Promise.all(runners);
      return { errors };
    } finally {
      setBulkExportRunning(false);
      setBulkProgress({ done: 0, total: 0 });
    }
  };

  const exportConsolidatedProject = async (projectId, midpId) => {
    await ApiService.exportConsolidatedProject(projectId, midpId);
  };

  return {
    templates,
    selectedTemplate,
    setSelectedTemplate,
    exportLoading,
    bulkExportRunning,
    bulkProgress,
    exportTidpExcel,
    exportTidpPdf,
    exportMidpExcel,
    exportMidpPdf,
    exportAllTidpPdfs,
    exportConsolidatedProject
  };
};