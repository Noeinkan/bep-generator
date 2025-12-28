import { useState, useCallback } from 'react';
import { validateDraftName, sanitizeText, sanitizeFileName } from '../utils/validationUtils';
import { draftStorageService } from '../services/draftStorageService';

export const useDraftOperations = (user, currentFormData, bepType, onLoadDraft, onClose) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const saveDraft = useCallback(async (name, data = currentFormData) => {
    const validation = validateDraftName(name);
    if (!validation.isValid) {
      setError(validation.error);
      return;
    }

    const sanitizedName = validation.sanitized;

    if (!user?.id) {
      setError('Cannot save draft - invalid user data');
      return;
    }

    if (!data || typeof data !== 'object') {
      setError('Cannot save draft - invalid form data');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const existingDrafts = draftStorageService.loadDrafts(user.id);

      const isDuplicateName = Object.values(existingDrafts).some(
        draft => sanitizeText(draft.name).toLowerCase() === sanitizedName.toLowerCase()
      );

      if (isDuplicateName) {
        setError(`A draft with the name "${sanitizedName}" already exists`);
        return false;
      }

      const sanitizedProjectName = data.projectName && typeof data.projectName === 'string'
        ? sanitizeText(data.projectName) || 'Unnamed Project'
        : 'Unnamed Project';

      const draft = {
        id: Date.now().toString(),
        name: sanitizedName,
        data: data,
        bepType: bepType || 'pre-appointment',
        lastModified: new Date().toISOString(),
        projectName: sanitizedProjectName
      };

      draftStorageService.saveDraft(user.id, draft);
      return true;
    } catch (error) {
      console.error('Error saving draft:', error);
      setError('Failed to save draft. Please try again.');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [user, currentFormData, bepType]);

  const deleteDraft = useCallback(async (draftId) => {
    if (!draftId || typeof draftId !== 'string') {
      setError('Invalid draft ID');
      return;
    }

    if (!user?.id) {
      setError('Cannot delete draft - invalid user data');
      return;
    }



    setIsLoading(true);
    setError(null);

    try {
      const success = draftStorageService.deleteDraft(user.id, draftId);
      if (!success) {
        setError('Draft not found');
      }
      return success;
    } catch (error) {
      console.error('Error deleting draft:', error);
      setError('Failed to delete draft. Please try again.');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [user]);

  const renameDraft = useCallback(async (draftId, newName) => {
    if (!draftId || typeof draftId !== 'string') {
      setError('Invalid draft ID');
      return;
    }

    const validation = validateDraftName(newName);
    if (!validation.isValid) {
      setError(validation.error);
      return;
    }

    const sanitizedName = validation.sanitized;

    if (!user?.id) {
      setError('Cannot rename draft - invalid user data');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const existingDrafts = draftStorageService.loadDrafts(user.id);

      if (!existingDrafts[draftId]) {
        setError('Draft not found');
        return false;
      }

      const isDuplicateName = Object.entries(existingDrafts).some(
        ([id, draft]) => id !== draftId && sanitizeText(draft.name).toLowerCase() === sanitizedName.toLowerCase()
      );

      if (isDuplicateName) {
        setError(`A draft with the name "${sanitizedName}" already exists`);
        return false;
      }

      const success = draftStorageService.updateDraft(user.id, draftId, {
        name: sanitizedName,
        lastModified: new Date().toISOString()
      });

      return success;
    } catch (error) {
      console.error('Error renaming draft:', error);
      setError('Failed to rename draft. Please try again.');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [user]);

  const loadDraft = useCallback((draft) => {
    if (!draft || typeof draft !== 'object' || !draft.data) {
      setError('Invalid draft data - cannot load');
      return;
    }

    if (typeof onLoadDraft !== 'function' || typeof onClose !== 'function') {
      setError('Invalid callback functions - cannot load draft');
      return;
    }

    try {
      onLoadDraft(draft.data, draft.bepType);
      onClose();
    } catch (error) {
      console.error('Error loading draft:', error);
      setError('Failed to load draft. Please try again.');
    }
  }, [onLoadDraft, onClose]);

  const exportDraft = useCallback((draft) => {
    if (!draft || typeof draft !== 'object' || !draft.data) {
      setError('Invalid draft data - cannot export');
      return;
    }

    try {
      const dataStr = JSON.stringify(draft.data, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });

      const sanitizedFileName = sanitizeFileName(draft.name) || 'unnamed_draft';
      const finalFileName = `${sanitizedFileName}_draft.json`;

      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = finalFileName;
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error exporting draft:', error);
      setError('Failed to export draft. Please try again.');
    }
  }, []);

  const importBepFromJson = useCallback((file) => {
    return new Promise((resolve, reject) => {
      if (!file) {
        setError('No file selected');
        reject(new Error('No file selected'));
        return;
      }

      if (!file.name.endsWith('.json')) {
        setError('Invalid file type. Please select a JSON file.');
        reject(new Error('Invalid file type'));
        return;
      }

      setIsLoading(true);
      setError(null);

      const reader = new FileReader();

      reader.onload = (e) => {
        try {
          const jsonData = JSON.parse(e.target.result);

          // Validate that the imported data has the expected structure
          if (!jsonData || typeof jsonData !== 'object') {
            throw new Error('Invalid JSON structure');
          }

          // Load the imported data into the form
          if (typeof onLoadDraft === 'function') {
            // Try to detect BEP type from the data, default to 'pre-appointment'
            const detectedBepType = jsonData.bepType || 'pre-appointment';
            onLoadDraft(jsonData, detectedBepType);

            if (typeof onClose === 'function') {
              onClose();
            }
          }

          setIsLoading(false);
          resolve(jsonData);
        } catch (error) {
          console.error('Error parsing JSON:', error);
          setError('Failed to import BEP. Invalid JSON format.');
          setIsLoading(false);
          reject(error);
        }
      };

      reader.onerror = () => {
        setError('Failed to read file');
        setIsLoading(false);
        reject(new Error('Failed to read file'));
      };

      reader.readAsText(file);
    });
  }, [onLoadDraft, onClose]);

  return {
    isLoading,
    error,
    setError,
    saveDraft,
    deleteDraft,
    renameDraft,
    loadDraft,
    exportDraft,
    importBepFromJson
  };
};