import { useState, useCallback } from 'react';
import { validateDraftName, sanitizeText } from '../utils/validationUtils';
import { draftApiService } from '../services/draftApiService';

/**
 * Hook for draft operations using API backend
 * Replaces the old localStorage-based system
 */
export const useDraftOperations = (user, currentFormData, bepType, onLoadDraft, onClose) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Save a draft to the server
   */
  const saveDraft = useCallback(async (name, data = currentFormData, draftId = null) => {
    const validation = validateDraftName(name);
    if (!validation.isValid) {
      setError(validation.error);
      return false;
    }

    const sanitizedName = validation.sanitized;

    if (!user?.id) {
      setError('Cannot save draft - invalid user data');
      return false;
    }

    if (!data || typeof data !== 'object') {
      setError('Cannot save draft - invalid form data');
      return false;
    }

    if (!bepType) {
      setError('Cannot save draft - BEP type not selected');
      return false;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Check for duplicate names (only if creating new draft)
      if (!draftId) {
        const existingDrafts = await draftApiService.getAllDrafts(user.id);
        const isDuplicateName = existingDrafts.some(
          draft => sanitizeText(draft.title).toLowerCase() === sanitizedName.toLowerCase()
        );

        if (isDuplicateName) {
          setError(`A draft with the name "${sanitizedName}" already exists`);
          return false;
        }
      }

      // Extract project name from data if available
      const projectId = data.projectId || null;

      // Save or update draft
      await draftApiService.saveDraft(
        user.id,
        sanitizedName,
        bepType,
        data,
        draftId,
        projectId
      );

      return true;
    } catch (error) {
      console.error('Error saving draft:', error);
      setError(error.message || 'Failed to save draft. Please try again.');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [user, currentFormData, bepType]);

  /**
   * Delete a draft from the server
   */
  const deleteDraft = useCallback(async (draftId) => {
    if (!draftId || typeof draftId !== 'string') {
      setError('Invalid draft ID');
      return false;
    }

    if (!user?.id) {
      setError('Cannot delete draft - invalid user data');
      return false;
    }

    setIsLoading(true);
    setError(null);

    try {
      await draftApiService.deleteDraft(draftId, user.id);
      return true;
    } catch (error) {
      console.error('Error deleting draft:', error);
      setError(error.message || 'Failed to delete draft. Please try again.');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [user]);

  /**
   * Rename a draft
   */
  const renameDraft = useCallback(async (draftId, newName) => {
    if (!draftId || typeof draftId !== 'string') {
      setError('Invalid draft ID');
      return false;
    }

    const validation = validateDraftName(newName);
    if (!validation.isValid) {
      setError(validation.error);
      return false;
    }

    const sanitizedName = validation.sanitized;

    if (!user?.id) {
      setError('Cannot rename draft - invalid user data');
      return false;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Check for duplicate names
      const existingDrafts = await draftApiService.getAllDrafts(user.id);
      const isDuplicateName = existingDrafts.some(
        draft => draft.id !== draftId &&
                 sanitizeText(draft.title).toLowerCase() === sanitizedName.toLowerCase()
      );

      if (isDuplicateName) {
        setError(`A draft with the name "${sanitizedName}" already exists`);
        return false;
      }

      await draftApiService.updateDraft(draftId, user.id, { title: sanitizedName });
      return true;
    } catch (error) {
      console.error('Error renaming draft:', error);
      setError(error.message || 'Failed to rename draft. Please try again.');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [user]);

  /**
   * Load a draft into the form
   */
  const loadDraft = useCallback((draft) => {
    if (!draft || typeof draft !== 'object' || !draft.data) {
      setError('Invalid draft data - cannot load');
      return false;
    }

    if (typeof onLoadDraft !== 'function' || typeof onClose !== 'function') {
      setError('Invalid callback functions - cannot load draft');
      return false;
    }

    try {
      onLoadDraft(draft.data, draft.type);
      onClose();
      return true;
    } catch (error) {
      console.error('Error loading draft:', error);
      setError('Failed to load draft. Please try again.');
      return false;
    }
  }, [onLoadDraft, onClose]);

  /**
   * Export a draft as JSON
   */
  const exportDraft = useCallback((draft) => {
    if (!draft || typeof draft !== 'object' || !draft.data) {
      setError('Invalid draft data - cannot export');
      return;
    }

    try {
      const dataStr = JSON.stringify(draft.data, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });

      const sanitizedFileName = (draft.title || 'unnamed_draft').replace(/[^a-z0-9_-]/gi, '_');
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

  /**
   * Import BEP from JSON file
   */
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

          if (!jsonData || typeof jsonData !== 'object') {
            throw new Error('Invalid JSON structure');
          }

          if (typeof onLoadDraft === 'function') {
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

  /**
   * Get all drafts for current user
   */
  const getAllDrafts = useCallback(async () => {
    if (!user?.id) {
      setError('Cannot load drafts - invalid user data');
      return [];
    }

    setIsLoading(true);
    setError(null);

    try {
      const drafts = await draftApiService.getAllDrafts(user.id);
      return drafts;
    } catch (error) {
      console.error('Error loading drafts:', error);
      setError(error.message || 'Failed to load drafts. Please try again.');
      return [];
    } finally {
      setIsLoading(false);
    }
  }, [user]);

  /**
   * Migrate localStorage drafts to server
   */
  const migrateDraftsToServer = useCallback(async () => {
    if (!user?.id) {
      console.warn('Cannot migrate - no user logged in');
      return { migrated: 0, skipped: 0, failed: 0 };
    }

    try {
      // Get drafts from localStorage
      const draftsKey = `bepDrafts_${user.id}`;
      const savedDrafts = localStorage.getItem(draftsKey);

      if (!savedDrafts) {
        console.log('No localStorage drafts to migrate');
        return { migrated: 0, skipped: 0, failed: 0 };
      }

      const parsedDrafts = JSON.parse(savedDrafts);

      if (Object.keys(parsedDrafts).length === 0) {
        console.log('No drafts found in localStorage');
        return { migrated: 0, skipped: 0, failed: 0 };
      }

      console.log(`Migrating ${Object.keys(parsedDrafts).length} drafts to server...`);

      const results = await draftApiService.migrateDrafts(user.id, parsedDrafts);

      // Clear localStorage after successful migration
      if (results.migrated.length > 0) {
        localStorage.removeItem(draftsKey);
        console.log(`Migration complete. Cleared localStorage.`);
      }

      return {
        migrated: results.migrated.length,
        skipped: results.skipped.length,
        failed: results.failed.length
      };
    } catch (error) {
      console.error('Error migrating drafts:', error);
      return { migrated: 0, skipped: 0, failed: Object.keys(parsedDrafts || {}).length };
    }
  }, [user]);

  return {
    isLoading,
    error,
    setError,
    saveDraft,
    deleteDraft,
    renameDraft,
    loadDraft,
    exportDraft,
    importBepFromJson,
    getAllDrafts,
    migrateDraftsToServer
  };
};
