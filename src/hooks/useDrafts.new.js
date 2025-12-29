import { useState, useEffect, useCallback, useMemo } from 'react';
import { draftApiService } from '../services/draftApiService';
import { validateUser, validateFormData, validateCallbacks } from '../utils/validationUtils';
import { useDraftFilters } from './useDraftFilters';

/**
 * Hook for loading and managing drafts from the API backend
 * Replaces the old localStorage-based system
 */
export const useDrafts = (user, currentFormData, onLoadDraft, onClose) => {
  const [rawDrafts, setRawDrafts] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasMigrated, setHasMigrated] = useState(false);

  // Comprehensive validation
  const isValidComponent = useMemo(() => {
    return validateUser(user) && validateFormData(currentFormData) && validateCallbacks(onLoadDraft, onClose);
  }, [user, currentFormData, onLoadDraft, onClose]);

  // Safe user ID extraction
  const safeUserId = useMemo(() => {
    try {
      return validateUser(user) ? user.id.trim() : null;
    } catch (error) {
      console.error('Error extracting user ID:', error);
      return null;
    }
  }, [user]);

  // Migrate localStorage drafts to server (one-time operation)
  const migrateLocalStorageDrafts = useCallback(async () => {
    if (!safeUserId || hasMigrated) {
      return;
    }

    try {
      // Check if there are drafts in localStorage
      const draftsKey = `bepDrafts_${safeUserId}`;
      const savedDrafts = localStorage.getItem(draftsKey);

      if (!savedDrafts) {
        setHasMigrated(true);
        return;
      }

      const parsedDrafts = JSON.parse(savedDrafts);

      if (Object.keys(parsedDrafts).length === 0) {
        setHasMigrated(true);
        return;
      }

      console.log(`Migrating ${Object.keys(parsedDrafts).length} drafts from localStorage to server...`);

      const results = await draftApiService.migrateDrafts(safeUserId, parsedDrafts);

      console.log(`Migration complete: ${results.migrated.length} migrated, ${results.skipped.length} skipped, ${results.failed.length} failed`);

      // Clear localStorage after successful migration
      if (results.migrated.length > 0) {
        localStorage.removeItem(draftsKey);
        console.log('Cleared localStorage after migration');
      }

      setHasMigrated(true);
    } catch (error) {
      console.error('Error migrating drafts:', error);
      // Don't block - continue with normal loading
      setHasMigrated(true);
    }
  }, [safeUserId, hasMigrated]);

  // Load drafts from API
  const loadDrafts = useCallback(async () => {
    if (!safeUserId) {
      setError('Invalid user data - cannot load drafts');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const draftsArray = await draftApiService.getAllDrafts(safeUserId);

      // Convert array to object with id as key (for compatibility with existing code)
      const draftsObject = {};
      draftsArray.forEach(draft => {
        draftsObject[draft.id] = {
          id: draft.id,
          name: draft.title,
          data: draft.data,
          bepType: draft.type,
          lastModified: draft.updated_at,
          projectName: draft.data?.projectName || 'Unnamed Project',
          projectId: draft.project_id,
          createdAt: draft.created_at,
          status: draft.status,
          version: draft.version
        };
      });

      setRawDrafts(draftsObject);
    } catch (error) {
      console.error('Error loading drafts:', error);
      setError(error.message || 'Failed to load drafts from server.');
      setRawDrafts({});
    } finally {
      setIsLoading(false);
    }
  }, [safeUserId]);

  // Migrate localStorage drafts on first load, then load from API
  useEffect(() => {
    const initializeDrafts = async () => {
      if (!safeUserId) return;

      // First, try to migrate localStorage drafts
      await migrateLocalStorageDrafts();

      // Then load drafts from API
      await loadDrafts();
    };

    initializeDrafts();
  }, [safeUserId, migrateLocalStorageDrafts, loadDrafts]);

  // Use the filter hook
  const filterHook = useDraftFilters(rawDrafts);

  // Refresh drafts function
  const refreshDrafts = useCallback(() => {
    loadDrafts();
  }, [loadDrafts]);

  return {
    // Basic state
    rawDrafts,
    setRawDrafts,
    isLoading,
    error,
    setError,
    isValidComponent,
    safeUserId,

    // Filtered drafts and filter controls
    ...filterHook,
    drafts: filterHook.filteredAndSortedDrafts,

    // Functions
    refreshDrafts
  };
};
