// Help Content Registry - Smart system using bepConfig as source of truth
// This ensures help content is always correctly associated with BEP fields

import CONFIG from '../../config/bepConfig.js';

// Import modular help content
import { projectInfoHelp } from './projectInfo.js';

// Import legacy content as fallback
import LEGACY_HELP_CONTENT from '../helpContentData.js';

/**
 * Extract all unique field names from bepConfig
 * This is the authoritative list of fields that should have help content
 */
const extractAllFieldsFromConfig = () => {
  const allFields = new Map();
  
  // Extract from pre-appointment BEP
  if (CONFIG.formFields['pre-appointment']) {
    Object.values(CONFIG.formFields['pre-appointment']).forEach(step => {
      if (step.fields) {
        step.fields.forEach(field => {
          if (!allFields.has(field.name)) {
            allFields.set(field.name, {
              name: field.name,
              label: field.label,
              type: field.type,
              required: field.required || false,
              step: step.title,
              bepType: 'pre-appointment'
            });
          }
        });
      }
    });
  }
  
  // Extract from post-appointment BEP
  if (CONFIG.formFields['post-appointment']) {
    Object.values(CONFIG.formFields['post-appointment']).forEach(step => {
      if (step.fields) {
        step.fields.forEach(field => {
          if (!allFields.has(field.name)) {
            allFields.set(field.name, {
              name: field.name,
              label: field.label,
              type: field.type,
              required: field.required || false,
              step: step.title,
              bepType: 'post-appointment'
            });
          } else {
            // Field exists in both, mark as shared
            const existing = allFields.get(field.name);
            existing.bepType = 'shared';
            existing.appearsIn = ['pre-appointment', 'post-appointment'];
          }
        });
      }
    });
  }
  
  // Extract from shared fields
  if (CONFIG.sharedFormFields) {
    Object.values(CONFIG.sharedFormFields).forEach(step => {
      if (step.fields) {
        step.fields.forEach(field => {
          if (!allFields.has(field.name)) {
            allFields.set(field.name, {
              name: field.name,
              label: field.label,
              type: field.type,
              required: field.required || false,
              step: step.title,
              bepType: 'shared'
            });
          }
        });
      }
    });
  }
  
  return allFields;
};

/**
 * Build the unified help content map
 * Priority: Modular content > Legacy content > null
 */
const buildHelpContentMap = () => {
  const allFields = extractAllFieldsFromConfig();
  const helpContentMap = new Map();

  // Combine all available help content sources
  const allHelpContent = {
    ...LEGACY_HELP_CONTENT,  // Base layer
    ...projectInfoHelp,       // Modular overrides
    // Add more modules here as they're created
  };

  // For each field in bepConfig, try to find help content
  allFields.forEach((fieldInfo, fieldName) => {
    const helpContent = allHelpContent[fieldName];

    helpContentMap.set(fieldName, {
      field: fieldInfo,
      helpContent: helpContent || null,
      hasHelp: !!helpContent,
      source: helpContent ? (projectInfoHelp[fieldName] ? 'modular' : 'legacy') : 'missing'
    });
  });

  // Also add help content for fields that exist ONLY in helpContentData (virtual subsection fields)
  // These are fields like namingConventions_overview that are not in bepConfig but have help
  Object.keys(allHelpContent).forEach(fieldName => {
    if (!helpContentMap.has(fieldName)) {
      helpContentMap.set(fieldName, {
        field: {
          name: fieldName,
          label: fieldName,
          type: 'virtual',
          required: false,
          step: 'N/A',
          bepType: 'virtual'
        },
        helpContent: allHelpContent[fieldName],
        hasHelp: true,
        source: projectInfoHelp[fieldName] ? 'modular' : 'legacy'
      });
    }
  });

  return helpContentMap;
};

// Build the registry once on module load
const HELP_CONTENT_REGISTRY = buildHelpContentMap();

/**
 * Get help content for a specific field
 * Returns null if field doesn't exist in bepConfig or has no help content
 * @param {string} fieldName - The field name from bepConfig
 * @returns {Object|null} Help content object or null
 */
export const getHelpContent = (fieldName) => {
  const entry = HELP_CONTENT_REGISTRY.get(fieldName);
  return entry ? entry.helpContent : null;
};

/**
 * Check if help content exists for a field
 * @param {string} fieldName - Field name to check
 * @returns {boolean} True if help content exists
 */
export const hasHelpContent = (fieldName) => {
  const entry = HELP_CONTENT_REGISTRY.get(fieldName);
  return entry ? entry.hasHelp : false;
};

/**
 * Get all field names that exist in bepConfig
 * This is the authoritative list of what SHOULD have help content
 * @returns {string[]} Array of field names
 */
export const getAllBepFields = () => {
  return Array.from(HELP_CONTENT_REGISTRY.keys());
};

/**
 * Get all fields that have help content
 * @returns {string[]} Array of field names with help
 */
export const getFieldsWithHelp = () => {
  return Array.from(HELP_CONTENT_REGISTRY.entries())
    .filter(([_, entry]) => entry.hasHelp)
    .map(([fieldName, _]) => fieldName);
};

/**
 * Get all fields that are MISSING help content
 * These are fields defined in bepConfig but without help
 * @returns {Array} Array of objects with field info
 */
export const getFieldsWithoutHelp = () => {
  return Array.from(HELP_CONTENT_REGISTRY.entries())
    .filter(([_, entry]) => !entry.hasHelp)
    .map(([fieldName, entry]) => ({
      name: fieldName,
      label: entry.field.label,
      type: entry.field.type,
      step: entry.field.step,
      bepType: entry.field.bepType
    }));
};

/**
 * Get comprehensive statistics about help content coverage
 * @returns {Object} Detailed statistics
 */
export const getHelpContentStats = () => {
  const allFields = Array.from(HELP_CONTENT_REGISTRY.entries());
  const withHelp = allFields.filter(([_, entry]) => entry.hasHelp);
  const withoutHelp = allFields.filter(([_, entry]) => !entry.hasHelp);
  const modular = allFields.filter(([_, entry]) => entry.source === 'modular');
  const legacy = allFields.filter(([_, entry]) => entry.source === 'legacy');
  
  return {
    totalFields: allFields.length,
    withHelp: withHelp.length,
    withoutHelp: withoutHelp.length,
    modular: modular.length,
    legacy: legacy.length,
    coveragePercent: ((withHelp.length / allFields.length) * 100).toFixed(1),
    modularPercent: ((modular.length / allFields.length) * 100).toFixed(1),
    fieldsWithoutHelp: withoutHelp.map(([name, entry]) => ({
      name,
      label: entry.field.label,
      step: entry.field.step
    })),
    modularFields: modular.map(([name]) => name),
    legacyFields: legacy.map(([name]) => name)
  };
};

/**
 * Get detailed information about a specific field
 * Includes both bepConfig metadata and help content
 * @param {string} fieldName - Field name
 * @returns {Object|null} Complete field information
 */
export const getFieldInfo = (fieldName) => {
  const entry = HELP_CONTENT_REGISTRY.get(fieldName);
  if (!entry) return null;
  
  return {
    name: fieldName,
    ...entry.field,
    hasHelp: entry.hasHelp,
    helpSource: entry.source,
    helpContent: entry.helpContent
  };
};

/**
 * Get all fields for a specific BEP step
 * @param {number} stepNumber - Step number (1-14)
 * @param {string} bepType - 'pre-appointment' or 'post-appointment' or 'shared'
 * @returns {Array} Array of field information objects
 */
export const getFieldsByStep = (stepNumber, bepType = 'shared') => {
  const stepIndex = stepNumber - 1;
  let stepFields = [];
  
  if (bepType === 'shared' && CONFIG.sharedFormFields[stepIndex]) {
    stepFields = CONFIG.sharedFormFields[stepIndex].fields || [];
  } else if (CONFIG.formFields[bepType] && CONFIG.formFields[bepType][stepIndex]) {
    stepFields = CONFIG.formFields[bepType][stepIndex].fields || [];
  }
  
  return stepFields.map(field => ({
    ...field,
    hasHelp: hasHelpContent(field.name),
    helpContent: getHelpContent(field.name)
  }));
};

/**
 * Preload help content for multiple fields
 * @param {string[]} fieldNames - Array of field names
 * @returns {Object} Map of field names to help content
 */
export const preloadHelpContent = (fieldNames) => {
  const preloaded = {};
  fieldNames.forEach(fieldName => {
    const content = getHelpContent(fieldName);
    if (content) {
      preloaded[fieldName] = content;
    }
  });
  return preloaded;
};

/**
 * Validate that all required fields have help content
 * @returns {Object} Validation results
 */
export const validateHelpContentCoverage = () => {
  const allFields = Array.from(HELP_CONTENT_REGISTRY.entries());
  const requiredFields = allFields.filter(([_, entry]) => entry.field.required);
  const requiredWithoutHelp = requiredFields.filter(([_, entry]) => !entry.hasHelp);
  
  return {
    valid: requiredWithoutHelp.length === 0,
    totalRequired: requiredFields.length,
    requiredWithHelp: requiredFields.length - requiredWithoutHelp.length,
    requiredWithoutHelp: requiredWithoutHelp.map(([name, entry]) => ({
      name,
      label: entry.field.label,
      step: entry.field.step
    }))
  };
};

// === BACKWARD COMPATIBILITY ===
// Export all available fields for components that use the old pattern
export const getAvailableFields = getAllBepFields;

// Default export of all help content as object (backward compatible)
const HELP_CONTENT = {};
HELP_CONTENT_REGISTRY.forEach((entry, fieldName) => {
  if (entry.helpContent) {
    HELP_CONTENT[fieldName] = entry.helpContent;
  }
});

export default HELP_CONTENT;

// === DEVELOPMENT UTILITIES ===
// Expose registry for debugging (only in development)
if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
  window.__BEP_HELP_REGISTRY__ = {
    registry: HELP_CONTENT_REGISTRY,
    stats: getHelpContentStats(),
    missing: getFieldsWithoutHelp(),
    validate: validateHelpContentCoverage()
  };
  
  console.log('📊 BEP Help Content Registry loaded');
  console.log('Run window.__BEP_HELP_REGISTRY__ in console to inspect');
}
