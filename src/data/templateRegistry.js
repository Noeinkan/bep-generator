import EMPTY_BEP_DATA from './emptyBepData';
import COMMERCIAL_OFFICE_TEMPLATE from './templates/commercialOfficeTemplate';

/**
 * Template Registry
 * Defines all available BEP templates with metadata
 */
export const TEMPLATE_REGISTRY = [
  {
    id: 'commercial-office',
    name: 'Commercial Office Complex',
    category: 'Commercial',
    description: 'Modern office building with sustainable design, flexible workspaces, and smart building technologies',
    bepTypes: ['pre', 'post'], // Which BEP types this template supports
    thumbnail: null, // Path to thumbnail image (future)
    tags: ['commercial', 'office', 'sustainable', 'smart building'],
    data: COMMERCIAL_OFFICE_TEMPLATE
  },
  // Future templates can be added here:
  // {
  //   id: 'residential-complex',
  //   name: 'Residential Complex',
  //   category: 'Residential',
  //   description: 'Multi-unit residential building with shared amenities',
  //   bepTypes: ['pre', 'post'],
  //   tags: ['residential', 'housing', 'apartments'],
  //   data: RESIDENTIAL_TEMPLATE
  // },
  // {
  //   id: 'hospital',
  //   name: 'Healthcare Facility',
  //   category: 'Healthcare',
  //   description: 'Hospital or medical center with specialized requirements',
  //   bepTypes: ['pre', 'post'],
  //   tags: ['healthcare', 'hospital', 'medical'],
  //   data: HOSPITAL_TEMPLATE
  // }
];

/**
 * Get empty BEP data structure
 * @returns {Object} Empty BEP data object
 */
export const getEmptyBepData = () => {
  return { ...EMPTY_BEP_DATA };
};

/**
 * Get template by ID
 * @param {string} templateId - Template identifier
 * @returns {Object|null} Template data merged with empty base, or null if not found
 */
export const getTemplateById = (templateId) => {
  const template = TEMPLATE_REGISTRY.find(t => t.id === templateId);
  if (!template) {
    console.warn(`Template not found: ${templateId}`);
    return null;
  }

  // Merge template data with empty base to ensure all fields exist
  return {
    ...EMPTY_BEP_DATA,
    ...template.data
  };
};

/**
 * Get all templates, optionally filtered by category or BEP type
 * @param {Object} filters - Optional filters
 * @param {string} filters.category - Filter by category
 * @param {string} filters.bepType - Filter by BEP type ('pre' or 'post')
 * @returns {Array} Array of template metadata (without full data)
 */
export const getAvailableTemplates = (filters = {}) => {
  let templates = TEMPLATE_REGISTRY;

  if (filters.category) {
    templates = templates.filter(t => t.category === filters.category);
  }

  if (filters.bepType) {
    templates = templates.filter(t => t.bepTypes.includes(filters.bepType));
  }

  // Return metadata only (without full data payload)
  return templates.map(({ id, name, category, description, bepTypes, thumbnail, tags }) => ({
    id,
    name,
    category,
    description,
    bepTypes,
    thumbnail,
    tags
  }));
};

/**
 * Get template categories
 * @returns {Array} Array of unique categories
 */
export const getTemplateCategories = () => {
  const categories = new Set(TEMPLATE_REGISTRY.map(t => t.category));
  return Array.from(categories).sort();
};
