export const sanitizeText = (text) => {
  if (!text || typeof text !== 'string') return '';

  return text
    // Remove HTML tags
    .replace(/<[^>]*>/g, '')
    // Remove potentially dangerous characters
    .replace(/[<>"'&]/g, '')
  // Remove control characters excluding common whitespace (tab/newline) to avoid accidental unprintable chars
  // eslint-disable-next-line no-control-regex
  .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, '')
    // Remove zero-width characters
    .replace(/[\u200B-\u200D\uFEFF]/g, '')
    // Normalize whitespace
    .replace(/\s+/g, ' ')
    // Trim
    .trim();
};

export const sanitizeFileName = (name) => {
  if (!name || typeof name !== 'string') return '';

  return name
    // Apply basic sanitization first
    .replace(/<[^>]*>/g, '')
    .replace(/[<>"'&]/g, '')
    // Remove file system dangerous characters
    .replace(/[\\/:*?"<>|]/g, '_')
    // Remove dots at start/end to prevent hidden files
    .replace(/^\.+|\.+$/g, '')
    // Remove multiple consecutive underscores/spaces
    .replace(/[_\s]+/g, '_')
    // Ensure it doesn't start with special characters
    .replace(/^[^a-zA-Z0-9]/, '')
    .trim();
};

export const validateDraftName = (name) => {
  const sanitized = sanitizeText(name);

  // Check length
  if (sanitized.length === 0) {
    return { isValid: false, error: 'Draft name cannot be empty' };
  }

  if (sanitized.length > 100) {
    return { isValid: false, error: 'Draft name cannot exceed 100 characters' };
  }

  // Check for reserved names
  const reservedNames = ['con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4', 'com5', 'com6', 'com7', 'com8', 'com9', 'lpt1', 'lpt2', 'lpt3', 'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'];
  if (reservedNames.includes(sanitized.toLowerCase())) {
    return { isValid: false, error: 'This name is reserved by the system' };
  }

  return { isValid: true, sanitized };
};

export const validateUser = (user) => {
  return user && typeof user === 'object' && user.id && typeof user.id === 'string' && user.id.trim().length > 0;
};

export const validateFormData = (data) => {
  return data && typeof data === 'object' && !Array.isArray(data);
};

export const validateBepType = (type) => {
  return type && typeof type === 'string' && ['pre-appointment', 'post-appointment'].includes(type);
};

export const validateCallbacks = (onLoadDraft, onClose) => {
  return typeof onLoadDraft === 'function' && typeof onClose === 'function';
};

export const isDraftValid = (draft) => {
  return (
    draft &&
    typeof draft === 'object' &&
    typeof draft.id === 'string' &&
    typeof draft.name === 'string' &&
    draft.name.trim().length > 0 &&
    typeof draft.lastModified === 'string' &&
    draft.data &&
    typeof draft.data === 'object'
  );
};

export const formatDate = (dateString) => {
  try {
    if (!dateString || typeof dateString !== 'string') {
      return 'Invalid date';
    }

    const date = new Date(dateString);
    if (isNaN(date.getTime())) {
      return 'Invalid date';
    }

    return date.toLocaleString('en-US', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  } catch (error) {
    console.error('Error formatting date:', error);
    return 'Invalid date';
  }
};