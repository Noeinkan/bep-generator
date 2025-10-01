// Simple slugify utility for building URL-friendly names
const slugify = (input = '') => {
  if (!input) return '';
  return String(input)
    .toLowerCase()
    .trim()
    // replace spaces and underscores with dashes
    .replace(/[\s_]+/g, '-')
    // remove characters that are not alphanumeric or dashes
    .replace(/[^a-z0-9-]/g, '-')
    // collapse multiple dashes
    .replace(/-+/g, '-')
    // trim leading/trailing dashes
    .replace(/^-+|-+$/g, '')
    // limit length to 60 chars
    .slice(0, 60);
};

export default slugify;
