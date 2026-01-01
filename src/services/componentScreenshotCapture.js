import html2canvas from 'html2canvas';

/**
 * Simplified screenshot capture service for custom visual components
 * Captures screenshots from components rendered by HiddenComponentsRenderer
 */

// List of custom visual component field names and their types
const VISUAL_COMPONENTS = [
  { name: 'organizationalStructure', type: 'orgchart' },
  { name: 'cdeStrategy', type: 'cdeDiagram' },
  { name: 'volumeStrategy', type: 'mindmap' },
  { name: 'fileStructureDiagram', type: 'fileStructure' },
  { name: 'namingConventions', type: 'naming-conventions' },
  { name: 'federationStrategy', type: 'federation-strategy' }
];

/**
 * Wait for an element to be present in the DOM
 */
const waitForElement = (selector, timeout = 5000) => {
  return new Promise((resolve, reject) => {
    const element = document.querySelector(selector);
    if (element) {
      return resolve(element);
    }

    const startTime = Date.now();
    const interval = setInterval(() => {
      const el = document.querySelector(selector);
      if (el) {
        clearInterval(interval);
        resolve(el);
      } else if (Date.now() - startTime > timeout) {
        clearInterval(interval);
        reject(new Error(`Element ${selector} not found after ${timeout}ms`));
      }
    }, 100);
  });
};

/**
 * Capture a single component as a screenshot
 */
const captureComponent = async (element) => {
  try {
    // Wait a bit for component to fully render
    await new Promise(resolve => setTimeout(resolve, 300));

    const canvas = await html2canvas(element, {
      scale: 2, // High quality
      backgroundColor: '#ffffff',
      logging: false,
      useCORS: true,
      allowTaint: true,
      width: element.scrollWidth,
      height: element.scrollHeight
    });

    return canvas.toDataURL('image/png');
  } catch (error) {
    console.error('Error capturing component:', error);
    throw error;
  }
};

/**
 * Main function: captures all custom visual components
 * Returns a map of fieldName -> base64 image data
 *
 * Tries to capture from both the visible preview and hidden components container
 */
export const captureCustomComponentScreenshots = async (formData) => {
  console.log('🎬 Starting screenshot capture...');
  console.log('FormData keys:', Object.keys(formData));

  const screenshots = {};

  // Capture each component that has data
  for (const { name, type } of VISUAL_COMPONENTS) {
    // Skip if no data for this field
    if (!formData[name]) {
      console.log(`⊘ Skipping ${name} - no data`);
      continue;
    }

    console.log(`📸 Capturing ${name} (${type})...`);

    // Try multiple selector strategies
    const selectors = [
      `[data-field-name="${name}"]`, // Primary selector
      `#${name}`,                     // ID selector
      `.${type}-component`            // Type-based class
    ];

    let captured = false;

    for (const selector of selectors) {
      try {
        const element = await waitForElement(selector, 1000);

        if (!element) continue;

        // Check if element is actually visible and has dimensions
        if (element.offsetWidth === 0 || element.offsetHeight === 0) {
          console.log(`  ⊘ Element found but has no dimensions: ${selector}`);
          continue;
        }

        console.log(`  ✓ Found element (${element.offsetWidth}x${element.offsetHeight}px) via ${selector}`);

        const imageData = await captureComponent(element);
        screenshots[name] = imageData;

        console.log(`  ✓ Captured successfully (${(imageData.length / 1024).toFixed(1)} KB)`);
        captured = true;
        break;
      } catch (error) {
        // Continue to next selector
        continue;
      }
    }

    if (!captured) {
      console.warn(`  ✗ Failed to capture ${name} with any selector`);
    }
  }

  console.log(`\n✅ Screenshot capture complete: ${Object.keys(screenshots).length}/${VISUAL_COMPONENTS.filter(c => formData[c.name]).length} components captured`);
  console.log('Captured components:', Object.keys(screenshots));

  return screenshots;
};

/**
 * Capture a component by direct element reference (for future use)
 */
export const captureComponentByRef = async (componentRef) => {
  if (!componentRef || !componentRef.current) {
    throw new Error('Invalid component reference');
  }
  return await captureComponent(componentRef.current);
};

/**
 * Capture a component by CSS selector (for future use)
 */
export const captureComponentBySelector = async (selector) => {
  const element = document.querySelector(selector);
  if (!element) {
    throw new Error(`Component not found: ${selector}`);
  }
  return await captureComponent(element);
};
