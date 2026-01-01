import html2canvas from 'html2canvas';
import { toPng } from 'html-to-image';

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
 * Capture a single component as a screenshot using html-to-image
 * This library handles SVG and Canvas elements better than html2canvas
 */
const captureComponent = async (element) => {
  try {
    // Wait a bit for component to fully render
    await new Promise(resolve => setTimeout(resolve, 500));

    console.log('    🔧 Capture options:', {
      width: element.scrollWidth,
      height: element.scrollHeight,
      pixelRatio: 2
    });

    // Use html-to-image (toPng) which handles SVG/Canvas better
    const dataUrl = await toPng(element, {
      pixelRatio: 2, // High quality (2x)
      backgroundColor: '#ffffff',
      width: element.scrollWidth,
      height: element.scrollHeight,
      style: {
        transform: 'scale(1)',
        transformOrigin: 'top left'
      }
    });

    console.log('    ✓ Image captured, DataURL length:', dataUrl.length, 'chars');

    return dataUrl;
  } catch (error) {
    console.error('    ❌ Error capturing component:', error);
    throw error;
  }
};

/**
 * Main function: captures all custom visual components
 * Returns a map of fieldName -> base64 image data
 *
 * Strategy: Prioritize hidden components container for reliable capture
 */
export const captureCustomComponentScreenshots = async (formData) => {
  console.log('🎬 Starting screenshot capture...');
  console.log('FormData keys:', Object.keys(formData));

  const screenshots = {};

  // First, check if hidden components container exists
  const hiddenContainer = document.querySelector('#hidden-components-for-pdf');
  console.log('Hidden container found:', !!hiddenContainer);

  // Capture each component that has data
  for (const { name, type } of VISUAL_COMPONENTS) {
    // Skip if no data for this field
    if (!formData[name]) {
      console.log(`⊘ Skipping ${name} - no data`);
      continue;
    }

    console.log(`📸 Capturing ${name} (${type})...`);

    // Find all elements with this field name
    const allElements = Array.from(document.querySelectorAll(`[data-field-name="${name}"]`));
    console.log(`  🔍 Found ${allElements.length} elements with data-field-name="${name}"`);

    // Sort elements: visible first, hidden last
    allElements.sort((a, b) => {
      const aInHidden = a.closest('#hidden-components-for-pdf') !== null;
      const bInHidden = b.closest('#hidden-components-for-pdf') !== null;
      return (aInHidden ? 1 : 0) - (bInHidden ? 1 : 0);
    });

    let captured = false;

    // Try each element, starting with visible ones
    for (let i = 0; i < allElements.length; i++) {
      const element = allElements[i];
      const isInHidden = element.closest('#hidden-components-for-pdf') !== null;

      console.log(`  🔍 Trying element ${i + 1}/${allElements.length} [${isInHidden ? 'HIDDEN' : 'VISIBLE'}]...`);

      // Check if element has dimensions
      if (element.offsetWidth === 0 || element.offsetHeight === 0) {
        console.log(`  ⊘ Element has no dimensions (${element.offsetWidth}x${element.offsetHeight})`);
        continue;
      }

      console.log(`  ✓ Element has dimensions (${element.offsetWidth}x${element.offsetHeight}px)`);

      try {
        // Wait a bit for rendering
        await new Promise(resolve => setTimeout(resolve, 200));

        const imageData = await captureComponent(element);
        screenshots[name] = imageData;

        console.log(`  ✅ Captured successfully from ${isInHidden ? 'HIDDEN' : 'VISIBLE'} element (${(imageData.length / 1024).toFixed(1)} KB)`);
        captured = true;
        break;
      } catch (error) {
        console.log(`  ⚠️ Error capturing element:`, error.message);
        continue;
      }
    }

    if (!captured) {
      console.warn(`  ❌ Failed to capture ${name} with any selector`);
    }
  }

  const capturedCount = Object.keys(screenshots).length;
  const totalCount = VISUAL_COMPONENTS.filter(c => formData[c.name]).length;

  console.log(`\n✅ Screenshot capture complete: ${capturedCount}/${totalCount} components captured`);
  console.log('Captured components:', Object.keys(screenshots));

  if (capturedCount === 0 && totalCount > 0) {
    console.error('⚠️ WARNING: No screenshots were captured! Check that components are rendering.');
  }

  // Save to window for debugging
  if (typeof window !== 'undefined') {
    window.lastCapturedScreenshots = screenshots;
    console.log('💾 Screenshots saved to window.lastCapturedScreenshots for debugging');
  }

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
