/**
 * Test the complete AI suggestion chain:
 * Frontend -> Node.js Backend -> Python ML Service
 */

const axios = require('axios');

const COLORS = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
};

function log(message, color = 'reset') {
  console.log(`${COLORS[color]}${message}${COLORS.reset}`);
}

async function testMLServiceDirect() {
  log('\n[TEST 1] Testing Python ML Service directly (port 8000)...', 'cyan');

  try {
    // Test health endpoint
    const healthResponse = await axios.get('http://localhost:8000/health', { timeout: 5000 });
    log('✓ ML Service health check passed', 'green');
    log(`  Status: ${healthResponse.data.status}`, 'cyan');
    log(`  Model loaded: ${healthResponse.data.model_loaded}`, 'cyan');
    log(`  Device: ${healthResponse.data.device}`, 'cyan');

    // Test suggest endpoint
    log('\n  Testing /suggest endpoint...', 'yellow');
    const suggestResponse = await axios.post('http://localhost:8000/suggest', {
      field_type: 'executiveSummary',
      partial_text: '',
      max_length: 100
    }, { timeout: 30000 });

    log('✓ ML Service /suggest endpoint passed', 'green');
    log(`  Generated text: "${suggestResponse.data.text.substring(0, 80)}..."`, 'cyan');

    return true;
  } catch (error) {
    log('✗ ML Service test FAILED', 'red');
    log(`  Error: ${error.message}`, 'red');
    if (error.response) {
      log(`  Status: ${error.response.status}`, 'red');
      log(`  Response: ${JSON.stringify(error.response.data)}`, 'red');
    }
    return false;
  }
}

async function testBackendProxy() {
  log('\n[TEST 2] Testing Node.js Backend proxy (port 3001)...', 'cyan');

  try {
    // Test health endpoint
    const healthResponse = await axios.get('http://localhost:3001/api/ai/health', { timeout: 5000 });
    log('✓ Backend health check passed', 'green');
    log(`  Status: ${healthResponse.data.status}`, 'cyan');

    // Test suggest endpoint through backend
    log('\n  Testing /api/ai/suggest endpoint...', 'yellow');
    const suggestResponse = await axios.post('http://localhost:3001/api/ai/suggest', {
      field_type: 'executiveSummary',
      partial_text: '',
      max_length: 100
    }, { timeout: 30000 });

    log('✓ Backend /api/ai/suggest endpoint passed', 'green');
    log(`  Success: ${suggestResponse.data.success}`, 'cyan');
    log(`  Generated text: "${suggestResponse.data.text.substring(0, 80)}..."`, 'cyan');

    return true;
  } catch (error) {
    log('✗ Backend proxy test FAILED', 'red');
    log(`  Error: ${error.message}`, 'red');
    if (error.response) {
      log(`  Status: ${error.response.status}`, 'red');
      log(`  Response: ${JSON.stringify(error.response.data)}`, 'red');
    } else if (error.code === 'ECONNREFUSED') {
      log('  The backend server is not responding. Is it running?', 'red');
    }
    return false;
  }
}

async function testFrontendIntegration() {
  log('\n[TEST 3] Simulating Frontend request...', 'cyan');
  log('  This simulates what happens when you click "Generate AI suggestion"', 'yellow');

  try {
    // This is exactly what the frontend AISuggestionButton does
    const response = await axios.post('http://localhost:3001/api/ai/suggest', {
      field_type: 'projectDescription',
      partial_text: '',
      max_length: 200
    }, { timeout: 30000 });

    if (response.data.success) {
      log('✓ Frontend simulation PASSED', 'green');
      log(`  The "Generate AI suggestion" button SHOULD work!`, 'green');
      log(`  Generated text: "${response.data.text.substring(0, 100)}..."`, 'cyan');
      return true;
    } else {
      log('✗ Frontend simulation FAILED', 'red');
      log(`  Response: ${JSON.stringify(response.data)}`, 'red');
      return false;
    }
  } catch (error) {
    log('✗ Frontend simulation FAILED', 'red');
    log(`  Error: ${error.message}`, 'red');
    if (error.response) {
      log(`  Status: ${error.response.status}`, 'red');
      log(`  Response: ${JSON.stringify(error.response.data)}`, 'red');
    }
    return false;
  }
}

async function runAllTests() {
  log('\n' + '='.repeat(60), 'blue');
  log('BEP AI Suggestion - End-to-End Testing', 'blue');
  log('='.repeat(60) + '\n', 'blue');

  const results = {
    mlService: false,
    backend: false,
    frontend: false
  };

  results.mlService = await testMLServiceDirect();

  if (results.mlService) {
    results.backend = await testBackendProxy();
  } else {
    log('\n[SKIP] Skipping backend test - ML service not available', 'yellow');
  }

  if (results.backend) {
    results.frontend = await testFrontendIntegration();
  } else {
    log('\n[SKIP] Skipping frontend test - backend not available', 'yellow');
  }

  // Summary
  log('\n' + '='.repeat(60), 'blue');
  log('TEST SUMMARY', 'blue');
  log('='.repeat(60), 'blue');
  log(`ML Service (port 8000):     ${results.mlService ? '✓ PASS' : '✗ FAIL'}`, results.mlService ? 'green' : 'red');
  log(`Backend Proxy (port 3001):  ${results.backend ? '✓ PASS' : '✗ FAIL'}`, results.backend ? 'green' : 'red');
  log(`Frontend Integration:       ${results.frontend ? '✓ PASS' : '✗ FAIL'}`, results.frontend ? 'green' : 'red');
  log('='.repeat(60) + '\n', 'blue');

  if (results.mlService && results.backend && results.frontend) {
    log('🎉 ALL TESTS PASSED! The "Generate AI suggestion" button should work!', 'green');
    log('Go to http://localhost:3000 and test it in the browser!\n', 'cyan');
    process.exit(0);
  } else {
    log('⚠️  Some tests failed. Check the errors above.', 'yellow');

    if (!results.mlService) {
      log('\nML Service troubleshooting:', 'yellow');
      log('  1. Check if the service is running: netstat -ano | findstr :8000', 'cyan');
      log('  2. Try starting it manually: cd ml-service && venv\\Scripts\\python.exe api.py', 'cyan');
    }

    if (!results.backend) {
      log('\nBackend troubleshooting:', 'yellow');
      log('  1. Check if the service is running: netstat -ano | findstr :3001', 'cyan');
      log('  2. Check backend logs for errors', 'cyan');
    }

    process.exit(1);
  }
}

// Run tests
runAllTests().catch(error => {
  log(`\n✗ Unexpected error: ${error.message}`, 'red');
  console.error(error);
  process.exit(1);
});
