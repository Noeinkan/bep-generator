const http = require('http');

const payload = {
  teamName: 'API Test Team',
  discipline: 'Architecture',
  leader: 'API Tester',
  company: 'TestCo',
  responsibilities: 'Testing create',
  description: 'Created via API test',
  status: 'Active',
  containers: [
    {
      'Information Container ID': 'IC-API-1',
      'Information Container Name/Title': 'API Container 1',
      'Container Name': 'API Container 1',
      'Description': 'desc',
      'Task Name': 'API Task',
      'Responsible Task Team/Party': 'API Test Team',
      'Author': 'API Tester',
      'Level of Information Need (LOIN)': 'LOD 300',
      'Estimated Production Time': '1 day',
      'Delivery Milestone': 'Stage 2 - Concept Design',
      'Due Date': new Date().toISOString(),
      'Format/Type': 'PDF',
      'Purpose': 'Testing',
      'Acceptance Criteria': 'OK',
      'Review and Authorization Process': 'S3 - Issue for comment',
      'Status': 'Planned'
    }
  ]
};

const data = JSON.stringify(payload);

const options = {
  hostname: 'localhost',
  port: 3001,
  path: '/api/tidp',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(data)
  }
};

const req = http.request(options, (res) => {
  let body = '';
  res.on('data', (chunk) => { body += chunk; });
  res.on('end', () => {
    console.log('STATUS', res.statusCode);
    try {
      console.log('BODY', JSON.parse(body));
    } catch (e) {
      console.log('BODY (raw)', body);
    }
  });
});

req.on('error', (err) => {
  console.error('REQUEST ERROR', err);
});

req.write(data);
req.end();
