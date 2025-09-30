const { aggregateTIDPs, checkDelayImpact } = require('../../server/services/midpService');

describe('MIDP Aggregation', () => {
  test('should aggregate TIDPs correctly', () => {
    const mockTidps = [
      {
        id: 'tidp1',
        discipline: 'Architecture',
        teamName: 'Arch Team',
        containers: [
          {
            id: 'c1',
            'Container Name': 'Model A',
            Type: 'IFC',
            Format: 'IFC 4.0',
            'LOI Level': 'LOD 300',
            Author: 'John',
            'Milestone': 'Stage 3',
            'Due Date': '2025-10-01',
            'Est. Time': '5 days',
            Status: 'Planned'
          }
        ]
      }
    ];

    const result = aggregateTIDPs(mockTidps);

    expect(result.containers).toHaveLength(1);
    expect(result.milestones).toHaveLength(1);
    expect(result.disciplines).toContain('Architecture');
  });

  test('should flag delay impact when TIDP due date exceeds milestone', () => {
    const mockMilestone = {
      name: 'Stage 3',
      latestDate: '2025-09-30',
      containers: [
        { dueDate: '2025-10-01' } // After milestone
      ]
    };
    const mockTidps = []; // Not needed for this test

    const hasDelay = checkDelayImpact(mockMilestone, mockTidps);
    expect(hasDelay).toBe(true);
  });

  test('should not flag delay when all dates are within milestone', () => {
    const mockMilestone = {
      name: 'Stage 3',
      latestDate: '2025-10-15',
      containers: [
        { dueDate: '2025-10-01' } // Before milestone
      ]
    };
    const mockTidps = [];

    const hasDelay = checkDelayImpact(mockMilestone, mockTidps);
    expect(hasDelay).toBe(false);
  });

  // Add more tests for compliance, dependencies, etc.
});