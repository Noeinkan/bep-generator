import Papa from 'papaparse';

export const exportTidpCsvTemplate = () => {
  const csvData = [
    {
      'Information Container ID': 'IC-ARCH-001',
      'Information Container Name/Title': 'Federated Architectural Model',
      'Description': 'Complete architectural model including all building elements',
      'Task Name': 'Architectural Modeling',
      'Responsible Task Team/Party': 'Architecture Team',
      'Author': 'John Smith',
      'Dependencies/Predecessors': 'Site Survey, Structural Grid',
      'Level of Information Need (LOIN)': 'LOD 300',
      'Classification': 'Pr_20_30_60 - Building model',
      'Estimated Production Time': '3 days',
      'Delivery Milestone': 'Stage 3 - Developed Design',
      'Due Date': '2025-12-31',
      'Format/Type': 'IFC 4.0',
      'Purpose': 'Coordination and visualization',
      'Acceptance Criteria': 'Model validation passed, no clashes',
      'Review and Authorization Process': 'S4 - Issue for approval',
      'Status': 'Planned'
    },
    {
      'Information Container ID': 'IC-STRUC-001',
      'Information Container Name/Title': 'Structural Model',
      'Description': 'Complete structural model with foundations, columns, beams, and slabs',
      'Task Name': 'Structural Modeling',
      'Responsible Task Team/Party': 'Structural Engineering Team',
      'Author': 'Jane Doe',
      'Dependencies/Predecessors': 'Architectural Model',
      'Level of Information Need (LOIN)': 'LOD 350',
      'Classification': 'Pr_20_30_60 - Building model',
      'Estimated Production Time': '5 days',
      'Delivery Milestone': 'Stage 4 - Technical Design',
      'Due Date': '2026-01-15',
      'Format/Type': 'IFC 4.0',
      'Purpose': 'Structural analysis and coordination',
      'Acceptance Criteria': 'Structural analysis completed, coordination resolved',
      'Review and Authorization Process': 'S4 - Issue for approval',
      'Status': 'Planned'
    }
  ];

  const csv = Papa.unparse(csvData);
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.href = url;
  link.download = 'tidp-deliverables-template.csv';
  link.style.display = 'none';

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

export const importTidpFromCsv = (file, onSuccess, onError) => {
  if (!file) return;

  Papa.parse(file, {
    header: true,
    skipEmptyLines: true,
    complete: (results) => {
      try {
        const containers = results.data.map((row, index) => ({
          id: `IC-${Date.now()}-${index}`,
          'Information Container ID': row['Information Container ID'] || `IC-${Date.now()}-${index}`,
          'Information Container Name/Title': row['Information Container Name/Title'] || '',
          'Description': row['Description'] || '',
          'Task Name': row['Task Name'] || '',
          'Responsible Task Team/Party': row['Responsible Task Team/Party'] || '',
          'Author': row['Author'] || '',
          'Dependencies/Predecessors': row['Dependencies/Predecessors'] || '',
          'Level of Information Need (LOIN)': row['Level of Information Need (LOIN)'] || 'LOD 200',
          'Classification': row['Classification'] || '',
          'Estimated Production Time': row['Estimated Production Time'] || '1 day',
          'Delivery Milestone': row['Delivery Milestone'] || '',
          'Due Date': row['Due Date'] || '',
          'Format/Type': row['Format/Type'] || 'IFC 4.0',
          'Purpose': row['Purpose'] || '',
          'Acceptance Criteria': row['Acceptance Criteria'] || '',
          'Review and Authorization Process': row['Review and Authorization Process'] || 'S1 - Work in progress',
          'Status': row['Status'] || 'Planned'
        })).filter(container => container['Information Container Name/Title'].trim() !== '');

        if (containers.length === 0) {
          onError(new Error('No valid deliverables found in CSV'));
          return;
        }

        onSuccess(containers);
      } catch (error) {
        console.error('CSV import error:', error);
        onError(error);
      }
    },
    error: (error) => {
      console.error('CSV parsing error:', error);
      onError(error);
    }
  });
};

export const getDefaultContainer = () => ({
  id: `IC-${Date.now()}`,
  'Information Container ID': `IC-${Date.now()}`,
  'Information Container Name/Title': '',
  'Description': '',
  'Task Name': '',
  'Responsible Task Team/Party': '',
  'Author': '',
  'Dependencies/Predecessors': '',
  'Level of Information Need (LOIN)': 'LOD 200',
  'Classification': '',
  'Estimated Production Time': '1 day',
  'Delivery Milestone': '',
  'Due Date': '',
  'Format/Type': 'IFC 4.0',
  'Purpose': '',
  'Acceptance Criteria': '',
  'Review and Authorization Process': 'S1 - Work in progress',
  'Status': 'Planned'
});