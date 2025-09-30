import Papa from 'papaparse';
import { TIDP_CSV_TEMPLATE_DATA } from '../constants/tidpTemplates';

export const exportTidpCsvTemplate = () => {
  try {
    const csv = Papa.unparse(TIDP_CSV_TEMPLATE_DATA);
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

    return { success: true };
  } catch (error) {
    console.error('Download failed:', error);
    return { success: false, error };
  }
};

export const exportTidpToCSV = (tidp) => {
  try {
    const csvData = tidp.containers?.map(container => ({
      'Information Container ID': container['Information Container ID'] || '',
      'Container Name': container['Container Name'] || container['Information Container Name/Title'] || '',
      'Description': container['Description'] || '',
      'Task Name': container['Task Name'] || '',
      'Responsible Party': container['Responsible Task Team/Party'] || '',
      'Author': container['Author'] || '',
      'Dependencies': container['Dependencies/Predecessors'] || '',
      'LOIN': container['Level of Information Need (LOIN)'] || container['LOI Level'] || '',
      'Classification': container['Classification'] || '',
      'Estimated Time': container['Estimated Production Time'] || container['Est. Time'] || '',
      'Milestone': container['Delivery Milestone'] || container['Milestone'] || '',
      'Due Date': container['Due Date'] || '',
      'Format': container['Format/Type'] || container['Format'] || '',
      'Purpose': container['Purpose'] || '',
      'Acceptance Criteria': container['Acceptance Criteria'] || '',
      'Review Process': container['Review and Authorization Process'] || '',
      'Status': container['Status'] || ''
    })) || [];

    const csv = Papa.unparse(csvData);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.href = url;
    link.download = `TIDP-${tidp.teamName?.replace(/\s+/g, '-')}-${new Date().toISOString().split('T')[0]}.csv`;
    link.style.display = 'none';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    return { success: true };
  } catch (error) {
    console.error('Download failed:', error);
    return { success: false, error };
  }
};

export const exportMidpToCSV = (midp) => {
  try {
    const reportData = {
      'Project Name': midp.projectName || '',
      'Description': midp.description || '',
      'Total TIDPs': midp.includedTIDPs?.length || 0,
      'Total Containers': midp.aggregatedData?.totalContainers || 0,
      'Total Estimated Hours': midp.aggregatedData?.totalEstimatedHours || 0,
      'Status': midp.status || '',
      'Version': midp.version || '',
      'Created': midp.createdAt || '',
      'Last Updated': midp.updatedAt || ''
    };

    const csv = Papa.unparse([reportData]);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.href = url;
    link.download = `MIDP-${midp.projectName?.replace(/\s+/g, '-')}-${new Date().toISOString().split('T')[0]}.csv`;
    link.style.display = 'none';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    return { success: true };
  } catch (error) {
    console.error('Download failed:', error);
    return { success: false, error };
  }
};
