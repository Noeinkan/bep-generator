import React from 'react';
import '@testing-library/jest-dom/extend-expect';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import TidpMidpManager from '../components/pages/TidpMidpManager';

// External mutable state used by the mocked hook so the component can observe updates
let tidpsState = [];

const createdTidp = {
  id: 'tidp-created-1',
  taskTeam: 'Test Team',
  discipline: 'architecture',
  teamLeader: 'Jane Doe',
  description: 'Created in test',
  status: 'Active',
  containers: [
    { id: 'c1', 'Information Container Name/Title': 'Deliverable 1', 'Due Date': '2025-10-01' }
  ]
};

const mockLoadTidps = jest.fn(async () => {
  // Simulate that loading tidps populates the list with the newly created tidp
  tidpsState = [createdTidp];
});

const mockCreateTidp = jest.fn(async (data) => {
  // Simulate API returning created tidp under .data
  return { data: createdTidp };
});

jest.mock('../hooks/useTidpData', () => ({
  useTidpData: () => ({
    tidps: tidpsState,
    loading: false,
    loadTidps: mockLoadTidps,
    createTidp: mockCreateTidp,
    updateTidp: jest.fn(),
    deleteTidp: jest.fn()
  })
}));

jest.mock('../hooks/useMidpData', () => ({
  useMidpData: () => ({
    midps: [],
    loading: false,
    loadMidps: jest.fn(),
    createMidp: jest.fn()
  })
}));

jest.mock('../hooks/useExport', () => ({
  useExport: () => ({
    templates: [],
    selectedTemplate: null,
    setSelectedTemplate: jest.fn(),
    exportLoading: {},
    bulkExportRunning: false,
    bulkProgress: { done: 0, total: 0 },
    exportTidpExcel: jest.fn(),
    exportTidpPdf: jest.fn(),
    exportMidpExcel: jest.fn(),
    exportMidpPdf: jest.fn(),
    exportAllTidpPdfs: jest.fn(),
    exportConsolidatedProject: jest.fn()
  })
}));

test('create flow switches to TIDPs tab and shows Attivo badge', async () => {
  // Ensure initial state is empty
  tidpsState = [];

  render(<TidpMidpManager initialShowTidpForm={true} />);

  // Fill the minimal required fields in the form
  fireEvent.change(screen.getByLabelText(/Task Team/i), { target: { value: 'Test Team' } });
  fireEvent.change(screen.getByLabelText(/Discipline/i), { target: { value: 'architecture' } });
  fireEvent.change(screen.getByLabelText(/Team Leader/i), { target: { value: 'Jane Doe' } });

  // Submit create
  fireEvent.click(screen.getByText(/Create TIDP/i));

  // Expect the hook create function to have been called
  await waitFor(() => expect(mockCreateTidp).toHaveBeenCalled());

  // The manager calls loadTidps after create; wait for that
  await waitFor(() => expect(mockLoadTidps).toHaveBeenCalled());

  // Now the TIDPs tab content should be rendered
  expect(await screen.findByText(/Task Information Delivery Plans/i)).toBeInTheDocument();

  // The 'Attivo' badge (for status === 'Active') should be visible
  expect(await screen.findByText(/Attivo/i)).toBeInTheDocument();

  // And the created TIDP team name should be present
  expect(await screen.findByText(/Test Team/i)).toBeInTheDocument();
});
