import '@testing-library/jest-dom';
import React, { useState } from 'react';
import { render, screen, fireEvent, within } from '@testing-library/react';

import TIDPForm from '../components/tidp/TIDPForm';
import TIDPList from '../components/tidp/TIDPList';

// Simple harness: renders the form and list, keeps local tidps state and switches to 'tidps' tab after create
const TestHarness = ({ initialShowForm = true }) => {
  const [tidps, setTidps] = useState([]);
  const [showForm, setShowForm] = useState(initialShowForm);
  const [activeTab, setActiveTab] = useState(showForm ? 'editor' : 'tidps');
  const [tidpForm, setTidpForm] = useState({
    taskTeam: '',
    discipline: '',
    teamLeader: '',
    description: '',
    containers: [{ id: 'c1', 'Information Container Name/Title': 'Initial', 'Due Date': '2025-10-01' }]
  });

  const createTidp = async (form) => {
    const newTidp = {
      id: `tidp-${Date.now()}`,
      taskTeam: form.taskTeam,
      discipline: form.discipline,
      teamLeader: form.teamLeader,
      description: form.description,
      status: 'Active',
      containers: form.containers || []
    };
    setTidps((s) => [newTidp, ...s]);
    setShowForm(false);
    setActiveTab('tidps');
    return newTidp;
  };

  return (
    <div>
      <nav>
        <button onClick={() => setActiveTab('editor')}>Editor</button>
        <button onClick={() => setActiveTab('tidps')}>TIDPs</button>
      </nav>

      {showForm && activeTab === 'editor' && (
        <TIDPForm tidpForm={tidpForm} onTidpFormChange={setTidpForm} onSubmit={() => createTidp(tidpForm)} onCancel={() => setShowForm(false)} />
      )}

      {activeTab === 'tidps' && (
        <TIDPList
          tidps={tidps}
          templates={[]}
          selectedTemplate={null}
          onTemplateChange={() => {}}
          exportLoading={{}}
          onExportPdf={() => {}}
          onExportExcel={() => {}}
          onViewDetails={() => {}}
          onShowTidpForm={() => setShowForm(true)}
          onShowImportDialog={() => {}}
          onImportCsv={() => {}}
          onExportAllPdfs={() => {}}
          onExportConsolidated={() => {}}
          bulkExportRunning={false}
          bulkProgress={{}}
          midps={[]}
          onToast={() => {}}
        />
      )}
    </div>
  );
};

test('create flow switches to TIDPs tab and shows Attivo badge', async () => {
  const { container } = render(<TestHarness initialShowForm={true} />);

  // Scope into the form to find inputs (labels are not linked with htmlFor in the component)
  const form = container.querySelector('form');
  const f = within(form);

  // Fill required fields. The form and container rows share some placeholders, so pick the first matching inputs.
  const taskTeamInputs = f.getAllByPlaceholderText(/Architecture Team/i);
  fireEvent.change(taskTeamInputs[0], { target: { value: 'Test Team' } });

  // Find the discipline select by examining options for the value 'architecture'
  const selects = f.getAllByRole('combobox');
  const disciplineSelect = selects.find((s) => Array.from(s.options).some((o) => o.value === 'architecture' || /architecture/i.test(o.textContent)));
  if (disciplineSelect) fireEvent.change(disciplineSelect, { target: { value: 'architecture' } });

  const teamLeaderInputs = f.getAllByPlaceholderText(/John Smith/i);
  fireEvent.change(teamLeaderInputs[0], { target: { value: 'Jane Doe' } });

  // Submit create
  fireEvent.click(screen.getByText(/Create TIDP/i));

  // The harness switches to the TIDPs tab and the list should show the created tidp and the 'Attivo' badge
  expect(await screen.findByText(/Attivo/i)).toBeInTheDocument();
  expect(await screen.findByText(/Test Team/i)).toBeInTheDocument();
});

