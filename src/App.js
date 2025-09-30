import React from 'react';
import { BrowserRouter } from 'react-router-dom';

// Import layout and page components
import { AuthProvider } from './contexts/AuthContext';
import { PageProvider, usePage } from './contexts/PageContext';
import MainLayout from './components/layout/MainLayout';
import HomePage from './components/pages/HomePage';
import BEPGeneratorWrapper from './components/pages/BEPGeneratorWrapper';
import TIDPMIDPDashboard from './components/pages/TIDPMIDPDashboard';
import TidpEditorPage from './components/pages/TidpEditorPage';

const AppContent = () => {
  const { currentPage } = usePage();

  // Use a mock user when authentication is disabled
  // const mockUser = { id: 'demo-user', name: 'Demo User', email: 'demo@example.com' };

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage />;
      case 'bep-generator':
        return <BEPGeneratorWrapper />;
      case 'tidp-midp':
        return <TIDPMIDPDashboard />;
      case 'tidp-editor':
        return <TidpEditorPage />;
      default:
        return <HomePage />;
    }
  };

  return (
    <BrowserRouter>
      <MainLayout>
        {renderCurrentPage()}
      </MainLayout>
    </BrowserRouter>
  );
};

function App() {
  return (
    <AuthProvider>
      <PageProvider>
        <AppContent />
      </PageProvider>
    </AuthProvider>
  );
}

export default App;