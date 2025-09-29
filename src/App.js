import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

// Import layout and page components
import { AuthProvider, useAuth } from './contexts/AuthContext';
import MainLayout from './components/layout/MainLayout';
import HomePage from './components/pages/HomePage';
import BEPGeneratorWrapper from './components/pages/BEPGeneratorWrapper';
import TIDPMIDPDashboard from './components/pages/TIDPMIDPDashboard';



const AppContent = () => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
            <div className="w-8 h-8 border-4 border-white border-t-transparent rounded-full animate-spin"></div>
          </div>
          <p className="text-gray-600">Loading BEP Suite...</p>
        </div>
      </div>
    );
  }

  // Use a mock user when authentication is disabled
  const mockUser = user || { id: 'demo-user', name: 'Demo User', email: 'demo@example.com' };

  return (
    <Router>
      <MainLayout>
        <Routes>
          {/* Home Page */}
          <Route path="/" element={<HomePage />} />

          {/* BEP Generator Routes */}
          <Route path="/bep-generator" element={<BEPGeneratorWrapper />} />
          <Route path="/bep-generator/*" element={<BEPGeneratorWrapper />} />

          {/* TIDP/MIDP Dashboard Routes */}
          <Route path="/tidp-midp" element={<TIDPMIDPDashboard />} />
          <Route path="/tidp-midp/dashboard" element={<TIDPMIDPDashboard />} />
          <Route path="/tidp-midp/tidps" element={<TIDPMIDPDashboard />} />
          <Route path="/tidp-midp/midps" element={<TIDPMIDPDashboard />} />
          <Route path="/tidp-midp/import" element={<TIDPMIDPDashboard />} />
          <Route path="/tidp-midp/evolution/:midpId" element={<TIDPMIDPDashboard />} />

          {/* Fallback redirect */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </MainLayout>
    </Router>
  );
};

// Main App Component

const App = () => {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
};

export default App;