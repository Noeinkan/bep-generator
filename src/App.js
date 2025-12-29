import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

// Import layout and page components
import { AuthProvider } from './contexts/AuthContext';
import MainLayout from './components/layout/MainLayout';
import HomePage from './components/pages/HomePage';
import BEPGeneratorWrapper from './components/pages/BEPGeneratorWrapper';
import TIDPMIDPDashboard from './components/pages/tidp-midp/TIDPMIDPDashboard';
import TidpEditorPage from './components/pages/TidpEditorPage';

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Root redirect to /home */}
          <Route path="/" element={<Navigate to="/home" replace />} />

          {/* Main layout wrapper with nested routes */}
          <Route element={<MainLayout />}>
            <Route path="/home" element={<HomePage />} />

            {/* BEP Generator with sub-routes */}
            <Route path="/bep-generator" element={<BEPGeneratorWrapper />}>
              <Route index element={null} /> {/* Start menu is default */}
              <Route path="select-type" element={null} />
              <Route path="form" element={null} />
            </Route>

            {/* TIDP/MIDP Dashboard with sub-routes */}
            <Route path="/tidp-midp">
              <Route index element={<TIDPMIDPDashboard />} />
              <Route path="tidps" element={<TIDPMIDPDashboard />} />
              <Route path="midps" element={<TIDPMIDPDashboard />} />
              <Route path="import" element={<TIDPMIDPDashboard />} />
            </Route>

            {/* TIDP Editor routes */}
            <Route path="/tidp-editor" element={<TidpEditorPage />} />
            <Route path="/tidp-editor/:id" element={<TidpEditorPage />} />
          </Route>

          {/* Catch-all redirect for 404 */}
          <Route path="*" element={<Navigate to="/home" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;