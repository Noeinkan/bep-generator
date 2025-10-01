import React from 'react';
import TidpMidpManager from './TidpMidpManager';

const TidpEditorPage = () => {
  // If URL is /tidp-editor/:id, extract the id and forward to the manager so it can load the existing TIDP
  const path = window.location.pathname || '';
  const match = path.match(/^\/tidp-editor\/([^\/]+)/);
  const initialTidpId = match ? match[1] : null;

  return <TidpMidpManager initialShowTidpForm={true} initialTidpId={initialTidpId} />;
};

export default TidpEditorPage;