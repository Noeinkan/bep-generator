import React from 'react';
import TidpMidpManager from './TidpMidpManager';

const TidpEditorPage = () => {
  // If URL is /tidp-editor/:id, extract the id and forward to the manager so it can load the existing TIDP
  const path = window.location.pathname || '';
  const match = path.match(/^\/tidp-editor\/([^/]+)/);
  const raw = match ? match[1] : null;
  // raw may be in the form id-slug; manager expects the id or id-slug but will extract id itself
  const initialTidpId = raw;

  return (
    <div data-page-uri={initialTidpId ? `/tidp-editor/${initialTidpId}` : '/tidp-editor'}>
      <TidpMidpManager initialShowTidpForm={true} initialTidpId={initialTidpId} />
    </div>
  );
};

export default TidpEditorPage;