import React, { createContext, useContext, useState, useEffect } from 'react';

const PageContext = createContext();

export const usePage = () => {
  const context = useContext(PageContext);
  if (!context) {
    throw new Error('usePage must be used within a PageProvider');
  }
  return context;
};

export const PageProvider = ({ children }) => {
  // Initialize page based on current URL path
  const getInitialPage = () => {
    const path = window.location.pathname;
    if (path.startsWith('/tidp-editor')) return 'tidp-editor';
    if (path.startsWith('/tidp-midp')) return 'tidp-midp';
    if (path.startsWith('/bep-generator')) return 'bep-generator';
    return 'home';
  };

  const [currentPage, setCurrentPage] = useState(getInitialPage);

  // Update URL when page changes
  useEffect(() => {
    // Only update if currentPage is a simple page name (not a full path)
    if (!currentPage.startsWith('/')) {
      const path = `/${currentPage}`;
      if (window.location.pathname !== path) {
        window.history.replaceState(null, '', path);
      }
    }
  }, [currentPage]);

  // Listen for browser back/forward buttons
  useEffect(() => {
    const handlePopState = () => {
      const path = window.location.pathname;
      if (path.startsWith('/tidp-editor')) setCurrentPage('tidp-editor');
      else if (path.startsWith('/tidp-midp')) setCurrentPage('tidp-midp');
      else if (path.startsWith('/bep-generator')) setCurrentPage('bep-generator');
      else setCurrentPage('home');
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  const navigateTo = (page) => {
    console.log(`Navigating from ${currentPage} to ${page}`);

    // If page is a full path (starts with /), use it directly
    if (page.startsWith('/')) {
      // Extract the base page from the path
      if (page.startsWith('/tidp-editor')) {
        setCurrentPage('tidp-editor');
        window.history.pushState(null, '', page);
      } else if (page.startsWith('/tidp-midp')) {
        setCurrentPage('tidp-midp');
        window.history.pushState(null, '', page);
      } else if (page.startsWith('/bep-generator')) {
        setCurrentPage('bep-generator');
        window.history.pushState(null, '', page);
      } else {
        setCurrentPage(page.substring(1)); // Remove leading /
      }
    } else {
      setCurrentPage(page);
    }
  };

  return (
    <PageContext.Provider value={{ currentPage, navigateTo }}>
      {children}
    </PageContext.Provider>
  );
};