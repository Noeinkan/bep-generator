import { useState, useMemo } from 'react';

export const useTIDPFilters = (tidps) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterDiscipline, setFilterDiscipline] = useState('all');

  const disciplines = useMemo(() => {
    const disciplineSet = new Set(tidps.map(tidp => tidp.discipline).filter(Boolean));
    return Array.from(disciplineSet);
  }, [tidps]);

  const filteredTidps = useMemo(() => {
    return tidps.filter(tidp => {
      const matchesSearch = !searchTerm ||
        tidp.teamName?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        tidp.description?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        tidp.discipline?.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesDiscipline = filterDiscipline === 'all' || tidp.discipline === filterDiscipline;

      return matchesSearch && matchesDiscipline;
    });
  }, [tidps, searchTerm, filterDiscipline]);

  return {
    searchTerm,
    setSearchTerm,
    filterDiscipline,
    setFilterDiscipline,
    disciplines,
    filteredTidps
  };
};
