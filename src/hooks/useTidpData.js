import { useState, useEffect, useRef, useCallback } from 'react';
import ApiService from '../services/apiService';

export const useTidpData = () => {
  const [tidps, setTidps] = useState([]);
  const [loading, setLoading] = useState(false);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  const loadTidps = useCallback(async () => {
    setLoading(true);
    try {
      const tidpData = await ApiService.getAllTIDPs();
      if (!mountedRef.current) return;
      setTidps(tidpData.tidps || []);
    } catch (error) {
      if (mountedRef.current) console.error('Failed to load TIDP data:', error);
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, []);

  const createTidp = async (tidpData) => {
    const payload = {
      teamName: tidpData.taskTeam,
      discipline: tidpData.discipline,
      leader: tidpData.teamLeader,
      company: 'TBD',
      responsibilities: tidpData.description || 'TBD',
      description: tidpData.description,
      containers: tidpData.containers
    };
    const created = await ApiService.createTIDP(payload);
    await loadTidps();
    return created && (created.data || created.tidp) || created;
  };

  const updateTidp = async (id, update) => {
    await ApiService.updateTIDP(id, update);
    await loadTidps();
  };

  const deleteTidp = async (id) => {
    await ApiService.deleteTIDP(id);
    await loadTidps();
  };

  return {
    tidps,
    loading,
    loadTidps,
    createTidp,
    updateTidp,
    deleteTidp
  };
};