import { useState, useCallback } from 'react';

export const useUndoRedo = (initialState, maxHistorySize = 50) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [history, setHistory] = useState([initialState]);

  const canUndo = currentIndex > 0;
  const canRedo = currentIndex < history.length - 1;

  const pushToHistory = useCallback((newState) => {
    setHistory(prev => {
      // Remove any future history if we're not at the end
      const newHistory = prev.slice(0, currentIndex + 1);

      // Add new state
      newHistory.push(newState);

      // Limit history size
      if (newHistory.length > maxHistorySize) {
        newHistory.shift();
        setCurrentIndex(prev => prev); // Keep same relative position
        return newHistory;
      }

      setCurrentIndex(newHistory.length - 1);
      return newHistory;
    });
  }, [currentIndex, maxHistorySize]);

  const undo = useCallback(() => {
    if (canUndo) {
      setCurrentIndex(prev => prev - 1);
      return history[currentIndex - 1];
    }
    return null;
  }, [canUndo, currentIndex, history]);

  const redo = useCallback(() => {
    if (canRedo) {
      setCurrentIndex(prev => prev + 1);
      return history[currentIndex + 1];
    }
    return null;
  }, [canRedo, currentIndex, history]);

  const getCurrentState = useCallback(() => {
    return history[currentIndex];
  }, [history, currentIndex]);

  const reset = useCallback((newInitialState) => {
    setHistory([newInitialState]);
    setCurrentIndex(0);
  }, []);

  return {
    canUndo,
    canRedo,
    undo,
    redo,
    pushToHistory,
    getCurrentState,
    reset,
    historyLength: history.length,
    currentIndex
  };
};