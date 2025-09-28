import React, { useState, useEffect } from 'react';
import { Save, FolderOpen, Trash2, Edit3, Plus, ArrowLeft, Download, Calendar } from 'lucide-react';

const DraftManager = ({ user, currentFormData, onLoadDraft, onClose, bepType }) => {
  const [drafts, setDrafts] = useState([]);
  const [newDraftName, setNewDraftName] = useState('');
  const [editingId, setEditingId] = useState(null);
  const [editingName, setEditingName] = useState('');
  const [showSaveDialog, setShowSaveDialog] = useState(false);

  useEffect(() => {
    loadDrafts();
  }, [user.id]);

  const loadDrafts = () => {
    const draftsKey = `bepDrafts_${user.id}`;
    const savedDrafts = localStorage.getItem(draftsKey);
    if (savedDrafts) {
      try {
        const parsedDrafts = JSON.parse(savedDrafts);
        setDrafts(Object.values(parsedDrafts).sort((a, b) => new Date(b.lastModified) - new Date(a.lastModified)));
      } catch (error) {
        console.error('Error loading drafts:', error);
        setDrafts([]);
      }
    }
  };

  const saveDraft = (name, data = currentFormData) => {
    if (!name.trim()) return;

    const draftsKey = `bepDrafts_${user.id}`;
    const existingDrafts = JSON.parse(localStorage.getItem(draftsKey) || '{}');

    const draftId = Date.now().toString();
    const draft = {
      id: draftId,
      name: name.trim(),
      data: data,
      bepType: bepType,
      lastModified: new Date().toISOString(),
      projectName: data.projectName || 'Progetto senza nome'
    };

    existingDrafts[draftId] = draft;
    localStorage.setItem(draftsKey, JSON.stringify(existingDrafts));

    loadDrafts();
    setNewDraftName('');
    setShowSaveDialog(false);
  };

  const deleteDraft = (draftId) => {
    if (!window.confirm('Sei sicuro di voler eliminare questo draft?')) return;

    const draftsKey = `bepDrafts_${user.id}`;
    const existingDrafts = JSON.parse(localStorage.getItem(draftsKey) || '{}');
    delete existingDrafts[draftId];
    localStorage.setItem(draftsKey, JSON.stringify(existingDrafts));

    loadDrafts();
  };

  const renameDraft = (draftId, newName) => {
    if (!newName.trim()) return;

    const draftsKey = `bepDrafts_${user.id}`;
    const existingDrafts = JSON.parse(localStorage.getItem(draftsKey) || '{}');

    if (existingDrafts[draftId]) {
      existingDrafts[draftId].name = newName.trim();
      existingDrafts[draftId].lastModified = new Date().toISOString();
      localStorage.setItem(draftsKey, JSON.stringify(existingDrafts));
      loadDrafts();
    }

    setEditingId(null);
    setEditingName('');
  };

  const loadDraft = (draft) => {
    if (window.confirm(`Caricare il draft "${draft.name}"? I dati correnti verranno sostituiti.`)) {
      onLoadDraft(draft.data, draft.bepType);
      onClose();
    }
  };

  const exportDraft = (draft) => {
    const dataStr = JSON.stringify(draft.data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });

    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${draft.name}_draft.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString('it-IT', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <button
                onClick={onClose}
                className="flex items-center space-x-2 text-gray-600 hover:text-gray-800 transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>Torna al BEP</span>
              </button>
            </div>
            <h1 className="text-2xl font-bold text-gray-900">Gestione Draft BEP</h1>
            <div className="flex items-center space-x-3">
              <span className="text-sm text-gray-600">
                {user.name}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Save Current Work Section */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Salva Lavoro Corrente</h2>
            <button
              onClick={() => setShowSaveDialog(true)}
              className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
              <Save className="w-4 h-4" />
              <span>Salva Draft</span>
            </button>
          </div>
          <p className="text-gray-600">
            Salva il tuo lavoro corrente come draft con un nome personalizzato per poterlo recuperare in seguito.
          </p>
        </div>

        {/* Save Dialog */}
        {showSaveDialog && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 w-full max-w-md">
              <h3 className="text-lg font-semibold mb-4">Salva Draft</h3>
              <input
                type="text"
                value={newDraftName}
                onChange={(e) => setNewDraftName(e.target.value)}
                placeholder="Nome del draft..."
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                autoFocus
              />
              <div className="flex space-x-3 mt-4">
                <button
                  onClick={() => saveDraft(newDraftName)}
                  disabled={!newDraftName.trim()}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  Salva
                </button>
                <button
                  onClick={() => {
                    setShowSaveDialog(false);
                    setNewDraftName('');
                  }}
                  className="flex-1 bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  Annulla
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Drafts List */}
        <div className="bg-white rounded-lg shadow-sm">
          <div className="p-6 border-b">
            <h2 className="text-xl font-semibold text-gray-900">Draft Salvati ({drafts.length})</h2>
            <p className="text-gray-600 mt-1">Gestisci i tuoi draft salvati</p>
          </div>

          {drafts.length === 0 ? (
            <div className="p-12 text-center">
              <FolderOpen className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Nessun draft salvato</h3>
              <p className="text-gray-600">Salva il tuo primo draft per iniziare a gestire le tue bozze di BEP.</p>
            </div>
          ) : (
            <div className="divide-y divide-gray-200">
              {drafts.map((draft) => (
                <div key={draft.id} className="p-6 hover:bg-gray-50 transition-colors">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      {editingId === draft.id ? (
                        <div className="flex items-center space-x-2">
                          <input
                            type="text"
                            value={editingName}
                            onChange={(e) => setEditingName(e.target.value)}
                            className="flex-1 p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            autoFocus
                            onKeyPress={(e) => {
                              if (e.key === 'Enter') {
                                renameDraft(draft.id, editingName);
                              }
                            }}
                          />
                          <button
                            onClick={() => renameDraft(draft.id, editingName)}
                            className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded transition-colors"
                          >
                            Salva
                          </button>
                          <button
                            onClick={() => {
                              setEditingId(null);
                              setEditingName('');
                            }}
                            className="bg-gray-500 hover:bg-gray-600 text-white px-3 py-2 rounded transition-colors"
                          >
                            Annulla
                          </button>
                        </div>
                      ) : (
                        <div>
                          <h3 className="text-lg font-medium text-gray-900">{draft.name}</h3>
                          <div className="flex items-center space-x-4 mt-2 text-sm text-gray-500">
                            <div className="flex items-center space-x-1">
                              <Calendar className="w-4 h-4" />
                              <span>{formatDate(draft.lastModified)}</span>
                            </div>
                            <span>•</span>
                            <span>{draft.projectName}</span>
                            <span>•</span>
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                              {draft.bepType === 'pre-appointment' ? 'Pre-Appointment' : 'Post-Appointment'}
                            </span>
                          </div>
                        </div>
                      )}
                    </div>

                    {editingId !== draft.id && (
                      <div className="flex items-center space-x-2 ml-4">
                        <button
                          onClick={() => loadDraft(draft)}
                          className="flex items-center space-x-1 bg-green-600 hover:bg-green-700 text-white px-3 py-2 rounded transition-colors"
                          title="Carica draft"
                        >
                          <FolderOpen className="w-4 h-4" />
                          <span>Carica</span>
                        </button>

                        <button
                          onClick={() => exportDraft(draft)}
                          className="flex items-center space-x-1 bg-purple-600 hover:bg-purple-700 text-white px-3 py-2 rounded transition-colors"
                          title="Esporta draft"
                        >
                          <Download className="w-4 h-4" />
                        </button>

                        <button
                          onClick={() => {
                            setEditingId(draft.id);
                            setEditingName(draft.name);
                          }}
                          className="flex items-center space-x-1 bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded transition-colors"
                          title="Rinomina"
                        >
                          <Edit3 className="w-4 h-4" />
                        </button>

                        <button
                          onClick={() => deleteDraft(draft.id)}
                          className="flex items-center space-x-1 bg-red-600 hover:bg-red-700 text-white px-3 py-2 rounded transition-colors"
                          title="Elimina"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DraftManager;