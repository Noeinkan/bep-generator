import React, { useState } from 'react';
import { X, Edit2, Check } from 'lucide-react';

/**
 * ClashMatrixHeatmap
 * Visual square matrix for clash detection (similar to RACI matrix)
 *
 * Features:
 * - Square table with disciplines on both axes
 * - Checkbox cells to enable/disable clash detection
 * - Click checked cell to edit details (tolerance, responsible party, notes)
 * - Color-coded cells based on tolerance levels
 * - Symmetric matrix (Architecture ↔ Structure = Structure ↔ Architecture)
 * - Diagonal cells disabled (same discipline)
 */
const ClashMatrixHeatmap = ({ disciplines, clashes, onChange, disabled = false }) => {

  const [editingClash, setEditingClash] = useState(null); // {disciplineA: 0, disciplineB: 1}
  const [editForm, setEditForm] = useState({ tolerance: '', responsibleParty: '', notes: '' });

  // Helper: Get clash for a discipline pair
  const getClash = (indexA, indexB) => {
    if (indexA === indexB) return null; // Diagonal - same discipline

    return clashes.find(c =>
      (c.disciplineA === indexA && c.disciplineB === indexB) ||
      (c.disciplineA === indexB && c.disciplineB === indexA)
    );
  };

  // Helper: Get color class based on tolerance
  const getColorClass = (tolerance) => {
    if (!tolerance) return 'bg-gray-100';
    if (tolerance <= 25) return 'bg-green-200 hover:bg-green-300';
    if (tolerance <= 50) return 'bg-yellow-200 hover:bg-yellow-300';
    return 'bg-orange-200 hover:bg-orange-300';
  };

  // Helper: Get color indicator emoji
  const getColorIndicator = (tolerance) => {
    if (!tolerance) return '';
    if (tolerance <= 25) return '🟢';
    if (tolerance <= 50) return '🟡';
    return '🟠';
  };

  // Toggle clash detection for a cell
  const toggleClash = (indexA, indexB) => {
    if (disabled || indexA === indexB) return;

    const existing = getClash(indexA, indexB);

    if (existing) {
      // Remove clash
      const updated = clashes.filter(c =>
        !((c.disciplineA === indexA && c.disciplineB === indexB) ||
          (c.disciplineA === indexB && c.disciplineB === indexA))
      );
      onChange(updated);
    } else {
      // Add new clash with default values
      const newClash = {
        disciplineA: indexA,
        disciplineB: indexB,
        enabled: true,
        tolerance: 50, // Default to standard coordination
        responsibleParty: 'BIM Coordinator',
        notes: ''
      };
      onChange([...clashes, newClash]);
    }
  };

  // Open edit modal for a clash
  const openEditModal = (indexA, indexB, e) => {
    e.stopPropagation();
    const clash = getClash(indexA, indexB);
    if (!clash || disabled) return;

    setEditingClash({ disciplineA: indexA, disciplineB: indexB });
    setEditForm({
      tolerance: clash.tolerance || '',
      responsibleParty: clash.responsibleParty || '',
      notes: clash.notes || ''
    });
  };

  // Save edited clash
  const saveEditedClash = () => {
    const updated = clashes.map(c => {
      const isTargetClash =
        (c.disciplineA === editingClash.disciplineA && c.disciplineB === editingClash.disciplineB) ||
        (c.disciplineA === editingClash.disciplineB && c.disciplineB === editingClash.disciplineA);

      if (isTargetClash) {
        return {
          ...c,
          tolerance: parseInt(editForm.tolerance) || 50,
          responsibleParty: editForm.responsibleParty,
          notes: editForm.notes
        };
      }
      return c;
    });

    onChange(updated);
    setEditingClash(null);
    setEditForm({ tolerance: '', responsibleParty: '', notes: '' });
  };

  // Cancel editing
  const cancelEdit = () => {
    setEditingClash(null);
    setEditForm({ tolerance: '', responsibleParty: '', notes: '' });
  };

  return (
    <div className="relative">
      {/* Matrix Table */}
      <div className="overflow-x-auto">
        <table className="border-collapse w-full text-sm">
          <thead>
            <tr>
              <th className="border border-gray-300 bg-gray-100 p-2 font-semibold text-xs sticky left-0 z-10"></th>
              {disciplines.map((disc, idx) => (
                <th
                  key={idx}
                  className="border border-gray-300 bg-gray-100 p-2 font-semibold text-xs text-center min-w-[80px]"
                  title={disc}
                >
                  {disc.split(' ')[0]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {disciplines.map((rowDisc, rowIdx) => (
              <tr key={rowIdx}>
                <td className="border border-gray-300 bg-gray-100 p-2 font-semibold text-xs sticky left-0 z-10 whitespace-nowrap">
                  {rowDisc}
                </td>
                {disciplines.map((colDisc, colIdx) => {
                  const clash = getClash(rowIdx, colIdx);
                  const isDiagonal = rowIdx === colIdx;
                  const isChecked = !!clash;

                  return (
                    <td
                      key={colIdx}
                      className={`border border-gray-300 p-2 text-center cursor-pointer ${
                        isDiagonal
                          ? 'bg-gray-200 cursor-not-allowed'
                          : isChecked
                          ? getColorClass(clash?.tolerance)
                          : 'bg-white hover:bg-gray-50'
                      }`}
                      onClick={() => !isDiagonal && (isChecked ? openEditModal(rowIdx, colIdx, { stopPropagation: () => {} }) : toggleClash(rowIdx, colIdx))}
                      title={
                        isDiagonal
                          ? '-'
                          : isChecked
                          ? `${rowDisc} ↔ ${colDisc}\nTolerance: ${clash?.tolerance}mm\nResponsible: ${clash?.responsibleParty}`
                          : `Click to enable clash detection between ${rowDisc} and ${colDisc}`
                      }
                    >
                      {isDiagonal ? (
                        <span className="text-gray-400">-</span>
                      ) : isChecked ? (
                        <div className="flex items-center justify-center gap-1">
                          <span>{getColorIndicator(clash?.tolerance)}</span>
                          <Check className="w-4 h-4 text-gray-700" />
                        </div>
                      ) : (
                        <input
                          type="checkbox"
                          checked={false}
                          onChange={() => toggleClash(rowIdx, colIdx)}
                          className="w-4 h-4 cursor-pointer"
                          disabled={disabled}
                        />
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div className="mt-4 flex items-center gap-4 text-xs text-gray-600">
        <span className="font-semibold">Legend:</span>
        <span className="flex items-center gap-1">
          🟢 <span>10-25mm (High precision)</span>
        </span>
        <span className="flex items-center gap-1">
          🟡 <span>50mm (Standard coordination)</span>
        </span>
        <span className="flex items-center gap-1">
          🟠 <span>75-100mm (Low priority)</span>
        </span>
      </div>

      {/* Edit Modal */}
      {editingClash && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-lg max-w-lg w-full p-6">
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-lg font-semibold text-gray-900">
                Edit Clash Detection
              </h3>
              <button
                onClick={cancelEdit}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="mb-4">
              <p className="text-sm text-gray-600">
                <strong>Disciplines:</strong> {disciplines[editingClash.disciplineA]} ↔ {disciplines[editingClash.disciplineB]}
              </p>
            </div>

            <div className="space-y-4">
              {/* Tolerance */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Clash Tolerance (mm) <span className="text-red-500">*</span>
                </label>
                <input
                  type="number"
                  value={editForm.tolerance}
                  onChange={(e) => setEditForm({ ...editForm, tolerance: e.target.value })}
                  placeholder="e.g., 50"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  min="0"
                  step="5"
                />
                <p className="text-xs text-gray-500 mt-1">
                  🟢 10-25mm | 🟡 50mm | 🟠 75-100mm
                </p>
              </div>

              {/* Responsible Party */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Responsible Party <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  value={editForm.responsibleParty}
                  onChange={(e) => setEditForm({ ...editForm, responsibleParty: e.target.value })}
                  placeholder="e.g., MEP Coordinator"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              {/* Notes */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Notes
                </label>
                <textarea
                  value={editForm.notes}
                  onChange={(e) => setEditForm({ ...editForm, notes: e.target.value })}
                  placeholder="Additional coordination notes..."
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>

            {/* Actions */}
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={cancelEdit}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={saveEditedClash}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700"
                disabled={!editForm.tolerance || !editForm.responsibleParty}
              >
                Save Changes
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ClashMatrixHeatmap;
