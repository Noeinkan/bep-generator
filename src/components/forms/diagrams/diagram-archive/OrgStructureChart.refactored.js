import React, { useReducer, useEffect, useCallback, useMemo, useRef } from 'react';
import styles from '../diagram-styles/OrgStructureChart.module.css';
import {
  buildOrgChartData,
  validateNodeData,
  getColorPalette,
  convertTreeToFinalizedParties,
  deepEqual
} from '../diagram-utils/orgChartUtils';
import { orgChartReducer, initialState, ACTIONS } from '../diagram-utils/orgChartReducer';

/**
 * Memoized Lead Node Component
 */
const LeadNode = React.memo(({ 
  lead, 
  leadIndex, 
  colors, 
  editable, 
  editing, 
  editValues, 
  errors,
  onStartEdit, 
  onSaveEdit, 
  onCancelEdit, 
  onUpdateEditValues,
  onDelete, 
  onAddAppointed 
}) => {
  const isEditing = editing?.type === 'lead' && editing.path.leadIndex === leadIndex;
  const inputRef = useRef(null);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSaveEdit();
    } else if (e.key === 'Escape') {
      onCancelEdit();
    }
  };

  return (
    <div 
      className={styles.leadCard}
      style={{ 
        background: colors.lead, 
        border: `2px solid ${colors.border}` 
      }}
      role="treeitem"
      aria-label={`Lead: ${lead.name}`}
    >
      {isEditing ? (
        <form 
          className={styles.editForm}
          onSubmit={(e) => { e.preventDefault(); onSaveEdit(); }}
        >
          <div>
            <label htmlFor={`lead-name-${leadIndex}`} className={styles.srOnly}>
              Lead Name
            </label>
            <input
              ref={inputRef}
              id={`lead-name-${leadIndex}`}
              type="text"
              className={`${styles.input} ${errors.name ? styles.inputError : ''}`}
              value={editValues.name}
              onChange={(e) => onUpdateEditValues({ name: e.target.value })}
              onKeyDown={handleKeyDown}
              placeholder="Lead Name"
              aria-invalid={!!errors.name}
              aria-describedby={errors.name ? `lead-name-error-${leadIndex}` : undefined}
            />
            {errors.name && (
              <div id={`lead-name-error-${leadIndex}`} className={styles.errorMessage} role="alert">
                {errors.name}
              </div>
            )}
          </div>

          <div>
            <label htmlFor={`lead-role-${leadIndex}`} className={styles.srOnly}>
              Role
            </label>
            <input
              id={`lead-role-${leadIndex}`}
              type="text"
              className={`${styles.input} ${styles.inputSmall} ${errors.role ? styles.inputError : ''}`}
              value={editValues.role}
              onChange={(e) => onUpdateEditValues({ role: e.target.value })}
              onKeyDown={handleKeyDown}
              placeholder="Role"
              aria-invalid={!!errors.role}
            />
          </div>

          <div>
            <label htmlFor={`lead-contact-${leadIndex}`} className={styles.srOnly}>
              Information Manager
            </label>
            <input
              id={`lead-contact-${leadIndex}`}
              type="text"
              className={`${styles.input} ${styles.inputSmall}`}
              value={editValues.contact}
              onChange={(e) => onUpdateEditValues({ contact: e.target.value })}
              onKeyDown={handleKeyDown}
              placeholder="Information Manager"
            />
          </div>

          <div className={styles.buttonGroup}>
            <button
              type="submit"
              className={`${styles.button} ${styles.buttonSuccess} ${styles.buttonSmall}`}
              aria-label="Save changes"
            >
              ✓ Save
            </button>
            <button
              type="button"
              onClick={onCancelEdit}
              className={`${styles.button} ${styles.buttonSecondary} ${styles.buttonSmall}`}
              aria-label="Cancel editing"
            >
              ✕ Cancel
            </button>
          </div>
        </form>
      ) : (
        <>
          <div className={styles.leadTitle}>{lead.name}</div>
          <div className={styles.leadRole}>{lead.role}</div>
          {lead.contact && (
            <div className={styles.leadContact}>IM: {lead.contact}</div>
          )}
          {editable && (
            <div className={styles.buttonGroup}>
              <button
                onClick={onStartEdit}
                className={`${styles.button} ${styles.buttonPrimary} ${styles.buttonSmall}`}
                aria-label={`Edit ${lead.name}`}
              >
                ✏️ Edit
              </button>
              <button
                onClick={onDelete}
                className={`${styles.button} ${styles.buttonDanger} ${styles.buttonSmall}`}
                aria-label={`Delete ${lead.name}`}
              >
                🗑️ Delete
              </button>
              <button
                onClick={onAddAppointed}
                className={`${styles.button} ${styles.buttonWarning} ${styles.buttonSmall}`}
                aria-label={`Add appointed party to ${lead.name}`}
              >
                ➕ Add Party
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
});

LeadNode.displayName = 'LeadNode';

/**
 * Memoized Appointed Party Node Component
 */
const AppointedNode = React.memo(({ 
  appointed, 
  leadIndex, 
  appointedIndex, 
  colors, 
  editable, 
  editing,
  editValues,
  errors,
  onStartEdit, 
  onSaveEdit, 
  onCancelEdit,
  onUpdateEditValues,
  onDelete 
}) => {
  const isEditing = editing?.type === 'appointed' && 
                    editing.path.leadIndex === leadIndex && 
                    editing.path.appointedIndex === appointedIndex;
  const inputRef = useRef(null);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSaveEdit();
    } else if (e.key === 'Escape') {
      onCancelEdit();
    }
  };

  return (
    <div 
      className={styles.appointedCard}
      style={{ 
        background: colors.appointed, 
        border: `1px solid ${colors.border}` 
      }}
      role="treeitem"
      aria-label={`Appointed party: ${appointed.name}`}
    >
      {isEditing ? (
        <form 
          className={styles.editForm}
          onSubmit={(e) => { e.preventDefault(); onSaveEdit(); }}
        >
          <div>
            <label htmlFor={`appointed-name-${leadIndex}-${appointedIndex}`} className={styles.srOnly}>
              Company Name
            </label>
            <input
              ref={inputRef}
              id={`appointed-name-${leadIndex}-${appointedIndex}`}
              type="text"
              className={`${styles.input} ${styles.inputSmall} ${errors.name ? styles.inputError : ''}`}
              value={editValues.name}
              onChange={(e) => onUpdateEditValues({ name: e.target.value })}
              onKeyDown={handleKeyDown}
              placeholder="Company Name"
              aria-invalid={!!errors.name}
            />
            {errors.name && (
              <div className={styles.errorMessage} role="alert">{errors.name}</div>
            )}
          </div>

          <div>
            <label htmlFor={`appointed-role-${leadIndex}-${appointedIndex}`} className={styles.srOnly}>
              Role/Service
            </label>
            <input
              id={`appointed-role-${leadIndex}-${appointedIndex}`}
              type="text"
              className={`${styles.input} ${styles.inputSmall} ${errors.role ? styles.inputError : ''}`}
              value={editValues.role}
              onChange={(e) => onUpdateEditValues({ role: e.target.value })}
              onKeyDown={handleKeyDown}
              placeholder="Role/Service"
              aria-invalid={!!errors.role}
            />
          </div>

          <div>
            <label htmlFor={`appointed-contact-${leadIndex}-${appointedIndex}`} className={styles.srOnly}>
              Information Manager
            </label>
            <input
              id={`appointed-contact-${leadIndex}-${appointedIndex}`}
              type="text"
              className={`${styles.input} ${styles.inputSmall}`}
              value={editValues.contact}
              onChange={(e) => onUpdateEditValues({ contact: e.target.value })}
              onKeyDown={handleKeyDown}
              placeholder="Information Manager"
            />
          </div>

          <div className={styles.buttonGroup}>
            <button
              type="submit"
              className={`${styles.button} ${styles.buttonSuccess} ${styles.buttonTiny}`}
              aria-label="Save changes"
            >
              ✓ Save
            </button>
            <button
              type="button"
              onClick={onCancelEdit}
              className={`${styles.button} ${styles.buttonSecondary} ${styles.buttonTiny}`}
              aria-label="Cancel editing"
            >
              ✕ Cancel
            </button>
          </div>
        </form>
      ) : (
        <>
          <div className={styles.appointedTitle} style={{ color: colors.border }}>
            {appointed.name}
          </div>
          <div className={styles.appointedRole} style={{ color: colors.border }}>
            {appointed.role}
          </div>
          {appointed.contact && (
            <div className={styles.appointedContact} style={{ color: colors.border }}>
              IM: {appointed.contact}
            </div>
          )}
          {editable && (
            <div className={styles.buttonGroup}>
              <button
                onClick={onStartEdit}
                className={`${styles.button} ${styles.buttonPrimary} ${styles.buttonTiny}`}
                aria-label={`Edit ${appointed.name}`}
              >
                ✏️ Edit
              </button>
              <button
                onClick={onDelete}
                className={`${styles.button} ${styles.buttonDanger} ${styles.buttonTiny}`}
                aria-label={`Delete ${appointed.name}`}
              >
                🗑️ Del
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
});

AppointedNode.displayName = 'AppointedNode';

/**
 * Main OrgStructureChart Component
 */
const OrgStructureChart = ({ data, onChange, editable = false }) => {
  // Initialize state with reducer
  const [state, dispatch] = useReducer(orgChartReducer, {
    ...initialState,
    orgData: data && data.appointingParty && data.leadAppointedParty 
      ? buildOrgChartData(data) 
      : data
  });

  const { orgData, editing, editValues, errors } = state;
  
  // Track previous data to detect changes
  const previousDataRef = useRef(data);

  // Update when external data changes
  useEffect(() => {
    if (!data) return;
    
    // Only update if data actually changed
    if (deepEqual(data, previousDataRef.current)) return;
    
    previousDataRef.current = data;
    const newTree = buildOrgChartData(data);
    
    // Check if the structure actually changed
    if (!deepEqual(newTree, orgData)) {
      dispatch({ type: ACTIONS.SET_ORG_DATA, payload: newTree });
    }
  }, [data, orgData]);

  // Notify parent of changes
  const notifyChange = useCallback((newData) => {
    if (onChange && newData && newData.leadGroups) {
      const leads = newData.leadGroups.map(g => g.name);
      const finalizedParties = convertTreeToFinalizedParties(newData);

      onChange({
        tree: newData,
        leadAppointedParty: leads,
        finalizedParties
      });
    }
  }, [onChange]);

  // Handle editing
  const handleStartEdit = useCallback((type, path, currentNode) => {
    dispatch({
      type: ACTIONS.START_EDIT,
      payload: {
        nodeId: currentNode.id,
        type,
        path,
        currentValues: {
          name: currentNode.name || '',
          role: currentNode.role || '',
          contact: currentNode.contact || ''
        }
      }
    });
  }, []);

  const handleCancelEdit = useCallback(() => {
    dispatch({ type: ACTIONS.CANCEL_EDIT });
  }, []);

  const handleUpdateEditValues = useCallback((updates) => {
    dispatch({ type: ACTIONS.UPDATE_EDIT_VALUES, payload: updates });
  }, []);

  const handleSaveEdit = useCallback(() => {
    // Validate
    const validation = validateNodeData(editValues, editing.type);
    
    if (!validation.isValid) {
      const errorObj = {};
      validation.errors.forEach(error => {
        if (error.includes('Name')) errorObj.name = error;
        if (error.includes('Role')) errorObj.role = error;
        if (error.includes('Contact')) errorObj.contact = error;
      });
      dispatch({ type: ACTIONS.SET_ERRORS, payload: errorObj });
      return;
    }

    // Update node
    dispatch({
      type: ACTIONS.UPDATE_NODE,
      payload: {
        path: editing.path,
        updates: editValues
      }
    });

    // Notify parent
    const updatedData = { ...orgData };
    if (editing.type === 'appointing') {
      Object.assign(updatedData, editValues);
    } else if (editing.type === 'lead') {
      updatedData.leadGroups[editing.path.leadIndex] = {
        ...updatedData.leadGroups[editing.path.leadIndex],
        ...editValues
      };
    } else if (editing.type === 'appointed') {
      updatedData.leadGroups[editing.path.leadIndex].children[editing.path.appointedIndex] = {
        ...updatedData.leadGroups[editing.path.leadIndex].children[editing.path.appointedIndex],
        ...editValues
      };
    }
    notifyChange(updatedData);

    // Clear editing state
    dispatch({ type: ACTIONS.CANCEL_EDIT });
  }, [editValues, editing, orgData, notifyChange]);

  // Handle add/delete operations
  const handleAddLead = useCallback(() => {
    dispatch({ type: ACTIONS.ADD_LEAD });
    const newData = {
      ...orgData,
      leadGroups: [...(orgData.leadGroups || []), {
        name: 'New Lead',
        role: 'Lead Appointed Party',
        contact: '',
        children: []
      }]
    };
    notifyChange(newData);
  }, [orgData, notifyChange]);

  const handleDeleteLead = useCallback((leadIndex) => {
    const confirmed = window.confirm('Delete this lead and all its appointed parties?');
    if (!confirmed) return;

    dispatch({ type: ACTIONS.DELETE_LEAD, payload: { leadIndex } });
    const newData = {
      ...orgData,
      leadGroups: orgData.leadGroups.filter((_, idx) => idx !== leadIndex)
    };
    notifyChange(newData);
  }, [orgData, notifyChange]);

  const handleAddAppointed = useCallback((leadIndex) => {
    dispatch({ type: ACTIONS.ADD_APPOINTED, payload: { leadIndex } });
    const newData = {
      ...orgData,
      leadGroups: orgData.leadGroups.map((group, idx) =>
        idx === leadIndex
          ? {
              ...group,
              children: [...(group.children || []), {
                name: 'New Appointed Party',
                role: 'Appointed Party',
                contact: ''
              }]
            }
          : group
      )
    };
    notifyChange(newData);
  }, [orgData, notifyChange]);

  const handleDeleteAppointed = useCallback((leadIndex, appointedIndex) => {
    const confirmed = window.confirm('Delete this appointed party?');
    if (!confirmed) return;

    dispatch({ type: ACTIONS.DELETE_APPOINTED, payload: { leadIndex, appointedIndex } });
    const newData = {
      ...orgData,
      leadGroups: orgData.leadGroups.map((group, idx) =>
        idx === leadIndex
          ? { ...group, children: group.children.filter((_, childIdx) => childIdx !== appointedIndex) }
          : group
      )
    };
    notifyChange(newData);
  }, [orgData, notifyChange]);

  // Render empty state
  if (!orgData || !orgData.leadGroups) {
    return (
      <div className={styles.emptyState}>
        <div className={styles.emptyStateIcon}>📊</div>
        <div className={styles.emptyStateTitle}>No organizational data available</div>
        <div className={styles.emptyStateDescription}>
          Please provide organizational structure data to visualize the chart.
        </div>
      </div>
    );
  }

  // Render appointing party
  const isEditingAppointing = editing?.type === 'appointing';
  const appointingInputRef = useRef(null);

  useEffect(() => {
    if (isEditingAppointing && appointingInputRef.current) {
      appointingInputRef.current.focus();
      appointingInputRef.current.select();
    }
  }, [isEditingAppointing]);

  return (
    <div className={styles.container} role="tree" aria-label="Organization Structure Chart">
      {/* Appointing Party */}
      <div className={styles.appointingPartyWrapper}>
        <div className={styles.card} role="treeitem" aria-label={`Appointing party: ${orgData.name}`}>
          {isEditingAppointing ? (
            <form 
              className={styles.editForm}
              onSubmit={(e) => { e.preventDefault(); handleSaveEdit(); }}
            >
              <div>
                <label htmlFor="appointing-name" className={styles.srOnly}>
                  Appointing Party Name
                </label>
                <input
                  ref={appointingInputRef}
                  id="appointing-name"
                  type="text"
                  className={`${styles.input} ${errors.name ? styles.inputError : ''}`}
                  value={editValues.name}
                  onChange={(e) => handleUpdateEditValues({ name: e.target.value })}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSaveEdit();
                    } else if (e.key === 'Escape') {
                      handleCancelEdit();
                    }
                  }}
                  placeholder="Appointing Party Name"
                  aria-invalid={!!errors.name}
                />
                {errors.name && (
                  <div className={styles.errorMessage} role="alert">{errors.name}</div>
                )}
              </div>

              <div className={styles.buttonGroup}>
                <button
                  type="submit"
                  className={`${styles.button} ${styles.buttonSuccess}`}
                  aria-label="Save changes"
                >
                  ✓ Save
                </button>
                <button
                  type="button"
                  onClick={handleCancelEdit}
                  className={`${styles.button} ${styles.buttonSecondary}`}
                  aria-label="Cancel editing"
                >
                  ✕ Cancel
                </button>
              </div>
            </form>
          ) : (
            <>
              <div className={styles.cardTitle}>{orgData.name}</div>
              <div className={styles.cardRole}>{orgData.role}</div>
              {editable && (
                <div className={styles.buttonGroup}>
                  <button
                    onClick={() => handleStartEdit('appointing', { type: 'appointing' }, orgData)}
                    className={`${styles.button} ${styles.buttonPrimary}`}
                    aria-label={`Edit ${orgData.name}`}
                  >
                    ✏️ Edit
                  </button>
                  <button
                    onClick={handleAddLead}
                    className={`${styles.button} ${styles.buttonSuccess}`}
                    aria-label="Add new lead"
                  >
                    ➕ Add Lead
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Lead Groups Grid */}
      <div className={styles.leadsGrid}>
        {orgData.leadGroups.map((lead, leadIndex) => {
          const colors = getColorPalette(leadIndex);

          return (
            <div key={lead.id} className={styles.leadColumn}>
              <LeadNode
                lead={lead}
                leadIndex={leadIndex}
                colors={colors}
                editable={editable}
                editing={editing}
                editValues={editValues}
                errors={errors}
                onStartEdit={() => handleStartEdit('lead', { type: 'lead', leadIndex }, lead)}
                onSaveEdit={handleSaveEdit}
                onCancelEdit={handleCancelEdit}
                onUpdateEditValues={handleUpdateEditValues}
                onDelete={() => handleDeleteLead(leadIndex)}
                onAddAppointed={() => handleAddAppointed(leadIndex)}
              />

              {/* Appointed Parties */}
              <div className={styles.appointedPartiesColumn}>
                {(lead.children || []).map((appointed, appointedIndex) => (
                  <AppointedNode
                    key={appointed.id}
                    appointed={appointed}
                    leadIndex={leadIndex}
                    appointedIndex={appointedIndex}
                    colors={colors}
                    editable={editable}
                    editing={editing}
                    editValues={editValues}
                    errors={errors}
                    onStartEdit={() => 
                      handleStartEdit('appointed', { type: 'appointed', leadIndex, appointedIndex }, appointed)
                    }
                    onSaveEdit={handleSaveEdit}
                    onCancelEdit={handleCancelEdit}
                    onUpdateEditValues={handleUpdateEditValues}
                    onDelete={() => handleDeleteAppointed(leadIndex, appointedIndex)}
                  />
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default OrgStructureChart;
