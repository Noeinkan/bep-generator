import React, { useState, useEffect, useRef } from 'react';
import { Building2, ChevronDown, ChevronUp, Trash2, Plus, ArrowLeft, ArrowRight } from 'lucide-react';

// Helper to build org chart data
function buildOrgChartData(data) {
  if (!data) return null;

  const appointingParty = data.appointingParty || 'Appointing Party';

  // Normalize leads
  let leads = [];
  if (Array.isArray(data.leadAppointedParty)) {
    leads = data.leadAppointedParty;
  } else if (typeof data.leadAppointedParty === 'string') {
    leads = [data.leadAppointedParty];
  }

  // Map appointed parties to leads
  let appointedMap = {};
  leads.forEach(lead => {
    appointedMap[lead] = [];
  });

  // Handle finalizedParties with strict mapping
  if (Array.isArray(data.finalizedParties) && data.finalizedParties.length > 0) {
    data.finalizedParties.forEach((party, index) => {
      // Only assign to a lead if explicitly specified
      const parentLead = party['Parent Lead'] || party['Lead Appointed Party'] || party['Lead'];
      if (parentLead && leads.includes(parentLead)) {
        appointedMap[parentLead].push({
          id: `appointed_${Date.now()}_${index}_${Math.random().toString(36).slice(2)}`,
          name: party['Company Name'] || party['Role/Service'] || 'Appointed Party',
          role: party['Role/Service'] || 'Appointed Party',
          contact: party['Lead Contact'] || ''
        });
      }
    });
  }

  return {
    id: `appointing_${Date.now()}_${Math.random().toString(36).slice(2)}`,
    name: appointingParty,
    role: 'Appointing Party',
    leadGroups: leads.map((lead, index) => ({
      id: `lead_${Date.now()}_${index}_${Math.random().toString(36).slice(2)}`,
      name: lead,
      role: 'Lead Appointed Party',
      children: appointedMap[lead] || []
    }))
  };
}

const OrgStructureChart = ({ data, onChange, editable = false }) => {
  const initialTree = data && data.appointingParty && data.leadAppointedParty
    ? buildOrgChartData(data)
    : data;

  const [orgData, setOrgData] = useState(initialTree);
  const [editing, setEditing] = useState(null);
  const [editValues, setEditValues] = useState({ name: '', role: '', contact: '' });
  const [collapsedLeads, setCollapsedLeads] = useState(new Set());
  const [svgKey, setSvgKey] = useState(0);
  const appointingRef = useRef(null);
  const leadRefs = useRef([]);

  useEffect(() => {
    if (!data) return;
    const newTree = buildOrgChartData(data);

    const currentLeadNames = orgData?.leadGroups ? orgData.leadGroups.map(g => g.name) : [];
    const newLeadNames = newTree?.leadGroups ? newTree.leadGroups.map(g => g.name) : [];

    if (newTree && (newTree.name !== orgData?.name || JSON.stringify(newLeadNames) !== JSON.stringify(currentLeadNames))) {
      setOrgData(newTree);
    }
  }, [data, orgData?.name, orgData?.leadGroups]);

  useEffect(() => {
    // Force SVG update after refs are populated
    const timer = setTimeout(() => {
      setSvgKey(prev => prev + 1);
    }, 50);
    return () => clearTimeout(timer);
  }, [orgData?.leadGroups?.length]);

  const notifyChange = (newData) => {
    if (onChange && newData && newData.leadGroups) {
      const leads = newData.leadGroups.map(g => g.name);
      const finalizedParties = [];

      newData.leadGroups.forEach(group => {
        (group.children || []).forEach(child => {
          finalizedParties.push({
            'Role/Service': child.role || 'Appointed Party',
            'Company Name': child.name,
            'Lead Contact': child.contact || '',
            'Parent Lead': group.name
          });
        });
      });

      onChange({
        tree: newData,
        leadAppointedParty: leads,
        finalizedParties
      });
    }
  };

  const startEditing = (type, index, appointedIndex = null) => {
    if (type === 'appointing') {
      setEditValues({ name: orgData.name, role: orgData.role, contact: '' });
    } else if (type === 'lead') {
      setEditValues({ name: orgData.leadGroups[index].name, role: orgData.leadGroups[index].role, contact: '' });
    } else if (type === 'appointed') {
      const appointed = orgData.leadGroups[index].children[appointedIndex];
      setEditValues({ name: appointed.name, role: appointed.role, contact: appointed.contact || '' });
    }
    setEditing({ type, index, appointedIndex });
  };

  const saveEdits = () => {
    if (!editValues.name.trim()) {
      alert('Name cannot be empty');
      return;
    }
    if (!editValues.role.trim() && editing.type !== 'appointing') {
      alert('Role cannot be empty');
      return;
    }

    let newOrgData = { ...orgData };
    if (editing.type === 'appointing') {
      newOrgData = { ...orgData, name: editValues.name };
    } else if (editing.type === 'lead') {
      const newLeadGroups = orgData.leadGroups.map((group, index) =>
        index === editing.index ? { ...group, name: editValues.name, role: editValues.role } : group
      );
      newOrgData = { ...orgData, leadGroups: newLeadGroups };
    } else if (editing.type === 'appointed') {
      const newLeadGroups = orgData.leadGroups.map((group, index) =>
        index === editing.index
          ? {
              ...group,
              children: group.children.map((child, i) =>
                i === editing.appointedIndex
                  ? { ...child, name: editValues.name, role: editValues.role, contact: editValues.contact }
                  : child
              )
            }
          : group
      );
      newOrgData = { ...orgData, leadGroups: newLeadGroups };
    }

    setOrgData(newOrgData);
    notifyChange(newOrgData);
    setEditing(null);
    setEditValues({ name: '', role: '', contact: '' });
  };

  const cancelEditing = () => {
    setEditing(null);
    setEditValues({ name: '', role: '', contact: '' });
  };

  const addLead = () => {
    const newLead = {
      id: `lead_${Date.now()}_${Math.random().toString(36).slice(2)}`,
      name: 'New Lead',
      role: 'Lead Appointed Party',
      children: []
    };

    const newOrgData = {
      ...orgData,
      leadGroups: [...(orgData.leadGroups || []), newLead]
    };

    setOrgData(newOrgData);
    notifyChange(newOrgData);
  };

  const deleteLead = (leadIndex) => {
    const newLeadGroups = orgData.leadGroups.filter((_, index) => index !== leadIndex);
    const newOrgData = { ...orgData, leadGroups: newLeadGroups };

    leadRefs.current = [];
    setSvgKey(prev => prev + 1);
    setOrgData(newOrgData);
    notifyChange(newOrgData);
  };

  const addAppointedParty = (leadIndex) => {
    const newAppointed = {
      id: `appointed_${Date.now()}_${Math.random().toString(36).slice(2)}`,
      name: 'New Appointed Party',
      role: 'Appointed Party',
      contact: ''
    };

    const newLeadGroups = orgData.leadGroups.map((group, index) =>
      index === leadIndex
        ? { ...group, children: [...(group.children || []), newAppointed] }
        : group
    );

    const newOrgData = { ...orgData, leadGroups: newLeadGroups };
    setOrgData(newOrgData);
    notifyChange(newOrgData);
  };

  const deleteAppointedParty = (leadIndex, appointedIndex) => {
    const newLeadGroups = orgData.leadGroups.map((group, index) =>
      index === leadIndex
        ? { ...group, children: group.children.filter((_, i) => i !== appointedIndex) }
        : group
    );

    const newOrgData = { ...orgData, leadGroups: newLeadGroups };
    setOrgData(newOrgData);
    notifyChange(newOrgData);
  };

  const moveLeadLeft = (leadIndex) => {
    if (leadIndex === 0) return;
    const newLeadGroups = [...orgData.leadGroups];
    const temp = newLeadGroups[leadIndex];
    newLeadGroups[leadIndex] = newLeadGroups[leadIndex - 1];
    newLeadGroups[leadIndex - 1] = temp;

    const newOrgData = { ...orgData, leadGroups: newLeadGroups };
    leadRefs.current = [];
    setSvgKey(prev => prev + 1);
    setOrgData(newOrgData);
    notifyChange(newOrgData);
  };

  const moveLeadRight = (leadIndex) => {
    if (leadIndex === orgData.leadGroups.length - 1) return;
    const newLeadGroups = [...orgData.leadGroups];
    const temp = newLeadGroups[leadIndex];
    newLeadGroups[leadIndex] = newLeadGroups[leadIndex + 1];
    newLeadGroups[leadIndex + 1] = temp;

    const newOrgData = { ...orgData, leadGroups: newLeadGroups };
    leadRefs.current = [];
    setSvgKey(prev => prev + 1);
    setOrgData(newOrgData);
    notifyChange(newOrgData);
  };

  const toggleCollapse = (leadIndex) => {
    const newCollapsed = new Set(collapsedLeads);
    if (newCollapsed.has(leadIndex)) {
      newCollapsed.delete(leadIndex);
    } else {
      newCollapsed.add(leadIndex);
    }
    setCollapsedLeads(newCollapsed);
  };

  const colorPalettes = [
    { bg: 'bg-blue-500', light: 'bg-blue-50', border: 'border-blue-300', text: 'text-blue-700', hover: 'hover:bg-blue-600', ring: 'ring-blue-400' },
    { bg: 'bg-green-500', light: 'bg-green-50', border: 'border-green-300', text: 'text-green-700', hover: 'hover:bg-green-600', ring: 'ring-green-400' },
    { bg: 'bg-orange-500', light: 'bg-orange-50', border: 'border-orange-300', text: 'text-orange-700', hover: 'hover:bg-orange-600', ring: 'ring-orange-400' },
    { bg: 'bg-purple-500', light: 'bg-purple-50', border: 'border-purple-300', text: 'text-purple-700', hover: 'hover:bg-purple-600', ring: 'ring-purple-400' },
    { bg: 'bg-red-500', light: 'bg-red-50', border: 'border-red-300', text: 'text-red-700', hover: 'hover:bg-red-600', ring: 'ring-red-400' },
    { bg: 'bg-cyan-500', light: 'bg-cyan-50', border: 'border-cyan-300', text: 'text-cyan-700', hover: 'hover:bg-cyan-600', ring: 'ring-cyan-400' },
    { bg: 'bg-amber-500', light: 'bg-amber-50', border: 'border-amber-300', text: 'text-amber-700', hover: 'hover:bg-amber-600', ring: 'ring-amber-400' },
    { bg: 'bg-slate-500', light: 'bg-slate-50', border: 'border-slate-300', text: 'text-slate-700', hover: 'hover:bg-slate-600', ring: 'ring-slate-400' }
  ];

  if (!orgData || !orgData.leadGroups) {
    return <div className="text-gray-500 p-4">No organizational data available</div>;
  }

  return (
    <div className="w-full p-6 bg-gradient-to-br from-gray-50 via-gray-100 to-gray-50 rounded-2xl relative overflow-hidden shadow-inner">
      {/* SVG for connectors */}
      <svg key={svgKey} className="absolute inset-0 pointer-events-none z-0 opacity-60" style={{ width: '100%', height: '100%' }}>
        {orgData.leadGroups.map((_, leadIndex) => {
          if (!appointingRef.current || !leadRefs.current[leadIndex]) return null;

          const appointingRect = appointingRef.current.getBoundingClientRect();
          const leadRect = leadRefs.current[leadIndex].getBoundingClientRect();
          const containerRect = appointingRef.current.closest('.relative').getBoundingClientRect();

          const startX = appointingRect.left + appointingRect.width / 2 - containerRect.left;
          const startY = appointingRect.bottom - containerRect.top + 8;
          const endX = leadRect.left + leadRect.width / 2 - containerRect.left;
          const endY = leadRect.top - containerRect.top - 8;

          return (
            <line
              key={`line-${leadIndex}`}
              x1={startX}
              y1={startY}
              x2={endX}
              y2={endY}
              stroke="#94a3b8"
              strokeWidth="2"
              strokeDasharray="6,4"
            />
          );
        })}
      </svg>

      {/* Appointing Party */}
      <div ref={appointingRef} className="flex justify-center mb-16 relative z-10">
        <div className="min-w-[220px] max-w-[380px] w-full">
          {editing?.type === 'appointing' ? (
            <div className="bg-white border-2 border-blue-500 rounded-xl p-5 shadow-2xl animate-in fade-in zoom-in duration-200">
              <input
                type="text"
                value={editValues.name}
                onChange={(e) => setEditValues({ ...editValues, name: e.target.value })}
                placeholder="Name"
                className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all mb-3 text-sm"
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Enter') saveEdits();
                  if (e.key === 'Escape') cancelEditing();
                }}
              />
              <div className="flex gap-2">
                <button
                  onClick={saveEdits}
                  className="flex-1 px-4 py-2 bg-green-500 hover:bg-green-600 hover:shadow-lg text-white text-sm font-medium rounded-lg transition-all transform hover:scale-105 active:scale-95"
                >
                  Save
                </button>
                <button
                  onClick={cancelEditing}
                  className="flex-1 px-4 py-2 bg-gray-400 hover:bg-gray-500 hover:shadow-lg text-white text-sm font-medium rounded-lg transition-all transform hover:scale-105 active:scale-95"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className="bg-gradient-to-br from-blue-600 via-blue-700 to-blue-800 text-white rounded-xl p-5 shadow-2xl border-2 border-blue-400 transform transition-all duration-300 hover:scale-110 hover:shadow-3xl hover:rotate-1 cursor-pointer group">
              <div className="flex items-center justify-center gap-3 mb-2">
                <Building2 className="w-6 h-6 group-hover:scale-125 transition-transform" />
                <div className="text-xl font-bold">{orgData.name}</div>
              </div>
              <div className="text-sm opacity-90 text-center font-medium">{orgData.role}</div>
              {editable && (
                <div className="mt-4 flex gap-2 justify-center">
                  <button
                    onClick={() => startEditing('appointing', 0)}
                    className="px-4 py-1.5 bg-white/20 hover:bg-white/40 text-white text-xs font-medium rounded-lg transition-all backdrop-blur-sm transform hover:scale-110 hover:shadow-lg"
                  >
                    Edit
                  </button>
                  <button
                    onClick={addLead}
                    className="px-4 py-1.5 bg-green-500 hover:bg-green-400 text-white text-xs font-medium rounded-lg transition-all flex items-center gap-1.5 transform hover:scale-110 hover:shadow-lg"
                  >
                    <Plus className="w-3.5 h-3.5" /> Add Lead
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Lead Groups Grid */}
      <div
        className={`grid gap-6 relative z-10`}
        style={{ gridTemplateColumns: `repeat(${Math.min(orgData.leadGroups.length, 5)}, minmax(0, 1fr))` }}
      >
        {orgData.leadGroups.map((lead, leadIndex) => {
          const colors = colorPalettes[leadIndex % colorPalettes.length];
          const isCollapsed = collapsedLeads.has(leadIndex);

          return (
            <div
              key={lead.id}
              className="flex flex-col transition-all duration-300"
            >
              {/* Lead Node */}
              <div
                ref={el => leadRefs.current[leadIndex] = el}
                className={`${colors.bg} text-white rounded-xl p-4 shadow-xl mb-4 transform transition-all duration-300 ${colors.hover} hover:scale-105 hover:shadow-2xl hover:-translate-y-1 cursor-pointer group`}
              >
                {editing?.type === 'lead' && editing.index === leadIndex ? (
                  <div className="space-y-2.5">
                    <input
                      type="text"
                      value={editValues.name}
                      onChange={(e) => setEditValues({ ...editValues, name: e.target.value })}
                      placeholder="Name"
                      className="w-full px-3 py-2 text-gray-900 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-white transition-all"
                      autoFocus
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') saveEdits();
                        if (e.key === 'Escape') cancelEditing();
                      }}
                    />
                    <input
                      type="text"
                      value={editValues.role}
                      onChange={(e) => setEditValues({ ...editValues, role: e.target.value })}
                      placeholder="Role"
                      className="w-full px-3 py-2 text-gray-900 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-white transition-all"
                    />
                    <div className="flex gap-2">
                      <button
                        onClick={saveEdits}
                        className="flex-1 px-3 py-1.5 bg-green-500 hover:bg-green-600 text-white text-xs font-medium rounded-lg transition-all transform hover:scale-105"
                      >
                        Save
                      </button>
                      <button
                        onClick={cancelEditing}
                        className="flex-1 px-3 py-1.5 bg-gray-400 hover:bg-gray-500 text-white text-xs font-medium rounded-lg transition-all transform hover:scale-105"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="flex items-center justify-center gap-2.5 mb-1.5">
                      <div className="font-bold text-base text-center">{lead.name}</div>
                    </div>
                    <div className="text-xs opacity-90 text-center mb-3 font-medium">{lead.role}</div>
                    {editable && (
                      <>
                        <div className="flex gap-1.5 flex-wrap justify-center mb-2">
                          <button
                            onClick={() => startEditing('lead', leadIndex)}
                            className="px-2.5 py-1 bg-white/20 hover:bg-white/40 text-white text-xs font-medium rounded-md transition-all backdrop-blur-sm transform hover:scale-110"
                          >
                            Edit
                          </button>
                          <button
                            onClick={() => deleteLead(leadIndex)}
                            className="px-2.5 py-1 bg-red-500/80 hover:bg-red-600 text-white text-xs font-medium rounded-md transition-all flex items-center gap-1 transform hover:scale-110"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                          <button
                            onClick={() => addAppointedParty(leadIndex)}
                            className="px-2.5 py-1 bg-green-500/80 hover:bg-green-600 text-white text-xs font-medium rounded-md transition-all flex items-center gap-1 transform hover:scale-110"
                          >
                            <Plus className="w-3 h-3" /> Add
                          </button>
                          {(lead.children || []).length > 0 && (
                            <button
                              onClick={() => toggleCollapse(leadIndex)}
                              className="px-2.5 py-1 bg-white/20 hover:bg-white/40 text-white text-xs font-medium rounded-md transition-all transform hover:scale-110"
                            >
                              {isCollapsed ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
                            </button>
                          )}
                        </div>
                        <div className="flex gap-1.5 justify-center">
                          <button
                            onClick={() => moveLeadLeft(leadIndex)}
                            disabled={leadIndex === 0}
                            className={`px-2.5 py-1 text-white text-xs font-medium rounded-md transition-all flex items-center gap-1 ${
                              leadIndex === 0
                                ? 'bg-white/10 cursor-not-allowed opacity-50'
                                : 'bg-white/20 hover:bg-white/40 transform hover:scale-110'
                            }`}
                          >
                            <ArrowLeft className="w-3 h-3" /> Move Left
                          </button>
                          <button
                            onClick={() => moveLeadRight(leadIndex)}
                            disabled={leadIndex === orgData.leadGroups.length - 1}
                            className={`px-2.5 py-1 text-white text-xs font-medium rounded-md transition-all flex items-center gap-1 ${
                              leadIndex === orgData.leadGroups.length - 1
                                ? 'bg-white/10 cursor-not-allowed opacity-50'
                                : 'bg-white/20 hover:bg-white/40 transform hover:scale-110'
                            }`}
                          >
                            Move Right <ArrowRight className="w-3 h-3" />
                          </button>
                        </div>
                      </>
                    )}
                  </>
                )}
              </div>

              {/* Appointed Parties */}
              {!isCollapsed && (
                <div className="space-y-3 animate-in slide-in-from-top duration-300">
                  {(lead.children || []).map((appointed, appointedIndex) => (
                    <div
                      key={appointed.id}
                      className={`${colors.light} border-2 ${colors.border} rounded-xl p-3.5 shadow-md transition-all duration-300 hover:shadow-xl hover:scale-105 hover:-translate-y-0.5 cursor-pointer group`}
                    >
                      {editing?.type === 'appointed' && editing.index === leadIndex && editing.appointedIndex === appointedIndex ? (
                        <div className="space-y-2.5">
                          <input
                            type="text"
                            value={editValues.name}
                            onChange={(e) => setEditValues({ ...editValues, name: e.target.value })}
                            placeholder="Name"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 transition-all"
                            autoFocus
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') saveEdits();
                              if (e.key === 'Escape') cancelEditing();
                            }}
                          />
                          <input
                            type="text"
                            value={editValues.role}
                            onChange={(e) => setEditValues({ ...editValues, role: e.target.value })}
                            placeholder="Role"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 transition-all"
                          />
                          <input
                            type="text"
                            value={editValues.contact}
                            onChange={(e) => setEditValues({ ...editValues, contact: e.target.value })}
                            placeholder="Contact"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 transition-all"
                          />
                          <div className="flex gap-2">
                            <button
                              onClick={saveEdits}
                              className="flex-1 px-3 py-1.5 bg-green-500 hover:bg-green-600 text-white text-xs font-medium rounded-lg transition-all transform hover:scale-105"
                            >
                              Save
                            </button>
                            <button
                              onClick={cancelEditing}
                              className="flex-1 px-3 py-1.5 bg-gray-400 hover:bg-gray-500 text-white text-xs font-medium rounded-lg transition-all transform hover:scale-105"
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      ) : (
                        <>
                          <div className={`flex items-center gap-2 mb-1.5 ${colors.text}`}>
                            <div className="font-semibold text-sm">{appointed.name}</div>
                          </div>
                          <div className={`text-xs ${colors.text} opacity-80 mb-1 font-medium`}>{appointed.role}</div>
                          {appointed.contact && (
                            <div className={`text-xs ${colors.text} opacity-70 mb-2.5`}>{appointed.contact}</div>
                          )}
                          {editable && (
                            <div className="flex gap-1.5">
                              <button
                                onClick={() => startEditing('appointed', leadIndex, appointedIndex)}
                                className="flex-1 px-2.5 py-1.5 bg-blue-500 hover:bg-blue-600 text-white text-xs font-medium rounded-lg transition-all transform hover:scale-105 hover:shadow-lg"
                              >
                                Edit
                              </button>
                              <button
                                onClick={() => deleteAppointedParty(leadIndex, appointedIndex)}
                                className="px-2.5 py-1.5 bg-red-500 hover:bg-red-600 text-white text-xs font-medium rounded-lg transition-all flex items-center justify-center transform hover:scale-105 hover:shadow-lg"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default OrgStructureChart;
