import React, { useMemo } from 'react';
import FieldHeader from '../base/FieldHeader';

/**
 * Dynamic table that reads data from organizationalStructure (OrgStructureChart)
 * and displays Lead Appointed Parties and their Information Managers
 */
const OrgStructureDataTable = ({ value, field, formData }) => {
  // Extract data from organizationalStructure
  const tableData = useMemo(() => {
    const orgStructure = formData?.organizationalStructure;
    
    if (!orgStructure) return [];

    // Check if we have the tree structure
    const tree = orgStructure.tree || orgStructure;
    
    if (!tree || !tree.leadGroups) return [];

    // Map lead groups to table rows
    return tree.leadGroups.map((lead, index) => ({
      id: lead.id || `lead_${index}`,
      leadAppointedParty: lead.name || '',
      role: lead.role || 'Lead Appointed Party',
      informationManager: lead.contact || '',
      appointedPartiesCount: (lead.children || []).length
    }));
  }, [formData?.organizationalStructure]);

  return (
    <div className="w-full max-w-full">
      <FieldHeader 
        fieldName={field.name}
        label={field.label}
        number={field.number}
        required={field.required}
      />
      
      <div className="mt-2 w-full">
        {tableData.length === 0 ? (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
            <p className="text-blue-700 text-sm">
              📊 This table will automatically populate when you add Lead Appointed Parties in the Organizational Structure diagram above.
            </p>
          </div>
        ) : (
          <div className="w-full overflow-x-auto border border-gray-300 rounded-lg shadow-sm">
            <table className="w-full table-fixed divide-y divide-gray-300">
              <thead className="bg-gradient-to-r from-indigo-50 to-blue-50">
                <tr>
                  <th className="w-12 px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-r border-gray-300">
                    #
                  </th>
                  <th className="w-[30%] px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-r border-gray-300">
                    Lead Appointed Party
                  </th>
                  <th className="w-[25%] px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-r border-gray-300">
                    Role
                  </th>
                  <th className="w-[30%] px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-r border-gray-300">
                    Information Manager
                  </th>
                  <th className="w-[15%] px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider">
                    Appointed Parties
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {tableData.map((row, index) => (
                  <tr key={row.id} className="hover:bg-blue-50 transition-colors">
                    <td className="w-12 px-4 py-3 text-sm text-gray-700 border-r border-gray-200 font-medium">
                      {index + 1}
                    </td>
                    <td className="w-[30%] px-4 py-3 text-sm text-gray-900 border-r border-gray-200 font-medium break-words">
                      {row.leadAppointedParty}
                    </td>
                    <td className="w-[25%] px-4 py-3 text-sm text-gray-600 border-r border-gray-200 break-words">
                      {row.role}
                    </td>
                    <td className="w-[30%] px-4 py-3 text-sm text-gray-600 border-r border-gray-200 break-words">
                      {row.informationManager || (
                        <span className="text-gray-400 italic">Not specified</span>
                      )}
                    </td>
                    <td className="w-[15%] px-4 py-3 text-sm text-center">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800">
                        {row.appointedPartiesCount}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        
        <div className="mt-3 flex items-start space-x-2 text-sm text-gray-600 bg-gray-50 border border-gray-200 rounded-lg p-3">
          <div className="flex-shrink-0 mt-0.5">
            <svg className="w-4 h-4 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
          </div>
          <div>
            <p className="font-medium text-gray-700 mb-1">📝 Auto-synced Data</p>
            <p className="text-xs">
              This table automatically reflects the Lead Appointed Parties and Information Managers 
              defined in the <strong>Delivery Team's Organisational Structure and Composition</strong> diagram above. 
              Any changes made to the organizational chart will be instantly reflected here.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OrgStructureDataTable;
