import React from 'react';
import OrgStructureField from '../forms/specialized/OrgStructureField';
import OrgStructureDataTable from '../forms/specialized/OrgStructureDataTable';
import CDEDiagramBuilderV2 from '../forms/diagrams/diagram-components/CDEDiagramBuilder';
import VolumeStrategyMindmap from '../forms/diagrams/diagram-components/VolumeStrategyMindmap';
import FolderStructureDiagram from '../forms/diagrams/diagram-components/FolderStructureDiagram';
import NamingConventionBuilder from '../forms/custom/NamingConventionBuilder';
import FederationStrategyBuilder from '../forms/custom/FederationStrategyBuilder';

/**
 * Hidden component renderer for PDF export
 * Renders all custom visual components off-screen so they can be captured as screenshots
 *
 * IMPORTANT: Uses position fixed and off-screen positioning instead of visibility:hidden
 * because html2canvas cannot capture elements with visibility:hidden
 */
const HiddenComponentsRenderer = ({ formData, bepType }) => {
  const noop = () => {}; // No-op onChange since we're just rendering for screenshot

  return (
    <div
      id="hidden-components-for-pdf"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '1200px',
        maxHeight: '100vh',
        overflow: 'hidden',
        pointerEvents: 'none',
        zIndex: -1000,
        backgroundColor: 'white',
        // Make it invisible but keep it in the viewport for html2canvas
        clipPath: 'inset(0 0 100% 100%)', // Clips the content so it's not visible
        opacity: 0.01 // Very low opacity instead of 0
      }}
    >
      {/* Organizational Structure Chart */}
      {formData.organizationalStructure && (
        <div data-field-name="organizationalStructure" data-component-type="orgchart" style={{ marginBottom: '50px' }}>
          <OrgStructureField
            field={{ name: 'organizationalStructure', label: 'Organizational Structure' }}
            value={formData.organizationalStructure}
            onChange={noop}
            formData={formData}
            exportMode={true}
          />
        </div>
      )}

      {/* Lead Appointed Parties Table (Section 3.2) - Auto-populated from Org Structure */}
      {formData.organizationalStructure && (
        <div data-field-name="leadAppointedPartiesTable" data-component-type="orgstructure-data-table" style={{ marginBottom: '50px' }}>
          <OrgStructureDataTable
            field={{ name: 'leadAppointedPartiesTable', label: 'Lead Appointed Parties and Information Managers', readOnly: true }}
            value={formData.leadAppointedPartiesTable}
            formData={formData}
            exportMode={true}
          />
        </div>
      )}

      {/* CDE Diagram */}
      {formData.cdeStrategy && (
        <div data-field-name="cdeStrategy" data-component-type="cdeDiagram" style={{ marginBottom: '50px' }}>
          <CDEDiagramBuilderV2
            field={{ name: 'cdeStrategy', label: 'CDE Strategy' }}
            value={formData.cdeStrategy}
            onChange={noop}
          />
        </div>
      )}

      {/* Volume Strategy Mindmap */}
      {formData.volumeStrategy && (
        <div data-field-name="volumeStrategy" data-component-type="mindmap" style={{ marginBottom: '50px' }}>
          <VolumeStrategyMindmap
            field={{ name: 'volumeStrategy', label: 'Volume Strategy' }}
            value={formData.volumeStrategy}
            onChange={noop}
          />
        </div>
      )}

      {/* Folder Structure Diagram */}
      {formData.fileStructureDiagram && (
        <div data-field-name="fileStructureDiagram" data-component-type="fileStructure" style={{ marginBottom: '50px' }}>
          <FolderStructureDiagram
            field={{ name: 'fileStructureDiagram', label: 'Folder Structure' }}
            value={formData.fileStructureDiagram}
            onChange={noop}
          />
        </div>
      )}

      {/* Naming Conventions */}
      {formData.namingConventions && (
        <div data-field-name="namingConventions" data-component-type="naming-conventions" style={{ marginBottom: '50px' }}>
          <NamingConventionBuilder
            field={{ name: 'namingConventions', label: 'Naming Conventions' }}
            value={formData.namingConventions}
            onChange={noop}
          />
        </div>
      )}

      {/* Federation Strategy */}
      {formData.federationStrategy && (
        <div data-field-name="federationStrategy" data-component-type="federation-strategy" style={{ marginBottom: '50px' }}>
          <FederationStrategyBuilder
            field={{ name: 'federationStrategy', label: 'Federation Strategy' }}
            value={formData.federationStrategy}
            onChange={noop}
          />
        </div>
      )}
    </div>
  );
};

export default HiddenComponentsRenderer;
