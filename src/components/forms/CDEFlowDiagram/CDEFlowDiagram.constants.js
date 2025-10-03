// Swimlane configuration
export const SWIMLANES = [
  {
    id: 'wip',
    label: 'WIP',
    fullLabel: 'Work in Progress',
    color: '#fef3c7',
    borderColor: '#f59e0b',
    textColor: '#92400e',
    x: 0
  },
  {
    id: 'shared',
    label: 'SHARED',
    fullLabel: 'Coordination',
    color: '#dbeafe',
    borderColor: '#3b82f6',
    textColor: '#1e40af',
    x: 250
  },
  {
    id: 'published',
    label: 'PUBLISHED',
    fullLabel: 'Approved',
    color: '#d1fae5',
    borderColor: '#10b981',
    textColor: '#065f46',
    x: 500
  },
  {
    id: 'archived',
    label: 'ARCHIVED',
    fullLabel: 'Reference',
    color: '#e5e7eb',
    borderColor: '#6b7280',
    textColor: '#374151',
    x: 750
  },
];

// Initial nodes with swimlane structure
export const getInitialNodes = () => [
  // Swimlane backgrounds (non-interactive, behind everything)
  ...SWIMLANES.map(lane => ({
    id: `bg-${lane.id}`,
    type: 'swimlaneBackground',
    data: { ...lane },
    position: { x: lane.x, y: 50 },
    draggable: false,
    selectable: false,
    zIndex: -1,
  })),
  // Swimlane headers (anchored, not draggable)
  ...SWIMLANES.map(lane => ({
    id: `header-${lane.id}`,
    type: 'swimlaneHeader',
    data: { ...lane },
    position: { x: lane.x + 15, y: 0 },
    draggable: false,
    selectable: false,
  })),
  // Example solutions in swimlanes
  {
    id: 'sol1',
    type: 'solution',
    data: { label: 'SharePoint', swimlane: 'wip' },
    position: { x: 25, y: 80 },
  },
  {
    id: 'sol2',
    type: 'solution',
    data: { label: 'Autodesk Docs', swimlane: 'shared' },
    position: { x: 275, y: 80 },
  },
  {
    id: 'sol3',
    type: 'solution',
    data: { label: 'BIMcollab', swimlane: 'shared' },
    position: { x: 275, y: 160 },
  },
  {
    id: 'sol4',
    type: 'solution',
    data: { label: 'Aconex', swimlane: 'published' },
    position: { x: 525, y: 80 },
  },
  {
    id: 'sol5',
    type: 'solution',
    data: { label: 'Document Archive', swimlane: 'archived' },
    position: { x: 775, y: 80 },
  },
];

// Initial edges showing workflow
export const getInitialEdges = () => [
  { id: 'e1', source: 'sol1', target: 'sol2', type: 'labeledStraight', style: { stroke: '#3b82f6', strokeWidth: 2 }, data: { label: 'review' } },
  { id: 'e2', source: 'sol2', target: 'sol4', type: 'labeledStraight', style: { stroke: '#10b981', strokeWidth: 2 }, data: { label: 'approve' } },
  { id: 'e3', source: 'sol3', target: 'sol4', type: 'labeledStraight', style: { stroke: '#10b981', strokeWidth: 2 }, data: { label: 'publish' } },
  { id: 'e4', source: 'sol4', target: 'sol5', type: 'labeledStraight', style: { stroke: '#6b7280', strokeWidth: 2 }, data: { label: 'archive' } },
];

// Default node style
export const DEFAULT_NODE_STYLE = {
  background: '#ffffff',
  borderColor: '#6b7280',
  textColor: '#000000'
};
