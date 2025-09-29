import { Building, Layers, Zap, Wrench, MapPin, Folder, Users, FileText, Calendar, AlertTriangle, CheckCircle, Clock, Truck } from 'lucide-react';

export const NODE_TYPES = {
  ROOT: 'root',
  DISCIPLINE: 'discipline',
  ZONE: 'zone',
  LEVEL: 'level',
  SYSTEM: 'system',
  EQUIPMENT: 'equipment',
  LOCATION: 'location',
  STAKEHOLDER: 'stakeholder',
  DELIVERABLE: 'deliverable',
  MILESTONE: 'milestone',
  ISSUE: 'issue',
  APPROVAL: 'approval',
  PHASE: 'phase',
  LOGISTICS: 'logistics'
};

export const NODE_TYPE_CONFIG = {
  [NODE_TYPES.ROOT]: {
    label: 'Root',
    color: '#3B82F6',
    bgColor: '#3B82F6',
    textColor: '#FFFFFF',
    icon: Building,
    shape: 'circle'
  },
  [NODE_TYPES.DISCIPLINE]: {
    label: 'Discipline',
    color: '#10B981',
    bgColor: '#10B981',
    textColor: '#FFFFFF',
    icon: Folder,
    shape: 'rect'
  },
  [NODE_TYPES.ZONE]: {
    label: 'Zone',
    color: '#F59E0B',
    bgColor: '#F59E0B',
    textColor: '#FFFFFF',
    icon: MapPin,
    shape: 'rect'
  },
  [NODE_TYPES.LEVEL]: {
    label: 'Level',
    color: '#8B5CF6',
    bgColor: '#8B5CF6',
    textColor: '#FFFFFF',
    icon: Layers,
    shape: 'rect'
  },
  [NODE_TYPES.SYSTEM]: {
    label: 'System',
    color: '#EF4444',
    bgColor: '#EF4444',
    textColor: '#FFFFFF',
    icon: Zap,
    shape: 'rect'
  },
  [NODE_TYPES.EQUIPMENT]: {
    label: 'Equipment',
    color: '#6B7280',
    bgColor: '#6B7280',
    textColor: '#FFFFFF',
    icon: Wrench,
    shape: 'rect'
  },
  [NODE_TYPES.LOCATION]: {
    label: 'Location',
    color: '#14B8A6',
    bgColor: '#14B8A6',
    textColor: '#FFFFFF',
    icon: MapPin,
    shape: 'rect'
  },
  [NODE_TYPES.STAKEHOLDER]: {
    label: 'Stakeholder',
    color: '#0EA5E9',
    bgColor: '#0EA5E9',
    textColor: '#FFFFFF',
    icon: Users,
    shape: 'rect'
  },
  [NODE_TYPES.DELIVERABLE]: {
    label: 'Deliverable',
    color: '#8B5CF6',
    bgColor: '#8B5CF6',
    textColor: '#FFFFFF',
    icon: FileText,
    shape: 'rect'
  },
  [NODE_TYPES.MILESTONE]: {
    label: 'Milestone',
    color: '#F59E0B',
    bgColor: '#F59E0B',
    textColor: '#FFFFFF',
    icon: Calendar,
    shape: 'rect'
  },
  [NODE_TYPES.ISSUE]: {
    label: 'Issue',
    color: '#EF4444',
    bgColor: '#EF4444',
    textColor: '#FFFFFF',
    icon: AlertTriangle,
    shape: 'rect'
  },
  [NODE_TYPES.APPROVAL]: {
    label: 'Approval',
    color: '#22C55E',
    bgColor: '#22C55E',
    textColor: '#FFFFFF',
    icon: CheckCircle,
    shape: 'rect'
  },
  [NODE_TYPES.PHASE]: {
    label: 'Phase',
    color: '#A855F7',
    bgColor: '#A855F7',
    textColor: '#FFFFFF',
    icon: Clock,
    shape: 'rect'
  },
  [NODE_TYPES.LOGISTICS]: {
    label: 'Logistics',
    color: '#FB7185',
    bgColor: '#FB7185',
    textColor: '#FFFFFF',
    icon: Truck,
    shape: 'rect'
  }
};

export const getNodeTypeConfig = (nodeType) => {
  return NODE_TYPE_CONFIG[nodeType] || NODE_TYPE_CONFIG[NODE_TYPES.DISCIPLINE];
};

export const getNodeTypeOptions = () => {
  return Object.keys(NODE_TYPE_CONFIG)
    .filter(type => type !== NODE_TYPES.ROOT)
    .map(type => ({
      value: type,
      label: NODE_TYPE_CONFIG[type].label,
      color: NODE_TYPE_CONFIG[type].color
    }));
};