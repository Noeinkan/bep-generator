import React, { useState } from 'react';
import { ChevronRight, ChevronLeft, Download, FileText, Users, Settings, CheckCircle, AlertCircle, Building, Zap, Shield, Database, Calendar, Target, BookOpen, Monitor } from 'lucide-react';

const ProfessionalBEPGenerator = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [bepType, setBepType] = useState('pre-appointment');
  const [formData, setFormData] = useState({
    // Project Information
    projectName: '',
    projectNumber: '',
    projectDescription: '',
    projectType: '',
    projectTimeline: '',
    projectBudget: '',
    
    // Stakeholders
    appointingParty: '',
    leadAppointedParty: '',
    appointedParties: '',
    informationManager: '',
    taskTeamLeaders: '',
    
    // BIM Goals and Uses
    bimGoals: '',
    bimUses: [],
    primaryObjectives: '',
    
    // Level of Information Need (LOIN)
    informationPurposes: [],
    geometricalInfo: '',
    alphanumericalInfo: '',
    documentationInfo: '',
    informationFormats: [],
    
    // Information Delivery Planning
    midpDescription: '',
    keyMilestones: '',
    deliverySchedule: '',
    tidpRequirements: '',
    
    // Common Data Environment
    cdeProvider: '',
    cdePlatform: '',
    workflowStates: '',
    accessControl: '',
    securityMeasures: '',
    backupProcedures: '',
    
    // Technology Requirements
    bimSoftware: [],
    fileFormats: [],
    hardwareRequirements: '',
    networkRequirements: '',
    interoperabilityNeeds: '',
    
    // Information Production Methods
    modelingStandards: '',
    namingConventions: '',
    fileStructure: '',
    versionControl: '',
    dataExchangeProtocols: '',
    
    // Quality Assurance
    qaFramework: '',
    modelValidation: '',
    reviewProcesses: '',
    approvalWorkflows: '',
    complianceVerification: '',
    
    // Information Security & Privacy
    dataClassification: '',
    accessPermissions: '',
    encryptionRequirements: '',
    dataTransferProtocols: '',
    privacyConsiderations: '',
    
    // Training and Competency
    bimCompetencyLevels: '',
    trainingRequirements: '',
    certificationNeeds: '',
    projectSpecificTraining: '',
    
    // Coordination and Collaboration
    coordinationMeetings: '',
    clashDetectionWorkflow: '',
    issueResolution: '',
    communicationProtocols: '',
    federationStrategy: '',
    
    // Risk Management
    informationRisks: '',
    technologyRisks: '',
    riskMitigation: '',
    contingencyPlans: '',
    
    // Compliance and Monitoring
    performanceMetrics: '',
    monitoringProcedures: '',
    auditTrails: '',
    updateProcesses: ''
  });

  const [completedSections, setCompletedSections] = useState(new Set());

  const steps = [
    {
      title: 'BEP Type & Project Info',
      icon: <Building className="w-5 h-5" />,
      description: 'Define BEP type and basic project information',
      category: 'Commercial'
    },
    {
      title: 'Stakeholders & Roles',
      icon: <Users className="w-5 h-5" />,
      description: 'Define project stakeholders and responsibilities',
      category: 'Commercial'
    },
    {
      title: 'BIM Goals & Uses',
      icon: <Target className="w-5 h-5" />,
      description: 'Define BIM objectives and applications',
      category: 'Commercial'
    },
    {
      title: 'Level of Information Need',
      icon: <Database className="w-5 h-5" />,
      description: 'Specify LOIN requirements and content',
      category: 'Management'
    },
    {
      title: 'Information Delivery Planning',
      icon: <Calendar className="w-5 h-5" />,
      description: 'MIDP, TIDPs and delivery schedules',
      category: 'Management'
    },
    {
      title: 'Common Data Environment',
      icon: <Monitor className="w-5 h-5" />,
      description: 'CDE specification and workflows',
      category: 'Technical'
    },
    {
      title: 'Technology Requirements',
      icon: <Settings className="w-5 h-5" />,
      description: 'Software, hardware and technical specs',
      category: 'Technical'
    },
    {
      title: 'Information Production',
      icon: <FileText className="w-5 h-5" />,
      description: 'Methods, standards and procedures',
      category: 'Management'
    },
    {
      title: 'Quality Assurance',
      icon: <CheckCircle className="w-5 h-5" />,
      description: 'QA framework and validation processes',
      category: 'Management'
    },
    {
      title: 'Security & Privacy',
      icon: <Shield className="w-5 h-5" />,
      description: 'Information security and privacy measures',
      category: 'Management'
    },
    {
      title: 'Training & Competency',
      icon: <BookOpen className="w-5 h-5" />,
      description: 'Training requirements and competency levels',
      category: 'Management'
    },
    {
      title: 'Coordination & Risk',
      icon: <AlertCircle className="w-5 h-5" />,
      description: 'Collaboration procedures and risk management',
      category: 'Management'
    }
  ];

  const bimUsesOptions = [
    'Design Authoring', 'Design Reviews', '3D Coordination', 'Clash Detection',
    'Quantity Take-off', '4D Planning', '5D Cost Management', 'Asset Management',
    'Construction Sequencing', 'Facility Management Handover', 'Energy Analysis',
    'Code Validation', 'Space Planning', 'Site Analysis', 'Structural Analysis',
    'MEP Analysis', 'Lighting Analysis', 'Acoustical Analysis', 'Other Analysis'
  ];

  const informationPurposesOptions = [
    'Design Development', 'Construction Planning', 'Quantity Surveying', 'Cost Estimation',
    'Facility Management', 'Asset Management', 'Carbon Footprint Analysis', 'Fire Strategy',
    'Structural Analysis', 'MEP Coordination', 'Space Management', 'Maintenance Planning',
    'Energy Performance', 'Code Compliance', 'Safety Planning', 'Sustainability Assessment'
  ];

  const softwareOptions = [
    'Autodesk Revit', 'ArchiCAD', 'Tekla Structures', 'Bentley MicroStation', 'Bentley AECOsim',
    'SketchUp Pro', 'Rhino', 'Navisworks', 'Solibri Model Checker', 'BIM 360', 'Trimble Connect',
    'Synchro Pro', 'Vico Office', 'CostX', 'Innovaya', 'dRofus', 'BIMcollab', 'Aconex',
    'PowerBI', 'Tableau', 'FME', 'Safe Software', 'Other'
  ];

  const fileFormatsOptions = [
    'IFC 2x3', 'IFC 4', 'IFC 4.1', 'IFC 4.3', 'DWG', 'DXF', 'PDF', 'PDF/A', 'BCF 2.1', 'BCF 3.0',
    'NWD', 'NWC', 'NWF', 'RVT', 'PLN', 'DGN', 'SKP', 'COBie', 'XML', 'JSON', 'CSV', 'XLS', 'XLSX'
  ];

  const updateFormData = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const toggleArrayValue = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: prev[field].includes(value)
        ? prev[field].filter(item => item !== value)
        : [...prev[field], value]
    }));
  };

  const validateStep = (stepIndex) => {
    const requiredFields = {
      0: ['projectName', 'appointingParty', 'projectType'],
      1: ['leadAppointedParty', 'informationManager'],
      2: ['bimGoals', 'bimUses'],
      3: ['informationPurposes'],
      4: ['midpDescription', 'keyMilestones'],
      5: ['cdeProvider', 'workflowStates'],
      6: ['bimSoftware', 'fileFormats'],
      7: ['modelingStandards', 'namingConventions'],
      8: ['qaFramework', 'modelValidation'],
      9: ['dataClassification', 'accessPermissions'],
      10: ['bimCompetencyLevels'],
      11: ['coordinationMeetings', 'informationRisks']
    };

    const required = requiredFields[stepIndex] || [];
    return required.every(field => {
      const value = formData[field];
      return Array.isArray(value) ? value.length > 0 : value && value.trim() !== '';
    });
  };

  const nextStep = () => {
    if (validateStep(currentStep)) {
      setCompletedSections(prev => new Set([...prev, currentStep]));
    }
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const generateProfessionalBEP = () => {
    const currentDate = new Date();
    const formattedDate = currentDate.toLocaleDateString();
    const formattedTime = currentDate.toLocaleTimeString();

    const bepContent = `
      <html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
      <head>
        <meta charset="utf-8">
        <title>BIM Execution Plan - ${formData.projectName}</title>
        <style>
          body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }
          .header { text-align: center; border-bottom: 3px solid #2563eb; padding-bottom: 20px; margin-bottom: 30px; }
          h1 { color: #1e40af; font-size: 28px; margin-bottom: 10px; }
          .subtitle { color: #059669; font-size: 18px; font-weight: bold; margin-bottom: 5px; }
          .bep-type { background: #dbeafe; padding: 10px; border-radius: 8px; display: inline-block; margin: 10px 0; font-weight: bold; }
          h2 { color: #1e40af; margin-top: 35px; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; font-size: 20px; }
          h3 { color: #374151; margin-top: 25px; font-size: 16px; border-left: 4px solid #2563eb; padding-left: 12px; }
          .section { margin: 25px 0; }
          .info-box { background-color: #f8fafc; padding: 20px; border-left: 4px solid #2563eb; margin: 20px 0; border-radius: 0 8px 8px 0; }
          .category-header { background: linear-gradient(135deg, #2563eb, #1e40af); color: white; padding: 15px; margin: 30px 0 20px 0; border-radius: 8px; font-weight: bold; font-size: 18px; }
          ul { margin: 10px 0; padding-left: 25px; }
          li { margin: 8px 0; }
          .footer { margin-top: 50px; padding-top: 25px; border-top: 2px solid #e5e7eb; background: #f9fafb; padding: 25px; border-radius: 8px; }
          table { width: 100%; border-collapse: collapse; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
          th, td { padding: 12px; border-bottom: 1px solid #e5e7eb; text-align: left; }
          th { background-color: #f8fafc; font-weight: bold; color: #374151; }
          .label { font-weight: bold; width: 250px; color: #4b5563; }
          .highlight { background-color: #fef3c7; padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b; margin: 15px 0; }
          .compliance-box { background: #ecfdf5; border: 2px solid #10b981; padding: 20px; border-radius: 8px; margin: 20px 0; }
          .risk-box { background: #fef2f2; border: 2px solid #ef4444; padding: 15px; border-radius: 8px; margin: 15px 0; }
          .page-break { page-break-before: always; }
        </style>
      </head>
      <body>
        <div class="header">
          <h1>BIM EXECUTION PLAN (BEP)</h1>
          <div class="subtitle">ISO 19650-2 Compliant</div>
          <div class="bep-type">${bepType === 'pre-appointment' ? 'Pre-Appointment BEP' : 'Post-Appointment BEP'}</div>
        </div>

        <div class="compliance-box">
          <h3>Document Information</h3>
          <table>
            <tr><td class="label">Document Type:</td><td>${bepType === 'pre-appointment' ? 'Pre-Appointment BEP' : 'Post-Appointment BEP'}</td></tr>
            <tr><td class="label">Project Name:</td><td>${formData.projectName || 'Not specified'}</td></tr>
            <tr><td class="label">Project Number:</td><td>${formData.projectNumber || 'Not specified'}</td></tr>
            <tr><td class="label">Generated Date:</td><td>${formattedDate} at ${formattedTime}</td></tr>
            <tr><td class="label">Status:</td><td>Work in Progress</td></tr>
          </table>
        </div>

        <div class="category-header">COMMERCIAL ASPECTS</div>
        
        <div class="section">
          <h2>1. PROJECT INFORMATION AND OBJECTIVES</h2>
          <div class="info-box">
            <table>
              <tr><td class="label">Project Name:</td><td>${formData.projectName || 'Not specified'}</td></tr>
              <tr><td class="label">Project Number:</td><td>${formData.projectNumber || 'Not specified'}</td></tr>
              <tr><td class="label">Project Type:</td><td>${formData.projectType || 'Not specified'}</td></tr>
              <tr><td class="label">Timeline:</td><td>${formData.projectTimeline || 'Not specified'}</td></tr>
              <tr><td class="label">Budget:</td><td>${formData.projectBudget || 'Not specified'}</td></tr>
            </table>
            
            ${formData.projectDescription ? `
              <h3>Project Description</h3>
              <p>${formData.projectDescription}</p>
            ` : ''}
          </div>
        </div>

        <div class="section">
          <h2>2. STAKEHOLDERS AND RESPONSIBILITIES</h2>
          <div class="info-box">
            <h3>Key Stakeholders</h3>
            <table>
              <tr><td class="label">Appointing Party:</td><td>${formData.appointingParty || 'Not specified'}</td></tr>
              <tr><td class="label">Lead Appointed Party:</td><td>${formData.leadAppointedParty || 'Not specified'}</td></tr>
              <tr><td class="label">Information Manager:</td><td>${formData.informationManager || 'Not specified'}</td></tr>
              <tr><td class="label">Task Team Leaders:</td><td>${formData.taskTeamLeaders || 'Not specified'}</td></tr>
            </table>
            
            ${formData.appointedParties ? `
              <h3>Appointed Parties</h3>
              <p>${formData.appointedParties}</p>
            ` : ''}
          </div>
        </div>

        <div class="section">
          <h2>3. BIM GOALS AND OBJECTIVES</h2>
          <div class="info-box">
            ${formData.bimGoals ? `
              <h3>BIM Goals</h3>
              <p>${formData.bimGoals}</p>
            ` : ''}
            
            ${formData.primaryObjectives ? `
              <h3>Primary Objectives</h3>
              <p>${formData.primaryObjectives}</p>
            ` : ''}
            
            ${formData.bimUses.length > 0 ? `
              <h3>BIM Uses</h3>
              <ul>
                ${formData.bimUses.map(use => `<li>${use}</li>`).join('')}
              </ul>
            ` : ''}
          </div>
        </div>

        <div class="page-break"></div>
        <div class="category-header">MANAGEMENT ASPECTS</div>

        <div class="section">
          <h2>4. LEVEL OF INFORMATION NEED (LOIN)</h2>
          <div class="info-box">
            ${formData.informationPurposes.length > 0 ? `
              <h3>Information Purposes</h3>
              <ul>
                ${formData.informationPurposes.map(purpose => `<li>${purpose}</li>`).join('')}
              </ul>
            ` : ''}

            ${formData.geometricalInfo ? `
              <h3>Geometrical Information Requirements</h3>
              <p>${formData.geometricalInfo}</p>
            ` : ''}

            ${formData.alphanumericalInfo ? `
              <h3>Alphanumerical Information Requirements</h3>
              <p>${formData.alphanumericalInfo}</p>
            ` : ''}

            ${formData.documentationInfo ? `
              <h3>Documentation Requirements</h3>
              <p>${formData.documentationInfo}</p>
            ` : ''}

            ${formData.informationFormats.length > 0 ? `
              <h3>Information Formats</h3>
              <ul>
                ${formData.informationFormats.map(format => `<li>${format}</li>`).join('')}
              </ul>
            ` : ''}
          </div>
        </div>

        <div class="section">
          <h2>5. INFORMATION DELIVERY PLANNING</h2>
          <div class="info-box">
            <h3>Master Information Delivery Plan (MIDP)</h3>
            ${formData.midpDescription ? `<p>${formData.midpDescription}</p>` : ''}
            
            ${formData.keyMilestones ? `
              <h3>Key Information Delivery Milestones</h3>
              <p>${formData.keyMilestones}</p>
            ` : ''}

            ${formData.deliverySchedule ? `
              <h3>Delivery Schedule</h3>
              <p>${formData.deliverySchedule}</p>
            ` : ''}

            ${formData.tidpRequirements ? `
              <h3>Task Information Delivery Plans (TIDPs)</h3>
              <p>${formData.tidpRequirements}</p>
            ` : ''}
          </div>
        </div>

        <div class="section">
          <h2>6. INFORMATION PRODUCTION METHODS AND PROCEDURES</h2>
          <div class="info-box">
            ${formData.modelingStandards ? `
              <h3>Modeling Standards</h3>
              <p>${formData.modelingStandards}</p>
            ` : ''}

            ${formData.namingConventions ? `
              <h3>Naming Conventions</h3>
              <p>${formData.namingConventions}</p>
            ` : ''}

            ${formData.fileStructure ? `
              <h3>File Structure</h3>
              <p>${formData.fileStructure}</p>
            ` : ''}

            ${formData.versionControl ? `
              <h3>Version Control</h3>
              <p>${formData.versionControl}</p>
            ` : ''}

            ${formData.dataExchangeProtocols ? `
              <h3>Data Exchange Protocols</h3>
              <p>${formData.dataExchangeProtocols}</p>
            ` : ''}
          </div>
        </div>

        <div class="section">
          <h2>7. QUALITY ASSURANCE AND CONTROL</h2>
          <div class="info-box">
            ${formData.qaFramework ? `
              <h3>Quality Assurance Framework</h3>
              <p>${formData.qaFramework}</p>
            ` : ''}

            ${formData.modelValidation ? `
              <h3>Model Validation Procedures</h3>
              <p>${formData.modelValidation}</p>
            ` : ''}

            ${formData.reviewProcesses ? `
              <h3>Review Processes</h3>
              <p>${formData.reviewProcesses}</p>
            ` : ''}

            ${formData.approvalWorkflows ? `
              <h3>Approval Workflows</h3>
              <p>${formData.approvalWorkflows}</p>
            ` : ''}

            ${formData.complianceVerification ? `
              <h3>Compliance Verification</h3>
              <p>${formData.complianceVerification}</p>
            ` : ''}
          </div>
        </div>

        <div class="section">
          <h2>8. INFORMATION SECURITY AND PRIVACY</h2>
          <div class="info-box">
            ${formData.dataClassification ? `
              <h3>Data Classification</h3>
              <p>${formData.dataClassification}</p>
            ` : ''}

            ${formData.accessPermissions ? `
              <h3>Access Permissions</h3>
              <p>${formData.accessPermissions}</p>
            ` : ''}

            ${formData.encryptionRequirements ? `
              <h3>Encryption Requirements</h3>
              <p>${formData.encryptionRequirements}</p>
            ` : ''}

            ${formData.dataTransferProtocols ? `
              <h3>Data Transfer Protocols</h3>
              <p>${formData.dataTransferProtocols}</p>
            ` : ''}

            ${formData.privacyConsiderations ? `
              <h3>Privacy Considerations</h3>
              <p>${formData.privacyConsiderations}</p>
            ` : ''}
          </div>
        </div>

        <div class="section">
          <h2>9. TRAINING AND COMPETENCY</h2>
          <div class="info-box">
            ${formData.bimCompetencyLevels ? `
              <h3>BIM Competency Levels</h3>
              <p>${formData.bimCompetencyLevels}</p>
            ` : ''}

            ${formData.trainingRequirements ? `
              <h3>Training Requirements</h3>
              <p>${formData.trainingRequirements}</p>
            ` : ''}

            ${formData.certificationNeeds ? `
              <h3>Certification Requirements</h3>
              <p>${formData.certificationNeeds}</p>
            ` : ''}

            ${formData.projectSpecificTraining ? `
              <h3>Project-Specific Training</h3>
              <p>${formData.projectSpecificTraining}</p>
            ` : ''}
          </div>
        </div>

        <div class="page-break"></div>
        <div class="category-header">TECHNICAL ASPECTS</div>

        <div class="section">
          <h2>10. COMMON DATA ENVIRONMENT (CDE)</h2>
          <div class="info-box">
            <table>
              <tr><td class="label">CDE Provider:</td><td>${formData.cdeProvider || 'Not specified'}</td></tr>
              <tr><td class="label">CDE Platform:</td><td>${formData.cdePlatform || 'Not specified'}</td></tr>
            </table>

            ${formData.workflowStates ? `
              <h3>Workflow States</h3>
              <p>${formData.workflowStates}</p>
            ` : ''}

            ${formData.accessControl ? `
              <h3>Access Control</h3>
              <p>${formData.accessControl}</p>
            ` : ''}

            ${formData.securityMeasures ? `
              <h3>Security Measures</h3>
              <p>${formData.securityMeasures}</p>
            ` : ''}

            ${formData.backupProcedures ? `
              <h3>Backup Procedures</h3>
              <p>${formData.backupProcedures}</p>
            ` : ''}
          </div>
        </div>

        <div class="section">
          <h2>11. TECHNOLOGY AND SOFTWARE REQUIREMENTS</h2>
          <div class="info-box">
            ${formData.bimSoftware.length > 0 ? `
              <h3>BIM Software Applications</h3>
              <ul>
                ${formData.bimSoftware.map(software => `<li>${software}</li>`).join('')}
              </ul>
            ` : ''}

            ${formData.fileFormats.length > 0 ? `
              <h3>File Formats</h3>
              <ul>
                ${formData.fileFormats.map(format => `<li>${format}</li>`).join('')}
              </ul>
            ` : ''}

            ${formData.hardwareRequirements ? `
              <h3>Hardware Requirements</h3>
              <p>${formData.hardwareRequirements}</p>
            ` : ''}

            ${formData.networkRequirements ? `
              <h3>Network Requirements</h3>
              <p>${formData.networkRequirements}</p>
            ` : ''}

            ${formData.interoperabilityNeeds ? `
              <h3>Interoperability Requirements</h3>
              <p>${formData.interoperabilityNeeds}</p>
            ` : ''}
          </div>
        </div>

        <div class="section">
          <h2>12. COORDINATION AND COLLABORATION PROCEDURES</h2>
          <div class="info-box">
            ${formData.coordinationMeetings ? `
              <h3>Coordination Meetings</h3>
              <p>${formData.coordinationMeetings}</p>
            ` : ''}

            ${formData.clashDetectionWorkflow ? `
              <h3>Clash Detection Workflow</h3>
              <p>${formData.clashDetectionWorkflow}</p>
            ` : ''}

            ${formData.issueResolution ? `
              <h3>Issue Resolution Process</h3>
              <p>${formData.issueResolution}</p>
            ` : ''}

            ${formData.communicationProtocols ? `
              <h3>Communication Protocols</h3>
              <p>${formData.communicationProtocols}</p>
            ` : ''}

            ${formData.federationStrategy ? `
              <h3>Model Federation Strategy</h3>
              <p>${formData.federationStrategy}</p>
            ` : ''}
          </div>
        </div>

        <div class="section">
          <h2>13. RISK MANAGEMENT</h2>
          <div class="risk-box">
            ${formData.informationRisks ? `
              <h3>Information-Related Risks</h3>
              <p>${formData.informationRisks}</p>
            ` : ''}

            ${formData.technologyRisks ? `
              <h3>Technology-Related Risks</h3>
              <p>${formData.technologyRisks}</p>
            ` : ''}

            ${formData.riskMitigation ? `
              <h3>Risk Mitigation Strategies</h3>
              <p>${formData.riskMitigation}</p>
            ` : ''}

            ${formData.contingencyPlans ? `
              <h3>Contingency Plans</h3>
              <p>${formData.contingencyPlans}</p>
            ` : ''}
          </div>
        </div>

        <div class="section">
          <h2>14. COMPLIANCE AND MONITORING</h2>
          <div class="compliance-box">
            ${formData.performanceMetrics ? `
              <h3>Performance Metrics and KPIs</h3>
              <p>${formData.performanceMetrics}</p>
            ` : ''}

            ${formData.monitoringProcedures ? `
              <h3>Monitoring Procedures</h3>
              <p>${formData.monitoringProcedures}</p>
            ` : ''}

            ${formData.auditTrails ? `
              <h3>Audit Trails</h3>
              <p>${formData.auditTrails}</p>
            ` : ''}

            ${formData.updateProcesses ? `
              <h3>Update Processes</h3>
              <p>${formData.updateProcesses}</p>
            ` : ''}
          </div>
        </div>

        <div class="footer">
          <h3>Document Control Information</h3>
          <table>
            <tr><td class="label">Document Type:</td><td>BIM Execution Plan (BEP)</td></tr>
            <tr><td class="label">ISO Standard:</td><td>ISO 19650-2:2018</td></tr>
            <tr><td class="label">Document Status:</td><td>Work in Progress</td></tr>
            <tr><td class="label">Generated By:</td><td>Professional BEP Generator Tool</td></tr>
            <tr><td class="label">Generated Date:</td><td>${formattedDate}</td></tr>
            <tr><td class="label">Generated Time:</td><td>${formattedTime}</td></tr>
            <tr><td class="label">Next Review:</td><td>To be determined by project team</td></tr>
          </table>
          
          <div class="highlight">
            <strong>Important Notes:</strong>
            <ul>
              <li>This BEP is a living document that should be continuously updated throughout the project lifecycle</li>
              <li>All stakeholders must review and approve this document before project commencement</li>
              <li>Regular compliance checks should be performed against ISO 19650-2 requirements</li>
              <li>This document should be read in conjunction with the Exchange Information Requirements (EIR)</li>
            </ul>
          </div>
        </div>
      </body>
      </html>
    `;

    const blob = new Blob([bepContent], { 
      type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' 
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Professional_BEP_${formData.projectName || 'Project'}_${new Date().toISOString().split('T')[0]}.doc`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const renderStep = () => {
    switch (currentStep) {
      case 0:
        return (
          <div className="space-y-6">
            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
              <h3 className="text-lg font-semibold text-blue-900 mb-3">BEP Type Selection</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <label className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                  bepType === 'pre-appointment' ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
                }`}>
                  <input
                    type="radio"
                    value="pre-appointment"
                    checked={bepType === 'pre-appointment'}
                    onChange={(e) => setBepType(e.target.value)}
                    className="sr-only"
                  />
                  <div className="font-medium text-gray-900">Pre-Appointment BEP</div>
                  <div className="text-sm text-gray-600 mt-1">
                    Demonstrates capability and proposed approach during tender phase
                  </div>
                </label>
                
                <label className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                  bepType === 'post-appointment' ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
                }`}>
                  <input
                    type="radio"
                    value="post-appointment"
                    checked={bepType === 'post-appointment'}
                    onChange={(e) => setBepType(e.target.value)}
                    className="sr-only"
                  />
                  <div className="font-medium text-gray-900">Post-Appointment BEP</div>
                  <div className="text-sm text-gray-600 mt-1">
                    Confirms delivery approach with detailed planning and schedules
                  </div>
                </label>
              </div>
            </div>

            <h3 className="text-xl font-semibold">Project Information and Objectives</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Project Name *</label>
                <input
                  type="text"
                  value={formData.projectName}
                  onChange={(e) => updateFormData('projectName', e.target.value)}
                  className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Enter project name"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">Project Number</label>
                <input
                  type="text"
                  value={formData.projectNumber}
                  onChange={(e) => updateFormData('projectNumber', e.target.value)}
                  className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Enter project number"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Project Type *</label>
                <select
                  value={formData.projectType}
                  onChange={(e) => updateFormData('projectType', e.target.value)}
                  className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="">Select project type</option>
                  <option value="Commercial Building">Commercial Building</option>
                  <option value="Residential">Residential</option>
                  <option value="Infrastructure">Infrastructure</option>
                  <option value="Industrial">Industrial</option>
                  <option value="Healthcare">Healthcare</option>
                  <option value="Education">Education</option>
                  <option value="Mixed Use">Mixed Use</option>
                  <option value="Renovation/Retrofit">Renovation/Retrofit</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Appointing Party *</label>
                <input
                  type="text"
                  value={formData.appointingParty}
                  onChange={(e) => updateFormData('appointingParty', e.target.value)}
                  className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Client/Appointing party name"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Project Timeline</label>
                <input
                  type="text"
                  value={formData.projectTimeline}
                  onChange={(e) => updateFormData('projectTimeline', e.target.value)}
                  className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="e.g., 24 months (Jan 2025 - Dec 2026)"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Project Budget</label>
                <input
                  type="text"
                  value={formData.projectBudget}
                  onChange={(e) => updateFormData('projectBudget', e.target.value)}
                  className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="e.g., £5M - £10M"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Project Description</label>
              <textarea
                value={formData.projectDescription}
                onChange={(e) => updateFormData('projectDescription', e.target.value)}
                rows={4}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Provide a detailed project description including scope, objectives, and key requirements..."
              />
            </div>
          </div>
        );

      case 1:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold mb-4">Stakeholders and Responsibilities</h3>
            
            <div className="bg-amber-50 p-4 rounded-lg border border-amber-200">
              <h4 className="font-medium text-amber-900 mb-2">ISO 19650 Key Roles</h4>
              <div className="text-sm text-amber-800 space-y-1">
                <p><strong>Appointing Party:</strong> The client who appoints the delivery team</p>
                <p><strong>Lead Appointed Party:</strong> Main contractor responsible for delivery coordination</p>
                <p><strong>Information Manager:</strong> Manages information delivery and standards</p>
                <p><strong>Task Teams:</strong> Individual disciplines responsible for specific deliverables</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Lead Appointed Party *</label>
                <input
                  type="text"
                  value={formData.leadAppointedParty}
                  onChange={(e) => updateFormData('leadAppointedParty', e.target.value)}
                  className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Main contractor/Lead consultant"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Information Manager *</label>
                <input
                  type="text"
                  value={formData.informationManager}
                  onChange={(e) => updateFormData('informationManager', e.target.value)}
                  className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Information Manager name and organization"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Task Team Leaders</label>
              <textarea
                value={formData.taskTeamLeaders}
                onChange={(e) => updateFormData('taskTeamLeaders', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="List the task team leaders by discipline (Architecture, Structure, MEP, etc.) with their organizations and contact information..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Appointed Parties</label>
              <textarea
                value={formData.appointedParties}
                onChange={(e) => updateFormData('appointedParties', e.target.value)}
                rows={4}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="List all appointed parties (consultants, contractors, specialists) with their roles and responsibilities in the information delivery process..."
              />
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold mb-4">BIM Goals and Objectives</h3>
            
            <div>
              <label className="block text-sm font-medium mb-2">BIM Goals *</label>
              <textarea
                value={formData.bimGoals}
                onChange={(e) => updateFormData('bimGoals', e.target.value)}
                rows={4}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define the overall BIM goals for this project, aligning with project objectives and organizational requirements..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Primary Objectives</label>
              <textarea
                value={formData.primaryObjectives}
                onChange={(e) => updateFormData('primaryObjectives', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Outline the primary objectives that BIM implementation will support..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">BIM Uses *</label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-60 overflow-y-auto border rounded-lg p-3">
                {bimUsesOptions.map(use => (
                  <label key={use} className="flex items-center space-x-2 p-2 border rounded cursor-pointer hover:bg-gray-50">
                    <input
                      type="checkbox"
                      checked={formData.bimUses.includes(use)}
                      onChange={() => toggleArrayValue('bimUses', use)}
                      className="rounded"
                    />
                    <span className="text-sm">{use}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold mb-4">Level of Information Need (LOIN)</h3>
            
            <div className="bg-green-50 p-4 rounded-lg border border-green-200">
              <h4 className="font-medium text-green-900 mb-2">LOIN Framework</h4>
              <p className="text-sm text-green-800">
                LOIN specifies the information requirements according to purpose, content, form, and format for information exchange.
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Information Purposes *</label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-48 overflow-y-auto border rounded-lg p-3">
                {informationPurposesOptions.map(purpose => (
                  <label key={purpose} className="flex items-center space-x-2 p-2 border rounded cursor-pointer hover:bg-gray-50">
                    <input
                      type="checkbox"
                      checked={formData.informationPurposes.includes(purpose)}
                      onChange={() => toggleArrayValue('informationPurposes', purpose)}
                      className="rounded"
                    />
                    <span className="text-sm">{purpose}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Geometrical Information Requirements</label>
              <textarea
                value={formData.geometricalInfo}
                onChange={(e) => updateFormData('geometricalInfo', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Specify the geometrical information requirements including level of detail, accuracy, and dimensional tolerances..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Alphanumerical Information Requirements</label>
              <textarea
                value={formData.alphanumericalInfo}
                onChange={(e) => updateFormData('alphanumericalInfo', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define the alphanumerical data requirements including properties, parameters, and metadata..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Documentation Requirements</label>
              <textarea
                value={formData.documentationInfo}
                onChange={(e) => updateFormData('documentationInfo', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Specify documentation requirements including drawings, schedules, reports, and other deliverables..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Information Formats</label>
              <div className="grid grid-cols-3 md:grid-cols-4 gap-2 max-h-40 overflow-y-auto border rounded-lg p-3">
                {fileFormatsOptions.map(format => (
                  <label key={format} className="flex items-center space-x-2 p-2 border rounded cursor-pointer hover:bg-gray-50">
                    <input
                      type="checkbox"
                      checked={formData.informationFormats.includes(format)}
                      onChange={() => toggleArrayValue('informationFormats', format)}
                      className="rounded"
                    />
                    <span className="text-sm">{format}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        );

      case 4:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold mb-4">Information Delivery Planning</h3>
            
            <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
              <h4 className="font-medium text-purple-900 mb-2">Information Delivery Framework</h4>
              <p className="text-sm text-purple-800">
                The MIDP combines individual TIDPs into a comprehensive delivery schedule for all project information.
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Master Information Delivery Plan (MIDP) *</label>
              <textarea
                value={formData.midpDescription}
                onChange={(e) => updateFormData('midpDescription', e.target.value)}
                rows={4}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Describe the Master Information Delivery Plan including how it coordinates all task team deliverables and aligns with project milestones..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Key Information Delivery Milestones *</label>
              <textarea
                value={formData.keyMilestones}
                onChange={(e) => updateFormData('keyMilestones', e.target.value)}
                rows={4}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define key information delivery milestones aligned with project stages (RIBA stages, construction phases, etc.)..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Delivery Schedule</label>
              <textarea
                value={formData.deliverySchedule}
                onChange={(e) => updateFormData('deliverySchedule', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Outline the overall delivery schedule including timelines, dependencies, and critical path items..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Task Information Delivery Plans (TIDPs)</label>
              <textarea
                value={formData.tidpRequirements}
                onChange={(e) => updateFormData('tidpRequirements', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Describe how individual TIDPs will be developed and coordinated by each task team..."
              />
            </div>
          </div>
        );

      case 5:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold mb-4">Common Data Environment (CDE)</h3>
            
            <div className="bg-indigo-50 p-4 rounded-lg border border-indigo-200">
              <h4 className="font-medium text-indigo-900 mb-2">CDE Requirements</h4>
              <p className="text-sm text-indigo-800">
                The CDE provides a central platform for information creation, sharing, and archiving throughout the project lifecycle.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">CDE Provider *</label>
                <input
                  type="text"
                  value={formData.cdeProvider}
                  onChange={(e) => updateFormData('cdeProvider', e.target.value)}
                  className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="e.g., Autodesk BIM 360, Bentley ProjectWise"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">CDE Platform Version</label>
                <input
                  type="text"
                  value={formData.cdePlatform}
                  onChange={(e) => updateFormData('cdePlatform', e.target.value)}
                  className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Platform version and configuration"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Workflow States *</label>
              <textarea
                value={formData.workflowStates}
                onChange={(e) => updateFormData('workflowStates', e.target.value)}
                rows={4}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define the workflow states: Work in Progress (WIP), Shared, Published, and Archived. Include approval processes and responsibilities..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Access Control</label>
              <textarea
                value={formData.accessControl}
                onChange={(e) => updateFormData('accessControl', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define user roles, permissions, and access levels for different project stakeholders..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Security Measures</label>
              <textarea
                value={formData.securityMeasures}
                onChange={(e) => updateFormData('securityMeasures', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Describe security measures including authentication, encryption, and access monitoring..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Backup Procedures</label>
              <textarea
                value={formData.backupProcedures}
                onChange={(e) => updateFormData('backupProcedures', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define backup and disaster recovery procedures for information preservation..."
              />
            </div>
          </div>
        );

      case 6:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold mb-4">Technology and Software Requirements</h3>
            
            <div>
              <label className="block text-sm font-medium mb-2">BIM Software Applications *</label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-60 overflow-y-auto border rounded-lg p-3">
                {softwareOptions.map(software => (
                  <label key={software} className="flex items-center space-x-2 p-2 border rounded cursor-pointer hover:bg-gray-50">
                    <input
                      type="checkbox"
                      checked={formData.bimSoftware.includes(software)}
                      onChange={() => toggleArrayValue('bimSoftware', software)}
                      className="rounded"
                    />
                    <span className="text-sm">{software}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">File Formats *</label>
              <div className="grid grid-cols-3 md:grid-cols-4 gap-2 max-h-48 overflow-y-auto border rounded-lg p-3">
                {fileFormatsOptions.map(format => (
                  <label key={format} className="flex items-center space-x-2 p-2 border rounded cursor-pointer hover:bg-gray-50">
                    <input
                      type="checkbox"
                      checked={formData.fileFormats.includes(format)}
                      onChange={() => toggleArrayValue('fileFormats', format)}
                      className="rounded"
                    />
                    <span className="text-sm">{format}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Hardware Requirements</label>
              <textarea
                value={formData.hardwareRequirements}
                onChange={(e) => updateFormData('hardwareRequirements', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Specify minimum hardware requirements for BIM software, including CPU, RAM, graphics cards, and storage..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Network Requirements</label>
              <textarea
                value={formData.networkRequirements}
                onChange={(e) => updateFormData('networkRequirements', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define network bandwidth, connectivity, and performance requirements for collaborative working..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Interoperability Requirements</label>
              <textarea
                value={formData.interoperabilityNeeds}
                onChange={(e) => updateFormData('interoperabilityNeeds', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Describe interoperability requirements between different software platforms and file formats..."
              />
            </div>
          </div>
        );

      case 7:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold mb-4">Information Production Methods and Procedures</h3>
            
            <div>
              <label className="block text-sm font-medium mb-2">Modeling Standards *</label>
              <textarea
                value={formData.modelingStandards}
                onChange={(e) => updateFormData('modelingStandards', e.target.value)}
                rows={4}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define modeling standards including object libraries, classification systems (Uniclass, Omniclass), and modeling conventions..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Naming Conventions *</label>
              <textarea
                value={formData.namingConventions}
                onChange={(e) => updateFormData('namingConventions', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Specify file naming conventions, folder structures, and identifier systems (e.g., [Project]-[Originator]-[Volume]-[Level]-[Type]-[Role]-[Number])..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">File Structure</label>
              <textarea
                value={formData.fileStructure}
                onChange={(e) => updateFormData('fileStructure', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define the file and folder structure within the CDE, including organization by discipline, level, zone, etc..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Version Control</label>
              <textarea
                value={formData.versionControl}
                onChange={(e) => updateFormData('versionControl', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Describe version control procedures including revision numbering, change tracking, and approval processes..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Data Exchange Protocols</label>
              <textarea
                value={formData.dataExchangeProtocols}
                onChange={(e) => updateFormData('dataExchangeProtocols', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define data exchange protocols including frequency, formats, quality checks, and validation procedures..."
              />
            </div>
          </div>
        );

      case 8:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold mb-4">Quality Assurance and Control</h3>
            
            <div>
              <label className="block text-sm font-medium mb-2">Quality Assurance Framework *</label>
              <textarea
                value={formData.qaFramework}
                onChange={(e) => updateFormData('qaFramework', e.target.value)}
                rows={4}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define the overall QA framework including quality objectives, standards, and compliance requirements..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Model Validation Procedures *</label>
              <textarea
                value={formData.modelValidation}
                onChange={(e) => updateFormData('modelValidation', e.target.value)}
                rows={4}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Describe model validation procedures including automated checks, manual reviews, and validation tools..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Review Processes</label>
              <textarea
                value={formData.reviewProcesses}
                onChange={(e) => updateFormData('reviewProcesses', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define review processes including design reviews, coordination reviews, and quality checkpoints..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Approval Workflows</label>
              <textarea
                value={formData.approvalWorkflows}
                onChange={(e) => updateFormData('approvalWorkflows', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Describe approval workflows including authorization levels, sign-off procedures, and escalation processes..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Compliance Verification</label>
              <textarea
                value={formData.complianceVerification}
                onChange={(e) => updateFormData('complianceVerification', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define compliance verification methods including audits, checks against standards, and certification processes..."
              />
            </div>
          </div>
        );

      case 9:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold mb-4">Information Security and Privacy</h3>
            
            <div>
              <label className="block text-sm font-medium mb-2">Data Classification *</label>
              <textarea
                value={formData.dataClassification}
                onChange={(e) => updateFormData('dataClassification', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define data classification levels (Public, Internal, Confidential, Restricted) and handling requirements..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Access Permissions *</label>
              <textarea
                value={formData.accessPermissions}
                onChange={(e) => updateFormData('accessPermissions', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define access permission matrix including user roles, data access levels, and authorization procedures..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Encryption Requirements</label>
              <textarea
                value={formData.encryptionRequirements}
                onChange={(e) => updateFormData('encryptionRequirements', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Specify encryption requirements for data at rest and in transit, including encryption standards and key management..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Data Transfer Protocols</label>
              <textarea
                value={formData.dataTransferProtocols}
                onChange={(e) => updateFormData('dataTransferProtocols', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define secure data transfer protocols including approved methods, security checks, and audit trails..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Privacy Considerations</label>
              <textarea
                value={formData.privacyConsiderations}
                onChange={(e) => updateFormData('privacyConsiderations', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Address privacy considerations including GDPR compliance, personal data handling, and consent requirements..."
              />
            </div>
          </div>
        );

      case 10:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold mb-4">Training and Competency</h3>
            
            <div>
              <label className="block text-sm font-medium mb-2">BIM Competency Levels *</label>
              <textarea
                value={formData.bimCompetencyLevels}
                onChange={(e) => updateFormData('bimCompetencyLevels', e.target.value)}
                rows={4}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define required BIM competency levels for different roles including awareness, knowledge, and skill requirements..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Training Requirements</label>
              <textarea
                value={formData.trainingRequirements}
                onChange={(e) => updateFormData('trainingRequirements', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Specify training requirements including software training, BIM standards, and project-specific procedures..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Certification Requirements</label>
              <textarea
                value={formData.certificationNeeds}
                onChange={(e) => updateFormData('certificationNeeds', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define certification requirements including professional certifications, software certifications, and continuing education..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Project-Specific Training</label>
              <textarea
                value={formData.projectSpecificTraining}
                onChange={(e) => updateFormData('projectSpecificTraining', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Describe project-specific training including induction programs, tool training, and ongoing support..."
              />
            </div>
          </div>
        );

      case 11:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold mb-4">Coordination, Collaboration & Risk Management</h3>
            
            <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200 mb-6">
              <h4 className="font-medium text-yellow-900 mb-2">Final Section</h4>
              <p className="text-sm text-yellow-800">
                This section covers coordination procedures and risk management to complete your ISO 19650 compliant BEP.
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Coordination Meetings *</label>
              <textarea
                value={formData.coordinationMeetings}
                onChange={(e) => updateFormData('coordinationMeetings', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define coordination meeting schedules, participants, agendas, and decision-making processes..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Clash Detection Workflow</label>
              <textarea
                value={formData.clashDetectionWorkflow}
                onChange={(e) => updateFormData('clashDetectionWorkflow', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Describe clash detection procedures including frequency, tools, reporting, and resolution workflows..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Issue Resolution Process</label>
              <textarea
                value={formData.issueResolution}
                onChange={(e) => updateFormData('issueResolution', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define issue resolution processes including identification, tracking, escalation, and closure procedures..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Communication Protocols</label>
              <textarea
                value={formData.communicationProtocols}
                onChange={(e) => updateFormData('communicationProtocols', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Specify communication protocols including channels, frequency, reporting formats, and escalation procedures..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Model Federation Strategy</label>
              <textarea
                value={formData.federationStrategy}
                onChange={(e) => updateFormData('federationStrategy', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Describe model federation strategy including reference points, coordination methods, and integration procedures..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Information-Related Risks *</label>
              <textarea
                value={formData.informationRisks}
                onChange={(e) => updateFormData('informationRisks', e.target.value)}
                rows={4}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Identify information-related risks including data loss, security breaches, quality issues, and interoperability challenges..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Technology-Related Risks</label>
              <textarea
                value={formData.technologyRisks}
                onChange={(e) => updateFormData('technologyRisks', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Identify technology-related risks including software compatibility, hardware failures, and network issues..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Risk Mitigation Strategies</label>
              <textarea
                value={formData.riskMitigation}
                onChange={(e) => updateFormData('riskMitigation', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define risk mitigation strategies including preventive measures, monitoring procedures, and response plans..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Contingency Plans</label>
              <textarea
                value={formData.contingencyPlans}
                onChange={(e) => updateFormData('contingencyPlans', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Describe contingency plans including backup procedures, alternative workflows, and emergency protocols..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Performance Metrics and KPIs</label>
              <textarea
                value={formData.performanceMetrics}
                onChange={(e) => updateFormData('performanceMetrics', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define performance metrics and KPIs for monitoring BIM implementation success..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Monitoring Procedures</label>
              <textarea
                value={formData.monitoringProcedures}
                onChange={(e) => updateFormData('monitoringProcedures', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Describe monitoring procedures including progress tracking, compliance checking, and performance reviews..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Audit Trails</label>
              <textarea
                value={formData.auditTrails}
                onChange={(e) => updateFormData('auditTrails', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Define audit trail requirements including logging, documentation, and traceability procedures..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Update Processes</label>
              <textarea
                value={formData.updateProcesses}
                onChange={(e) => updateFormData('updateProcesses', e.target.value)}
                rows={3}
                className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Describe BEP update processes including review schedules, change management, and version control..."
              />
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  const isStepComplete = (stepIndex) => completedSections.has(stepIndex);
  const isStepValid = (stepIndex) => validateStep(stepIndex);

  const getCategoryColor = (category) => {
    switch (category) {
      case 'Commercial':
        return 'bg-blue-100 text-blue-800';
      case 'Management':
        return 'bg-green-100 text-green-800';
      case 'Technical':
        return 'bg-purple-100 text-purple-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2">
                <Zap className="w-8 h-8 text-blue-600" />
                <h1 className="text-2xl font-bold text-gray-900">Professional BEP Generator</h1>
              </div>
              <span className="text-sm text-gray-500">ISO 19650-2 Compliant</span>
            </div>
            <div className="text-sm text-gray-600">
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                {bepType === 'pre-appointment' ? 'Pre-Appointment BEP' : 'Post-Appointment BEP'}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Progress Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-sm p-6 sticky top-8">
              <h2 className="text-lg font-semibold mb-4">Progress Overview</h2>
              <div className="space-y-3">
                {steps.map((step, index) => (
                  <div
                    key={index}
                    className={`flex items-start space-x-3 p-3 rounded-lg cursor-pointer transition-colors
                      ${currentStep === index 
                        ? 'bg-blue-50 border border-blue-200' 
                        : isStepComplete(index)
                        ? 'bg-green-50 border border-green-200'
                        : 'hover:bg-gray-50'
                      }`}
                    onClick={() => setCurrentStep(index)}
                  >
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center
                      ${currentStep === index 
                        ? 'bg-blue-600 text-white' 
                        : isStepComplete(index)
                        ? 'bg-green-600 text-white'
                        : 'bg-gray-200 text-gray-600'
                      }`}>
                      {isStepComplete(index) ? (
                        <CheckCircle className="w-4 h-4" />
                      ) : (
                        step.icon
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className={`text-sm font-medium ${
                        currentStep === index ? 'text-blue-900' :
                        isStepComplete(index) ? 'text-green-900' : 'text-gray-900'
                      }`}>
                        {step.title}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">{step.description}</p>
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium mt-1 ${getCategoryColor(step.category)}`}>
                        {step.category}
                      </span>
                    </div>
                    {!isStepValid(index) && index !== currentStep && (
                      <AlertCircle className="w-4 h-4 text-orange-500 flex-shrink-0" />
                    )}
                  </div>
                ))}
              </div>

              <div className="mt-6 pt-4 border-t">
                <div className="text-sm text-gray-600 mb-2">
                  Completion: {Math.round((completedSections.size / steps.length) * 100)}%
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(completedSections.size / steps.length) * 100}%` }}
                  />
                </div>
              </div>

              <div className="mt-4 pt-4 border-t">
                <div className="text-xs text-gray-500 space-y-1">
                  <div className="flex justify-between">
                    <span>Commercial:</span>
                    <span>{steps.filter((s, i) => s.category === 'Commercial' && completedSections.has(i)).length}/{steps.filter(s => s.category === 'Commercial').length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Management:</span>
                    <span>{steps.filter((s, i) => s.category === 'Management' && completedSections.has(i)).length}/{steps.filter(s => s.category === 'Management').length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Technical:</span>
                    <span>{steps.filter((s, i) => s.category === 'Technical' && completedSections.has(i)).length}/{steps.filter(s => s.category === 'Technical').length}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-lg shadow-sm p-8">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">{steps[currentStep].title}</h2>
                  <p className="text-gray-600 mt-1">{steps[currentStep].description}</p>
                </div>
                <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getCategoryColor(steps[currentStep].category)}`}>
                  {steps[currentStep].category} Aspects
                </span>
              </div>

              {renderStep()}

              {/* Navigation */}
              <div className="flex justify-between items-center mt-8 pt-6 border-t">
                <button
                  onClick={prevStep}
                  disabled={currentStep === 0}
                  className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <ChevronLeft className="w-4 h-4" />
                  <span>Previous</span>
                </button>

                <div className="text-sm text-gray-500">
                  Step {currentStep + 1} of {steps.length}
                </div>

                <div className="flex space-x-3">
                  {currentStep === steps.length - 1 ? (
                    <button
                      onClick={generateProfessionalBEP}
                      className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-8 py-3 rounded-lg transition-all transform hover:scale-105 shadow-lg"
                    >
                      <Download className="w-5 h-5" />
                      <span>Generate Professional BEP</span>
                    </button>
                  ) : (
                    <button
                      onClick={nextStep}
                      className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors"
                    >
                      <span>Next</span>
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfessionalBEPGenerator;