import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  FileText,
  BarChart3,
  ArrowRight,
  Zap,
  Users,
  Calendar,
  Download,
  TrendingUp,
  CheckCircle,
  Sparkles,
  ChevronDown,
  Play
} from 'lucide-react';

const HomePage = () => {
  const navigate = useNavigate();
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  const features = {
    bep: [
      { icon: Zap, title: 'AI Text Generation', desc: 'PyTorch LSTM/GRU model with 24+ field types', color: 'text-yellow-600' },
      { icon: FileText, title: 'Rich Text Editor', desc: 'TipTap editor with tables, images & formatting', color: 'text-blue-600' },
      { icon: Download, title: 'Professional Export', desc: 'High-quality PDF/DOCX with TOC & diagrams', color: 'text-green-600' },
      { icon: CheckCircle, title: 'Command Palette', desc: 'Quick navigation (Ctrl+K) & onboarding', color: 'text-purple-600' }
    ],
    tidp: [
      { icon: BarChart3, title: 'Auto MIDP Generation', desc: 'Consolidate TIDPs into master plans', color: 'text-teal-600' },
      { icon: TrendingUp, title: 'Evolution Dashboard', desc: 'Real-time progress tracking & analytics', color: 'text-orange-600' },
      { icon: Users, title: 'Batch Operations', desc: 'Excel/CSV import & bulk TIDP creation', color: 'text-indigo-600' },
      { icon: Calendar, title: 'Dependency Matrix', desc: 'Cross-team visualization & risk register', color: 'text-red-600' }
    ]
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100" data-page-uri="/">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        {/* Background with animated gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-600 via-blue-700 to-indigo-800">
          <div className="absolute inset-0 bg-black opacity-10"></div>
          <div className="absolute inset-0" style={{
            backgroundImage: 'url("data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%23ffffff" fill-opacity="0.1"%3E%3Ccircle cx="30" cy="30" r="2"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")'
          }}></div>
        </div>

        {/* Floating elements */}
        <div className="absolute top-20 left-10 w-20 h-20 bg-white opacity-10 rounded-full animate-pulse"></div>
        <div className="absolute top-40 right-20 w-16 h-16 bg-white opacity-10 rounded-full animate-pulse delay-1000"></div>
        <div className="absolute bottom-20 left-1/4 w-12 h-12 bg-white opacity-10 rounded-full animate-pulse delay-2000"></div>

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-16">
          <div className={`text-center transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
            <div className="inline-flex items-center px-3 py-1.5 rounded-full bg-white bg-opacity-10 backdrop-blur-sm border border-white border-opacity-20 mb-4">
              <Sparkles className="w-3.5 h-3.5 text-yellow-300 mr-1.5" />
              <span className="text-xs lg:text-sm font-medium text-white">ISO 19650-Compliant Information Management Platform</span>
            </div>

            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-4 leading-tight">
              BEP Suite
            </h1>
            <p className="text-base md:text-lg lg:text-xl text-blue-100 mb-6 max-w-4xl mx-auto leading-relaxed">
              Comprehensive platform for BIM Execution Plans and Information Delivery Planning.
              Designed for BIM Managers and Information Managers to streamline ISO 19650 compliance across projects.
            </p>

            <div className="flex flex-col sm:flex-row gap-3 justify-center items-center mb-8">
              <button
                onClick={() => navigate('/bep-generator')}
                className="group inline-flex items-center px-6 py-3 bg-white text-blue-600 font-semibold rounded-lg hover:bg-blue-50 transform hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                <Play className="w-4 h-4 mr-2" />
                Launch BEP Generator
                <ArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </button>

              <button
                onClick={() => navigate('/tidp-midp')}
                className="group inline-flex items-center px-6 py-3 border-2 border-white text-white font-semibold rounded-lg hover:bg-white hover:text-blue-600 transform hover:scale-105 transition-all duration-200"
              >
                TIDP/MIDP Manager
                <ArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>

            {/* Key Features Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 lg:gap-6 max-w-4xl mx-auto">
              <div className="text-center">
                <div className="text-2xl lg:text-3xl font-bold text-white mb-0.5">ISO 19650</div>
                <div className="text-blue-200 text-xs lg:text-sm">Full Compliance</div>
              </div>
              <div className="text-center">
                <div className="text-2xl lg:text-3xl font-bold text-white mb-0.5">AI-Powered</div>
                <div className="text-blue-200 text-xs lg:text-sm">Content Generation</div>
              </div>
              <div className="text-center">
                <div className="text-2xl lg:text-3xl font-bold text-white mb-0.5">Auto MIDP</div>
                <div className="text-blue-200 text-xs lg:text-sm">From TIDPs</div>
              </div>
              <div className="text-center">
                <div className="text-2xl lg:text-3xl font-bold text-white mb-0.5">Multi-Export</div>
                <div className="text-blue-200 text-xs lg:text-sm">PDF • DOCX • Excel</div>
              </div>
            </div>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
          <ChevronDown className="w-6 h-6 text-white opacity-70" />
        </div>
      </div>

      {/* Products Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-16">
        <div className="text-center mb-8 lg:mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Complete ISO 19650 Information Management Solution
          </h2>
          <p className="text-base lg:text-lg text-gray-600 max-w-3xl mx-auto leading-relaxed">
            From pre-appointment BEPs to comprehensive TIDP/MIDP coordination - everything you need
            to manage information requirements and deliverables across the project lifecycle.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
          {/* BEP Generator */}
          <div className="group bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1 border border-gray-100 overflow-hidden">
            <div className="bg-gradient-to-r from-blue-500 to-blue-600 p-5 lg:p-6 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-24 h-24 bg-white opacity-10 rounded-full -translate-y-12 translate-x-12"></div>
              <div className="relative z-10 flex items-center text-white">
                <div className="w-12 h-12 lg:w-14 lg:h-14 bg-white bg-opacity-20 rounded-xl flex items-center justify-center mr-3">
                  <FileText className="w-6 h-6 lg:w-7 lg:h-7" />
                </div>
                <div>
                  <h3 className="text-2xl lg:text-3xl font-bold mb-0.5">BEP Generator</h3>
                  <p className="text-blue-100 text-sm lg:text-base">Professional BIM Execution Plans</p>
                </div>
              </div>
            </div>

            <div className="p-5 lg:p-6">
              <p className="text-gray-600 mb-5 lg:mb-6 text-sm lg:text-base leading-relaxed">
                Generate ISO 19650-compliant BEPs with intelligent wizards for both pre-appointment and post-appointment phases.
                AI-assisted content generation helps you articulate information requirements, CDE workflows, and delivery strategies efficiently.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 lg:gap-5 mb-5 lg:mb-6">
                {features.bep.map((feature, index) => {
                  const IconComponent = feature.icon;
                  return (
                    <div key={index} className="flex items-start p-3 lg:p-3.5 rounded-lg bg-gray-50 hover:bg-blue-50 transition-colors duration-200">
                      <div className="flex-shrink-0 mr-3">
                        <div className={`w-8 h-8 lg:w-9 lg:h-9 rounded-lg bg-white flex items-center justify-center shadow-sm`}>
                          <IconComponent className={`w-4 h-4 lg:w-5 lg:h-5 ${feature.color}`} />
                        </div>
                      </div>
                      <div className="flex-1 min-w-0">
                        <h4 className="font-semibold text-gray-900 mb-0.5 text-sm lg:text-base">{feature.title}</h4>
                        <p className="text-gray-600 text-xs lg:text-sm leading-relaxed">{feature.desc}</p>
                      </div>
                    </div>
                  );
                })}
              </div>

              <button
                onClick={() => navigate('/bep-generator')}
                className="group inline-flex items-center justify-center w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white font-semibold rounded-xl hover:from-blue-700 hover:to-blue-800 transform hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                <span className="text-base lg:text-lg">Start Creating BEP</span>
                <ArrowRight className="ml-2 lg:ml-3 w-4 h-4 lg:w-5 lg:h-5 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          </div>

          {/* TIDP/MIDP Manager */}
          <div className="group bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1 border border-gray-100 overflow-hidden">
            <div className="bg-gradient-to-r from-green-500 to-green-600 p-5 lg:p-6 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-24 h-24 bg-white opacity-10 rounded-full -translate-y-12 translate-x-12"></div>
              <div className="relative z-10 flex items-center text-white">
                <div className="w-12 h-12 lg:w-14 lg:h-14 bg-white bg-opacity-20 rounded-xl flex items-center justify-center mr-3">
                  <BarChart3 className="w-6 h-6 lg:w-7 lg:h-7" />
                </div>
                <div>
                  <h3 className="text-2xl lg:text-3xl font-bold mb-0.5">TIDP/MIDP Manager</h3>
                  <p className="text-green-100 text-sm lg:text-base">Information Delivery Planning</p>
                </div>
              </div>
            </div>

            <div className="p-5 lg:p-6">
              <p className="text-gray-600 mb-5 lg:mb-6 text-sm lg:text-base leading-relaxed">
                Coordinate information delivery across task teams with comprehensive TIDP management.
                Automatically consolidate into MIDPs, track dependencies, manage LOINs, and maintain responsibility matrices aligned with ISO 19650 requirements.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 lg:gap-5 mb-5 lg:mb-6">
                {features.tidp.map((feature, index) => {
                  const IconComponent = feature.icon;
                  return (
                    <div key={index} className="flex items-start p-3 lg:p-3.5 rounded-lg bg-gray-50 hover:bg-green-50 transition-colors duration-200">
                      <div className="flex-shrink-0 mr-3">
                        <div className={`w-8 h-8 lg:w-9 lg:h-9 rounded-lg bg-white flex items-center justify-center shadow-sm`}>
                          <IconComponent className={`w-4 h-4 lg:w-5 lg:h-5 ${feature.color}`} />
                        </div>
                      </div>
                      <div className="flex-1 min-w-0">
                        <h4 className="font-semibold text-gray-900 mb-0.5 text-sm lg:text-base">{feature.title}</h4>
                        <p className="text-gray-600 text-xs lg:text-sm leading-relaxed">{feature.desc}</p>
                      </div>
                    </div>
                  );
                })}
              </div>

              <button
                onClick={() => navigate('/tidp-midp')}
                className="group inline-flex items-center justify-center w-full px-6 py-3 bg-gradient-to-r from-green-600 to-green-700 text-white font-semibold rounded-xl hover:from-green-700 hover:to-green-800 transform hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                <span className="text-base lg:text-lg">Manage Information Delivery</span>
                <ArrowRight className="ml-2 lg:ml-3 w-4 h-4 lg:w-5 lg:h-5 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Integration Section */}
      <div className="bg-gradient-to-r from-gray-900 to-gray-800 py-12 lg:py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-8 lg:mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Seamless Integration
            </h2>
            <p className="text-base lg:text-lg text-gray-300 mb-8 lg:mb-10 max-w-3xl mx-auto leading-relaxed">
              Both products work together seamlessly. Reference your TIDP/MIDP plans directly in your BEP documents,
              and navigate between systems with ease for maximum productivity.
            </p>

            <div className="flex flex-col md:flex-row items-center justify-center space-y-6 md:space-y-0 md:space-x-6">
              <div className="flex items-center space-x-3 bg-white bg-opacity-10 backdrop-blur-sm px-5 py-4 rounded-xl border border-white border-opacity-20">
                <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
                  <FileText className="w-5 h-5 text-white" />
                </div>
                <div className="text-left">
                  <div className="font-semibold text-white text-base lg:text-lg">BEP Generator</div>
                  <div className="text-blue-200 text-xs lg:text-sm">Create Professional BEPs</div>
                </div>
              </div>

              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-green-500 rounded-full flex items-center justify-center shadow-lg">
                  <ArrowRight className="w-5 h-5 text-white" />
                </div>
                <div className="text-center">
                  <div className="font-semibold text-white text-base lg:text-lg">Integrated</div>
                  <div className="text-gray-300 text-xs lg:text-sm">Seamless Workflow</div>
                </div>
                <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-blue-500 rounded-full flex items-center justify-center shadow-lg">
                  <ArrowRight className="w-5 h-5 text-white" />
                </div>
              </div>

              <div className="flex items-center space-x-3 bg-white bg-opacity-10 backdrop-blur-sm px-5 py-4 rounded-xl border border-white border-opacity-20">
                <div className="w-10 h-10 bg-green-500 rounded-lg flex items-center justify-center">
                  <BarChart3 className="w-5 h-5 text-white" />
                </div>
                <div className="text-left">
                  <div className="font-semibold text-white text-base lg:text-lg">TIDP/MIDP Manager</div>
                  <div className="text-green-200 text-xs lg:text-sm">Manage Information Delivery</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ISO 19650 Compliance Section */}
      <div className="bg-white py-12 lg:py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-8 lg:mb-10">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
              ISO 19650 Compliance Built-In
            </h2>
            <p className="text-base lg:text-lg text-gray-600 max-w-2xl mx-auto">
              Comprehensive implementation of ISO 19650-2:2018 requirements for information management
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-6 border-2 border-blue-200">
              <div className="w-12 h-12 bg-blue-500 rounded-lg flex items-center justify-center mb-4">
                <CheckCircle className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-bold text-gray-900 mb-2">Clause 5.1 & 5.3</h3>
              <p className="text-sm text-gray-700">Information management process and information requirements framework</p>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-6 border-2 border-green-200">
              <div className="w-12 h-12 bg-green-500 rounded-lg flex items-center justify-center mb-4">
                <Calendar className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-bold text-gray-900 mb-2">Clause 5.4</h3>
              <p className="text-sm text-gray-700">Information delivery planning with TIDP/MIDP framework</p>
            </div>

            <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-6 border-2 border-purple-200">
              <div className="w-12 h-12 bg-purple-500 rounded-lg flex items-center justify-center mb-4">
                <FileText className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-bold text-gray-900 mb-2">Clause 5.6 & 5.7</h3>
              <p className="text-sm text-gray-700">Information production methods and Common Data Environment workflows</p>
            </div>

            <div className="bg-gradient-to-br from-orange-50 to-orange-100 rounded-xl p-6 border-2 border-orange-200">
              <div className="w-12 h-12 bg-orange-500 rounded-lg flex items-center justify-center mb-4">
                <Users className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-bold text-gray-900 mb-2">Annex A</h3>
              <p className="text-sm text-gray-700">Responsibility matrices for information management activities and deliverables</p>
            </div>

            <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-xl p-6 border-2 border-indigo-200">
              <div className="w-12 h-12 bg-indigo-500 rounded-lg flex items-center justify-center mb-4">
                <TrendingUp className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-bold text-gray-900 mb-2">LOIN Management</h3>
              <p className="text-sm text-gray-700">Level of Information Need specification and tracking for all deliverables</p>
            </div>

            <div className="bg-gradient-to-br from-teal-50 to-teal-100 rounded-xl p-6 border-2 border-teal-200">
              <div className="w-12 h-12 bg-teal-500 rounded-lg flex items-center justify-center mb-4">
                <BarChart3 className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-bold text-gray-900 mb-2">Validation & QA</h3>
              <p className="text-sm text-gray-700">Quality gates, acceptance criteria, and comprehensive validation checks</p>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 py-12 lg:py-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Streamline Your Information Management
          </h2>
          <p className="text-base lg:text-lg text-blue-100 mb-6 leading-relaxed">
            Professional tools for BIM Managers and Information Managers to deliver ISO 19650-compliant projects with confidence.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <button
              onClick={() => navigate('/bep-generator')}
              className="inline-flex items-center px-6 py-3 bg-white text-blue-600 font-semibold rounded-xl hover:bg-blue-50 transform hover:scale-105 transition-all duration-200 shadow-lg"
            >
              <FileText className="w-4 h-4 mr-2" />
              Create BEP
            </button>
            <button
              onClick={() => navigate('/tidp-midp')}
              className="inline-flex items-center px-6 py-3 border-2 border-white text-white font-semibold rounded-xl hover:bg-white hover:text-blue-600 transform hover:scale-105 transition-all duration-200"
            >
              <BarChart3 className="w-4 h-4 mr-2" />
              Manage TIDP/MIDP
            </button>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-900 py-8 lg:py-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-2 mb-3">
              <div className="w-7 h-7 lg:w-8 lg:h-8 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-lg flex items-center justify-center">
                <FileText className="w-3.5 h-3.5 lg:w-4 lg:h-4 text-white" />
              </div>
              <span className="text-white font-bold text-lg lg:text-xl">BEP Suite</span>
            </div>
            <p className="text-gray-400 mb-4 lg:mb-5 max-w-md mx-auto text-sm lg:text-base">
              ISO 19650-compliant information management platform for BIM professionals in the AEC industry.
            </p>
            <div className="flex items-center justify-center space-x-4 lg:space-x-6 text-xs lg:text-sm text-gray-400 mb-4">
              <span className="text-gray-500">React 19 • PyTorch 2.6 • FastAPI</span>
              <span>•</span>
              <span className="text-gray-500">Version 2.0.0</span>
            </div>
            <div className="mt-4 lg:mt-5 pt-4 lg:pt-5 border-t border-gray-800">
              <p className="text-gray-500 text-xs lg:text-sm">
                © 2024-2025 BEP Suite. Professional ISO 19650 information management tools.
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;