import React from 'react';
import { Link } from 'react-router-dom';
import {
  FileText,
  BarChart3,
  ArrowRight,
  Zap,
  Users,
  Calendar,
  Download,
  TrendingUp,
  CheckCircle
} from 'lucide-react';

const HomePage = () => {
  const features = {
    bep: [
      { icon: FileText, title: 'Professional Templates', desc: 'Industry-standard BEP templates' },
      { icon: Zap, title: 'Smart Generation', desc: 'AI-powered content generation' },
      { icon: Download, title: 'Multi-format Export', desc: 'PDF, Word, Excel export options' },
      { icon: CheckCircle, title: 'Validation', desc: 'Built-in compliance checking' }
    ],
    tidp: [
      { icon: Users, title: 'Team Collaboration', desc: 'Multi-team TIDP management' },
      { icon: Calendar, title: 'Schedule Tracking', desc: 'Milestone and delivery tracking' },
      { icon: TrendingUp, title: 'Evolution Dashboard', desc: 'Project progress visualization' },
      { icon: BarChart3, title: 'MIDP Aggregation', desc: 'Automatic MIDP generation' }
    ]
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <div className="bg-gradient-to-br from-blue-600 via-blue-700 to-indigo-800 relative overflow-hidden">
        <div className="absolute inset-0 bg-black opacity-10"></div>
        <div className="absolute inset-0" style={{backgroundImage: 'url("data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%23ffffff" fill-opacity="0.1"%3E%3Ccircle cx="30" cy="30" r="2"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")'}}></div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 relative z-10">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-bold text-white mb-6">
              BEP Suite
            </h1>
            <p className="text-xl md:text-2xl text-blue-100 mb-8 max-w-3xl mx-auto">
              Complete solution for BIM Execution Plans and Information Delivery Planning
            </p>
          </div>
        </div>
      </div>

      {/* Products Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Two Powerful Products, One Integrated Suite
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Whether you need to create comprehensive BEP documents or manage complex information delivery workflows, we've got you covered.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* BEP Generator */}
          <div className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-300 transform hover:-translate-y-1 transition-transform duration-300">
            <div className="bg-gradient-to-r from-blue-500 to-blue-600 px-6 py-8">
              <div className="flex items-center text-white">
                <FileText className="w-8 h-8 mr-3" />
                <div>
                  <h3 className="text-2xl font-bold">BEP Generator</h3>
                  <p className="text-blue-100">Professional BIM Execution Plans</p>
                </div>
              </div>
            </div>

            <div className="p-6">
              <p className="text-gray-600 mb-6">
                Create comprehensive, professional BEP documents with our intelligent form-based wizard.
                Ensure compliance and consistency across all your projects.
              </p>

              <div className="space-y-4 mb-8">
                {features.bep.map((feature, index) => {
                  const IconComponent = feature.icon;
                  return (
                    <div key={index} className="flex items-start">
                      <div className="flex-shrink-0 mr-3">
                        <IconComponent className="w-5 h-5 text-blue-600 mt-0.5" />
                      </div>
                      <div>
                        <h4 className="font-medium text-gray-900">{feature.title}</h4>
                        <p className="text-gray-600 text-sm">{feature.desc}</p>
                      </div>
                    </div>
                  );
                })}
              </div>

              <Link
                to="/bep-generator"
                className="inline-flex items-center justify-center w-full px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 hover:scale-105 transition-all duration-200 shadow-md hover:shadow-lg"
              >
                Start Creating BEP
                <ArrowRight className="ml-2 w-4 h-4" />
              </Link>
            </div>
          </div>

          {/* TIDP/MIDP Manager */}
          <div className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-300 transform hover:-translate-y-1 transition-transform duration-300">
            <div className="bg-gradient-to-r from-green-500 to-green-600 px-6 py-8">
              <div className="flex items-center text-white">
                <BarChart3 className="w-8 h-8 mr-3" />
                <div>
                  <h3 className="text-2xl font-bold">TIDP/MIDP Manager</h3>
                  <p className="text-green-100">Information Delivery Planning</p>
                </div>
              </div>
            </div>

            <div className="p-6">
              <p className="text-gray-600 mb-6">
                Manage Task Information Delivery Plans and automatically generate Master Information
                Delivery Plans. Track progress and coordinate across multiple teams.
              </p>

              <div className="space-y-4 mb-8">
                {features.tidp.map((feature, index) => {
                  const IconComponent = feature.icon;
                  return (
                    <div key={index} className="flex items-start">
                      <div className="flex-shrink-0 mr-3">
                        <IconComponent className="w-5 h-5 text-green-600 mt-0.5" />
                      </div>
                      <div>
                        <h4 className="font-medium text-gray-900">{feature.title}</h4>
                        <p className="text-gray-600 text-sm">{feature.desc}</p>
                      </div>
                    </div>
                  );
                })}
              </div>

              <Link
                to="/tidp-midp"
                className="inline-flex items-center justify-center w-full px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-green-600 hover:bg-green-700 hover:scale-105 transition-all duration-200 shadow-md hover:shadow-lg"
              >
                Manage Information Delivery
                <ArrowRight className="ml-2 w-4 h-4" />
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Integration Section */}
      <div className="bg-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-gray-900 mb-6">
              Seamless Integration
            </h2>
            <p className="text-lg text-gray-600 mb-8 max-w-3xl mx-auto">
              Both products work together seamlessly. Reference your TIDP/MIDP plans directly in your BEP documents,
              and navigate between systems with ease.
            </p>

            <div className="flex items-center justify-center space-x-8">
              <div className="flex items-center space-x-3 bg-white px-6 py-3 rounded-lg shadow">
                <FileText className="w-6 h-6 text-blue-600" />
                <span className="font-medium text-gray-900">BEP Generator</span>
              </div>

              <div className="flex items-center space-x-2 text-gray-400">
                <ArrowRight className="w-5 h-5" />
                <span className="text-sm font-medium">Integrated</span>
                <ArrowRight className="w-5 h-5" />
              </div>

              <div className="flex items-center space-x-3 bg-white px-6 py-3 rounded-lg shadow">
                <BarChart3 className="w-6 h-6 text-green-600" />
                <span className="font-medium text-gray-900">TIDP/MIDP Manager</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-400">
            <p>&copy; 2024 BEP Suite. Professional BIM execution planning tools.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;