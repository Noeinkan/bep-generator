import React, { useState, useEffect } from 'react';
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
  CheckCircle,
  Star,
  Shield,
  Clock,
  Sparkles,
  ChevronDown,
  Play
} from 'lucide-react';

const HomePage = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [activeTestimonial, setActiveTestimonial] = useState(0);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  const features = {
    bep: [
      { icon: FileText, title: 'Professional Templates', desc: 'Industry-standard BEP templates', color: 'text-blue-600' },
      { icon: Zap, title: 'Smart Generation', desc: 'AI-powered content generation', color: 'text-yellow-600' },
      { icon: Download, title: 'Multi-format Export', desc: 'PDF, Word, Excel export options', color: 'text-green-600' },
      { icon: CheckCircle, title: 'Validation', desc: 'Built-in compliance checking', color: 'text-purple-600' }
    ],
    tidp: [
      { icon: Users, title: 'Team Collaboration', desc: 'Multi-team TIDP management', color: 'text-indigo-600' },
      { icon: Calendar, title: 'Schedule Tracking', desc: 'Milestone and delivery tracking', color: 'text-red-600' },
      { icon: TrendingUp, title: 'Evolution Dashboard', desc: 'Project progress visualization', color: 'text-orange-600' },
      { icon: BarChart3, title: 'MIDP Aggregation', desc: 'Automatic MIDP generation', color: 'text-teal-600' }
    ]
  };

  const testimonials = [
    {
      name: "Sarah Johnson",
      role: "BIM Manager",
      company: "ArchDesign Studios",
      content: "BEP Suite transformed our BIM workflow. The professional templates and validation features ensure compliance across all our projects.",
      avatar: "SJ"
    },
    {
      name: "Michael Chen",
      role: "Project Director",
      company: "Urban Construction Co.",
      content: "The integration between BEP Generator and TIDP/MIDP Manager is seamless. It saves us hours of manual coordination work.",
      avatar: "MC"
    },
    {
      name: "Emma Rodriguez",
      role: "Digital Lead",
      company: "FutureBuild Ltd",
      content: "Outstanding tool for BIM execution planning. The AI-powered content generation and multi-format export capabilities are game-changing.",
      avatar: "ER"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
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

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-32">
          <div className={`text-center transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
            <div className="inline-flex items-center px-4 py-2 rounded-full bg-white bg-opacity-10 backdrop-blur-sm border border-white border-opacity-20 mb-6">
              <Sparkles className="w-4 h-4 text-yellow-300 mr-2" />
              <span className="text-sm font-medium text-white">Professional BIM Execution Planning</span>
            </div>

            <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
              BEP Suite
            </h1>
            <p className="text-xl md:text-2xl text-blue-100 mb-8 max-w-4xl mx-auto leading-relaxed">
              Complete solution for BIM Execution Plans and Information Delivery Planning.
              Streamline your BIM workflow with professional tools designed for modern construction teams.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
              <Link
                to="/bep-generator"
                className="group inline-flex items-center px-8 py-4 bg-white text-blue-600 font-semibold rounded-lg hover:bg-blue-50 transform hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                <Play className="w-5 h-5 mr-2" />
                Start Creating BEP
                <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>

              <Link
                to="/tidp-midp"
                className="group inline-flex items-center px-8 py-4 border-2 border-white text-white font-semibold rounded-lg hover:bg-white hover:text-blue-600 transform hover:scale-105 transition-all duration-200"
              >
                Manage Information Delivery
                <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto">
              <div className="text-center">
                <div className="text-3xl font-bold text-white mb-1">500+</div>
                <div className="text-blue-200 text-sm">Projects Completed</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-white mb-1">50+</div>
                <div className="text-blue-200 text-sm">Team Members</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-white mb-1">99%</div>
                <div className="text-blue-200 text-sm">Compliance Rate</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-white mb-1">24/7</div>
                <div className="text-blue-200 text-sm">Support</div>
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
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            Two Powerful Products, One Integrated Suite
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Whether you need to create comprehensive BEP documents or manage complex information delivery workflows,
            we've got you covered with industry-leading BIM tools.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12">
          {/* BEP Generator */}
          <div className="group bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border border-gray-100 overflow-hidden">
            <div className="bg-gradient-to-r from-blue-500 to-blue-600 p-8 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-white opacity-10 rounded-full -translate-y-16 translate-x-16"></div>
              <div className="relative z-10 flex items-center text-white">
                <div className="w-16 h-16 bg-white bg-opacity-20 rounded-xl flex items-center justify-center mr-4">
                  <FileText className="w-8 h-8" />
                </div>
                <div>
                  <h3 className="text-3xl font-bold mb-1">BEP Generator</h3>
                  <p className="text-blue-100 text-lg">Professional BIM Execution Plans</p>
                </div>
              </div>
            </div>

            <div className="p-8">
              <p className="text-gray-600 mb-8 text-lg leading-relaxed">
                Create comprehensive, professional BEP documents with our intelligent form-based wizard.
                Ensure compliance and consistency across all your projects with automated validation.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-8">
                {features.bep.map((feature, index) => {
                  const IconComponent = feature.icon;
                  return (
                    <div key={index} className="flex items-start p-4 rounded-lg bg-gray-50 hover:bg-blue-50 transition-colors duration-200">
                      <div className="flex-shrink-0 mr-4">
                        <div className={`w-10 h-10 rounded-lg bg-white flex items-center justify-center shadow-sm`}>
                          <IconComponent className={`w-5 h-5 ${feature.color}`} />
                        </div>
                      </div>
                      <div className="flex-1 min-w-0">
                        <h4 className="font-semibold text-gray-900 mb-1">{feature.title}</h4>
                        <p className="text-gray-600 text-sm leading-relaxed">{feature.desc}</p>
                      </div>
                    </div>
                  );
                })}
              </div>

              <Link
                to="/bep-generator"
                className="group inline-flex items-center justify-center w-full px-8 py-4 bg-gradient-to-r from-blue-600 to-blue-700 text-white font-semibold rounded-xl hover:from-blue-700 hover:to-blue-800 transform hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                <span className="text-lg">Start Creating BEP</span>
                <ArrowRight className="ml-3 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>
          </div>

          {/* TIDP/MIDP Manager */}
          <div className="group bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border border-gray-100 overflow-hidden">
            <div className="bg-gradient-to-r from-green-500 to-green-600 p-8 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-white opacity-10 rounded-full -translate-y-16 translate-x-16"></div>
              <div className="relative z-10 flex items-center text-white">
                <div className="w-16 h-16 bg-white bg-opacity-20 rounded-xl flex items-center justify-center mr-4">
                  <BarChart3 className="w-8 h-8" />
                </div>
                <div>
                  <h3 className="text-3xl font-bold mb-1">TIDP/MIDP Manager</h3>
                  <p className="text-green-100 text-lg">Information Delivery Planning</p>
                </div>
              </div>
            </div>

            <div className="p-8">
              <p className="text-gray-600 mb-8 text-lg leading-relaxed">
                Manage Task Information Delivery Plans and automatically generate Master Information
                Delivery Plans. Track progress and coordinate across multiple teams with ease.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-8">
                {features.tidp.map((feature, index) => {
                  const IconComponent = feature.icon;
                  return (
                    <div key={index} className="flex items-start p-4 rounded-lg bg-gray-50 hover:bg-green-50 transition-colors duration-200">
                      <div className="flex-shrink-0 mr-4">
                        <div className={`w-10 h-10 rounded-lg bg-white flex items-center justify-center shadow-sm`}>
                          <IconComponent className={`w-5 h-5 ${feature.color}`} />
                        </div>
                      </div>
                      <div className="flex-1 min-w-0">
                        <h4 className="font-semibold text-gray-900 mb-1">{feature.title}</h4>
                        <p className="text-gray-600 text-sm leading-relaxed">{feature.desc}</p>
                      </div>
                    </div>
                  );
                })}
              </div>

              <Link
                to="/tidp-midp"
                className="group inline-flex items-center justify-center w-full px-8 py-4 bg-gradient-to-r from-green-600 to-green-700 text-white font-semibold rounded-xl hover:from-green-700 hover:to-green-800 transform hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                <span className="text-lg">Manage Information Delivery</span>
                <ArrowRight className="ml-3 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Integration Section */}
      <div className="bg-gradient-to-r from-gray-900 to-gray-800 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Seamless Integration
            </h2>
            <p className="text-xl text-gray-300 mb-12 max-w-3xl mx-auto leading-relaxed">
              Both products work together seamlessly. Reference your TIDP/MIDP plans directly in your BEP documents,
              and navigate between systems with ease for maximum productivity.
            </p>

            <div className="flex flex-col md:flex-row items-center justify-center space-y-8 md:space-y-0 md:space-x-8">
              <div className="flex items-center space-x-4 bg-white bg-opacity-10 backdrop-blur-sm px-8 py-6 rounded-2xl border border-white border-opacity-20">
                <div className="w-12 h-12 bg-blue-500 rounded-xl flex items-center justify-center">
                  <FileText className="w-6 h-6 text-white" />
                </div>
                <div className="text-left">
                  <div className="font-semibold text-white text-lg">BEP Generator</div>
                  <div className="text-blue-200 text-sm">Create Professional BEPs</div>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-green-500 rounded-full flex items-center justify-center shadow-lg">
                  <ArrowRight className="w-6 h-6 text-white" />
                </div>
                <div className="text-center">
                  <div className="font-semibold text-white text-lg">Integrated</div>
                  <div className="text-gray-300 text-sm">Seamless Workflow</div>
                </div>
                <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-blue-500 rounded-full flex items-center justify-center shadow-lg">
                  <ArrowRight className="w-6 h-6 text-white" />
                </div>
              </div>

              <div className="flex items-center space-x-4 bg-white bg-opacity-10 backdrop-blur-sm px-8 py-6 rounded-2xl border border-white border-opacity-20">
                <div className="w-12 h-12 bg-green-500 rounded-xl flex items-center justify-center">
                  <BarChart3 className="w-6 h-6 text-white" />
                </div>
                <div className="text-left">
                  <div className="font-semibold text-white text-lg">TIDP/MIDP Manager</div>
                  <div className="text-green-200 text-sm">Manage Information Delivery</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Testimonials Section */}
      <div className="bg-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Trusted by BIM Professionals
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              See what industry leaders say about BEP Suite
            </p>
          </div>

          <div className="max-w-4xl mx-auto">
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-8 md:p-12 shadow-lg">
              <div className="text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full flex items-center justify-center mx-auto mb-6">
                  <span className="text-white font-bold text-xl">
                    {testimonials[activeTestimonial].avatar}
                  </span>
                </div>
                <blockquote className="text-xl md:text-2xl text-gray-900 font-medium mb-6 leading-relaxed">
                  "{testimonials[activeTestimonial].content}"
                </blockquote>
                <div className="text-center">
                  <div className="font-semibold text-gray-900 text-lg">{testimonials[activeTestimonial].name}</div>
                  <div className="text-gray-600">{testimonials[activeTestimonial].role}, {testimonials[activeTestimonial].company}</div>
                </div>
              </div>

              <div className="flex justify-center space-x-2 mt-8">
                {testimonials.map((_, index) => (
                  <button
                    key={index}
                    onClick={() => setActiveTestimonial(index)}
                    className={`w-3 h-3 rounded-full transition-all duration-200 ${
                      index === activeTestimonial
                        ? 'bg-blue-500 scale-125'
                        : 'bg-gray-300 hover:bg-gray-400'
                    }`}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Ready to Transform Your BIM Workflow?
          </h2>
          <p className="text-xl text-blue-100 mb-8 leading-relaxed">
            Join thousands of BIM professionals who trust BEP Suite for their execution planning needs.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/bep-generator"
              className="inline-flex items-center px-8 py-4 bg-white text-blue-600 font-semibold rounded-xl hover:bg-blue-50 transform hover:scale-105 transition-all duration-200 shadow-lg"
            >
              <FileText className="w-5 h-5 mr-2" />
              Start Free Trial
            </Link>
            <Link
              to="/tidp-midp"
              className="inline-flex items-center px-8 py-4 border-2 border-white text-white font-semibold rounded-xl hover:bg-white hover:text-blue-600 transform hover:scale-105 transition-all duration-200"
            >
              <BarChart3 className="w-5 h-5 mr-2" />
              Explore Features
            </Link>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-900 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-2 mb-4">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-lg flex items-center justify-center">
                <FileText className="w-4 h-4 text-white" />
              </div>
              <span className="text-white font-bold text-xl">BEP Suite</span>
            </div>
            <p className="text-gray-400 mb-6 max-w-md mx-auto">
              Professional BIM execution planning tools designed for modern construction teams.
            </p>
            <div className="flex items-center justify-center space-x-6 text-sm text-gray-400">
              <a href="#" className="hover:text-white transition-colors">Privacy Policy</a>
              <span>•</span>
              <a href="#" className="hover:text-white transition-colors">Terms of Service</a>
              <span>•</span>
              <a href="#" className="hover:text-white transition-colors">Support</a>
            </div>
            <div className="mt-6 pt-6 border-t border-gray-800">
              <p className="text-gray-500 text-sm">
                © 2024 BEP Suite. All rights reserved. Professional BIM execution planning tools.
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;