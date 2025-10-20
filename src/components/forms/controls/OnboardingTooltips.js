import React, { useState, useEffect } from 'react';
import { X, Lightbulb, ArrowRight } from 'lucide-react';

const OnboardingTooltips = ({ onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    // Check if user has already seen the onboarding
    const hasSeenOnboarding = localStorage.getItem('mindmap-onboarding-seen');
    if (hasSeenOnboarding) {
      setDismissed(true);
    }
  }, []);

  const steps = [
    {
      title: 'Welcome to Volume Strategy Mindmap!',
      description: 'Let\'s quickly show you how to create amazing mindmaps.',
      position: 'center'
    },
    {
      title: 'Click to Select',
      description: 'Click on any node to select it. The selected node will be highlighted in red.',
      position: 'center'
    },
    {
      title: 'Quick Add Menu',
      description: 'Press Space or Shift+Click on a node to open the quick add menu and create child nodes instantly.',
      position: 'center'
    },
    {
      title: 'Edit Nodes',
      description: 'Double-click any node or press Enter to edit its name inline. No more clunky modals!',
      position: 'center'
    },
    {
      title: 'Right-Click Magic',
      description: 'Right-click on any node for a context menu with all available actions.',
      position: 'center'
    },
    {
      title: 'Keyboard Shortcuts',
      description: 'Press Ctrl+K to open the command palette and see all available shortcuts.',
      position: 'center'
    },
    {
      title: 'Drag & Drop',
      description: 'Drag nodes around to organize your mindmap. Use right-click + drag to pan the canvas.',
      position: 'center'
    }
  ];

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };

  const handleComplete = () => {
    localStorage.setItem('mindmap-onboarding-seen', 'true');
    setDismissed(true);
    if (onComplete) onComplete();
  };

  const handleSkip = () => {
    handleComplete();
  };

  if (dismissed) return null;

  const step = steps[currentStep];

  return (
    <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50 animate-fade-in">
      <div className="bg-white rounded-2xl shadow-2xl max-w-lg mx-4 animate-scale-in">
        {/* Header */}
        <div className="bg-gradient-to-r from-green-500 to-green-600 px-6 py-4 rounded-t-2xl">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-white bg-opacity-20 p-2 rounded-lg">
                <Lightbulb className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-bold text-white">{step.title}</h3>
            </div>
            <button
              onClick={handleSkip}
              className="text-white hover:bg-white hover:bg-opacity-20 rounded-lg p-2 transition-colors"
              aria-label="Close onboarding"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="px-6 py-8">
          <p className="text-gray-600 text-lg leading-relaxed">
            {step.description}
          </p>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-gray-50 rounded-b-2xl">
          <div className="flex items-center justify-between">
            <div className="flex space-x-2">
              {steps.map((_, index) => (
                <div
                  key={index}
                  className={`h-2 rounded-full transition-all duration-300 ${
                    index === currentStep
                      ? 'w-8 bg-green-500'
                      : index < currentStep
                      ? 'w-2 bg-green-300'
                      : 'w-2 bg-gray-300'
                  }`}
                />
              ))}
            </div>
            <div className="flex items-center space-x-3">
              <button
                onClick={handleSkip}
                className="px-4 py-2 text-gray-600 hover:text-gray-800 font-medium transition-colors"
              >
                Skip Tour
              </button>
              <button
                onClick={handleNext}
                className="px-6 py-2 bg-green-500 hover:bg-green-600 text-white font-medium rounded-lg flex items-center space-x-2 transition-colors"
              >
                <span>{currentStep < steps.length - 1 ? 'Next' : 'Get Started'}</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
          <div className="mt-3 text-center">
            <p className="text-sm text-gray-500">
              Step {currentStep + 1} of {steps.length}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OnboardingTooltips;
