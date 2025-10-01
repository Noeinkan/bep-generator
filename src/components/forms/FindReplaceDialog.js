import React, { useState, useCallback } from 'react';
import { X, Search, Replace, ChevronDown, ChevronUp } from 'lucide-react';

const FindReplaceDialog = ({ editor, onClose }) => {
  const [findText, setFindText] = useState('');
  const [replaceText, setReplaceText] = useState('');
  const [matchCase, setMatchCase] = useState(false);
  const [matchCount, setMatchCount] = useState(0);
  const [currentMatch, setCurrentMatch] = useState(0);

  const highlightMatches = useCallback((searchTerm) => {
    if (!editor || !searchTerm) {
      setMatchCount(0);
      setCurrentMatch(0);
      return [];
    }

    const text = editor.getText();
    const flags = matchCase ? 'g' : 'gi';
    const regex = new RegExp(searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), flags);
    const matches = [...text.matchAll(regex)];

    setMatchCount(matches.length);
    return matches;
  }, [editor, matchCase]);

  const findNext = useCallback(() => {
    const matches = highlightMatches(findText);
    if (matches.length === 0) return;

    const nextIndex = (currentMatch + 1) % matches.length;
    setCurrentMatch(nextIndex);

    // Scroll to match (simplified - would need more work for actual positioning)
    // In a full implementation, you'd use ProseMirror's selection API
  }, [findText, currentMatch, highlightMatches]);

  const findPrevious = useCallback(() => {
    const matches = highlightMatches(findText);
    if (matches.length === 0) return;

    const prevIndex = currentMatch === 0 ? matches.length - 1 : currentMatch - 1;
    setCurrentMatch(prevIndex);
  }, [findText, currentMatch, highlightMatches]);

  const replaceOne = useCallback(() => {
    if (!editor || !findText) return;

    const { from, to } = editor.state.selection;
    const selectedText = editor.state.doc.textBetween(from, to);

    const textMatches = matchCase
      ? selectedText === findText
      : selectedText.toLowerCase() === findText.toLowerCase();

    if (textMatches) {
      editor.chain().focus().insertContentAt({ from, to }, replaceText).run();
      findNext();
    } else {
      findNext();
    }
  }, [editor, findText, replaceText, matchCase, findNext]);

  const replaceAll = useCallback(() => {
    if (!editor || !findText) return;

    const content = editor.getHTML();
    const flags = matchCase ? 'g' : 'gi';
    const regex = new RegExp(findText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), flags);
    const newContent = content.replace(regex, replaceText);

    editor.commands.setContent(newContent);
    setMatchCount(0);
    setCurrentMatch(0);
  }, [editor, findText, replaceText, matchCase]);

  const handleFindChange = (e) => {
    const value = e.target.value;
    setFindText(value);
    highlightMatches(value);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-2xl w-full max-w-md p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-800">Find & Replace</h3>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 rounded transition-colors"
            title="Close"
            type="button"
          >
            <X size={20} />
          </button>
        </div>

        <div className="space-y-4">
          {/* Find Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Find</label>
            <div className="relative">
              <input
                type="text"
                value={findText}
                onChange={handleFindChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Enter text to find..."
                autoFocus
              />
              <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
                <button
                  onClick={findPrevious}
                  disabled={matchCount === 0}
                  className="p-1 hover:bg-gray-100 rounded disabled:opacity-50"
                  title="Previous"
                  type="button"
                >
                  <ChevronUp size={16} />
                </button>
                <button
                  onClick={findNext}
                  disabled={matchCount === 0}
                  className="p-1 hover:bg-gray-100 rounded disabled:opacity-50"
                  title="Next"
                  type="button"
                >
                  <ChevronDown size={16} />
                </button>
              </div>
            </div>
            {findText && (
              <p className="text-xs text-gray-500 mt-1">
                {matchCount === 0 ? 'No matches found' : `${currentMatch + 1} of ${matchCount} matches`}
              </p>
            )}
          </div>

          {/* Replace Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Replace with</label>
            <input
              type="text"
              value={replaceText}
              onChange={(e) => setReplaceText(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              placeholder="Enter replacement text..."
            />
          </div>

          {/* Options */}
          <div className="flex items-center">
            <label className="flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={matchCase}
                onChange={(e) => {
                  setMatchCase(e.target.checked);
                  if (findText) highlightMatches(findText);
                }}
                className="rounded mr-2"
              />
              <span className="text-sm text-gray-700">Match case</span>
            </label>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2 pt-2">
            <button
              onClick={findNext}
              disabled={!findText || matchCount === 0}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              type="button"
            >
              <Search size={16} />
              Find Next
            </button>
            <button
              onClick={replaceOne}
              disabled={!findText || !replaceText || matchCount === 0}
              className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              type="button"
            >
              <Replace size={16} />
              Replace
            </button>
          </div>
          <button
            onClick={replaceAll}
            disabled={!findText || !replaceText || matchCount === 0}
            className="w-full px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed"
            type="button"
          >
            Replace All ({matchCount})
          </button>
        </div>
      </div>
    </div>
  );
};

export default FindReplaceDialog;
