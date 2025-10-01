import React, { useState, useEffect } from 'react';
import { Calendar, Clock } from 'lucide-react';

const TimelineInput = ({ field, value, onChange, error }) => {
  const { name, label, required, placeholder } = field;

  // Parse existing value if it exists (e.g., "24 months (Jan 2025 - Dec 2026)")
  const parseExistingValue = (val) => {
    if (!val) return { startDate: '', endDate: '', notes: '' };

    const dateMatch = val.match(/\(([^-]+)-([^)]+)\)/);
    if (dateMatch) {
      const start = dateMatch[1].trim();
      const end = dateMatch[2].trim();

      // Try to parse dates like "Jan 2025" or "January 2025"
      const startDate = parseDateString(start);
      const endDate = parseDateString(end);

      return {
        startDate: startDate || '',
        endDate: endDate || '',
        notes: ''
      };
    }

    return { startDate: '', endDate: '', notes: val };
  };

  const parseDateString = (str) => {
    const months = {
      'jan': '01', 'january': '01',
      'feb': '02', 'february': '02',
      'mar': '03', 'march': '03',
      'apr': '04', 'april': '04',
      'may': '05',
      'jun': '06', 'june': '06',
      'jul': '07', 'july': '07',
      'aug': '08', 'august': '08',
      'sep': '09', 'september': '09',
      'oct': '10', 'october': '10',
      'nov': '11', 'november': '11',
      'dec': '12', 'december': '12'
    };

    const parts = str.trim().toLowerCase().split(' ');
    if (parts.length === 2) {
      const month = months[parts[0]];
      const year = parts[1];
      if (month && year.length === 4) {
        return `${year}-${month}`;
      }
    }

    return null;
  };

  const [timeline, setTimeline] = useState(() => parseExistingValue(value));

  useEffect(() => {
    if (!value && timeline.startDate === '' && timeline.endDate === '') {
      return;
    }
    if (value && !timeline.startDate && !timeline.endDate) {
      const parsed = parseExistingValue(value);
      setTimeline(parsed);
    }
  }, [value]);

  const formatDate = (dateString) => {
    if (!dateString) return '';
    const [year, month] = dateString.split('-');
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return `${monthNames[parseInt(month) - 1]} ${year}`;
  };

  const calculateDuration = (start, end) => {
    if (!start || !end) return null;

    const [startYear, startMonth] = start.split('-').map(Number);
    const [endYear, endMonth] = end.split('-').map(Number);

    const months = (endYear - startYear) * 12 + (endMonth - startMonth);

    if (months <= 0) return null;

    return months;
  };

  const generateFormattedText = (start, end, notes) => {
    if (!start && !end) return notes || '';

    const duration = calculateDuration(start, end);
    const startFormatted = formatDate(start);
    const endFormatted = formatDate(end);

    if (start && end && duration) {
      let text = `${duration} month${duration !== 1 ? 's' : ''} (${startFormatted} - ${endFormatted})`;
      if (notes) {
        text += ` - ${notes}`;
      }
      return text;
    } else if (start && end) {
      let text = `${startFormatted} - ${endFormatted}`;
      if (notes) {
        text += ` - ${notes}`;
      }
      return text;
    } else if (notes) {
      return notes;
    }

    return '';
  };

  const handleChange = (field, value) => {
    const newTimeline = { ...timeline, [field]: value };
    setTimeline(newTimeline);

    const formattedText = generateFormattedText(
      newTimeline.startDate,
      newTimeline.endDate,
      newTimeline.notes
    );

    onChange(name, formattedText);
  };

  const duration = calculateDuration(timeline.startDate, timeline.endDate);
  const previewText = generateFormattedText(timeline.startDate, timeline.endDate, timeline.notes);

  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium text-gray-700">
        {label} {required && <span className="text-red-500">*</span>}
      </label>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Start Date */}
        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">
            <Calendar className="inline-block w-3 h-3 mr-1" />
            Start Date
          </label>
          <input
            type="month"
            value={timeline.startDate}
            onChange={(e) => handleChange('startDate', e.target.value)}
            className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="Select start month"
          />
        </div>

        {/* End Date */}
        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">
            <Calendar className="inline-block w-3 h-3 mr-1" />
            End Date
          </label>
          <input
            type="month"
            value={timeline.endDate}
            onChange={(e) => handleChange('endDate', e.target.value)}
            className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="Select end month"
          />
        </div>
      </div>

      {/* Duration Display */}
      {duration !== null && duration > 0 && (
        <div className="flex items-center gap-2 px-3 py-2 bg-blue-50 border border-blue-200 rounded-lg">
          <Clock className="w-4 h-4 text-blue-600" />
          <span className="text-sm font-medium text-blue-700">
            Duration: {duration} month{duration !== 1 ? 's' : ''}
          </span>
        </div>
      )}

      {duration !== null && duration <= 0 && (
        <div className="px-3 py-2 bg-amber-50 border border-amber-200 rounded-lg">
          <span className="text-sm text-amber-700">
            End date must be after start date
          </span>
        </div>
      )}

      {/* Optional Notes */}
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">
          Additional Notes (Optional)
        </label>
        <input
          type="text"
          value={timeline.notes}
          onChange={(e) => handleChange('notes', e.target.value)}
          className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          placeholder="e.g., Including 2-month contingency"
        />
      </div>

      {/* Preview */}
      {previewText && (
        <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
          <label className="block text-xs font-medium text-gray-600 mb-2">
            Preview (will appear in document):
          </label>
          <p className="text-sm font-medium text-gray-900">
            {previewText}
          </p>
        </div>
      )}

      {/* Placeholder hint when empty */}
      {!previewText && placeholder && (
        <div className="text-xs text-gray-500 italic">
          Example: {placeholder}
        </div>
      )}

      {error && (
        <p className="text-sm text-red-600 mt-1">{error}</p>
      )}
    </div>
  );
};

export default TimelineInput;
