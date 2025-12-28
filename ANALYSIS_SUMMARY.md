# Analysis Summary: Fields Missing aiPrompt in helpContentData.js

## Overview

A comprehensive analysis of `src/data/helpContentData.js` has identified all fields that are missing the `aiPrompt` property configuration needed for AI-powered content generation.

## Key Findings

- **File Size:** 5,614 lines (279 KB)
- **Total Fields:** 81
- **Fields WITH aiPrompt:** 21 (26%)
- **Fields WITHOUT aiPrompt:** 60 (74%) **[ACTION REQUIRED]**

## What is aiPrompt?

The `aiPrompt` property is a configuration object that enables AI-powered content generation for form fields. It consists of:

```javascript
aiPrompt: {
  system: 'Role/context for the AI',
  instructions: 'Specific guidance for content generation',
  style: 'Content style attributes'
}
```

Fields without this property cannot be auto-generated using AI and require manual entry.

## Generated Files

### 1. **fields_without_aiPrompt.json**
A structured JSON file containing all 60 fields missing aiPrompt, with:
- Field name
- Line number in the file
- Full description from the field definition

**Location:** `/bep-generator/fields_without_aiPrompt.json`

**Usage:** Import this JSON to programmatically add aiPrompt configurations or track which fields need updates.

### 2. **FIELDS_MISSING_AIPROMPT.md**
A comprehensive markdown document with:
- Summary statistics
- Complete list of all 60 fields organized by category
- Line numbers for easy navigation
- Field descriptions and purposes
- Quick reference table for all fields

**Location:** `/bep-generator/FIELDS_MISSING_AIPROMPT.md`

**Usage:** Review this document to understand which fields need aiPrompt added and their purpose.

## Field Categories

The 60 fields without aiPrompt are organized into 8 logical categories:

1. **Team & Organization** (15 fields)
   - Team composition, roles, contacts, and resource allocation

2. **BIM Uses & Value** (7 fields)
   - BIM application strategies and value metrics

3. **Information Requirements** (5 fields)
   - Geometrical, alphanumerical, and documentation requirements

4. **Information Delivery Planning** (10 fields)
   - Delivery schedules, milestones, and planning documents

5. **CDE & Infrastructure** (10 fields)
   - Common Data Environment setup and security

6. **Data Organization & Standards** (11 fields)
   - Naming conventions, file structures, and standards

7. **Quality & Security** (2 fields)
   - QA frameworks and data classification

## Next Steps

### To Add aiPrompt to Fields:

1. **Review the markdown document** (`FIELDS_MISSING_AIPROMPT.md`)
   - Understand each field's purpose
   - Review the existing examples in the field definitions

2. **Study existing aiPrompt patterns** in helpContentData.js
   - Look at fields that already have aiPrompt (lines 69-73, etc.)
   - Note the system prompt style and instruction patterns

3. **Create aiPrompt configurations**
   - For each field, create appropriate system prompts and instructions
   - Maintain consistency with existing patterns
   - Ensure instructions align with the field's purpose and examples

4. **Add to helpContentData.js**
   - Insert the aiPrompt object after other properties (before `relatedFields`)
   - Follow the existing formatting and structure

### Example Addition:

```javascript
// BEFORE (missing aiPrompt)
yourFieldName: {
  description: '...',
  iso19650: '...',
  bestPractices: [...],
  examples: {...},
  commonMistakes: [...],
  relatedFields: [...]
}

// AFTER (with aiPrompt)
yourFieldName: {
  description: '...',
  iso19650: '...',
  bestPractices: [...],
  examples: {...},
  commonMistakes: [...],

  aiPrompt: {
    system: 'You are a BIM expert...',
    instructions: 'Generate content that includes...',
    style: 'descriptive, technical, ISO 19650 compliant'
  },

  relatedFields: [...]
}
```

## File References

- **Main file being analyzed:** `c:\Users\andre\OneDrive\Documents\GitHub\bep-generator\src\data\helpContentData.js`
- **Output JSON:** `c:\Users\andre\OneDrive\Documents\GitHub\bep-generator\fields_without_aiPrompt.json`
- **Output Markdown:** `c:\Users\Andre\OneDrive\Documents\GitHub\bep-generator\FIELDS_MISSING_AIPROMPT.md`

## Statistics by Category

| Category | Count | Percentage |
|---|---|---|
| Team & Organization | 15 | 25% |
| Information Delivery Planning | 10 | 17% |
| CDE & Infrastructure | 10 | 17% |
| Data Organization & Standards | 11 | 18% |
| BIM Uses & Value | 7 | 12% |
| Information Requirements | 5 | 8% |
| Quality & Security | 2 | 3% |
| **TOTAL** | **60** | **100%** |

---

**Analysis completed:** 2025-12-28
**Total fields analyzed:** 81
**Action items identified:** 60
