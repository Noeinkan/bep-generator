const express = require('express');
const router = express.Router();
const tidpService = require('../services/tidpService');

/**
 * POST /api/migrate/tidps
 * Migrate TIDPs from localStorage to database
 */
router.post('/tidps', async (req, res) => {
  try {
    const { tidps } = req.body;

    if (!tidps || !Array.isArray(tidps)) {
      return res.status(400).json({
        success: false,
        message: 'Invalid request: tidps array is required'
      });
    }

    const results = {
      migrated: [],
      skipped: [],
      failed: []
    };

    tidps.forEach(draft => {
      try {
        // Check if already exists
        try {
          const existing = tidpService.getTIDP(draft.id);
          if (existing) {
            results.skipped.push({
              id: draft.id,
              teamName: draft.teamName,
              reason: 'Already exists in database'
            });
            return;
          }
        } catch (err) {
          // TIDP doesn't exist, continue with migration
        }

        // Create TIDP
        const tidpData = {
          teamName: draft.teamName || 'Untitled Team',
          discipline: draft.discipline || 'general',
          leader: draft.leader || '',
          company: draft.company || '',
          responsibilities: draft.responsibilities || '',
          projectId: draft.projectId || 'migrated-project',
          containers: draft.containers || []
        };

        const created = tidpService.createTIDP(tidpData);
        results.migrated.push({
          id: created.id,
          teamName: created.teamName,
          containers: created.containers?.length || 0
        });
      } catch (error) {
        results.failed.push({
          id: draft.id,
          teamName: draft.teamName,
          error: error.message
        });
      }
    });

    res.json({
      success: true,
      message: `Migration complete: ${results.migrated.length} migrated, ${results.skipped.length} skipped, ${results.failed.length} failed`,
      results
    });
  } catch (error) {
    console.error('Migration error:', error);
    res.status(500).json({
      success: false,
      message: 'Migration failed',
      error: error.message
    });
  }
});

module.exports = router;
