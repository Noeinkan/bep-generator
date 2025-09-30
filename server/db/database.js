const Database = require('better-sqlite3');
const path = require('path');
const fs = require('fs');

// Ensure the db directory exists
const dbDir = path.join(__dirname);
if (!fs.existsSync(dbDir)) {
  fs.mkdirSync(dbDir, { recursive: true });
}

const dbPath = path.join(__dirname, 'bep-generator.db');
const db = new Database(dbPath);

// Enable foreign keys
db.pragma('foreign_keys = ON');

// Create tables
db.exec(`
  CREATE TABLE IF NOT EXISTS tidps (
    id TEXT PRIMARY KEY,
    teamName TEXT NOT NULL,
    discipline TEXT NOT NULL,
    leader TEXT,
    company TEXT,
    responsibilities TEXT,
    projectId TEXT,
    createdAt TEXT NOT NULL,
    updatedAt TEXT NOT NULL,
    version TEXT DEFAULT '1.0',
    status TEXT DEFAULT 'Draft',
    source TEXT,
    createdVia TEXT
  );

  CREATE TABLE IF NOT EXISTS containers (
    id TEXT PRIMARY KEY,
    tidp_id TEXT NOT NULL,
    information_container_id TEXT,
    container_name TEXT,
    description TEXT,
    task_name TEXT,
    responsible_party TEXT,
    author TEXT,
    dependencies TEXT,
    loin TEXT,
    classification TEXT,
    estimated_time TEXT,
    delivery_milestone TEXT,
    due_date TEXT,
    format_type TEXT,
    purpose TEXT,
    acceptance_criteria TEXT,
    review_process TEXT,
    status TEXT,
    createdAt TEXT,
    FOREIGN KEY (tidp_id) REFERENCES tidps(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS midps (
    id TEXT PRIMARY KEY,
    projectName TEXT NOT NULL,
    modelUse TEXT NOT NULL,
    discipline TEXT NOT NULL,
    responsible TEXT,
    lod TEXT,
    milestone TEXT,
    dueDate TEXT,
    description TEXT,
    acceptanceCriteria TEXT,
    projectId TEXT,
    createdAt TEXT NOT NULL,
    updatedAt TEXT NOT NULL,
    version TEXT DEFAULT '1.0',
    status TEXT DEFAULT 'Draft'
  );

  CREATE INDEX IF NOT EXISTS idx_tidps_projectId ON tidps(projectId);
  CREATE INDEX IF NOT EXISTS idx_containers_tidp_id ON containers(tidp_id);
  CREATE INDEX IF NOT EXISTS idx_midps_projectId ON midps(projectId);
`);

console.log('Database initialized at:', dbPath);

module.exports = db;
