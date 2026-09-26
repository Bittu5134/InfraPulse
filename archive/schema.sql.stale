-- ==============================================================================
-- InfraPulse Relational Database DDL Schema
-- System: Photo-Based Infrastructure Defect & Priority Maintenance Platform
-- Target Database: SQLite 3 / PostgreSQL Compatible
-- ==============================================================================

-- 1. USERS TABLE
-- Stores public users who submit infrastructure defect complaints.
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(150) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    phone VARCHAR(50),
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- 2. STAFF TABLE
-- Stores maintenance squad staff members assigned to specific domain categories.
CREATE TABLE IF NOT EXISTS staff (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(150) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    domain VARCHAR(50) NOT NULL, -- 'Structural', 'Functional', or 'Performance'
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_staff_email ON staff(email);
CREATE INDEX IF NOT EXISTS idx_staff_domain ON staff(domain);

-- 3. ADMINS TABLE
-- Stores system administrators for staff oversight and ticket governance.
CREATE TABLE IF NOT EXISTS admins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(150) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_admins_email ON admins(email);

-- 4. COMPLAINTS (MAINTENANCE TICKETS) TABLE
-- Stores defect tickets with auto-calculated AI priority metrics and lifecycle states.
CREATE TABLE IF NOT EXISTS complaints (
    id BIGINT PRIMARY KEY, -- 10-Digit Random Reference Identifier
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    user_name VARCHAR(150) NOT NULL,
    user_email VARCHAR(255) NOT NULL,
    user_phone VARCHAR(50),
    address TEXT NOT NULL,
    description TEXT NOT NULL,
    photo_path VARCHAR(500) NOT NULL,
    category VARCHAR(50), -- 'Structural', 'Functional', 'Performance'
    defect_name VARCHAR(100), -- 'Spalling', 'Stagnant Water', 'Cracked Tiles', 'Paint Peeling'
    severity REAL DEFAULT 5.0, -- Scale [1.0, 10.0]
    extent REAL DEFAULT 20.0, -- Coverage Percentage [0.0%, 100.0%]
    priority_score REAL DEFAULT 0.0, -- Formulated Priority Score
    assigned_staff_id INTEGER REFERENCES staff(id) ON DELETE SET NULL,
    assigned_staff_name VARCHAR(150),
    status VARCHAR(50) DEFAULT 'Submitted', -- 'Submitted', 'Assigned', 'In Progress', 'Resolved'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_complaints_category ON complaints(category);
CREATE INDEX IF NOT EXISTS idx_complaints_status ON complaints(status);
CREATE INDEX IF NOT EXISTS idx_complaints_priority ON complaints(priority_score DESC);
CREATE INDEX IF NOT EXISTS idx_complaints_user_id ON complaints(user_id);
CREATE INDEX IF NOT EXISTS idx_complaints_assigned_staff ON complaints(assigned_staff_id);

-- 5. TICKET COMMENTS TABLE
-- Stores activity timeline and live communication messages per ticket.
CREATE TABLE IF NOT EXISTS ticket_comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id BIGINT NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
    sender_name VARCHAR(150) NOT NULL,
    sender_role VARCHAR(50) NOT NULL, -- 'User', 'Staff', 'Admin'
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_comments_ticket_id ON ticket_comments(ticket_id);

-- 6. NOTIFICATIONS TABLE
-- Stores in-app real-time notification alerts for assignment and status events.
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    staff_id INTEGER REFERENCES staff(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    link_url VARCHAR(500),
    is_read BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_notif_user ON notifications(user_id, is_read);
CREATE INDEX IF NOT EXISTS idx_notif_staff ON notifications(staff_id, is_read);
