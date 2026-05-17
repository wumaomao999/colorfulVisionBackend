PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS invite_codes (
    code TEXT PRIMARY KEY,
    description TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    referral_code_used TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (referral_code_used) REFERENCES invite_codes(code)
);

CREATE TABLE IF NOT EXISTS auth_tokens (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_used_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS vision_test_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    eye TEXT NOT NULL CHECK (eye IN ('left', 'right')),
    result_label TEXT NOT NULL,
    result_value REAL NOT NULL,
    correct_count INTEGER NOT NULL DEFAULT 0,
    wrong_count INTEGER NOT NULL DEFAULT 0,
    detected_distance REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_auth_tokens_user_id ON auth_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_records_user_id_created_at ON vision_test_records(user_id, created_at DESC);

INSERT OR IGNORE INTO invite_codes (code, description, is_active, created_at)
VALUES ('VISION2026', 'Default referral code', 1, '2026-04-28T00:00:00+00:00');
