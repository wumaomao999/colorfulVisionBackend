import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from werkzeug.security import generate_password_hash


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "vision_app.db"
DEFAULT_INVITE_CODE = "VISION2026"


def iso_days_ago(days: int, hour: int) -> str:
    target = datetime.now(timezone.utc) - timedelta(days=days)
    target = target.replace(hour=hour, minute=0, second=0, microsecond=0)
    return target.isoformat()


TEST_USERS = [
    {
        "username": "test_alice",
        "password": "Test123456",
        "records": [
            ("left", "4.8", 4.8, 2, 0, 1.41, iso_days_ago(0, 9)),
            ("right", "4.7", 4.7, 2, 1, 1.38, iso_days_ago(1, 21)),
            ("left", "4.9", 4.9, 2, 0, 1.43, iso_days_ago(3, 8)),
            ("right", "4.8", 4.8, 2, 0, 1.40, iso_days_ago(6, 20)),
        ],
    },
    {
        "username": "test_bob",
        "password": "Test123456",
        "records": [
            ("left", "4.6", 4.6, 2, 1, 1.36, iso_days_ago(0, 11)),
            ("right", "4.5", 4.5, 1, 2, 1.34, iso_days_ago(2, 19)),
            ("left", "4.7", 4.7, 2, 0, 1.39, iso_days_ago(5, 10)),
        ],
    },
    {
        "username": "test_cindy",
        "password": "Test123456",
        "records": [
            ("right", "5.0", 5.0, 2, 0, 1.44, iso_days_ago(0, 7)),
            ("left", "4.9", 4.9, 2, 0, 1.42, iso_days_ago(2, 7)),
            ("right", "4.9", 4.9, 2, 0, 1.41, iso_days_ago(4, 7)),
            ("left", "4.8", 4.8, 2, 1, 1.40, iso_days_ago(8, 7)),
            ("right", "5.0", 5.0, 2, 0, 1.43, iso_days_ago(12, 7)),
        ],
    },
]


def ensure_invite_code(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO invite_codes (code, description, is_active, created_at)
        VALUES (?, ?, 1, ?)
        """,
        (DEFAULT_INVITE_CODE, "Default referral code", datetime.now(timezone.utc).isoformat()),
    )


def get_or_create_user(conn: sqlite3.Connection, username: str, password: str) -> int:
    row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if row:
        return row[0]

    cursor = conn.execute(
        """
        INSERT INTO users (username, password_hash, referral_code_used, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            username,
            generate_password_hash(password),
            DEFAULT_INVITE_CODE,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    return cursor.lastrowid


def ensure_record(
    conn: sqlite3.Connection,
    user_id: int,
    eye: str,
    result_label: str,
    result_value: float,
    correct_count: int,
    wrong_count: int,
    detected_distance: float,
    created_at: str,
) -> None:
    existing = conn.execute(
        """
        SELECT id
        FROM vision_test_records
        WHERE user_id = ? AND eye = ? AND result_label = ? AND created_at = ?
        """,
        (user_id, eye, result_label, created_at),
    ).fetchone()
    if existing:
        return

    conn.execute(
        """
        INSERT INTO vision_test_records (
            user_id,
            eye,
            result_label,
            result_value,
            correct_count,
            wrong_count,
            detected_distance,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, eye, result_label, result_value, correct_count, wrong_count, detected_distance, created_at),
    )


def main() -> None:
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        ensure_invite_code(conn)

        created_users = 0
        inserted_records = 0

        for user in TEST_USERS:
            existing_before = conn.execute("SELECT id FROM users WHERE username = ?", (user["username"],)).fetchone()
            user_id = get_or_create_user(conn, user["username"], user["password"])
            if existing_before is None:
                created_users += 1

            for record in user["records"]:
                before = conn.total_changes
                ensure_record(conn, user_id, *record)
                if conn.total_changes > before:
                    inserted_records += 1

        conn.commit()
        print(f"Created users: {created_users}")
        print(f"Inserted records: {inserted_records}")
        print("Test credentials:")
        for user in TEST_USERS:
            print(f"  {user['username']} / {user['password']}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
