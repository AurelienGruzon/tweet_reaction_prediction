import os
import sqlite3
from datetime import datetime
from typing import Optional

DB_PATH = os.getenv("FEEDBACK_DB_PATH", "feedback.db")

def _db_path() -> str:
    return os.getenv("FEEDBACK_DB_PATH", "feedback.db")


def _conn() -> sqlite3.Connection:
    # check_same_thread=False pour FastAPI multi-thread
    return sqlite3.connect(_db_path(), check_same_thread=False)


def init_db() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                text TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                true_label TEXT,
                proba_negative REAL,
                is_correct INTEGER NOT NULL
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS counters (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        con.execute("""
            INSERT OR IGNORE INTO counters(key, value, updated_at)
            VALUES('bad_streak', 0, ?);
        """, (datetime.utcnow().isoformat(),))


def add_feedback(
    *,
    text: str,
    predicted_label: str,
    proba_negative: Optional[float],
    is_correct: bool,
    true_label: Optional[str] = None,
) -> None:
    now = datetime.utcnow().isoformat()
    with _conn() as con:
        con.execute(
            """
            INSERT INTO feedback(created_at, text, predicted_label, true_label, proba_negative, is_correct)
            VALUES(?,?,?,?,?,?);
            """,
            (now, text, predicted_label, true_label, proba_negative, 1 if is_correct else 0),
        )


def update_bad_streak(is_correct: bool) -> int:
    now = datetime.utcnow().isoformat()
    with _conn() as con:
        cur = con.execute("SELECT value FROM counters WHERE key='bad_streak';")
        row = cur.fetchone()
        streak = int(row[0]) if row else 0

        streak = 0 if is_correct else streak + 1

        con.execute(
            "UPDATE counters SET value=?, updated_at=? WHERE key='bad_streak';",
            (streak, now),
        )

    return streak
