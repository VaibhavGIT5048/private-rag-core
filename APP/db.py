"""SQLite metadata store: users, linked auth identities, OTP codes, documents,
and chat history. Lives on the same Azure Files volume as the rest of data/,
so no new Azure resource. Single-writer is fine given maxReplicas: 1.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

DB_PATH = Path("data/rag.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    email_verified INTEGER NOT NULL DEFAULT 0,
    password_hash TEXT,
    created_at TEXT NOT NULL,
    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
    privacy_policy_version TEXT,
    privacy_policy_accepted_at TEXT
);

CREATE TABLE IF NOT EXISTS auth_identities (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    provider TEXT NOT NULL,
    provider_user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (provider, provider_user_id)
);

CREATE TABLE IF NOT EXISTS otp_codes (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    code_hash TEXT NOT NULL,
    purpose TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    consumed_at TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL REFERENCES users(id),
    filename TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    pages INTEGER,
    chunks INTEGER,
    indexed_chunks INTEGER
);

CREATE TABLE IF NOT EXISTS chat_turns (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    owner_id TEXT NOT NULL REFERENCES users(id),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources_json TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_auth_identities_user ON auth_identities(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id);
CREATE INDEX IF NOT EXISTS idx_chat_turns_document ON chat_turns(document_id, owner_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # Not WAL: data/ is an Azure Files (SMB) mount, and WAL needs shared-memory
    # mmap plus byte-range locks that SMB doesn't provide — every connection
    # dies with "database is locked". Single-writer holds given maxReplicas: 1.
    conn.execute("PRAGMA journal_mode = DELETE")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# Columns added after the table already existed in deployed databases —
# CREATE TABLE IF NOT EXISTS above never retrofits an existing table, so each
# one needs its own ALTER TABLE here. Failing with "duplicate column" is the
# expected outcome once a database has already picked a migration up.
_MIGRATIONS = (
    "ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE users ADD COLUMN locked_until TEXT",
    "ALTER TABLE users ADD COLUMN privacy_policy_version TEXT",
    "ALTER TABLE users ADD COLUMN privacy_policy_accepted_at TEXT",
)


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)
        for migration in _MIGRATIONS:
            try:
                conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # column already exists


# --------------------------------------------------------------------------- #
# Users / auth identities
# --------------------------------------------------------------------------- #

def get_user_by_email(email: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()


def get_user_by_id(user_id: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def get_user_by_identity(provider: str, provider_user_id: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute(
            """
            SELECT users.* FROM users
            JOIN auth_identities ON auth_identities.user_id = users.id
            WHERE auth_identities.provider = ? AND auth_identities.provider_user_id = ?
            """,
            (provider, provider_user_id),
        ).fetchone()


def create_user(email: str, *, password_hash: str | None = None, email_verified: bool = False) -> str:
    user_id = _new_id()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (id, email, email_verified, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, email, int(email_verified), password_hash, _now()),
        )
    return user_id


def set_email_verified(user_id: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (user_id,))


# --------------------------------------------------------------------------- #
# Login lockout — protects the password path even with a rate limit in front
# of it, since a limit alone still allows slow, distributed guessing.
# --------------------------------------------------------------------------- #

LOCKOUT_THRESHOLD = 5
LOCKOUT_MINUTES = 15


def is_locked(user: sqlite3.Row) -> bool:
    locked_until = user["locked_until"]
    if not locked_until:
        return False
    return datetime.now(timezone.utc) < datetime.fromisoformat(locked_until)


def record_failed_login(user_id: str) -> None:
    with _connect() as conn:
        row = conn.execute("SELECT failed_login_attempts FROM users WHERE id = ?", (user_id,)).fetchone()
        attempts = (row["failed_login_attempts"] if row else 0) + 1
        locked_until = None
        if attempts >= LOCKOUT_THRESHOLD:
            locked_until = (datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)).isoformat()
        conn.execute(
            "UPDATE users SET failed_login_attempts = ?, locked_until = ? WHERE id = ?",
            (attempts, locked_until, user_id),
        )


def reset_failed_logins(user_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET failed_login_attempts = 0, locked_until = NULL WHERE id = ?", (user_id,)
        )


# --------------------------------------------------------------------------- #
# Privacy policy consent
# --------------------------------------------------------------------------- #

def has_accepted_privacy_policy(user: sqlite3.Row, current_version: str) -> bool:
    """Versioned, not a one-time flag: bumping `current_version` re-prompts
    everyone who agreed to an older version rather than grandfathering them in.
    """
    return user["privacy_policy_version"] == current_version


def record_privacy_policy_acceptance(user_id: str, version: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET privacy_policy_version = ?, privacy_policy_accepted_at = ? WHERE id = ?",
            (version, _now(), user_id),
        )


def link_identity(user_id: str, provider: str, provider_user_id: str) -> None:
    """Idempotent: re-linking the same (provider, provider_user_id) is a no-op."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO auth_identities (id, user_id, provider, provider_user_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (provider, provider_user_id) DO NOTHING
            """,
            (_new_id(), user_id, provider, provider_user_id, _now()),
        )


def get_or_create_user_for_oauth(email: str, provider: str, provider_user_id: str) -> str:
    """Verified-email linking policy: an OAuth login matching an existing
    account's email links to that account instead of creating a duplicate.
    """
    existing_by_identity = get_user_by_identity(provider, provider_user_id)
    if existing_by_identity is not None:
        return existing_by_identity["id"]

    existing_by_email = get_user_by_email(email)
    if existing_by_email is not None:
        link_identity(existing_by_email["id"], provider, provider_user_id)
        return existing_by_email["id"]

    user_id = create_user(email, email_verified=True)
    link_identity(user_id, provider, provider_user_id)
    return user_id


# --------------------------------------------------------------------------- #
# OTP codes
# --------------------------------------------------------------------------- #

def create_otp(user_id: str, code_hash: str, *, purpose: str = "signup", ttl_minutes: int = 10) -> None:
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO otp_codes (id, user_id, code_hash, purpose, expires_at, consumed_at) VALUES (?, ?, ?, ?, ?, NULL)",
            (_new_id(), user_id, code_hash, purpose, expires_at),
        )


def get_latest_otp(user_id: str, purpose: str = "signup") -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute(
            """
            SELECT * FROM otp_codes
            WHERE user_id = ? AND purpose = ? AND consumed_at IS NULL
            ORDER BY expires_at DESC LIMIT 1
            """,
            (user_id, purpose),
        ).fetchone()


def consume_otp(otp_id: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE otp_codes SET consumed_at = ? WHERE id = ?", (_now(), otp_id))


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #

def create_document(owner_id: str, filename: str, *, pages: int | None = None) -> str:
    document_id = _new_id()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO documents (id, owner_id, filename, ingested_at, pages, chunks, indexed_chunks) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (document_id, owner_id, filename, _now(), pages, None, None),
        )
    return document_id


def update_document_counts(document_id: str, *, chunks: int, indexed_chunks: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE documents SET chunks = ?, indexed_chunks = ? WHERE id = ?",
            (chunks, indexed_chunks, document_id),
        )


def list_documents(owner_id: str) -> list[sqlite3.Row]:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM documents WHERE owner_id = ? ORDER BY ingested_at DESC", (owner_id,)
        ).fetchall()


def get_document(document_id: str, owner_id: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM documents WHERE id = ? AND owner_id = ?", (document_id, owner_id)
        ).fetchone()


def delete_document(document_id: str, owner_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM documents WHERE id = ? AND owner_id = ?", (document_id, owner_id)
        )
        conn.execute("DELETE FROM chat_turns WHERE document_id = ? AND owner_id = ?", (document_id, owner_id))
        return cur.rowcount > 0


# --------------------------------------------------------------------------- #
# Chat turns
# --------------------------------------------------------------------------- #

def create_chat_turn(document_id: str, owner_id: str, question: str, answer: str, sources_json: str) -> str:
    turn_id = _new_id()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_turns (id, document_id, owner_id, question, answer, sources_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (turn_id, document_id, owner_id, question, answer, sources_json, _now()),
        )
    return turn_id


def list_chat_turns(document_id: str, owner_id: str) -> list[sqlite3.Row]:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM chat_turns WHERE document_id = ? AND owner_id = ? ORDER BY created_at ASC",
            (document_id, owner_id),
        ).fetchall()
