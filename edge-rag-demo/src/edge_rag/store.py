"""Embedded SQLite evidence store with strict authorization, atomic transactions,
structured fact storage, and immutable citation resolution.

Architecture & Trust Boundary:
- This is a local single-user CLI evidence store. Owner and requester strings
  are identifiers, NOT authentication credentials.
- A caller supplying the string "system" is an unprivileged user named "system".
  System and administrative privileges are only granted to explicit Principal
  objects constructed via trusted factories (e.g. Principal.system()) initialized
  by the local host runner, CLI harness, or catalog loader.
- Physical database file security is governed by local operating-system permissions;
  this layer does not claim protection against direct local file tampering.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Optional, Sequence
import urllib.parse


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class StoreError(Exception):
    """Base class for evidence store exceptions."""


class AuthorizationError(StoreError, PermissionError):
    """Raised when an operation is denied under the authorization policy."""


class VersionConflictError(StoreError, ValueError):
    """Raised when an expected_version mismatch causes an update conflict."""


class CitationError(StoreError, ValueError):
    """Raised when a citation is missing, malformed, or fails resolution."""


class NotFoundError(StoreError, KeyError):
    """Raised when a requested document or chunk does not exist."""


# ---------------------------------------------------------------------------
# Principal & Trust Boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    """An identity within the local single-user CLI trust boundary.

    Owner and requester strings are plain identifiers. An unprivileged caller
    cannot acquire administrative or system access simply by supplying "system".
    Only instances created via Principal.system() or Principal.admin() carry
    privileged authorization.
    """

    id: str
    is_system: bool = False
    is_admin: bool = False

    @classmethod
    def user(cls, user_id: str) -> "Principal":
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a nonempty string")
        # Passing "system" as a user string does NOT grant system privileges
        return cls(id=user_id, is_system=False, is_admin=False)

    @classmethod
    def system(cls) -> "Principal":
        """Trusted system principal factory for local trusted boundaries."""
        return cls(id="system", is_system=True, is_admin=True)

    @classmethod
    def anonymous(cls) -> "Principal":
        return cls(id="anonymous", is_system=False, is_admin=False)

    def can_admin(self) -> bool:
        return self.is_system or self.is_admin

    def can_read_document(self, owner_id: str, is_public: bool) -> bool:
        if is_public or self.can_admin():
            return True
        return self.id == owner_id

    def can_modify_document(self, owner_id: str) -> bool:
        if self.can_admin():
            return True
        return self.id == owner_id


def resolve_principal(principal: Principal | str | None) -> Principal:
    """Normalize a principal argument.

    Strings are treated strictly as unprivileged user identifiers.
    """
    if principal is None:
        return Principal.anonymous()
    if isinstance(principal, Principal):
        return principal
    if isinstance(principal, str):
        return Principal.user(principal)
    raise TypeError(f"Invalid principal type: {type(principal).__name__}")


# ---------------------------------------------------------------------------
# Structured Facts
# ---------------------------------------------------------------------------


@dataclass
class Fact:
    """A structured, typed fact preserving original values and provenance."""

    attribute: str
    value: Any = None
    unit: Optional[str] = None
    status: str = "present"
    source_location: Optional[str] = None
    source_revision: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "value": self.value,
            "unit": self.unit,
            "status": self.status,
            "source_location": self.source_location,
            "source_revision": self.source_revision,
        }


def _extract_fact_value(raw_val: Any) -> tuple[str, Optional[str], Optional[float], Optional[int]]:
    """Validate and categorize supported fact value types.

    Supported types:
    - bool -> ("bool", "true"|"false", None, 1|0)
    - int -> ("int", str(val), None, val)
    - float -> ("float", str(val), val, None) [rejects NaN, Infinity]
    - str -> ("str", val, None, None)
    - None -> ("null", None, None, None)
    """
    if raw_val is None:
        return ("null", None, None, None)
    # Check bool before int, as bool subclasses int in Python
    if isinstance(raw_val, bool):
        return ("bool", "true" if raw_val else "false", None, 1 if raw_val else 0)
    if isinstance(raw_val, int):
        return ("int", str(raw_val), None, raw_val)
    if isinstance(raw_val, float):
        if math.isnan(raw_val) or math.isinf(raw_val):
            raise ValueError(f"Non-finite numbers (NaN, Infinity) are rejected in facts: {raw_val!r}")
        return ("float", str(raw_val), raw_val, None)
    if isinstance(raw_val, str):
        return ("str", raw_val, None, None)
    raise TypeError(
        f"Unsupported fact value type: {type(raw_val).__name__} for {raw_val!r}; "
        f"supported scalar types are bool, int, float, str, or None"
    )


def _reconstruct_fact_value(
    v_type: Optional[str],
    v_text: Optional[str],
    v_num: Optional[float],
    v_int: Optional[int],
) -> Any:
    """Reconstruct original typed value from stored columns."""
    if v_type == "bool":
        return bool(v_int)
    if v_type == "int":
        return int(v_int) if v_int is not None else int(v_text)
    if v_type == "float":
        return float(v_num) if v_num is not None else float(v_text)
    if v_type == "str":
        return str(v_text) if v_text is not None else ""
    if v_type == "null":
        return None
    # Legacy fallback when value_type column is NULL
    if v_num is not None:
        return v_num
    return v_text


# ---------------------------------------------------------------------------
# Citation Identifiers & Integrity Hashes
# ---------------------------------------------------------------------------


def encode_doc_id(doc_id: str) -> str:
    """Encode a document identifier so delimiters like ':' do not break citation parsing."""
    if not doc_id or not isinstance(doc_id, str):
        raise ValueError("document_id must be a nonempty string")
    return urllib.parse.quote(doc_id, safe="-_.~")


def decode_doc_id(encoded: str) -> str:
    """Decode a document identifier from an encoded citation string."""
    if not encoded or not isinstance(encoded, str):
        raise CitationError("Encoded document identifier must be a nonempty string")
    try:
        return urllib.parse.unquote(encoded)
    except Exception as exc:
        raise CitationError(f"Failed to decode document identifier {encoded!r}: {exc}") from exc


_CITATION_RE = re.compile(
    r"^cite:(?P<doc_id>.+?):v(?P<version>\d+):c(?P<chunk_idx>\d+)(?::(?P<hash>[a-fA-F0-9]+))?$"
)


def _compute_hash(data: str) -> str:
    """Compute SHA-256 over raw source text bytes."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _compute_integrity_hash(
    text: str,
    attribute: Optional[str] = None,
    value_type: Optional[str] = None,
    value_repr: Optional[str] = None,
    unit: Optional[str] = None,
    status: str = "present",
    item_id: Optional[str] = None,
    variant_id: Optional[str] = None,
    source_location: Optional[str] = None,
    source_revision: Optional[str] = None,
) -> str:
    """Compute SHA-256 over a canonical representation of text, structured facts, and provenance."""
    payload = {
        "text": text,
        "attribute": attribute,
        "value_type": value_type,
        "value_repr": value_repr,
        "unit": unit,
        "status": status,
        "item_id": item_id,
        "variant_id": variant_id,
        "source_location": source_location,
        "source_revision": source_revision,
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Evidence Store
# ---------------------------------------------------------------------------


class EvidenceStore:
    """Embedded SQLite evidence store with atomic transactions, authorization,
    structured fact preservation, and immutable citation resolution.
    """

    SCHEMA_VERSION = 3

    def __init__(self, db_path: str = ":memory:"):
        self.backend = "sqlite"
        self._lock = threading.RLock()
        if db_path == ":memory:":
            self._conn_str = f"file:qedge_mem_{id(self)}_{time.time_ns()}?mode=memory&cache=shared"
            self._connection = sqlite3.connect(
                self._conn_str, uri=True, check_same_thread=False, timeout=10.0, isolation_level=None
            )
        else:
            self._conn_str = str(db_path)
            self._connection = sqlite3.connect(
                db_path, check_same_thread=False, timeout=10.0, isolation_level=None
            )
            # Enable write-ahead logging for file databases for concurrency
            try:
                self._connection.execute("PRAGMA journal_mode = WAL;")
            except sqlite3.OperationalError:
                pass
        self._connection.execute("PRAGMA busy_timeout = 10000;")
        self._connection.row_factory = sqlite3.Row
        self._init_or_migrate_db()

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "EvidenceStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @contextlib.contextmanager
    def _read_transaction(self):
        """Execute read operations inside a transaction to ensure a coherent database snapshot."""
        with self._lock:
            self._connection.execute("BEGIN")
            try:
                yield
            finally:
                try:
                    self._connection.execute("COMMIT")
                except sqlite3.OperationalError:
                    pass

    @contextlib.contextmanager
    def _write_transaction(self):
        """Execute a write transaction using BEGIN IMMEDIATE.

        Acquires SQLite's write lock immediately, preventing lost updates and
        ensuring cross-connection version checks and writes are atomic.
        """
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._connection.execute("COMMIT")
            except Exception:
                try:
                    self._connection.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
                raise

    # -----------------------------------------------------------------------
    # Schema Migration
    # -----------------------------------------------------------------------

    def _init_or_migrate_db(self) -> None:
        """Initialize new database or migrate existing database transactionally."""
        with self._write_transaction():
            # Create migrations table
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )

            # Check existing migration version first
            mig_row = self._connection.execute(
                "SELECT MAX(version) AS ver FROM schema_migrations"
            ).fetchone()
            current_ver = mig_row["ver"] if (mig_row and mig_row["ver"] is not None) else None

            if current_ver is not None and current_ver > self.SCHEMA_VERSION:
                raise StoreError(
                    f"Unsupported future database schema version: {current_ver} "
                    f"(max supported version is {self.SCHEMA_VERSION})"
                )

            # Check if documents table exists
            table_row = self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
            ).fetchone()

            if table_row is None:
                # Fresh database: create current schema (version 3)
                self._create_v3_schema()
                now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                self._connection.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (self.SCHEMA_VERSION, now),
                )
                return

            doc_ver = current_ver if current_ver is not None else 1
            if doc_ver < 2:
                self._migrate_v1_to_v2()
                now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                self._connection.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (2, now),
                )
                doc_ver = 2

            if doc_ver < 3:
                self._migrate_v2_to_v3()
                now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                self._connection.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (3, now),
                )

    def _create_v3_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                is_public INTEGER NOT NULL DEFAULT 0,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE document_history (
                doc_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                owner_id TEXT NOT NULL,
                is_public INTEGER NOT NULL DEFAULT 0,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                committed_at TEXT NOT NULL,
                PRIMARY KEY (doc_id, version)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE chunks (
                doc_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                chunk_idx INTEGER NOT NULL,
                chunk_id TEXT NOT NULL,
                text TEXT NOT NULL,
                item_id TEXT,
                variant_id TEXT,
                attribute TEXT,
                value_type TEXT,
                value_text TEXT,
                value_num REAL,
                value_int INTEGER,
                unit TEXT,
                status TEXT NOT NULL DEFAULT 'present',
                source_location TEXT,
                source_revision TEXT NOT NULL DEFAULT 'unknown',
                source_hash TEXT NOT NULL,
                integrity_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (doc_id, version, chunk_idx)
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_ver ON chunks(doc_id, version)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id)"
        )

    def _migrate_v1_to_v2(self) -> None:
        """Migrate legacy v1 schema to v2."""
        # Check existing columns in documents
        doc_cols = {
            col["name"]
            for col in self._connection.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "is_public" not in doc_cols:
            self._connection.execute("ALTER TABLE documents ADD COLUMN is_public INTEGER NOT NULL DEFAULT 0")
        if "is_deleted" not in doc_cols:
            self._connection.execute("ALTER TABLE documents ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0")
        if "updated_at" not in doc_cols:
            self._connection.execute("ALTER TABLE documents ADD COLUMN updated_at TEXT")
            self._connection.execute("UPDATE documents SET updated_at = created_at WHERE updated_at IS NULL")

        # Create document_history if missing
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_history (
                doc_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                owner_id TEXT NOT NULL,
                is_public INTEGER NOT NULL DEFAULT 0,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                committed_at TEXT NOT NULL,
                PRIMARY KEY (doc_id, version)
            )
            """
        )
        # Backfill document_history from existing documents (active versions)
        self._connection.execute(
            """
            INSERT OR IGNORE INTO document_history (doc_id, version, owner_id, is_public, is_deleted, committed_at)
            SELECT id, version, owner_id, is_public, is_deleted, updated_at FROM documents
            """
        )

        # Check existing columns in chunks
        chunk_cols = {
            col["name"]
            for col in self._connection.execute("PRAGMA table_info(chunks)").fetchall()
        }
        v2_new_cols = [
            ("chunk_id", "TEXT"),
            ("item_id", "TEXT"),
            ("variant_id", "TEXT"),
            ("attribute", "TEXT"),
            ("value_text", "TEXT"),
            ("value_num", "REAL"),
            ("unit", "TEXT"),
            ("status", "TEXT NOT NULL DEFAULT 'unknown'"),
            ("source_location", "TEXT DEFAULT 'legacy'"),
            ("source_revision", "TEXT NOT NULL DEFAULT 'unknown'"),
            ("source_hash", "TEXT"),
            ("created_at", "TEXT"),
        ]
        for col_name, col_type in v2_new_cols:
            if col_name not in chunk_cols:
                self._connection.execute(f"ALTER TABLE chunks ADD COLUMN {col_name} {col_type}")

        # Backfill chunk_id, source_hash, created_at for legacy rows
        legacy_rows = self._connection.execute(
            "SELECT doc_id, chunk_idx, text, version FROM chunks WHERE chunk_id IS NULL OR source_hash IS NULL"
        ).fetchall()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for row in legacy_rows:
            d_id = row["doc_id"]
            c_idx = row["chunk_idx"]
            v = row["version"]
            c_id = f"{encode_doc_id(d_id)}:v{v}:c{c_idx}"
            s_hash = _compute_hash(row["text"])
            self._connection.execute(
                """
                UPDATE chunks
                SET chunk_id = ?, source_hash = ?, created_at = COALESCE(created_at, ?)
                WHERE doc_id = ? AND chunk_idx = ? AND version = ?
                """,
                (c_id, s_hash, now, d_id, c_idx, v),
            )

        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_ver ON chunks(doc_id, version)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id)"
        )

    def _migrate_v2_to_v3(self) -> None:
        """Migrate v2 schema to v3: add value_type, value_int, integrity_hash and backfill."""
        chunk_cols = {
            col["name"]
            for col in self._connection.execute("PRAGMA table_info(chunks)").fetchall()
        }
        v3_cols = [
            ("value_type", "TEXT"),
            ("value_int", "INTEGER"),
            ("integrity_hash", "TEXT"),
        ]
        for col_name, col_type in v3_cols:
            if col_name not in chunk_cols:
                self._connection.execute(f"ALTER TABLE chunks ADD COLUMN {col_name} {col_type}")

        # Check historical-version completeness for previously migrated v2 databases:
        # Any version in chunks that was omitted from document_history is backfilled
        distinct_chunk_versions = self._connection.execute(
            """
            SELECT DISTINCT c.doc_id, c.version, d.owner_id
            FROM chunks c
            LEFT JOIN documents d ON c.doc_id = d.id
            """
        ).fetchall()
        for cv in distinct_chunk_versions:
            d_id = cv["doc_id"]
            ver = cv["version"]
            owner = cv["owner_id"] or "unknown"
            self._connection.execute(
                """
                INSERT OR IGNORE INTO document_history (doc_id, version, owner_id, is_public, is_deleted, committed_at)
                VALUES (?, ?, ?, 0, 0, 'unknown')
                """,
                (d_id, ver, owner),
            )

        # Transactional backfill of canonical integrity hashes and typed values for all retained rows
        rows_to_backfill = self._connection.execute(
            """
            SELECT doc_id, chunk_idx, text, version, chunk_id, item_id, variant_id,
                   attribute, value_type, value_text, value_num, value_int, unit,
                   status, source_location, source_revision, source_hash, created_at
            FROM chunks
            WHERE integrity_hash IS NULL OR value_type IS NULL
            """
        ).fetchall()

        for row in rows_to_backfill:
            d_id = row["doc_id"]
            c_idx = row["chunk_idx"]
            ver = row["version"]

            # Preserve recoverable values/types, mark irrecoverable legacy metadata unknown.
            # Never infer that old string "True" was originally a boolean!
            existing_type = row["value_type"]
            val_text = row["value_text"]
            val_num = row["value_num"]
            val_int = row["value_int"]

            if existing_type is not None:
                v_type = existing_type
                v_text = val_text
                v_num = val_num
                v_int = val_int
            elif val_num is not None:
                v_type = "float"
                v_num = float(val_num)
                v_text = val_text or str(v_num)
                v_int = None
            elif val_text is not None:
                # String value preserved as string; never infer boolean from string "True"/"False"
                v_type = "str"
                v_text = str(val_text)
                v_num = None
                v_int = None
            else:
                v_type = "null"
                v_text = None
                v_num = None
                v_int = None

            c_id = row["chunk_id"] or f"{encode_doc_id(d_id)}:v{ver}:c{c_idx}"
            s_hash = row["source_hash"] or _compute_hash(row["text"])
            v_repr = str(v_int) if v_int is not None else (str(v_num) if v_num is not None else v_text)

            integ_hash = _compute_integrity_hash(
                text=row["text"],
                attribute=row["attribute"],
                value_type=v_type,
                value_repr=v_repr,
                unit=row["unit"],
                status=row["status"] or "unknown",
                item_id=row["item_id"],
                variant_id=row["variant_id"],
                source_location=row["source_location"] or "legacy",
                source_revision=row["source_revision"] or "unknown",
            )

            self._connection.execute(
                """
                UPDATE chunks
                SET chunk_id = ?, value_type = ?, value_text = ?, value_num = ?, value_int = ?,
                    source_hash = ?, integrity_hash = ?, created_at = COALESCE(created_at, 'unknown')
                WHERE doc_id = ? AND chunk_idx = ? AND version = ?
                """,
                (c_id, v_type, v_text, v_num, v_int, s_hash, integ_hash, d_id, c_idx, ver),
            )

        # Reject unexpected missing integrity metadata after migration
        missing_cnt = self._connection.execute(
            "SELECT COUNT(*) AS cnt FROM chunks WHERE integrity_hash IS NULL"
        ).fetchone()["cnt"]
        if missing_cnt > 0:
            raise StoreError(
                f"Migration to v3 failed: {missing_cnt} chunk(s) remain with missing integrity_hash"
            )

    # -----------------------------------------------------------------------
    # CRUD & Authorization Operations
    # -----------------------------------------------------------------------

    def upload(
        self,
        document_id: str,
        owner: str | Principal,
        text_chunks: list[str] | str,
        expected_version: Optional[int] = None,
        is_public: Optional[bool] = None,
        facts: Optional[Sequence[dict[str, Any] | Fact]] = None,
        item_id: Optional[str] = None,
        variant_id: Optional[str] = None,
        source_location: Optional[str] = None,
        source_revision: Optional[str] = None,
    ) -> int:
        """Store a new document version atomically across connections.

        Enforces:
        - Expected version comparison.
        - Cross-owner write denial.
        - Deleted document ownership takeover denial.
        - Facts and chunks alignment validation.
        - Typed fact validation and non-finite rejection.
        - Atomic insertion of document, history, and chunk facts.
        """
        if not document_id or not isinstance(document_id, str):
            raise ValueError("document_id must be a nonempty string")

        principal = resolve_principal(owner)
        if isinstance(text_chunks, str):
            chunks = [text_chunks] if text_chunks else []
        else:
            chunks = list(text_chunks)

        # Validate facts/chunks alignment
        if facts is not None:
            if len(facts) != len(chunks):
                raise ValueError(
                    f"Facts count ({len(facts)}) must match chunks count ({len(chunks)})"
                )

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with self._write_transaction():
            row = self._connection.execute(
                "SELECT version, owner_id, is_deleted, is_public FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()

            current = int(row["version"]) if row else 0

            # 1. Expected version check
            if expected_version is not None and expected_version != current:
                raise VersionConflictError(
                    f"version conflict for {document_id!r}: current={current}, "
                    f"expected={expected_version}"
                )

            # 2. Ownership & takeover checks
            if row is not None:
                existing_owner = row["owner_id"]
                is_deleted = bool(row["is_deleted"])

                if is_deleted:
                    # Prevent takeover of deleted objects by another principal
                    if not principal.can_modify_document(existing_owner):
                        raise AuthorizationError(
                            f"Deleted document {document_id!r} is owned by {existing_owner!r}; "
                            f"takeover denied for principal {principal.id!r}"
                        )
                else:
                    # Cross-owner overwrite denial
                    if not principal.can_modify_document(existing_owner):
                        raise AuthorizationError(
                            f"Document {document_id!r} is owned by {existing_owner!r}; "
                            f"write denied for principal {principal.id!r}"
                        )

            # Determine owner for new or updated document
            doc_owner = row["owner_id"] if row is not None else principal.id
            version = current + 1

            # Distinguish omitted visibility from explicit False
            if is_public is None:
                public_flag = int(row["is_public"]) if row is not None else 0
            else:
                public_flag = 1 if is_public else 0

            # 3. Update documents table
            if row is None:
                self._connection.execute(
                    """
                    INSERT INTO documents (id, owner_id, version, is_public, is_deleted, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 0, ?, ?)
                    """,
                    (document_id, doc_owner, version, public_flag, now, now),
                )
            else:
                self._connection.execute(
                    """
                    UPDATE documents
                    SET version = ?, is_public = ?, is_deleted = 0, updated_at = ?
                    WHERE id = ?
                    """,
                    (version, public_flag, now, document_id),
                )

            # 4. Record document_history
            self._connection.execute(
                """
                INSERT INTO document_history (doc_id, version, owner_id, is_public, is_deleted, committed_at)
                VALUES (?, ?, ?, ?, 0, ?)
                """,
                (document_id, version, doc_owner, public_flag, now),
            )

            # 5. Insert structured chunks
            facts_list = list(facts) if facts is not None else []
            for index, text in enumerate(chunks):
                fact_obj = facts_list[index] if index < len(facts_list) else None
                attr = None
                v_type = None
                v_text = None
                v_num = None
                v_int = None
                unit = None
                status = "present"
                src_loc = source_location
                src_rev = source_revision or "unknown"

                if fact_obj is not None:
                    if hasattr(fact_obj, "to_dict"):
                        f_dict = fact_obj.to_dict()
                    elif isinstance(fact_obj, dict):
                        f_dict = fact_obj
                    else:
                        raise TypeError(f"Invalid fact object at index {index}")

                    attr = f_dict.get("attribute")
                    raw_val = f_dict.get("value")
                    v_type, v_text, v_num, v_int = _extract_fact_value(raw_val)
                    unit = f_dict.get("unit")
                    status = f_dict.get("status", "present")
                    src_loc = f_dict.get("source_location") or src_loc
                    src_rev = f_dict.get("source_revision") or src_rev

                encoded_doc_id = encode_doc_id(document_id)
                chunk_id = f"{encoded_doc_id}:v{version}:c{index}"
                source_hash = _compute_hash(text)
                v_repr = str(v_int) if v_int is not None else (str(v_num) if v_num is not None else v_text)
                integrity_hash = _compute_integrity_hash(
                    text=text,
                    attribute=attr,
                    value_type=v_type,
                    value_repr=v_repr,
                    unit=unit,
                    status=status,
                    item_id=item_id,
                    variant_id=variant_id,
                    source_location=src_loc,
                    source_revision=src_rev,
                )

                self._connection.execute(
                    """
                    INSERT INTO chunks (
                        doc_id, version, chunk_idx, chunk_id, text, item_id, variant_id,
                        attribute, value_type, value_text, value_num, value_int, unit,
                        status, source_location, source_revision, source_hash, integrity_hash, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        version,
                        index,
                        chunk_id,
                        text,
                        item_id,
                        variant_id,
                        attr,
                        v_type,
                        v_text,
                        v_num,
                        v_int,
                        unit,
                        status,
                        src_loc,
                        src_rev,
                        source_hash,
                        integrity_hash,
                        now,
                    ),
                )

        return version

    def retrieve(
        self, document_id: str, requester: str | Principal | None = None
    ) -> list[dict[str, Any]]:
        """Return the active chunks for a document within a coherent snapshot.

        Enforces:
        - Cross-owner read denial on private documents.
        - Public documents accessible to all principals.
        - Soft-deleted documents return empty list.
        """
        with self._read_transaction():
            principal = resolve_principal(requester)
            row = self._connection.execute(
                "SELECT version, owner_id, is_public, is_deleted FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()

            if row is None or bool(row["is_deleted"]):
                return []

            owner_id = row["owner_id"]
            is_public = bool(row["is_public"])

            if not principal.can_read_document(owner_id, is_public):
                raise AuthorizationError(
                    f"Document {document_id!r} is private to {owner_id!r}; "
                    f"read denied for principal {principal.id!r}"
                )

            active_version = int(row["version"])
            rows = self._connection.execute(
                """
                SELECT chunk_idx, chunk_id, text, item_id, variant_id, attribute,
                       value_type, value_text, value_num, value_int, unit, status,
                       source_location, source_revision, source_hash, integrity_hash
                FROM chunks
                WHERE doc_id = ? AND version = ?
                ORDER BY chunk_idx
                """,
                (document_id, active_version),
            ).fetchall()

            results = []
            for item in rows:
                chunk_idx = int(item["chunk_idx"])
                integ_hash = item["integrity_hash"] or item["source_hash"]
                citation = f"cite:{encode_doc_id(document_id)}:v{active_version}:c{chunk_idx}:{integ_hash[:8]}"
                val = _reconstruct_fact_value(
                    item["value_type"], item["value_text"], item["value_num"], item["value_int"]
                )
                results.append(
                    {
                        "chunk_idx": chunk_idx,
                        "chunk_id": item["chunk_id"],
                        "text": item["text"],
                        "citation": citation,
                        "item_id": item["item_id"],
                        "variant_id": item["variant_id"],
                        "attribute": item["attribute"],
                        "value": val,
                        "unit": item["unit"],
                        "status": item["status"],
                        "source_location": item["source_location"],
                        "source_revision": item["source_revision"],
                        "source_hash": item["source_hash"],
                        "integrity_hash": item["integrity_hash"],
                    }
                )
            return results

    def delete(self, document_id: str, owner: str | Principal) -> bool:
        """Soft-delete an active document with authorization check.

        Only the owner or an administrator may delete a document.
        """
        principal = resolve_principal(owner)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with self._write_transaction():
            row = self._connection.execute(
                "SELECT version, owner_id, is_deleted, is_public FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()

            if row is None or bool(row["is_deleted"]):
                return False

            existing_owner = row["owner_id"]
            if not principal.can_modify_document(existing_owner):
                raise AuthorizationError(
                    f"Document {document_id!r} is owned by {existing_owner!r}; "
                    f"delete denied for principal {principal.id!r}"
                )

            version = int(row["version"])
            self._connection.execute(
                "UPDATE documents SET is_deleted = 1, updated_at = ? WHERE id = ?",
                (now, document_id),
            )
            self._connection.execute(
                "UPDATE document_history SET is_deleted = 1 WHERE doc_id = ? AND version = ?",
                (document_id, version),
            )
            return True

    def restore(self, document_id: str, owner: str | Principal) -> bool:
        """Restore a soft-deleted document.

        Only the original owner or an administrator may restore it.
        """
        principal = resolve_principal(owner)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with self._write_transaction():
            row = self._connection.execute(
                "SELECT version, owner_id, is_deleted, is_public FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()

            if row is None or not bool(row["is_deleted"]):
                return False

            existing_owner = row["owner_id"]
            if not principal.can_modify_document(existing_owner):
                raise AuthorizationError(
                    f"Document {document_id!r} is owned by {existing_owner!r}; "
                    f"restore denied for principal {principal.id!r}"
                )

            version = int(row["version"])
            self._connection.execute(
                "UPDATE documents SET is_deleted = 0, updated_at = ? WHERE id = ?",
                (now, document_id),
            )
            self._connection.execute(
                "UPDATE document_history SET is_deleted = 0 WHERE doc_id = ? AND version = ?",
                (document_id, version),
            )
            return True

    # -----------------------------------------------------------------------
    # Immutable Citation Resolution
    # -----------------------------------------------------------------------

    def resolve_citation(
        self, citation: str, requester: str | Principal | None = None
    ) -> dict[str, Any]:
        """Resolve an immutable citation to its exact historical fact and provenance.

        Enforces:
        - Citation string parsing and syntax validation (supporting encoded delimiters).
        - Current authorization and deletion policy (revoked public access or deleted
          documents cannot be bypassed via historical citations).
        - Historical version authorization check (unauthorized callers denied).
        - Payload integrity hash consistency check against canonical fact/provenance.
        - Source text hash check against original chunk bytes.
        """
        if not citation or not isinstance(citation, str):
            raise CitationError("Citation must be a nonempty string")

        match = _CITATION_RE.match(citation.strip())
        if not match:
            raise CitationError(
                f"Malformed citation: {citation!r}. Expected format 'cite:<doc_id>:v<ver>:c<idx>[:<hash>]'"
            )

        raw_doc_id = match.group("doc_id")
        doc_id = decode_doc_id(raw_doc_id)
        version = int(match.group("version"))
        chunk_idx = int(match.group("chunk_idx"))
        expected_hash_prefix = match.group("hash")

        with self._read_transaction():
            principal = resolve_principal(requester)

            # 1. Enforce current document authorization & deletion policy on historical resolution
            curr_row = self._connection.execute(
                "SELECT owner_id, is_public, is_deleted, version FROM documents WHERE id = ?",
                (doc_id,),
            ).fetchone()

            hist_row = self._connection.execute(
                """
                SELECT owner_id, is_public, is_deleted, committed_at
                FROM document_history
                WHERE doc_id = ? AND version = ?
                """,
                (doc_id, version),
            ).fetchone()

            if hist_row is None:
                raise NotFoundError(f"No history record found for citation {citation!r}")

            hist_owner = hist_row["owner_id"]
            hist_public = bool(hist_row["is_public"])

            if curr_row is not None:
                curr_owner = curr_row["owner_id"]
                curr_deleted = bool(curr_row["is_deleted"])
                curr_public = bool(curr_row["is_public"])

                if curr_deleted:
                    # Document is deleted: only owner or admin can resolve history
                    if not principal.can_modify_document(curr_owner):
                        raise AuthorizationError(
                            f"Document {doc_id!r} is deleted; historical resolution denied for principal {principal.id!r}"
                        )
                else:
                    # Document is active: if current visibility is private, historical public metadata must not bypass revocation!
                    if not curr_public and not principal.can_modify_document(curr_owner):
                        raise AuthorizationError(
                            f"Document {doc_id!r} is currently private to {curr_owner!r}; "
                            f"historical access denied for principal {principal.id!r}"
                        )
                    # If current document is public, but this historical version was private, non-owners cannot read it
                    if not hist_public and not principal.can_modify_document(hist_owner):
                        raise AuthorizationError(
                            f"Historical version {version} of {doc_id!r} was private; "
                            f"access denied for principal {principal.id!r}"
                        )
            else:
                # Standalone historical record (e.g. legacy historical version)
                if not hist_public and not principal.can_modify_document(hist_owner):
                    raise AuthorizationError(
                        f"Historical citation {citation!r} is private to {hist_owner!r}; "
                        f"access denied for principal {principal.id!r}"
                    )

            # 2. Fetch the exact historical chunk
            chunk_row = self._connection.execute(
                """
                SELECT chunk_id, text, item_id, variant_id, attribute,
                       value_type, value_text, value_num, value_int, unit,
                       status, source_location, source_revision,
                       source_hash, integrity_hash, created_at
                FROM chunks
                WHERE doc_id = ? AND version = ? AND chunk_idx = ?
                """,
                (doc_id, version, chunk_idx),
            ).fetchone()

            if chunk_row is None:
                raise NotFoundError(f"Chunk index {chunk_idx} not found for citation {citation!r}")

            source_hash = chunk_row["source_hash"]
            stored_integ_hash = chunk_row["integrity_hash"]

            # Reject unexpected missing integrity metadata after migration
            if stored_integ_hash is None:
                raise CitationError(
                    f"Stored chunk integrity error for {chunk_row['chunk_id']}: missing integrity metadata"
                )

            # 3. Verify integrity hash prefix if supplied in citation
            if expected_hash_prefix is not None:
                prefix_match = False
                if stored_integ_hash.lower().startswith(expected_hash_prefix.lower()):
                    prefix_match = True
                elif source_hash and source_hash.lower().startswith(expected_hash_prefix.lower()):
                    prefix_match = True
                if not prefix_match:
                    raise CitationError(
                        f"Integrity hash mismatch for citation {citation!r}: "
                        f"expected prefix {expected_hash_prefix!r}"
                    )

            # 4. Verify canonical payload integrity (unconditionally performed)
            v_int = chunk_row["value_int"]
            v_num = chunk_row["value_num"]
            v_text = chunk_row["value_text"]
            v_type = chunk_row["value_type"]
            v_repr = str(v_int) if v_int is not None else (str(v_num) if v_num is not None else v_text)

            actual_integ_hash = _compute_integrity_hash(
                text=chunk_row["text"],
                attribute=chunk_row["attribute"],
                value_type=v_type,
                value_repr=v_repr,
                unit=chunk_row["unit"],
                status=chunk_row["status"],
                item_id=chunk_row["item_id"],
                variant_id=chunk_row["variant_id"],
                source_location=chunk_row["source_location"],
                source_revision=chunk_row["source_revision"],
            )
            if actual_integ_hash != stored_integ_hash:
                raise CitationError(
                    f"Stored chunk integrity error for {chunk_row['chunk_id']}: "
                    f"stored integrity hash does not match canonical fact and provenance payload"
                )

            # Verify source text hash
            actual_source_hash = _compute_hash(chunk_row["text"])
            if actual_source_hash != source_hash:
                raise CitationError(
                    f"Stored chunk text integrity error for {chunk_row['chunk_id']}: "
                    f"stored source hash does not match text bytes"
                )

            val = _reconstruct_fact_value(v_type, v_text, v_num, v_int)
            return {
                "citation": citation,
                "doc_id": doc_id,
                "version": version,
                "chunk_idx": chunk_idx,
                "chunk_id": chunk_row["chunk_id"],
                "text": chunk_row["text"],
                "fact": {
                    "attribute": chunk_row["attribute"],
                    "value": val,
                    "unit": chunk_row["unit"],
                    "status": chunk_row["status"],
                },
                "provenance": {
                    "item_id": chunk_row["item_id"],
                    "variant_id": chunk_row["variant_id"],
                    "source_location": chunk_row["source_location"],
                    "source_revision": chunk_row["source_revision"],
                    "source_hash": source_hash,
                    "integrity_hash": stored_integ_hash,
                    "created_at": chunk_row["created_at"],
                    "committed_at": hist_row["committed_at"],
                },
                "document_status": {
                    "owner_id": hist_owner,
                    "is_public": hist_public,
                    "is_deleted": bool(hist_row["is_deleted"]),
                },
            }
