"""Durable SQLite inspection journal. Only server-owned relative image paths are stored."""
from __future__ import annotations
import csv
from contextlib import contextmanager
import io
import json
from pathlib import Path
import sqlite3
import time
import uuid


class InspectionStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions(
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT NOT NULL,
                    started REAL NOT NULL, ended REAL, state TEXT NOT NULL, config TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS inspections(
                    id TEXT PRIMARY KEY, event_key TEXT NOT NULL UNIQUE, session_id TEXT NOT NULL,
                    track_id TEXT, created REAL NOT NULL, completed REAL, state TEXT NOT NULL,
                    decision TEXT, class_name TEXT, image_path TEXT NOT NULL, quality TEXT NOT NULL,
                    identity TEXT NOT NULL, metadata TEXT NOT NULL, request_id TEXT,
                    result TEXT, error TEXT, FOREIGN KEY(session_id) REFERENCES sessions(id));
                CREATE INDEX IF NOT EXISTS inspections_created ON inspections(created DESC);
                CREATE INDEX IF NOT EXISTS inspections_session ON inspections(session_id);
                CREATE INDEX IF NOT EXISTS inspections_request ON inspections(request_id);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def session(self, kind, label, config=None, session_id=None):
        session_id = session_id or uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO sessions VALUES(?,?,?,?,?,?,?)",
                       (session_id, kind, label[:300], time.time(), None, "running", json.dumps(config or {})))
        return session_id

    def end_session(self, session_id, state="completed"):
        with self.connect() as db:
            db.execute("UPDATE sessions SET ended=?,state=? WHERE id=?", (time.time(), state, session_id))

    def add(self, event_key, session_id, track_id, image_path, quality, identity, metadata):
        inspection_id = uuid.uuid5(uuid.NAMESPACE_URL, event_key).hex
        with self.connect() as db:
            db.execute("""INSERT OR IGNORE INTO inspections
                (id,event_key,session_id,track_id,created,state,image_path,quality,identity,metadata)
                VALUES(?,?,?,?,?,'preparing',?,?,?,?)""",
                (inspection_id, event_key, session_id, str(track_id), time.time(), image_path,
                 json.dumps(quality), json.dumps(identity), json.dumps(metadata)))
        return self.get(inspection_id)

    def pending(self, inspection_id, request_id):
        with self.connect() as db:
            db.execute("UPDATE inspections SET state='pending',request_id=? WHERE id=? AND state='preparing'",
                       (request_id, inspection_id))

    def finish(self, inspection_id, decision, result=None, error=None):
        if decision not in ("good", "defect", "review", "error"):
            raise ValueError("Invalid inspection decision")
        with self.connect() as db:
            db.execute("""UPDATE inspections SET state='completed',completed=?,decision=?,
                class_name=?,result=?,error=? WHERE id=? AND state IN ('pending','preparing')""",
                (time.time(), decision, (result or {}).get("class"), json.dumps(result or {}), error, inspection_id))

    @staticmethod
    def decode(row):
        if row is None:
            return None
        row = dict(row)
        for field in ("quality", "identity", "metadata", "result", "config"):
            if field in row and row[field] is not None:
                row[field] = json.loads(row[field])
        row.pop("event_key", None)
        if "image_path" in row:
            row["image_url"] = f"/api/history/{row['id']}/image"
        return row

    def get(self, inspection_id):
        with self.connect() as db:
            return self.decode(db.execute("SELECT * FROM inspections WHERE id=?", (inspection_id,)).fetchone())

    def by_request(self, request_id):
        with self.connect() as db:
            return self.decode(db.execute("SELECT * FROM inspections WHERE request_id=?", (request_id,)).fetchone())

    def unfinished(self):
        with self.connect() as db:
            return [self.decode(row) for row in db.execute("SELECT * FROM inspections WHERE state IN ('pending','preparing')")]

    def query(self, session_id="", decision="", class_name="", search="", limit=50, offset=0):
        if decision and decision not in ("good", "defect", "review", "error", "pending"):
            raise ValueError("Unknown decision filter")
        where, args = [], []
        for field, value in (("session_id", session_id), ("decision", decision), ("class_name", class_name)):
            if value:
                where.append("state='pending'" if field == "decision" and value == "pending" else f"{field}=?")
                if not (field == "decision" and value == "pending"):
                    args.append(value)
        if search:
            where.append("(track_id LIKE ? OR id LIKE ? OR session_id LIKE ?)")
            args.extend(["%" + search[:100] + "%"] * 3)
        clause = " WHERE " + " AND ".join(where) if where else ""
        with self.connect() as db:
            total = db.execute("SELECT COUNT(*) FROM inspections" + clause, args).fetchone()[0]
            rows = db.execute("SELECT * FROM inspections" + clause + " ORDER BY created DESC,id DESC LIMIT ? OFFSET ?",
                              [*args, max(1, min(500, int(limit))), max(0, int(offset))]).fetchall()
        return {"total": total, "items": [self.decode(row) for row in rows]}

    def summary(self):
        with self.connect() as db:
            decisions = dict(db.execute("SELECT COALESCE(decision,'pending'),COUNT(*) FROM inspections GROUP BY decision"))
            classes = dict(db.execute("SELECT class_name,COUNT(*) FROM inspections WHERE class_name IS NOT NULL GROUP BY class_name"))
            sessions = [self.decode(row) for row in db.execute("SELECT * FROM sessions ORDER BY started DESC LIMIT 100")]
        return {"decisions": decisions, "classes": classes, "total": sum(decisions.values()), "sessions": sessions}

    def session_counts(self, session_id):
        with self.connect() as db:
            return dict(db.execute("SELECT COALESCE(decision,'pending'),COUNT(*) FROM inspections WHERE session_id=? GROUP BY decision", (session_id,)))

    def session_gpu_counts(self, session_id):
        with self.connect() as db:
            row = db.execute("""SELECT COUNT(request_id),
                SUM(CASE WHEN decision IN ('good','defect') THEN 1 ELSE 0 END),
                SUM(CASE WHEN request_id IS NOT NULL AND state='pending' THEN 1 ELSE 0 END),
                SUM(CASE WHEN request_id IS NOT NULL AND decision='error' THEN 1 ELSE 0 END)
                FROM inspections WHERE session_id=?""", (session_id,)).fetchone()
        return dict(zip(('submitted', 'completed', 'pending', 'failed'), (int(v or 0) for v in row)))

    def export_csv(self, **filters):
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["inspection_id", "session_id", "track_id", "created_unix", "decision", "class",
                         "source_kind", "source_label",
                         "confidence_raw", "quality_reasons", "source_pts_ns", "engine_sha256", "checkpoint_sha256",
                         "tensor_ms", "event_to_result_ms", "error"])
        offset = 0
        while True:
            page = self.query(**filters, limit=500, offset=offset)
            for row in page["items"]:
                result, meta, identity = row["result"] or {}, row["metadata"], row["identity"]
                values = [row["id"], row["session_id"], row["track_id"], row["created"], row["decision"], row["class_name"],
                          meta.get('source_kind'), meta.get('source_label'),
                          result.get("confidence"), ";".join(row["quality"]["reasons"]), meta.get("source_pts_ns"),
                          identity.get("engine_sha256"), identity.get("checkpoint_sha256"),
                          result.get("tensor_transfer_inference_ms"), result.get("server_event_to_result_ms"), row["error"]]
                # Prevent spreadsheet formula execution in user-controlled labels/errors.
                writer.writerow(["'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v for v in values])
            offset += len(page["items"])
            if offset >= page["total"] or not page["items"]:
                break
        return ("\ufeff" + output.getvalue()).encode("utf-8")
