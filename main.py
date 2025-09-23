"""
VIRGIL backend API (robust) with optional database.

This FastAPI app implements a chat endpoint (`/chat`) that processes user
messages and returns a JSON response containing a reply, mood, alert flag,
spark flag, and optional sources. The backend can persist logs, alerts,
and conversational memory either in a Postgres database (if configured
via `DATABASE_URL`) or in a local in-memory fallback if the database
is unreachable.

The app also exposes `/alerts`, `/alerts/count`, and `/logs` endpoints to
retrieve alert records and logs.

To run the app:
    uvicorn main:app --reload --port 8000
"""

import os
from datetime import datetime, timezone
from typing import List, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env (if present).
load_dotenv()

# Read database URL from environment. If unset or empty, the app will
# operate in in-memory mode.
DB_URL = os.getenv("DATABASE_URL", "")
USE_DB = bool(DB_URL)

# Attempt to import psycopg2 for database operations. If unavailable,
# the app will gracefully fall back to in-memory storage.
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None
    RealDictCursor = None
    USE_DB = False

def _final_db_url() -> str:
    """Return the DB URL with sslmode=require for Supabase if needed."""
    if not DB_URL:
        return ""
    # Ensure sslmode=require when connecting to Supabase
    if "sslmode=" not in DB_URL and "supabase.co" in DB_URL:
        return DB_URL + ("&" if "?" in DB_URL else "?") + "sslmode=require"
    return DB_URL

def db_ok() -> bool:
    """Check if a database connection can be established."""
    if not (USE_DB and psycopg2):
        return False
    try:
        con = psycopg2.connect(_final_db_url(), cursor_factory=RealDictCursor, connect_timeout=5)
        con.close()
        return True
    except Exception:
        return False

def get_conn():
    """Get a new database connection with appropriate options."""
    return psycopg2.connect(_final_db_url(), cursor_factory=RealDictCursor, connect_timeout=10)

# In-memory fallback structures
logs_mem: List[Dict[str, object]] = []
alerts_mem: List[Dict[str, object]] = []
memory_mem: List[Dict[str, str]] = []  # [{role, content}]

def log_event_local(msg: str, reply: str, mood: str, alert: bool = False, refused: bool = False) -> None:
    """Append a log entry to the in-memory logs structure."""
    ts = datetime.now(timezone.utc).isoformat()
    logs_mem.append({"ts": ts, "mood": mood, "message": msg, "reply": reply, "alert": alert, "refused": refused})
    if alert:
        alerts_mem.append({"ts": ts, "mood": mood, "message": msg})

def log_event_db(msg: str, reply: str, mood: str, alert: bool = False, refused: bool = False) -> None:
    """Insert a log entry into the database, creating tables if needed."""
    con = get_conn()
    cur = con.cursor()
    # Create tables if they don't exist
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMPTZ,
            mood TEXT,
            message TEXT,
            reply TEXT,
            alert BOOLEAN,
            refused BOOLEAN
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMPTZ,
            mood TEXT,
            message TEXT
        );
        """
    )
    ts = datetime.now(timezone.utc)
    cur.execute(
        "INSERT INTO logs (ts, mood, message, reply, alert, refused) VALUES (%s, %s, %s, %s, %s, %s)",
        (ts, mood, msg, reply, alert, refused),
    )
    if alert:
        cur.execute(
            "INSERT INTO alerts (ts, mood, message) VALUES (%s, %s, %s)",
            (ts, mood, msg),
        )
    con.commit()
    con.close()

def log_event(msg: str, reply: str, mood: str, alert: bool = False, refused: bool = False) -> None:
    """Persist a log entry either in DB or locally depending on availability."""
    if db_ok():
        try:
            log_event_db(msg, reply, mood, alert, refused)
            return
        except Exception:
            # Fallback to local on DB failure
            pass
    log_event_local(msg, reply, mood, alert, refused)

def memory_add(role: str, content: str) -> None:
    """Add a message to conversation history (DB or local)."""
    if not db_ok():
        memory_mem.append({"role": role, "content": content})
        # Limit to 50 messages
        if len(memory_mem) > 50:
            memory_mem.pop(0)
        return
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMPTZ,
            role TEXT,
            content TEXT
        );
        """
    )
    cur.execute(
        "INSERT INTO memory (ts, role, content) VALUES (%s, %s, %s)",
        (datetime.now(timezone.utc), role, content),
    )
    con.commit()
    con.close()

def memory_fetch(limit: int = 8) -> List[Dict[str, str]]:
    """Retrieve the last `limit` messages from memory."""
    if not db_ok():
        return memory_mem[-limit:].copy()
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT role, content FROM memory ORDER BY id DESC LIMIT %s",
        (limit,),
    )
    rows = cur.fetchall()
    con.close()
    # Reverse to return chronological order
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

# Instantiate the FastAPI app
app = FastAPI(title="VIRGIL API (robust)", version="0.9")

# Configure CORS (allow all for simplicity)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatIn(BaseModel):
    message: str
    proactivity: bool = True
    memory: bool = True
    spark: bool = False

class ChatOut(BaseModel):
    reply: str
    mood: str
    alert: bool = False
    spark: bool = False
    sources: Optional[List[dict]] = None

# Import the brain (think function). If unavailable, fallback to simple rules.
try:
    from brain import think  # type: ignore
except Exception:
    # Fallback think function if brain module is missing or errors
    def think(user_message: str, history: List[Dict[str, str]]):
        m = (user_message or "").strip().lower()
        reply, mood, alert, spark = "Compris.", "good", False, False
        # Simple logic for fallback
        if m.endswith("?"):
            reply, mood, spark = "Voici une réponse concise.", "analyze", True
        if "??" in m or "?!" in m:
            reply, mood, alert, spark = "Demande insistante détectée.", "notice", True, True
        if "mineur" in m and ("sex" in m or "sexe" in m or "explicit" in m):
            reply, mood, alert = "Désolé, sujet refusé. Incident consigné.", "alert", True
        return reply, mood, alert, spark, []

# Main chat endpoint
@app.post("/chat", response_model=ChatOut)
def chat(input: ChatIn):
    """Process a chat message and return a structured response."""
    history = memory_fetch(8) if input.memory else []
    reply, mood, alert, spark, sources = think(input.message, history)
    # Persist conversation memory if memory mode is enabled
    if input.memory:
        memory_add("user", input.message)
        memory_add("assistant", reply)
    # Persist log entry
    log_event(input.message, reply, mood, alert, refused=False)
    return ChatOut(reply=reply, mood=mood, alert=alert, spark=spark, sources=sources or [])

# Endpoint to list recent alerts
@app.get("/alerts")
def get_alerts(limit: int = 20):
    """Return the most recent `limit` alerts."""
    if db_ok():
        try:
            con = get_conn()
            cur = con.cursor()
            cur.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
            con.close()
            return rows
        except Exception:
            pass
    return list(reversed(alerts_mem[-limit:]))

# Endpoint to count alerts
@app.get("/alerts/count")
def get_alerts_count():
    """Return the total number of alerts."""
    if db_ok():
        try:
            con = get_conn()
            cur = con.cursor()
            cur.execute("SELECT COUNT(*) as count FROM alerts")
            c = cur.fetchone()["count"]
            con.close()
            return {"count": c}
        except Exception:
            pass
    return {"count": len(alerts_mem)}

# Endpoint to list recent logs
@app.get("/logs")
def get_logs(limit: int = 20):
    """Return the most recent `limit` log entries."""
    if db_ok():
        try:
            con = get_conn()
            cur = con.cursor()
            cur.execute("SELECT * FROM logs ORDER BY id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
            con.close()
            return rows
        except Exception:
            pass
    return list(reversed(logs_mem[-limit:]))