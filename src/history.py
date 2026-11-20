"""Small SQLite-backed local prediction history."""
from pathlib import Path
import json
import sqlite3
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "history.sqlite3"

def _connect(path=DB):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, model TEXT NOT NULL, outcome INTEGER NOT NULL, probability REAL, features TEXT NOT NULL)")
    return con

def add(model, outcome, probability, features, path=DB):
    con = _connect(path)
    try:
        con.execute("INSERT INTO predictions(created_at,model,outcome,probability,features) VALUES(?,?,?,?,?)", (datetime.now(timezone.utc).isoformat(timespec="seconds"), model, int(outcome), float(probability), json.dumps(features)))
        con.commit()
    finally:
        con.close()

def read(path=DB):
    con = _connect(path)
    try:
        rows = con.execute("SELECT id,created_at,model,outcome,probability,features FROM predictions ORDER BY id DESC").fetchall()
    finally:
        con.close()
    return [{"id":r[0],"timestamp":r[1],"model":r[2],"outcome":r[3],"probability":r[4],"features":json.loads(r[5])} for r in rows]

def clear(path=DB):
    con = _connect(path)
    try:
        con.execute("DELETE FROM predictions")
        con.commit()
    finally:
        con.close()
