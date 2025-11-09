# core/database.py
import sqlite3
from datetime import datetime
import json
from pathlib import Path

DB_PATH = Path("ml_runs.db")

def init_db():
    """Initialize the SQLite database with a table for ML run metadata."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_type TEXT NOT NULL,
            parameters TEXT,
            metrics TEXT,
            model_path TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_run(model_type, parameters=None, metrics=None, model_path=None):
    """
    Save metadata for a completed ML run.
    
    Args:
        model_type (str): e.g., 'cnn', 'random_forest'
        parameters (dict|None): training hyperparameters (epochs, lr, etc.)
        metrics (dict|None): performance metrics (RMSE, accuracy, etc.)
        model_path (str|None): filesystem path to saved model
    """
    init_db()  # ensure table exists before inserting

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Convert dicts to JSON for readability
    params_json = json.dumps(parameters or {}, indent=2)
    metrics_json = json.dumps(metrics or {}, indent=2)

    cursor.execute("""
        INSERT INTO runs (timestamp, model_type, parameters, metrics, model_path)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        model_type,
        params_json,
        metrics_json,
        model_path or "",
    ))

    conn.commit()
    conn.close()
    print(f"Saved run info for model: {model_type}")


def get_all_runs():
    """Fetch all past model runs sorted by most recent first."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM runs ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_latest_run(model_type=None):
    """Return the latest run, optionally filtered by model_type."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if model_type:
        cursor.execute(
            "SELECT * FROM runs WHERE model_type=? ORDER BY timestamp DESC LIMIT 1",
            (model_type,),
        )
    else:
        cursor.execute("SELECT * FROM runs ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return row
