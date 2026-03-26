"""
SQLite client for storing and querying detections data.
Parallel to vdms_client.py but uses SQLite backend.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class SQLiteClient:
    """SQLite client for detections database."""
    
    def __init__(self, db_path: str = "out/sql_data/detections.db"):
        """
        Initialize SQLite client.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._connect()
        self._create_schema()
    
    def _connect(self):
        """Connect to SQLite database."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    
    def _create_schema(self):
        """Create detections table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_frame_id ON detections(frame_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_label ON detections(label)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON detections(confidence)")
        
        self.conn.commit()
    
    def insert_detection(self, frame_id: int, label: str, confidence: float,
                        x: int, y: int, width: int, height: int,
                        metadata: Optional[Dict[str, Any]] = None):
        """
        Insert a single detection.
        
        Args:
            frame_id: Frame number
            label: Detection class label
            confidence: Detection confidence score
            x: Bounding box x coordinate
            y: Bounding box y coordinate
            width: Bounding box width
            height: Bounding box height
            metadata: Optional metadata dictionary
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO detections (frame_id, label, confidence, x, y, width, height, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (frame_id, label, confidence, x, y, width, height,
              json.dumps(metadata) if metadata else None))
        self.conn.commit()
    
    def insert_detections_batch(self, detections: List[Dict[str, Any]]):
        """
        Insert multiple detections in a batch.
        
        Args:
            detections: List of detection dictionaries
        """
        cursor = self.conn.cursor()
        data = [
            (
                d['frame_id'],
                d['label'],
                d['confidence'],
                d['x'],
                d['y'],
                d['width'],
                d['height'],
                json.dumps(d.get('metadata')) if d.get('metadata') else None
            )
            for d in detections
        ]
        
        cursor.executemany("""
            INSERT INTO detections (frame_id, label, confidence, x, y, width, height, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        self.conn.commit()
    
    def clear_detections(self):
        """
        Clear all detections from the database.
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM detections")
        self.conn.commit()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters tuple
            
        Returns:
            List of result dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert Row objects to dictionaries
        return [dict(row) for row in rows]
    
    def get_schema(self) -> str:
        """
        Get database schema as string for LLM context.
        
        Returns:
            Schema description string
        """
        return """
Database Schema:

Table: detections
Columns:
  - id (INTEGER): Primary key, auto-increment
  - frame_id (INTEGER): Frame number where detection occurred
  - label (TEXT): Detection class (e.g., 'Obstacle', 'Deformation', 'Rupture')
  - confidence (REAL): Detection confidence score (0.0 to 1.0)
  - x (INTEGER): Bounding box x coordinate
  - y (INTEGER): Bounding box y coordinate
  - width (INTEGER): Bounding box width in pixels
  - height (INTEGER): Bounding box height in pixels
  - metadata (TEXT): JSON string with additional metadata
  - created_at (TIMESTAMP): Insertion timestamp

Available labels: 'Obstacle', 'Deformation', 'Rupture', 'Deposition', 'Disconnect', 'Misalignment'

Indexes:
  - idx_frame_id on frame_id
  - idx_label on label
  - idx_confidence on confidence
"""
    
    def count_detections(self) -> int:
        """Get total count of detections."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM detections")
        return cursor.fetchone()[0]
    
    def clear_all(self):
        """Clear all detections from database."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM detections")
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
