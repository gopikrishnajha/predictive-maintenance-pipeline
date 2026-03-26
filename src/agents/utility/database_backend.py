"""
Backend abstraction for database operations.
Provides unified interface for SQLite backend.
"""

from typing import Dict, Any, List
from src.utility.sqlite_client import SQLiteClient


class DatabaseBackend:
    """Unified interface for database operations (SQLite)."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database backend.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        sqlite_cfg = config.get('sqlite', {})
        self.client = SQLiteClient(db_path=sqlite_cfg.get('db_path', 'out/sql_data/detections.db'))
        self.table_name = 'detections'
    
    def query_all(self, limit: int = None) -> List[Dict[str, Any]]:
        """Query all detections with optional limit."""
        query = "SELECT * FROM detections"
        if limit:
            query += f" LIMIT {limit}"
        results = self.client.execute_query(query)
        return self._normalize_sql_results(results)
    
    def query_by_confidence(self, min_conf: float, limit: int = None) -> List[Dict[str, Any]]:
        """Query detections above confidence threshold."""
        query = f"SELECT * FROM detections WHERE confidence >= {min_conf}"
        if limit:
            query += f" LIMIT {limit}"
        results = self.client.execute_query(query)
        return self._normalize_sql_results(results)
    
    def insert_audit(self, audit_data: Dict[str, Any]):
        """Insert audit record (not implemented for SQL yet)."""
        # TODO: Implement audit table for SQL
        pass
    
    def raw_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute raw SQL query."""
        if isinstance(query, str):
            return self.client.execute_query(query)
        else:
            raise ValueError("SQL backend requires string query")
    
    def get_client(self):
        """Get underlying client (for backward compatibility)."""
        return self.client
    
    def _normalize_sql_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize SQL flat structure to match VDMS nested bbox format."""
        normalized = []
        for row in results:
            normalized.append({
                "frame_id": int(row.get("frame_id", 0)),
                "label": row.get("label", ""),
                "confidence": float(row.get("confidence", 0.0)),
                "bbox": {
                    "x": float(row.get("x", 0)),
                    "y": float(row.get("y", 0)),
                    "width": float(row.get("width", 0)),
                    "height": float(row.get("height", 0)),
                }
            })
        return normalized
