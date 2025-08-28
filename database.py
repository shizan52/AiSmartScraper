"""
app/storage/database.py: SQLite-based project history management.
Manages a local SQLite database to store project metadata and file paths.
Integrates with orchestrator.py for saving projects and gui/window.py for loading history.
"""

import sqlite3
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import uuid
from datetime import datetime

# Configure logging to match other modules
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Database:
    """SQLite database manager for project history."""
    
    def __init__(self, db_path: str = "data/projects.db"):
        """
        Initialize the database connection and create tables if needed.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._ensure_data_folder()
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()

    def _ensure_data_folder(self):
        """Ensure the data/ folder exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data folder ensured: {self.db_path.parent}")

    def _connect(self):
        """Connect to the SQLite database."""
        try:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _create_tables(self):
        """Create the projects table if it doesn't exist."""
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    title TEXT,
                    timestamp TEXT NOT NULL,
                    raw_html_path TEXT,
                    cleaned_text_path TEXT,
                    summary_path TEXT,
                    analysis_path TEXT
                )
            """)
            self.conn.commit()
            logger.info("Projects table created or verified.")
        except sqlite3.Error as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def add_project(self, url: str, title: str, raw_html_path: str, cleaned_text_path: str, summary_path: str, analysis_path: str) -> str:
        """
        Add a new project to the database.
        
        Args:
            url: The processed URL.
            title: The page title.
            raw_html_path: Path to raw HTML file.
            cleaned_text_path: Path to cleaned text file.
            summary_path: Path to summary JSON file.
            analysis_path: Path to analysis JSON file.
        
        Returns:
            The generated project_id.
        """
        project_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        try:
            with self.conn:
                self.cursor.execute("""
                    INSERT INTO projects (project_id, url, title, timestamp, raw_html_path, cleaned_text_path, summary_path, analysis_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (project_id, url, title, timestamp, raw_html_path, cleaned_text_path, summary_path, analysis_path))
            logger.info(f"Added project: {project_id} for URL {url}")
            return project_id
        except sqlite3.Error as e:
            logger.error(f"Failed to add project: {e}")
            raise

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a project by project_id.
        
        Args:
            project_id: The unique project ID.
        
        Returns:
            Dictionary of project data or None if not found.
        """
        try:
            self.cursor.execute("""
                SELECT * FROM projects WHERE project_id = ?
            """, (project_id,))
            row = self.cursor.fetchone()
            if row:
                columns = [desc[0] for desc in self.cursor.description]
                return dict(zip(columns, row))
            else:
                logger.warning(f"Project not found: {project_id}")
                return None
        except sqlite3.Error as e:
            logger.error(f"Failed to get project: {e}")
            raise

    def list_projects(self, search_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all projects, optionally filtered by search query (URL or title).
        
        Args:
            search_query: Optional search string for URL or title.
        
        Returns:
            List of project dictionaries (project_id, url, title, timestamp).
        """
        try:
            if search_query:
                search_query = f"%{search_query}%"
                self.cursor.execute("""
                    SELECT project_id, url, title, timestamp FROM projects
                    WHERE url LIKE ? OR title LIKE ?
                    ORDER BY timestamp DESC
                """, (search_query, search_query))
            else:
                self.cursor.execute("""
                    SELECT project_id, url, title, timestamp FROM projects
                    ORDER BY timestamp DESC
                """)
            rows = self.cursor.fetchall()
            columns = ['project_id', 'url', 'title', 'timestamp']
            return [dict(zip(columns, row)) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to list projects: {e}")
            raise

    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project by project_id.
        
        Args:
            project_id: The unique project ID.
        
        Returns:
            True if deleted, False if not found.
        """
        try:
            with self.conn:
                self.cursor.execute("""
                    DELETE FROM projects WHERE project_id = ?
                """, (project_id,))
                if self.cursor.rowcount > 0:
                    logger.info(f"Deleted project: {project_id}")
                    return True
                else:
                    logger.warning(f"Project not found for deletion: {project_id}")
                    return False
        except sqlite3.Error as e:
            logger.error(f"Failed to delete project: {e}")
            raise

    def update_project(self, project_id: str, **kwargs) -> bool:
        """
        Update a project's fields by project_id.
        
        Args:
            project_id: The unique project ID.
            **kwargs: Fields to update (e.g., title='New Title', raw_html_path='new/path').
        
        Returns:
            True if updated, False if not found.
        """
        if not kwargs:
            logger.warning("No fields provided for update.")
            return False
        
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [project_id]
        
        try:
            with self.conn:
                self.cursor.execute(f"""
                    UPDATE projects SET {set_clause} WHERE project_id = ?
                """, values)
                if self.cursor.rowcount > 0:
                    logger.info(f"Updated project: {project_id}")
                    return True
                else:
                    logger.warning(f"Project not found for update: {project_id}")
                    return False
        except sqlite3.Error as e:
            logger.error(f"Failed to update project: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")

    def __del__(self):
        """Ensure connection is closed on object deletion."""
        self.close()

# Singleton instance for easy access
db_instance = Database()

def get_db() -> Database:
    """Get the singleton database instance."""
    return db_instance