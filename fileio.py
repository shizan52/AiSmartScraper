"""
app/storage/fileio.py: File input/output operations for project data.
Handles reading and writing of raw HTML, cleaned text, summary, and analysis files.
Integrates with orchestrator.py for project data persistence.
"""

import json
from typing import Any
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging to match other modules
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FileIO:
    """Manages file input/output operations for project data."""

    def __init__(self, data_dir: str = "data/projects"):
        """
        Initialize the FileIO with a base data directory.

        Args:
            data_dir: Base directory for storing project files.
        """
        self.data_dir = Path(data_dir)
        self._ensure_data_dir()
        logger.info(f"FileIO initialized with data directory: {self.data_dir}")

    def _ensure_data_dir(self):
        """Ensure the data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Data directory ensured: {self.data_dir}")

    def save_raw_html(self, project_id: str, html_content: str) -> str:
        """
        Save raw HTML content to a file.

        Args:
            project_id: Unique project identifier.
            html_content: Raw HTML content to save.

        Returns:
            Path to the saved HTML file.
        """
        try:
            file_path = self.data_dir / f"{project_id}_raw.html"
            file_path.write_text(html_content, encoding='utf-8')
            logger.info(f"Saved raw HTML for project {project_id} to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save raw HTML for project {project_id}: {e}")
            raise

    def save_cleaned_text(self, project_id: str, cleaned_text: str) -> str:
        """
        Save cleaned text content to a file.

        Args:
            project_id: Unique project identifier.
            cleaned_text: Cleaned text content to save.

        Returns:
            Path to the saved text file.
        """
        try:
            file_path = self.data_dir / f"{project_id}_cleaned.txt"
            file_path.write_text(cleaned_text, encoding='utf-8')
            logger.info(f"Saved cleaned text for project {project_id} to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save cleaned text for project {project_id}: {e}")
            raise

    def save_summary(self, project_id: str, summary: Dict[str, Any]) -> str:
        """
        Save summary data to a JSON file.

        Args:
            project_id: Unique project identifier.
            summary: Summary data as a dictionary.

        Returns:
            Path to the saved JSON file.
        """
        try:
            file_path = self.data_dir / f"{project_id}_summary.json"
            def _json_default(obj: Any):
                # Handle numpy types if numpy is available
                try:
                    import numpy as _np
                    if isinstance(obj, (_np.integer,)):
                        return int(obj)
                    if isinstance(obj, (_np.floating,)):
                        return float(obj)
                    if isinstance(obj, (_np.bool_,)):
                        return bool(obj)
                except Exception:
                    pass
                # Sets -> list
                if isinstance(obj, set):
                    return list(obj)
                # Fallback: try to convert to str
                return str(obj)

            with file_path.open('w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
            logger.info(f"Saved summary for project {project_id} to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save summary for project {project_id}: {e}")
            raise

    def save_analysis(self, project_id: str, analysis: Dict[str, Any]) -> str:
        """
        Save analysis data to a JSON file.

        Args:
            project_id: Unique project identifier.
            analysis: Analysis data as a dictionary.

        Returns:
            Path to the saved JSON file.
        """
        try:
            file_path = self.data_dir / f"{project_id}_analysis.json"
            def _json_default(obj: Any):
                try:
                    import numpy as _np
                    if isinstance(obj, (_np.integer,)):
                        return int(obj)
                    if isinstance(obj, (_np.floating,)):
                        return float(obj)
                    if isinstance(obj, (_np.bool_,)):
                        return bool(obj)
                except Exception:
                    pass
                if isinstance(obj, set):
                    return list(obj)
                return str(obj)

            with file_path.open('w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2, default=_json_default)
            logger.info(f"Saved analysis for project {project_id} to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save analysis for project {project_id}: {e}")
            raise

    def read_raw_html(self, project_id: str) -> Optional[str]:
        """
        Read raw HTML content from a file.

        Args:
            project_id: Unique project identifier.

        Returns:
            HTML content as a string or None if the file doesn't exist.
        """
        file_path = self.data_dir / f"{project_id}_raw.html"
        try:
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8')
                logger.debug(f"Read raw HTML for project {project_id} from {file_path}")
                return content
            logger.warning(f"Raw HTML file not found for project {project_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to read raw HTML for project {project_id}: {e}")
            raise

    def read_cleaned_text(self, project_id: str) -> Optional[str]:
        """
        Read cleaned text content from a file.

        Args:
            project_id: Unique project identifier.

        Returns:
            Cleaned text content as a string or None if the file doesn't exist.
        """
        file_path = self.data_dir / f"{project_id}_cleaned.txt"
        try:
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8')
                logger.debug(f"Read cleaned text for project {project_id} from {file_path}")
                return content
            logger.warning(f"Cleaned text file not found for project {project_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to read cleaned text for project {project_id}: {e}")
            raise

    def read_summary(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Read summary data from a JSON file.

        Args:
            project_id: Unique project identifier.

        Returns:
            Summary data as a dictionary or None if the file doesn't exist.
        """
        file_path = self.data_dir / f"{project_id}_summary.json"
        try:
            if file_path.exists():
                with file_path.open('r', encoding='utf-8') as f:
                    content = json.load(f)
                logger.debug(f"Read summary for project {project_id} from {file_path}")
                return content
            logger.warning(f"Summary file not found for project {project_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to read summary for project {project_id}: {e}")
            raise

    def read_analysis(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Read analysis data from a JSON file.

        Args:
            project_id: Unique project identifier.

        Returns:
            Analysis data as a dictionary or None if the file doesn't exist.
        """
        file_path = self.data_dir / f"{project_id}_analysis.json"
        try:
            if file_path.exists():
                with file_path.open('r', encoding='utf-8') as f:
                    content = json.load(f)
                logger.debug(f"Read analysis for project {project_id} from {file_path}")
                return content
            logger.warning(f"Analysis file not found for project {project_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to read analysis for project {project_id}: {e}")
            raise

    def delete_project_files(self, project_id: str) -> bool:
        """
        Delete all files associated with a project.

        Args:
            project_id: Unique project identifier.

        Returns:
            True if all files were deleted or didn't exist, False otherwise.
        """
        file_extensions = ['_raw.html', '_cleaned.txt', '_summary.json', '_analysis.json']
        success = True
        for ext in file_extensions:
            file_path = self.data_dir / f"{project_id}{ext}"
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted file {file_path} for project {project_id}")
            except Exception as e:
                logger.error(f"Failed to delete file {file_path} for project {project_id}: {e}")
                success = False
        return success

# Singleton instance for easy access
fileio_instance = FileIO()

def get_fileio() -> FileIO:
    """Get the singleton FileIO instance."""
    return fileio_instance