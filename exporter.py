"""
app/export/exporter.py: Export project data to various formats.
Handles exporting to TXT, JSON, CSV, PDF.
Integrates with database.py and fileio.py.
Requires reportlab for PDF (pip install reportlab).
"""

import json
import csv
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from app.storage.database import get_db, Database
from app.storage.fileio import get_fileio, FileIO

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Exporter:
    """Manages exporting project data to TXT, JSON, CSV, PDF."""

    def __init__(self, db: Database = None, fileio: FileIO = None):
        self.db = db or get_db()
        self.fileio = fileio or get_fileio()
        self.export_dir = Path("data/exports")
        self._ensure_export_dir()
        logger.info("Exporter initialized")

    def _ensure_export_dir(self):
        """Ensure export directory exists."""
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def _safe_fileio_call(self, method: str, project_id: str):
        """Safely call fileio read methods, fallback if missing."""
        try:
            func = getattr(self.fileio, method, None)
            if func:
                return func(project_id)
        except Exception:
            pass
        return {} if method in ('read_summary', 'read_analysis') else ""

    def export_to_txt(self, project_id: str, output_path: Optional[str] = None, include_summary: bool = True) -> str:
        """
        Export to TXT (cleaned text or summary).

        Args:
            project_id: Project ID.
            output_path: Custom output path.
            include_summary: If True, include summary; else cleaned text.

        Returns:
            Path to TXT file.
        """
        try:
            project = self.db.get_project(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            content = ""
            if include_summary:
                summary = self._safe_fileio_call('read_summary', project_id)
                content = "\n".join(summary.get('summary_sentences', [])) if summary else "No summary available"
            else:
                content = self._safe_fileio_call('read_cleaned_text', project_id) or "No cleaned text available"

            output_file = Path(output_path) if output_path else self.export_dir / f"{project_id}_export.txt"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content, encoding='utf-8')
            logger.info(f"Exported project {project_id} to TXT at {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Failed to export to TXT: {e}")
            raise

    def export_to_json(self, project_id: str, output_path: Optional[str] = None) -> str:
        """
        Export to JSON (structured data).
        """
        try:
            project = self.db.get_project(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            export_data = {
                "metadata": {
                    "project_id": project["project_id"],
                    "url": project["url"],
                    "title": project["title"],
                    "timestamp": project["timestamp"]
                },
                "text": self._safe_fileio_call('read_cleaned_text', project_id),
                "summary": self._safe_fileio_call('read_summary', project_id),
                "analysis": {
                    "sentiment": self._safe_fileio_call('read_analysis', project_id).get('sentiment', {}),
                    "bias": self._safe_fileio_call('read_analysis', project_id).get('bias', {}),
                    "entities": self._safe_fileio_call('read_analysis', project_id).get('entities', [])
                }
            }

            output_file = Path(output_path) if output_path else self.export_dir / f"{project_id}_export.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open('w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Exported project {project_id} to JSON at {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            raise

    def export_to_csv(self, project_ids: List[str], output_path: Optional[str] = None) -> str:
        """
        Batch export to CSV (url, title, summary, sentiment, bias_score).
        """
        try:
            projects = [self.db.get_project(pid) for pid in project_ids if self.db.get_project(pid)]
            if not projects:
                raise ValueError("No valid projects found")

            fields = ["url", "title", "summary", "sentiment", "bias_score"]
            rows = []
            for project in projects:
                pid = project["project_id"]
                summary = self._safe_fileio_call('read_summary', pid).get('summary_sentences', [])
                analysis = self._safe_fileio_call('read_analysis', pid)
                row = {
                    "url": project["url"],
                    "title": project["title"],
                    "summary": " | ".join(summary),
                    "sentiment": analysis.get('sentiment', {}).get('document', 'Neutral'),
                    "bias_score": analysis.get('bias', {}).get('bias_score', 0.0)
                }
                rows.append(row)

            output_file = Path(output_path) if output_path else self.export_dir / "batch_export.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"Exported {len(projects)} projects to CSV at {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            raise

    def export_to_pdf(self, project_id: str, output_path: Optional[str] = None) -> str:
        """
        Export to PDF report (title, summary, key metrics, evidence sentences).
        Uses reportlab.
        """
        try:
            project = self.db.get_project(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            summary = self._safe_fileio_call('read_summary', project_id)
            analysis = self._safe_fileio_call('read_analysis', project_id)

            output_file = Path(output_path) if output_path else self.export_dir / f"{project_id}_report.pdf"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            doc = SimpleDocTemplate(str(output_file), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            story.append(Paragraph(f"Report for {project['title'] or 'Untitled'}", styles['Title']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"URL: {project['url']}", styles['Normal']))
            story.append(Paragraph(f"Timestamp: {project['timestamp']}", styles['Normal']))
            story.append(Spacer(1, 24))

            # Summary
            story.append(Paragraph("Summary:", styles['Heading2']))
            for sent in summary.get('summary_sentences', []):
                story.append(Paragraph(sent, styles['Normal']))
            story.append(Spacer(1, 12))

            # Analysis
            story.append(Paragraph("Analysis:", styles['Heading2']))
            sentiment = analysis.get('sentiment', {})
            bias = analysis.get('bias', {})
            entities = analysis.get('entities', [])

            story.append(Paragraph(f"Sentiment: {sentiment.get('document', 'Neutral')} (Score: {sentiment.get('compound_score', 0.0):.2f})", styles['Normal']))
            story.append(Paragraph(f"Bias Detected: {bias.get('document_bias', False)} (Score: {bias.get('bias_score', 0.0):.2f})", styles['Normal']))
            story.append(Spacer(1, 12))

            if entities:
                story.append(Paragraph("Key Entities:", styles['Heading3']))
                for ent in entities:
                    story.append(Paragraph(f"{ent['text']} ({ent['label']})", styles['Normal']))

            doc.build(story)
            logger.info(f"Exported project {project_id} to PDF at {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Failed to export to PDF: {e}")
            raise

# Singleton instance
exporter_instance = Exporter()

def get_exporter() -> Exporter:
    return exporter_instance