"""
Orchestrator module to manage the pipeline of fetching, extracting, summarizing, and analyzing web content.
Integrates fetcher.py, extractor.py, summarizer.py, analyzer.py, database.py, fileio.py.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
import yaml
from urllib.parse import urlparse
import time
import csv
from pathlib import Path

# Import other module functions
from app.core.fetcher import fetch_url
from app.core.extractor import extract_content
from app.core.summarizer import summarize_text
from app.core.analyzer import analyze_text
from app.storage.database import get_db
from app.storage.fileio import get_fileio

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Orchestrator:
    """Manages the pipeline of fetching, extracting, summarizing, analyzing, and storing web content."""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        self._load_config(config_path)
        self.progress_callback = None
        self.db = get_db()
        self.fileio = get_fileio()

    def _load_config(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)
            self.default_params = config.get('orchestrator_params', {
                'summarization_algorithm': 'hybrid',
                'summary_length': 5,
                'max_length_ratio': 0.3,
                'diversity_penalty': 0.5,
                'position_bias': True,
                'entity_boost': True,
                'chunk_size': 100,
                'sentiment_threshold_positive': 0.1,
                'sentiment_threshold_negative': -0.1,
                'bias_score_threshold': 0.3,
                'batch_size': 10,
                'delay_between_urls': 1.0
            })
            logger.info("Orchestrator config loaded successfully.")
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            self.default_params = {
                'summarization_algorithm': 'hybrid',
                'summary_length': 5,
                'max_length_ratio': 0.3,
                'diversity_penalty': 0.5,
                'position_bias': True,
                'entity_boost': True,
                'chunk_size': 100,
                'sentiment_threshold_positive': 0.1,
                'sentiment_threshold_negative': -0.1,
                'bias_score_threshold': 0.3,
                'batch_size': 10,
                'delay_between_urls': 1.0
            }

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        self.progress_callback = callback
        logger.info("Progress callback set.")

    def _report_progress(self, progress: float, message: str):
        if self.progress_callback:
            self.progress_callback(progress, message)

    def _save_project(self, url: str, html_content: str, extraction: Dict[str, Any], summary: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Save project to files and database, return project_id."""
        title = extraction.get('metadata', {}).get('title', 'Untitled')
        project_id = urlparse(url).netloc + '_' + time.strftime("%Y%m%d_%H%M%S")  # Simple unique ID

        raw_path = self.fileio.save_raw_html(project_id, html_content)
        cleaned_path = self.fileio.save_cleaned_text(project_id, extraction.get('cleaned_text', ''))
        summary_path = self.fileio.save_summary(project_id, summary)
        analysis_path = self.fileio.save_analysis(project_id, analysis)

        project_id = self.db.add_project(url, title, raw_path, cleaned_path, summary_path, analysis_path)
        logger.info(f"Saved project {project_id} for URL {url}")
        return project_id

    def process_url(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Process a single URL through the pipeline and save to storage.

        Args:
            url: URL to process
            **kwargs: Override default parameters

        Returns:
            Result dictionary with project_id added
        """
        params = {**self.default_params, **kwargs}
        self._report_progress(0.0, f"Starting process for {url}")

        # Fetch
        html_content, error, metadata = fetch_url(url)
        if error:
            return {'url': url, 'error': error}
        self._report_progress(0.25, "Fetched content")

        # Extract
        extraction = extract_content(html_content, url)
        self._report_progress(0.5, "Extracted content")

        # Summarize
        sentences = extraction.get('sentences', [])
        summary = summarize_text(sentences, **params)
        self._report_progress(0.75, "Generated summary")

        # Analyze
        analysis = analyze_text(extraction.get('cleaned_text', ''), sentences, **params)
        self._report_progress(0.9, "Performed analysis")

        # Save to storage
        project_id = self._save_project(url, html_content, extraction, summary, analysis)

        result = {
            'url': url,
            'metadata': metadata,
            'extraction': extraction,
            'summary': summary,
            'analysis': analysis,
            'project_id': project_id
        }
        self._report_progress(1.0, f"Completed process for {url}")
        return result

    def process_batch(self, urls: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Process batch of URLs."""
        params = {**self.default_params, **kwargs}
        results = []
        for i, url in enumerate(urls):
            self._report_progress(i / len(urls), f"Processing URL {i+1}/{len(urls)}: {url}")
            result = self.process_url(url, **params)
            results.append(result)
            if i < len(urls) - 1:
                time.sleep(params['delay_between_urls'])
        self._report_progress(1.0, f"Batch processing completed for {len(urls)} URLs")
        return results

    def process_csv(self, csv_path: str, url_column: str = "url", **kwargs) -> List[Dict[str, Any]]:
        """Process URLs from CSV."""
        try:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                logger.error(f"CSV file not found: {csv_path}")
                return [{'error': f"CSV file not found: {csv_path}"}]

            urls = []
            with open(csv_path, 'r', encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                if url_column not in reader.fieldnames:
                    logger.error(f"URL column '{url_column}' not found in CSV")
                    return [{'error': f"URL column '{url_column}' not found in CSV"}]
                urls = [row[url_column].strip() for row in reader if row[url_column].strip()]

            logger.info(f"Loaded {len(urls)} URLs from {csv_path}")
            return self.process_batch(urls, **kwargs)
        except Exception as e:
            logger.error(f"Failed to process CSV: {str(e)}")
            return [{'error': f"Failed to process CSV: {str(e)}"}]

# Singleton instance
orchestrator_instance = Orchestrator()

def orchestrate_url(url: str, **kwargs) -> Dict[str, Any]:
    return orchestrator_instance.process_url(url, **kwargs)

def orchestrate_batch(urls: List[str], **kwargs) -> List[Dict[str, Any]]:
    return orchestrator_instance.process_batch(urls, **kwargs)

def orchestrate_csv(csv_path: str, url_column: str = "url", **kwargs) -> List[Dict[str, Any]]:
    return orchestrator_instance.process_csv(csv_path, url_column, **kwargs)