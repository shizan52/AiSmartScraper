"""
Main entrypoint for the Intelligent Web Scraper with Content Summarizer (2025 Update).
Handles CLI and GUI modes.
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import List, Dict
from PyQt6.QtWidgets import QApplication

# Import core modules
from app.core.orchestrator import orchestrate_url, orchestrate_batch, orchestrate_csv
from app.gui.window import MainWindow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intelligent Web Scraper with Content Summarizer (2025 Update)")
    
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--url', type=str, help='Single URL to process')
    input_group.add_argument('--batch', nargs='+', type=str, help='List of URLs to process in batch')
    input_group.add_argument('--csv', type=str, help='Path to CSV file containing URLs (column: "url")')
    
    parser.add_argument('--config', type=str, default="configs/default.yaml", help='Path to config YAML file')
    parser.add_argument('--summarization_algorithm', type=str, default=None, help='Override summarization algorithm (tfidf, textrank, sbert, hybrid)')
    parser.add_argument('--summary_length', type=int, default=None, help='Number of sentences in summary')
    parser.add_argument('--output', type=str, default=None, help='Output file path for JSON results')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    
    parser.add_argument('--gui', action='store_true', help='Launch GUI mode')
    
    return parser.parse_args()

def process_results(results: List[Dict], output_file: str = None) -> None:
    for result in results:
        if result.get('error'):
            logger.error(f"Error processing {result.get('url', 'unknown')}: {result['error']}")
        else:
            logger.info(f"Successfully processed {result['url']}")
            logger.info(f"Summary: {result['summary'].get('summary_sentences', [])}")
            logger.info(f"Analysis: {result['analysis']}")
    
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save output: {e}")

def main() -> None:
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    override_params = {}
    if args.summarization_algorithm:
        override_params['summarization_algorithm'] = args.summarization_algorithm
    if args.summary_length:
        override_params['summary_length'] = args.summary_length
    override_params['config_path'] = args.config
    
    if args.gui:
        logger.info("Launching GUI...")
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    
    results = []
    try:
        if args.url:
            logger.info(f"Processing single URL: {args.url}")
            result = orchestrate_url(args.url, **override_params)
            results = [result]
        elif args.batch:
            logger.info(f"Processing batch of {len(args.batch)} URLs")
            results = orchestrate_batch(args.batch, **override_params)
        elif args.csv:
            csv_path = Path(args.csv)
            if not csv_path.exists():
                logger.error(f"CSV file not found: {args.csv}")
                sys.exit(1)
            logger.info(f"Processing URLs from CSV: {args.csv}")
            results = orchestrate_csv(args.csv, **override_params)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    
    process_results(results, args.output)

if __name__ == "__main__":
    main()
"""
Main entrypoint for the Intelligent Web Scraper with Content Summarizer.
This script handles CLI mode for processing URLs (single, batch, or CSV) and can launch GUI if implemented.
It loads config from default.yaml and uses the orchestrator to run the pipeline.

Usage examples:
- Single URL: python main.py --url https://example.com
- Batch URLs: python main.py --batch url1 url2 url3
- CSV file: python main.py --csv path/to/urls.csv
- GUI mode: python main.py --gui (Note: GUI not implemented in this code; placeholder for future)

Output: Prints results to console; future exports can be added via exporter.py.
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import List, Dict

# Import core modules
from app.core.orchestrator import orchestrate_url, orchestrate_batch, orchestrate_csv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Intelligent Web Scraper with Content Summarizer")
    
    # Input options (mutually exclusive group for URL inputs)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--url', type=str, help='Single URL to process')
    input_group.add_argument('--batch', nargs='+', type=str, help='List of URLs to process in batch')
    input_group.add_argument('--csv', type=str, help='Path to CSV file containing URLs (column: "url")')
    
    # Processing options
    parser.add_argument('--config', type=str, default="configs/default.yaml", help='Path to config YAML file')
    parser.add_argument('--summarization_algorithm', type=str, default=None, help='Override summarization algorithm (tfidf, textrank, sbert, hybrid)')
    parser.add_argument('--summary_length', type=int, default=None, help='Number of sentences in summary')
    parser.add_argument('--output', type=str, default=None, help='Output file path for JSON results')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    
    # GUI option
    parser.add_argument('--gui', action='store_true', help='Launch GUI mode')
    
    args = parser.parse_args()
    
    # Check if input is provided when not in GUI mode
    if not args.gui and not (args.url or args.batch or args.csv):
        parser.error("one of the arguments --url --batch --csv is required")
    
    return args

def process_results(results: List[Dict], output_file: str = None) -> None:
    """
    Process and display results from orchestrator.
    Optionally save to JSON file.
    
    Args:
        results: List of result dictionaries from orchestrator
        output_file: Optional path to save JSON output
    """
    for result in results:
        if result.get('error'):
            logger.error(f"Error processing {result.get('url', 'unknown')}: {result['error']}")
        else:
            logger.info(f"Successfully processed {result['url']}")
            logger.info(f"Summary: {result['summary'].get('summary', [])}")
            logger.info(f"Analysis: {result['analysis']}")
    
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save output: {e}")

def main() -> None:
    """
    Main function to handle CLI arguments and run the pipeline.
    """
    args = parse_arguments()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Prepare override parameters from args
    override_params = {}
    if args.summarization_algorithm:
        override_params['summarization_algorithm'] = args.summarization_algorithm
    if args.summary_length:
        override_params['summary_length'] = args.summary_length
    # Always pass config_path to orchestrator functions
    override_params['config_path'] = args.config
    
    # Handle GUI mode (placeholder; implement gui/window.py for actual GUI)
    if args.gui:
        logger.info("GUI mode requested. Launching GUI...")
        # Placeholder: Import and launch GUI here
        from app.gui.window import MainWindow
        from PyQt6.QtWidgets import QApplication
        import sys as sys_module
        app = QApplication(sys_module.argv)
        window = MainWindow()
        window.show()
        sys_module.exit(app.exec())
        logger.warning("GUI not implemented yet. Use CLI mode.")
        sys.exit(0)
    
    # Run orchestrator based on input type
    results = []
    try:
        if args.url:
            logger.info(f"Processing single URL: {args.url}")
            result = orchestrate_url(args.url, **override_params)
            results = [result]
        
        elif args.batch:
            logger.info(f"Processing batch of {len(args.batch)} URLs")
            results = orchestrate_batch(args.batch, **override_params)
        
        elif args.csv:
            csv_path = Path(args.csv)
            if not csv_path.exists():
                logger.error(f"CSV file not found: {args.csv}")
                sys.exit(1)
            logger.info(f"Processing URLs from CSV: {args.csv}")
            results = orchestrate_csv(args.csv, **override_params)
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    
    # Process and output results
    process_results(results, args.output)

if __name__ == "__main__":
    main()