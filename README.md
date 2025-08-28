# Intelligent Web Scraper with Content Summarizer (2025 Update)

## Overview
Desktop app for scraping web content, summarizing it, analyzing sentiment/bias, and exporting results. Updated for 2025 with async fetching, latest libraries, and improved GUI.

## Installation
1. Create virtual env:

```powershell
python -m venv venv
```

2. Activate:

```powershell
venv\Scripts\activate
```

3. Install deps:

```powershell
pip install -r requirements.txt
```

4. Download models:

```powershell
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon'); nltk.download('stopwords')"
```

5. Run GUI:

```powershell
python -m app.main --gui
```

6. CLI example:

```powershell
python -m app.main --url https://example.com
```

## Updates in 2025
- Async fetching with aiohttp for faster batch processing.
- Latest SBERT model ('all-MiniLM-L12-v2') for better summarization.
- Hugging Face sentiment classifier in analyzer (optional).
- GUI with webview for previewing URLs.
- Enhanced PDF export with images.
- Improved error handling and caching.

## Usage
- GUI: Enter URL, fetch, view results, export.
- CLI: See `app/main.py` for options.

## Troubleshooting
- If NLTK/spaCy fails, download manually as shown above.
- For PyQt issues, ensure `PyQt6-WebEngine` is installed.

Report issues via GitHub Issues.
