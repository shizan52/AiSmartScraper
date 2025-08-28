"""
gui/window.py: Main UI layout for Intelligent Web Scraper with Content Summarizer (2025 Update).
Added webview for URL preview.
Uses PyQt6 and PyQt6-WebEngine.
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QProgressBar, QSplitter,
    QTextEdit, QListWidget, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QMenuBar, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QAction
from PyQt6.QtWebEngineWidgets import QWebEngineView  # New for webview
from pathlib import Path

# Import core and storage
from app.core.orchestrator import orchestrate_url, orchestrate_batch, orchestrate_csv
from app.export.exporter import get_exporter
from app.storage.database import get_db
from app.storage.fileio import get_fileio
from app.gui.widgets import SentimentMeter, BiasFlagWidget, HistoryWidget

class ProgressThread(QThread):
    progress_signal = pyqtSignal(float, str)
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, url=None, batch=None, csv_path=None):
        super().__init__()
        self.url = url
        self.batch = batch
        self.csv_path = csv_path

    def run(self):
        try:
            if self.url:
                result = orchestrate_url(self.url)
                self.result_signal.emit(result)
            elif self.batch:
                results = orchestrate_batch(self.batch)
                for result in results:
                    self.result_signal.emit(result)
            elif self.csv_path:
                results = orchestrate_csv(self.csv_path)
                for result in results:
                    self.result_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Web Scraper (2025)")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(self.get_stylesheet())

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Input layout
        input_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter URL or CSV path...")
        self.fetch_button = QPushButton("Fetch")
        self.fetch_button.clicked.connect(self.start_processing)
        input_layout.addWidget(self.url_input)
        input_layout.addWidget(self.fetch_button)
        main_layout.addLayout(input_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # Splitter for panes
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.raw_text_edit = QTextEdit()
        self.raw_text_edit.setReadOnly(True)
        splitter.addWidget(self.raw_text_edit)

        # New: Webview for URL preview
        self.web_view = QWebEngineView()
        splitter.addWidget(self.web_view)

        # Summary/Analysis pane
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        self.summary_list = QListWidget()
        analysis_layout.addWidget(QLabel("Summary:"))
        analysis_layout.addWidget(self.summary_list)
        self.sentiment_meter = SentimentMeter()
        analysis_layout.addWidget(self.sentiment_meter)
        self.bias_widget = BiasFlagWidget()
        analysis_layout.addWidget(self.bias_widget)
        splitter.addWidget(analysis_widget)
        main_layout.addWidget(splitter)

        # History pane
        self.history_widget = HistoryWidget()
        main_layout.addWidget(self.history_widget)
        self.history_widget.history_table.itemClicked.connect(self.load_project)

        # Menu for export
        menubar = QMenuBar()
        self.setMenuBar(menubar)
        export_menu = QMenu("Export", self)
        menubar.addMenu(export_menu)
        export_txt_action = QAction("TXT", self)
        export_txt_action.triggered.connect(lambda: self.export_file("TXT"))
        export_menu.addAction(export_txt_action)
        export_json_action = QAction("JSON", self)
        export_json_action.triggered.connect(lambda: self.export_file("JSON"))
        export_menu.addAction(export_json_action)
        export_csv_action = QAction("CSV (Batch)", self)
        export_csv_action.triggered.connect(lambda: self.export_file("CSV"))
        export_menu.addAction(export_csv_action)
        export_pdf_action = QAction("PDF", self)
        export_pdf_action.triggered.connect(lambda: self.export_file("PDF"))
        export_menu.addAction(export_pdf_action)

        self.exporter = get_exporter()
        self.db = get_db()
        self.fileio = get_fileio()
        self.history = []
        self.load_history_from_db()

    def load_history_from_db(self):
        projects = self.db.list_projects()
        for project in projects:
            self.history_widget.update_history(project)

    def start_processing(self):
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Input Error", "Please enter a URL or CSV path.")
            return
        if hasattr(self, 'progress_thread') and self.progress_thread.isRunning():
            QMessageBox.warning(self, "Processing", "A process is already running. Please wait.")
            return

        self.progress_thread = ProgressThread(url=url)
        self.progress_thread.progress_signal.connect(self.update_progress)
        self.progress_thread.result_signal.connect(self.display_result)
        self.progress_thread.error_signal.connect(self.show_error)
        self.progress_thread.start()

    def update_progress(self, value: float, message: str):
        self.progress_bar.setValue(int(value * 100))
        self.progress_bar.setFormat(message)

    def display_result(self, result: dict):
        if 'error' in result:
            self.show_error(result['error'])
            return

        self.raw_text_edit.setText(result['extraction'].get('cleaned_text', ''))
        try:
            self.web_view.setUrl(result['url'])  # New: Load URL in webview
        except Exception:
            pass
        self.summary_list.clear()
        for sent in result['summary'].get('summary_sentences', []):
            self.summary_list.addItem(sent)
        try:
            self.sentiment_meter.update_sentiment(result['analysis'].get('sentiment', {}))
            self.bias_widget.update_bias(result['analysis'].get('bias', {}))
        except Exception:
            pass
        self.history.append(result)
        self.history_widget.update_history({
            'project_id': result.get('project_id'),
            'url': result.get('url'),
            'title': result.get('extraction', {}).get('metadata', {}).get('title', 'Untitled'),
            'timestamp': result.get('metadata', {}).get('timestamp', '')
        })

    def load_project(self, item):
        row = item.row()
        project_id = self.history_widget.history_table.item(row, 0).text()
        project = self.db.get_project(project_id)
        if project:
            raw = self.fileio.read_raw_html(project_id)
            cleaned = self.fileio.read_cleaned_text(project_id)
            summary = self.fileio.read_summary(project_id)
            analysis = self.fileio.read_analysis(project_id)

            self.raw_text_edit.setText(cleaned or raw)
            self.summary_list.clear()
            for sent in (summary.get('summary_sentences', []) if summary else []):
                self.summary_list.addItem(sent)
            try:
                self.sentiment_meter.update_sentiment(analysis.get('sentiment', {}) if analysis else {})
                self.bias_widget.update_bias(analysis.get('bias', {}) if analysis else {})
            except Exception:
                pass

    def export_file(self, format_type):
        if not self.history:
            QMessageBox.warning(self, "Export", "No result to export.")
            return
        result = self.history[-1]
        project_id = result.get('project_id')
        file_path, _ = QFileDialog.getSaveFileName(self, f"Save as {format_type}", "", f"{format_type} Files (*.{format_type.lower()})")
        if file_path:
            try:
                if format_type == "TXT":
                    self.exporter.export_to_txt(project_id, file_path)
                elif format_type == "JSON":
                    self.exporter.export_to_json(project_id, file_path)
                elif format_type == "CSV":
                    self.exporter.export_to_csv([project_id], file_path)
                elif format_type == "PDF":
                    self.exporter.export_to_pdf(project_id, file_path)
                QMessageBox.information(self, "Export", f"Exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")

    def show_error(self, message: str):
        QMessageBox.critical(self, "Error", message)

    def get_stylesheet(self):
        return """
        QMainWindow { background-color: #f0f0f0; }
        QLineEdit { border: 1px solid #ccc; border-radius: 4px; padding: 5px; }
        QPushButton { background-color: #007bff; color: white; border: none; border-radius: 4px; padding: 8px; }
        QPushButton:hover { background-color: #0056b3; }
        QProgressBar { background-color: #e0e0e0; border: 1px solid #ccc; border-radius: 4px; text-align: center; }
        QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007bff, stop:1 #28a745); }
        QTextEdit, QListWidget { border: 1px solid #ddd; border-radius: 4px; padding: 5px; background: white; }
        QTableWidget { border: 1px solid #ddd; background: white; }
        QLabel { color: #333; }
        """

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
from app.export.exporter import exporter_instance
from app.storage.database import get_db
"""
gui/window.py: Main UI layout for Intelligent Web Scraper with Content Summarizer.
This file defines the main window with input controls, progress indicators,
splitter for raw text and summary/analysis panes, and history pane.
Uses PyQt6 for the GUI framework.
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QProgressBar, QSplitter,
    QTextEdit, QListWidget, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QToolBar, QToolButton, QMenuBar,
    QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QIcon, QColor, QPalette, QAction
from pathlib import Path

# Import core modules (assuming they are in the project structure)
from app.core.orchestrator import orchestrator_instance as orchestrator
from app.core.orchestrator import orchestrate_url, orchestrate_batch, orchestrate_csv

# For simplicity, we'll implement custom widgets inline here.
# In a full project, move them to widgets.py.

class ProgressThread(QThread):
    """Thread for processing URLs in background with progress updates."""
    progress_signal = pyqtSignal(float, str)
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, url=None, batch=None, csv_path=None):
        super().__init__()
        self.url = url
        self.batch = batch
        self.csv_path = csv_path

    def run(self):
        try:
            orchestrator.set_progress_callback(self.update_progress)
            if self.url:
                result = orchestrate_url(self.url)
                self.result_signal.emit(result)
            elif self.batch:
                results = orchestrate_batch(self.batch)
                # Emit all results for batch
                for result in results:
                    self.result_signal.emit(result)
            elif self.csv_path:
                results = orchestrate_csv(self.csv_path)
                # Emit all results for CSV
                for result in results:
                    self.result_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))

    def update_progress(self, progress: float, message: str):
        self.progress_signal.emit(progress, message)

class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Web Scraper with Content Summarizer")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(self.get_stylesheet())  # Apply custom stylesheet

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Menu bar
        self.create_menu_bar()

        # Top section: Input and controls
        top_layout = self.create_top_section()
        main_layout.addLayout(top_layout)

        # Middle section: Splitter with left (raw text) and right (summary/analysis)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.create_left_pane())
        splitter.addWidget(self.create_right_pane())
        splitter.setSizes([600, 600])  # Initial sizes
        main_layout.addWidget(splitter)

        # Bottom section: History pane
        history_layout = self.create_history_pane()
        main_layout.addLayout(history_layout)

        # Data storage: project history from database
        self.history = []  # List of project dicts from DB
        self.load_history_from_db()

    def load_history_from_db(self):
        """Load project history from database."""
        db = get_db()
        self.history = db.list_projects()
        self.refresh_history_table()

    def refresh_history_table(self):
        self.history_table.setRowCount(0)
        for idx, entry in enumerate(self.history):
            self.history_table.insertRow(idx)
            self.history_table.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))
            self.history_table.setItem(idx, 1, QTableWidgetItem(entry.get('url', 'Unknown')))
            self.history_table.setItem(idx, 2, QTableWidgetItem(entry.get('title', 'Untitled')))
            self.history_table.setItem(idx, 3, QTableWidgetItem(entry.get('timestamp', '')))

    def create_menu_bar(self):
        """Create menu bar with File, View, Help options."""
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        # File menu
        file_menu = QMenu("File", self)
        open_action = QAction("Open Project", self)
        save_action = QAction("Save Project", self)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        menu_bar.addMenu(file_menu)

        # View menu (e.g., theme toggle)
        view_menu = QMenu("View", self)
        dark_mode_action = QAction("Toggle Dark Mode", self)
        dark_mode_action.triggered.connect(self.toggle_dark_mode)
        view_menu.addAction(dark_mode_action)
        menu_bar.addMenu(view_menu)

        # Help menu
        help_menu = QMenu("Help", self)
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        menu_bar.addMenu(help_menu)

    def create_top_section(self):
        """Create top input section."""
        top_layout = QVBoxLayout()

        # Input row
        input_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter URL to scrape...")
        input_layout.addWidget(self.url_input, stretch=7)

        self.fetch_button = QPushButton("Fetch")
        self.fetch_button.clicked.connect(self.on_fetch_clicked)
        input_layout.addWidget(self.fetch_button, stretch=1)

        self.batch_button = QPushButton("Upload CSV")
        self.batch_button.clicked.connect(self.on_batch_upload)
        input_layout.addWidget(self.batch_button, stretch=1)

        top_layout.addLayout(input_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% - %v")
        top_layout.addWidget(self.progress_bar)

        # Status message
        self.status_label = QLabel("Ready")
        top_layout.addWidget(self.status_label)

        return top_layout

    def create_left_pane(self):
        """Create left pane for raw text."""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Toolbar
        toolbar = QToolBar()
        copy_action = QToolButton()
        copy_action.setText("Copy Text")
        copy_action.clicked.connect(self.copy_raw_text)
        toolbar.addWidget(copy_action)

        search_action = QToolButton()
        search_action.setText("Search")
        # Implement search functionality if needed
        toolbar.addWidget(search_action)
        left_layout.addWidget(toolbar)

        # Raw text editor
        self.raw_text_edit = QTextEdit()
        self.raw_text_edit.setReadOnly(True)
        left_layout.addWidget(self.raw_text_edit)

        return left_widget

    def create_right_pane(self):
        """Create right pane for summary and analysis."""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Summary section
        summary_label = QLabel("Summary")
        summary_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        right_layout.addWidget(summary_label)

        self.summary_list = QListWidget()
        self.summary_list.itemClicked.connect(self.on_summary_item_clicked)
        right_layout.addWidget(self.summary_list)

        # Sentiment meter (custom widget simulation with QProgressBar)
        sentiment_label = QLabel("Sentiment Score")
        sentiment_label.setStyleSheet("font-size: 14pt;")
        right_layout.addWidget(sentiment_label)

        self.sentiment_meter = QProgressBar()
        self.sentiment_meter.setRange(-100, 100)
        self.sentiment_meter.setValue(0)
        self.sentiment_meter.setFormat("Neutral")
        right_layout.addWidget(self.sentiment_meter)

        # Bias flags (simulation with QListWidget)
        bias_label = QLabel("Bias Flags")
        bias_label.setStyleSheet("font-size: 14pt;")
        right_layout.addWidget(bias_label)

        self.bias_list = QListWidget()
        right_layout.addWidget(self.bias_list)

        # Export buttons
        export_layout = QHBoxLayout()
        export_txt = QPushButton("Export TXT")
        export_txt.clicked.connect(lambda: self.export_file("TXT"))
        export_json = QPushButton("Export JSON")
        export_json.clicked.connect(lambda: self.export_file("JSON"))
        export_csv = QPushButton("Export CSV")
        export_csv.clicked.connect(lambda: self.export_file("CSV"))
        export_pdf = QPushButton("Export PDF")
        export_pdf.clicked.connect(lambda: self.export_file("PDF"))
        export_layout.addWidget(export_txt)
        export_layout.addWidget(export_json)
        export_layout.addWidget(export_csv)
        export_layout.addWidget(export_pdf)
        right_layout.addLayout(export_layout)

        return right_widget

    def create_history_pane(self):
        """Create bottom history pane."""
        history_layout = QVBoxLayout()

        history_label = QLabel("History")
        history_label.setStyleSheet("font-size: 14pt;")
        history_layout.addWidget(history_label)

        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(["ID", "URL", "Title", "Timestamp"])
        self.history_table.itemClicked.connect(self.on_history_item_clicked)
        history_layout.addWidget(self.history_table)

        return history_layout

    def on_fetch_clicked(self):
        """Handle Fetch button click."""
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Error", "Please enter a URL.")
            return

        # Prevent starting another processing job if one is running
        existing = getattr(self, 'worker_thread', None)
        if existing and isinstance(existing, QThread) and existing.isRunning():
            QMessageBox.warning(self, "Processing", "A process is already running. Please wait.")
            return

        self.progress_bar.setValue(0)
        self.status_label.setText("Fetching...")

        # disable buttons while running
        self.fetch_button.setEnabled(False)
        self.batch_button.setEnabled(False)

        # start background worker
        self.worker_thread = ProgressThread(url=url)
        self.worker_thread.progress_signal.connect(self.update_progress)
        self.worker_thread.result_signal.connect(self.display_result)
        self.worker_thread.error_signal.connect(self.show_error)
        self.worker_thread.finished.connect(self._on_worker_finished)
        self.worker_thread.start()

    def on_batch_upload(self):
        """Handle Batch Upload button click."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if file_path:
            # Prevent duplicate processing
            if getattr(self, 'worker_thread', None) and isinstance(getattr(self, 'worker_thread'), QThread) and self.worker_thread.isRunning():
                QMessageBox.warning(self, "Processing", "A process is already running. Please wait.")
                return

            self.progress_bar.setValue(0)
            self.status_label.setText("Processing CSV...")

            self.fetch_button.setEnabled(False)
            self.batch_button.setEnabled(False)

            self.worker_thread = ProgressThread(csv_path=file_path)
            self.worker_thread.progress_signal.connect(self.update_progress)
            self.worker_thread.result_signal.connect(self.display_result)
            self.worker_thread.error_signal.connect(self.show_error)
            self.worker_thread.finished.connect(self._on_worker_finished)
            self.worker_thread.start()

    def _on_worker_finished(self):
        """Re-enable buttons and clear worker thread reference when done."""
        try:
            self.fetch_button.setEnabled(True)
            self.batch_button.setEnabled(True)
        finally:
            # clear reference so future checks won't hit a non-thread object
            try:
                self.worker_thread = None
            except Exception:
                pass

    def update_progress(self, progress: float, message: str):
        """Update progress bar and status."""
        self.progress_bar.setValue(int(progress * 100))
        self.status_label.setText(message)

    def display_result(self, result: dict):
        """Display fetched result in UI."""
        if result.get('error'):
            self.show_error(result['error'])
            return

        # Update raw text
        self.raw_text_edit.setText(result.get('extraction', {}).get('clean_text', ''))

        # Update summary
        self.summary_list.clear()
        for sentence in result.get('summary', {}).get('summary', []):
            self.summary_list.addItem(sentence)

        # Update sentiment (simulate meter)
        sentiment = result.get('analysis', {}).get('sentiment', {}).get('compound_score', 0)
        self.sentiment_meter.setValue(int(sentiment * 100))
        label = result.get('analysis', {}).get('sentiment', {}).get('document', 'Neutral')
        self.sentiment_meter.setFormat(label)

        # Update bias
        self.bias_list.clear()
        bias_detected = result.get('analysis', {}).get('bias', {}).get('document_bias', False)
        if bias_detected:
            self.bias_list.addItem("Bias Detected (Score: {:.2f})".format(result['analysis']['bias']['bias_score']))
            for indicator, count in result.get('analysis', {}).get('bias', {}).get('indicator_counts', {}).items():
                self.bias_list.addItem(f"{indicator}: {count} instances")
        else:
            self.bias_list.addItem("No Bias Detected")

        # Add to history (DB integration: reload from DB)
        self.load_history_from_db()
        self.status_label.setText("Processing Complete")

    def on_summary_item_clicked(self, item):
        """Highlight selected summary sentence in raw text."""
        sentence = item.text()
        cursor = self.raw_text_edit.textCursor()
        cursor.clearSelection()
        text = self.raw_text_edit.toPlainText()
        pos = text.find(sentence)
        if pos != -1:
            cursor.setPosition(pos)
            cursor.movePosition(cursor.MoveOperation.Right, cursor.MoveMode.KeepAnchor, len(sentence))
            self.raw_text_edit.setTextCursor(cursor)
            self.raw_text_edit.ensureCursorVisible()

    def on_history_item_clicked(self, item):
        """Load selected history project from DB."""
        row = item.row()
        project_id = self.history[row].get('project_id')
        db = get_db()
        project = db.get_project(project_id)
        if project:
            # Compose result dict for display_result
            result = {
                'url': project.get('url', 'Unknown'),
                'extraction': {
                    'clean_text': self._safe_fileio_read('read_cleaned_text', project_id),
                    'metadata': {'title': project.get('title', 'Untitled')}
                },
                'summary': self._safe_fileio_read('read_summary', project_id),
                'analysis': self._safe_fileio_read('read_analysis', project_id),
                'timestamp': project.get('timestamp', '')
            }
            self.display_result(result)

    def _safe_fileio_read(self, method: str, project_id: str):
        """Safely call fileio read_* methods, fallback to empty string/dict if missing."""
        try:
            from app.storage.fileio import get_fileio
            fileio = get_fileio()
            func = getattr(fileio, method, None)
            if func:
                return func(project_id)
        except Exception:
            pass
        return {} if method in ('read_summary', 'read_analysis') else ""

    def copy_raw_text(self):
        """Copy raw text to clipboard."""
        QApplication.clipboard().setText(self.raw_text_edit.toPlainText())

    def export_file(self, format_type):
        """Export current project using exporter_instance."""
        if not self.history:
            QMessageBox.warning(self, "Export", "No result to export.")
            return
        project_id = self.history[-1].get('project_id')
        file_path, _ = QFileDialog.getSaveFileName(self, f"Save as {format_type}", "", f"{format_type} Files (*.{format_type.lower()})")
        if file_path and project_id:
            try:
                if format_type == "JSON":
                    exporter_instance.export_to_json(project_id, file_path)
                elif format_type == "CSV":
                    exporter_instance.export_to_csv([project_id], file_path)
                elif format_type == "Markdown":
                    exporter_instance.export_to_markdown(project_id, file_path)
                # TXT/PDF export can be added if implemented in exporter.py
                QMessageBox.information(self, "Export", f"Exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")

    def toggle_dark_mode(self):
        """Toggle dark mode (placeholder)."""
        # Implement palette change
        palette = self.palette()
        if palette.color(QPalette.ColorRole.Window).lightness() > 128:
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
            self.setPalette(dark_palette)
        else:
            self.setPalette(QPalette())  # Reset to light
        self.update()

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About", "Intelligent Web Scraper v1.0\nBuilt with PyQt6 and xAI tools.")

    def show_error(self, message: str):
        """Show error message."""
        QMessageBox.critical(self, "Error", message)
        self.status_label.setText("Error occurred")

    def get_stylesheet(self):
        """Custom stylesheet for modern look."""
        return """
        QMainWindow { background-color: #f0f0f0; }
        QLineEdit { border: 1px solid #ccc; border-radius: 4px; padding: 5px; }
        QPushButton { background-color: #007bff; color: white; border: none; border-radius: 4px; padding: 8px; }
        QPushButton:hover { background-color: #0056b3; }
        QProgressBar { background-color: #e0e0e0; border: 1px solid #ccc; border-radius: 4px; text-align: center; }
        QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007bff, stop:1 #28a745); }
        QTextEdit, QListWidget { border: 1px solid #ddd; border-radius: 4px; padding: 5px; background: white; }
        QTableWidget { border: 1px solid #ddd; background: white; }
        QLabel { color: #333; }
        """

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())