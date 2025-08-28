"""
gui/widgets.py: Custom widgets for the GUI (2025 Update).
Added tooltips and better styling.
"""

from typing import Dict, Any
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QListWidget, QTableWidget, QTableWidgetItem, QLineEdit, QPushButton
from PyQt6.QtCore import Qt

class SentimentMeter(QWidget):
    """Custom widget for sentiment score (2025: Added detailed tooltip)."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(50)
        self.score = 0.0
        self.label = "Neutral"
        self.compound = 0.0
        self.positive = 0.0
        self.negative = 0.0
        self.neutral = 0.0
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.title_label = QLabel("Sentiment Score")
        self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #333;")
        layout.addWidget(self.title_label)
        
        self.meter = QProgressBar()
        self.meter.setRange(-100, 100)
        self.meter.setValue(0)
        self.meter.setTextVisible(True)
        self.meter.setFormat("Neutral")
        layout.addWidget(self.meter)
        
        self.setToolTip("Hover for detailed sentiment scores")

    def update_sentiment(self, sentiment: Dict[str, Any]):
        self.compound = sentiment.get('compound_score', 0.0)
        self.label = sentiment.get('document', "Neutral")
        try:
            self.meter.setValue(int(self.compound * 100))
        except Exception:
            self.meter.setValue(0)
        self.meter.setFormat(self.label)
        # Detailed tooltip
        tooltip = f"Compound: {self.compound:.2f}\nPositive: {sentiment.get('positive', 0.0):.2f}\nNegative: {sentiment.get('negative', 0.0):.2f}\nNeutral: {sentiment.get('neutral', 0.0):.2f}"
        self.setToolTip(tooltip)

class BiasFlagWidget(QWidget):
    """Custom widget for bias detection (2025: Added color indicators)."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(50)
        self.bias_detected = False
        self.bias_score = 0.0
        self.indicators = {}
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.title_label = QLabel("Bias Detection")
        self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #333;")
        layout.addWidget(self.title_label)
        
        self.bias_label = QLabel("No Bias Detected")
        self.bias_label.setStyleSheet("font-size: 12pt; color: green;")
        layout.addWidget(self.bias_label)
        
        self.score_label = QLabel("Score: 0.00")
        layout.addWidget(self.score_label)
        
        self.indicator_list = QListWidget()
        layout.addWidget(self.indicator_list)
        
        self.setToolTip("Bias score and indicators")

    def update_bias(self, bias: Dict[str, Any]):
        self.bias_detected = bias.get('document_bias', False)
        self.bias_score = bias.get('bias_score', 0.0)
        self.indicators = bias.get('indicator_counts', {})
        
        if self.bias_detected:
            self.bias_label.setText("Bias Detected")
            self.bias_label.setStyleSheet("font-size: 12pt; color: red;")
        else:
            self.bias_label.setText("No Bias Detected")
            self.bias_label.setStyleSheet("font-size: 12pt; color: green;")
        
        self.score_label.setText(f"Score: {self.bias_score:.2f}")
        
        self.indicator_list.clear()
        for ind, count in self.indicators.items():
            self.indicator_list.addItem(f"{ind}: {count}")

class HistoryWidget(QWidget):
    """Widget for history pane (2025: Added search and delete)."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.title_label = QLabel("Project History")
        self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #333;")
        layout.addWidget(self.title_label)
        
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by URL or title...")
        self.search_input.textChanged.connect(self.filter_history)
        search_layout.addWidget(self.search_input)
        
        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self.delete_selected)
        search_layout.addWidget(self.delete_button)
        layout.addLayout(search_layout)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["ID", "URL", "Title", "Timestamp"])
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.history_table)
        
        self.history_data = []

    def update_history(self, entry: Dict[str, Any]):
        self.history_data.append(entry)
        self.refresh_history()

    def refresh_history(self):
        self.history_table.setRowCount(0)
        for i, entry in enumerate(self.history_data):
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)
            self.history_table.setItem(row, 0, QTableWidgetItem(entry.get('project_id', str(i + 1))))
            self.history_table.setItem(row, 1, QTableWidgetItem(entry.get('url', 'Unknown')))
            self.history_table.setItem(row, 2, QTableWidgetItem(entry.get('title', 'Untitled')))
            self.history_table.setItem(row, 3, QTableWidgetItem(entry.get('timestamp', '')))

    def filter_history(self, search_text: str):
        self.history_table.setRowCount(0)
        for entry in self.history_data:
            if search_text.lower() in entry.get('url', '').lower() or search_text.lower() in entry.get('title', '').lower():
                row = self.history_table.rowCount()
                self.history_table.insertRow(row)
                self.history_table.setItem(row, 0, QTableWidgetItem(entry.get('project_id', '')))
                self.history_table.setItem(row, 1, QTableWidgetItem(entry.get('url', 'Unknown')))
                self.history_table.setItem(row, 2, QTableWidgetItem(entry.get('title', 'Untitled')))
                self.history_table.setItem(row, 3, QTableWidgetItem(entry.get('timestamp', '')))

    def delete_selected(self) -> None:
        selected = self.history_table.selectedItems()
        if selected:
            try:
                row = selected[0].row()
                if 0 <= row < len(self.history_data):
                    self.history_data.pop(row)
                    self.refresh_history()
            except Exception as e:
                print(f"Error deleting history entry: {e}")

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # Test widgets
    window = QWidget()
    layout = QVBoxLayout(window)
    
    sentiment_meter = SentimentMeter()
    sentiment_meter.update_sentiment({
        'compound_score': 0.75,
        'document': 'Positive',
        'sentence_scores': [{'positive': 0.8, 'negative': 0.1, 'neutral': 0.1}]
    })
    layout.addWidget(sentiment_meter)
    
    bias_widget = BiasFlagWidget()
    bias_widget.update_bias({
        'document_bias': True,
        'bias_score': 0.45,
        'indicator_counts': {'emotional': 3, 'sensational': 2}
    })
    layout.addWidget(bias_widget)
    
    history_widget = HistoryWidget()
    history_widget.update_history({
        'project_id': 'test_id',
        'url': 'https://example.com',
        'title': 'Sample Article',
        'timestamp': '2025-08-26 22:34:00'
    })
    layout.addWidget(history_widget)
    
    window.show()
    sys.exit(app.exec())
"""
gui/widgets.py: Custom reusable widgets for the Intelligent Web Scraper GUI.
Includes SentimentMeter and BiasFlagWidget for displaying analysis results.
Uses PyQt6 for GUI components.
"""

try:
    from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QListWidget, QToolTip, QTableWidget, QTableWidgetItem, QLineEdit, QPushButton
    from PyQt6.QtCore import Qt, QSize
    from PyQt6.QtGui import QColor, QPainter, QLinearGradient, QFont
    HAS_PYQT6 = True
except Exception:
    # Allow module import even if PyQt6 is not installed. Instantiating widgets will raise a clear error.
    HAS_PYQT6 = False

    class _MissingPyQtBase:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyQt6 is not installed in this environment. Install it with 'pip install PyQt6' to use GUI widgets.")

    # Define placeholder names so file can be imported without PyQt6 present
    QWidget = _MissingPyQtBase
    QVBoxLayout = object
    QHBoxLayout = object
    QLabel = _MissingPyQtBase
    QProgressBar = _MissingPyQtBase
    QListWidget = _MissingPyQtBase
    QToolTip = _MissingPyQtBase
    QTableWidget = _MissingPyQtBase
    QTableWidgetItem = _MissingPyQtBase
    QLineEdit = _MissingPyQtBase
    QPushButton = _MissingPyQtBase
    Qt = None
    QSize = None
    QColor = None
    QPainter = None
    QLinearGradient = None
    QFont = None

class SentimentMeter(QWidget):
    """Custom widget to display sentiment score as a gradient meter."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(50)
        self.score = 0.0  # Sentiment score (-1 to +1)
        self.label = "Neutral"  # Sentiment label
        self.compound = 0.0
        self.positive = 0.0
        self.negative = 0.0
        self.neutral = 0.0
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        self.title_label = QLabel("Sentiment Score")
        self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #333;")
        layout.addWidget(self.title_label)
        
        # Progress bar
        self.meter = QProgressBar()
        self.meter.setRange(-100, 100)
        self.meter.setValue(0)
        self.meter.setTextVisible(True)
        self.meter.setFormat("Neutral")
        self.meter.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #e0e0e0;
                text-align: center;
                color: #333;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ff0000, stop:0.5 #cccccc, stop:1 #00ff00);
            }
        """)
        layout.addWidget(self.meter)
        
        # Set tooltip for detailed scores
        self.setToolTip("Click for detailed sentiment analysis")
        self.meter.mousePressEvent = self.show_detailed_scores
        
    def update_sentiment(self, sentiment_data: dict) -> None:
        """Update the meter with new sentiment data (average over all sentences)."""
        self.score = sentiment_data.get('compound_score', 0.0)
        self.label = sentiment_data.get('document', 'Neutral')
        self.compound = sentiment_data.get('compound_score', 0.0)
        scores = sentiment_data.get('sentence_scores', [])
        if scores:
            self.positive = sum(s.get('positive', 0.0) for s in scores) / len(scores)
            self.negative = sum(s.get('negative', 0.0) for s in scores) / len(scores)
            self.neutral = sum(s.get('neutral', 0.0) for s in scores) / len(scores)
        else:
            self.positive = self.negative = self.neutral = 0.0
        self.meter.setValue(int(self.score * 100))
        self.meter.setFormat(self.label)
        self.setToolTip(
            f"Sentiment: {self.label}\n"
            f"Compound: {self.compound:.2f}\n"
            f"Positive (avg): {self.positive:.2f}\n"
            f"Negative (avg): {self.negative:.2f}\n"
            f"Neutral (avg): {self.neutral:.2f}"
        )
        
    def show_detailed_scores(self, event):
        """Show detailed sentiment scores in a tooltip-like popup."""
        QToolTip.showText(
            self.mapToGlobal(self.meter.pos()),
            f"Detailed Sentiment Scores:\n"
            f"Compound: {self.compound:.2f}\n"
            f"Positive: {self.positive:.2f}\n"
            f"Negative: {self.negative:.2f}\n"
            f"Neutral: {self.neutral:.2f}",
            self,
            self.meter.geometry(),
            3000  # Show for 3 seconds
        )

class BiasFlagWidget(QWidget):
    """Custom widget to display bias indicators with evidence."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        self.title_label = QLabel("Bias Flags")
        self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #333;")
        layout.addWidget(self.title_label)
        
        # Bias list
        self.bias_list = QListWidget()
        self.bias_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                background: white;
                padding: 5px;
            }
            QListWidget::item:hover {
                background: #e6f3ff;
            }
        """)
        self.bias_list.itemClicked.connect(self.show_bias_evidence)
        layout.addWidget(self.bias_list)
        
    def update_bias(self, bias_data: dict) -> None:
        """Update the widget with new bias data."""
        self.bias_list.clear()
        bias_detected = bias_data.get('document_bias', False)
        bias_score = bias_data.get('bias_score', 0.0)
        indicator_counts = bias_data.get('indicator_counts', {})
        self.evidence = bias_data.get('evidence', {})  # evidence sentences per indicator
        if bias_detected:
            self.bias_list.addItem(f"Bias Detected (Score: {bias_score:.2f})")
            self.bias_list.setStyleSheet("""
                QListWidget {
                    border: 2px solid #ff6666;
                    border-radius: 4px;
                    background: #fff5f5;
                    padding: 5px;
                }
                QListWidget::item:hover {
                    background: #ffe6e6;
                }
            """)
            for indicator, count in indicator_counts.items():
                self.bias_list.addItem(f"{indicator}: {count} instances")
        else:
            self.bias_list.addItem("No Bias Detected")
            self.bias_list.setStyleSheet("""
                QListWidget {
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    background: white;
                    padding: 5px;
                }
                QListWidget::item:hover {
                    background: #e6f3ff;
                }
            """)
            
    def show_bias_evidence(self, item) -> None:
        """Show evidence sentences for selected bias indicator."""
        text = item.text()
        if "No Bias Detected" in text or "Bias Detected" in text:
            return
        indicator = text.split(":")[0]
        evidence_sentences = self.evidence.get(indicator, []) if hasattr(self, 'evidence') else []
        if evidence_sentences:
            msg = f"Evidence for '{indicator}':\n" + "\n".join(evidence_sentences[:3])
        else:
            msg = f"No evidence sentences found for '{indicator}'."
        QToolTip.showText(
            self.mapToGlobal(self.bias_list.pos()),
            msg,
            self,
            self.bias_list.geometry(),
            3000
        )

class HistoryWidget(QWidget):
    """Custom widget to display project history in a table."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        self.title_label = QLabel("History")
        self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #333;")
        layout.addWidget(self.title_label)
        
        # History table
        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(["ID", "URL", "Title", "Timestamp"])
        self.history_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                background: white;
            }
            QTableWidget::item:hover {
                background: #e6f3ff;
            }
        """)
        self.history_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.history_table)
        
        # Toolbar for history actions
        toolbar = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by URL or title...")
        self.search_input.textChanged.connect(self.filter_history)
        toolbar.addWidget(self.search_input)
        
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_selected)
        toolbar.addWidget(self.delete_button)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_history)
        toolbar.addWidget(self.refresh_button)
        
        layout.addLayout(toolbar)
        
        # Store history data (replace with database.py in full impl)
        self.history_data = []
        
    def update_history(self, history_entry: dict) -> None:
        """Add a new history entry to the table."""
        self.history_data.append(history_entry)
        self.refresh_history()
        
    def refresh_history(self) -> None:
        """Refresh the history table."""
        self.history_table.setRowCount(0)
        for entry in self.history_data:
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)
            self.history_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.history_table.setItem(row, 1, QTableWidgetItem(entry.get('url', 'Unknown')))
            self.history_table.setItem(row, 2, QTableWidgetItem(entry.get('title', 'Untitled')))
            self.history_table.setItem(row, 3, QTableWidgetItem(entry.get('timestamp', '')))
        
    def filter_history(self) -> None:
        """Filter history table based on search input."""
        search_text = self.search_input.text().lower()
        self.history_table.setRowCount(0)
        for entry in self.history_data:
            if search_text in entry.get('url', '').lower() or search_text in entry.get('title', '').lower():
                row = self.history_table.rowCount()
                self.history_table.insertRow(row)
                self.history_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
                self.history_table.setItem(row, 1, QTableWidgetItem(entry.get('url', 'Unknown')))
                self.history_table.setItem(row, 2, QTableWidgetItem(entry.get('title', 'Untitled')))
                self.history_table.setItem(row, 3, QTableWidgetItem(entry.get('timestamp', '')))
        
    def delete_selected(self) -> None:
        """Delete selected history entry (with exception handling)."""
        selected = self.history_table.selectedItems()
        if selected:
            try:
                row = selected[0].row()
                if 0 <= row < len(self.history_data):
                    self.history_data.pop(row)
                    self.refresh_history()
            except Exception as e:
                print(f"Error deleting history entry: {e}")

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # Test widgets
    window = QWidget()
    layout = QVBoxLayout(window)
    
    sentiment_meter = SentimentMeter()
    sentiment_meter.update_sentiment({
        'compound_score': 0.75,
        'document': 'Positive',
        'sentence_scores': [{'positive': 0.8, 'negative': 0.1, 'neutral': 0.1}]
    })
    layout.addWidget(sentiment_meter)
    
    bias_widget = BiasFlagWidget()
    bias_widget.update_bias({
        'document_bias': True,
        'bias_score': 0.45,
        'indicator_counts': {'emotional': 3, 'sensational': 2}
    })
    layout.addWidget(bias_widget)
    
    history_widget = HistoryWidget()
    history_widget.update_history({
        'url': 'https://example.com',
        'title': 'Sample Article',
        'timestamp': '2025-08-26 22:34:00'
    })
    layout.addWidget(history_widget)
    
    window.show()
    sys.exit(app.exec())