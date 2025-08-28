"""
Text analysis module for sentiment, bias, and entity analysis.
Integrates with extractor.py and summarizer.py for text processing.
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download as nltk_download
import yaml
import spacy
from collections import Counter

# Configure logging to match other modules
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download required NLTK data for VADER
try:
    nltk_download('vader_lexicon', quiet=True)
except Exception as e:
    logger.warning(f"NLTK VADER data download failed: {e}. Sentiment analysis may be limited.")

class TextAnalyzer:
    """Performs sentiment, bias, and entity analysis on text or summaries."""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        """Initialize the analyzer with parameters from config file."""
        self.nlp = None  # For NER
        self.sid = None  # For VADER sentiment analysis
        # Initialize stopwords for heuristic entity extraction
        try:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Failed to load NLTK stopwords: {e}. Using empty set.")
            self.stop_words = set()
        self._load_config(config_path)
        self._load_ner_model()
        self._load_sentiment_analyzer()

    def _load_config(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)
            self.default_params = config.get('analyzer_params', {
                'sentiment_threshold_positive': 0.1,
                'sentiment_threshold_negative': -0.1,
                'chunk_size': 100,  # For huge texts
                'bias_indicators': [
                    r'emotional', r'urgent', r'breaking', r'exclusive', r'shocking',
                    r'controversial', r'alleged', r'sensational'
                ],
                'entity_types': ['PERSON', 'ORG', 'GPE'],
                'bias_score_threshold': 0.3
            })
            logger.info("Analyzer config loaded successfully.")
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            self.default_params = {
                'sentiment_threshold_positive': 0.1,
                'sentiment_threshold_negative': -0.1,
                'chunk_size': 100,
                'bias_indicators': [
                    r'emotional', r'urgent', r'breaking', r'exclusive', r'shocking',
                    r'controversial', r'alleged', r'sensational'
                ],
                'entity_types': ['PERSON', 'ORG', 'GPE'],
                'bias_score_threshold': 0.3
            }

    def _load_ner_model(self):
        """Load spaCy NER model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy NER model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}. Entity analysis will use heuristic.")
            self.nlp = None

    def _load_sentiment_analyzer(self):
        """Load VADER sentiment analyzer."""
        try:
            self.sid = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load VADER: {e}. Sentiment analysis disabled.")
            self.sid = None

    def analyze(self, text: str = "", sentences: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Analyze text or sentences for sentiment, bias, and entities.
        
        Args:
            text: Raw text for analysis (optional if sentences provided)
            sentences: List of sentences for analysis (preferred)
            **kwargs: Analysis-specific parameters
            
        Returns:
            Dictionary containing sentiment, bias, and entity analysis results
        """
        params = {**self.default_params, **kwargs}
        chunk_size = params.get('chunk_size', 100)

        # Use sentences if provided, else split text into sentences
        if sentences is None:
            if not text:
                logger.error("No text or sentences provided for analysis.")
                return {
                    'sentiment': {},
                    'bias': {},
                    'entities': [],
                    'error': 'No text or sentences to analyze'
                }
            sentences = self._split_into_sentences(text)
        
        # Handle huge sentence lists with chunking
        if len(sentences) > chunk_size:
            logger.info(f"Large input detected ({len(sentences)} sentences). Processing in chunks of {chunk_size}.")
            return self._analyze_chunks(sentences, params)
        
        try:
            # Perform sentiment analysis
            sentiment_result = self._analyze_sentiment(sentences, params)
            
            # Perform bias detection
            bias_result = self._detect_bias(sentences, params)
            
            # Perform entity analysis
            entities_result = self._analyze_entities(sentences, params)
            
            logger.info(f"Analysis completed: {len(sentences)} sentences processed.")
            return {
                'sentiment': sentiment_result,
                'bias': bias_result,
                'entities': entities_result,
                'parameters': params,
                'sentence_count': len(sentences),
                'error': None
            }
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}. Returning partial results.")
            return {
                'sentiment': {},
                'bias': {},
                'entities': [],
                'parameters': params,
                'sentence_count': len(sentences),
                'error': str(e)
            }

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.warning(f"Sentence tokenization failed: {e}. Using simple split.")
            return [s.strip() for s in text.split('.') if s.strip()]

    def _analyze_chunks(self, sentences: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze large sentence lists by processing in chunks."""
        chunk_size = params.get('chunk_size', 100)
        chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
        
        sentiment_results = []
        bias_results = []
        all_entities = []
        
        for chunk_idx, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)} with {len(chunk)} sentences.")
            result = self.analyze(sentences=chunk, **params)
            sentiment_results.append(result['sentiment'])
            bias_results.append(result['bias'])
            all_entities.extend(result['entities'])
        
        # Aggregate results
        aggregated_sentiment = self._aggregate_sentiment(sentiment_results)
        aggregated_bias = self._aggregate_bias(bias_results)
        aggregated_entities = self._aggregate_entities(all_entities)
        
        return {
            'sentiment': aggregated_sentiment,
            'bias': aggregated_bias,
            'entities': aggregated_entities,
            'parameters': params,
            'sentence_count': len(sentences),
            'error': None
        }

    def _analyze_sentiment(self, sentences: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sentiment analysis on sentences."""
        if not self.sid:
            logger.warning("VADER not available. Sentiment analysis skipped.")
            return {'document': 'neutral', 'sentence_scores': []}
        
        sentence_scores = []
        for sentence in sentences:
            scores = self.sid.polarity_scores(sentence)
            compound = scores['compound']
            label = self._get_sentiment_label(compound, params)
            sentence_scores.append({
                'sentence': sentence,
                'compound_score': compound,
                'label': label,
                'details': scores
            })
        
        # Aggregate document-level sentiment
        if sentence_scores:
            avg_compound = float(np.mean([s['compound_score'] for s in sentence_scores]))
            doc_label = self._get_sentiment_label(avg_compound, params)
        else:
            avg_compound = 0.0
            doc_label = 'neutral'
        
        logger.debug(f"Document sentiment: {doc_label} (avg compound: {avg_compound:.3f})")
        return {
            'document': doc_label,
            'compound_score': avg_compound,
            'sentence_scores': sentence_scores
        }

    def _get_sentiment_label(self, compound_score: float, params: Dict[str, Any]) -> str:
        """Determine sentiment label based on compound score."""
        pos_threshold = params.get('sentiment_threshold_positive', 0.1)
        neg_threshold = params.get('sentiment_threshold_negative', -0.1)
        
        if compound_score >= pos_threshold:
            return 'positive'
        elif compound_score <= neg_threshold:
            return 'negative'
        else:
            return 'neutral'

    def _detect_bias(self, sentences: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential bias in text using heuristic-based approach."""
        bias_indicators = params.get('bias_indicators', [])
        bias_score_threshold = params.get('bias_score_threshold', 0.3)

        # Validate bias indicators as regex patterns
        compiled_indicators = []
        for indicator in bias_indicators:
            try:
                compiled_indicators.append(re.compile(indicator, re.I))
            except re.error as e:
                logger.warning(f"Invalid bias indicator regex '{indicator}': {e}")

        bias_scores = []
        total_indicators = 0
        indicator_counts = Counter()

        for sentence in sentences:
            sentence_lower = sentence.lower()
            indicators_found = []
            for regex in compiled_indicators:
                if regex.search(sentence_lower):
                    indicators_found.append(regex.pattern)
                    indicator_counts[regex.pattern] += 1
                    total_indicators += 1

            score = len(indicators_found) / max(1, len(compiled_indicators))
            bias_scores.append({
                'sentence': sentence,
                'bias_score': score,
                'indicators': indicators_found
            })

        # Aggregate document-level bias
        avg_bias_score = float(np.mean([s['bias_score'] for s in bias_scores]) if bias_scores else 0.0)
        bias_detected = avg_bias_score >= bias_score_threshold

        logger.debug(f"Bias detection: {'Detected' if bias_detected else 'Not detected'} (avg score: {avg_bias_score:.3f})")
        return {
            'document_bias': bias_detected,
            'bias_score': avg_bias_score,
            'sentence_scores': bias_scores,
            'indicator_counts': dict(indicator_counts)
        }

    def _analyze_entities(self, sentences: List[str], params: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract entities using spaCy or heuristic."""
        entity_types = params.get('entity_types', ['PERSON', 'ORG', 'GPE'])
        entities = []
        
        if not self.nlp:
            # Fallback to heuristic
            logger.warning("spaCy not available. Using heuristic for entity extraction.")
            from nltk.tokenize import word_tokenize
            for sentence in sentences:
                words = word_tokenize(sentence)
                proper_nouns = [word for word in words if word.istitle() and word.lower() not in self.stop_words]
                for noun in proper_nouns:
                    entities.append({
                        'text': noun,
                        'label': 'UNKNOWN',
                        'sentence': sentence
                    })
        else:
            # Use spaCy NER
            for sentence in sentences:
                doc = self.nlp(sentence)
                for ent in doc.ents:
                    if ent.label_ in entity_types:
                        entities.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'sentence': sentence
                        })
        
        logger.debug(f"Extracted {len(entities)} entities.")
        return entities

    def _aggregate_sentiment(self, sentiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate sentiment results from chunks."""
        if not sentiment_results:
            return {'document': 'neutral', 'compound_score': 0.0, 'sentence_scores': []}
        
        all_sentence_scores = []
        for result in sentiment_results:
            all_sentence_scores.extend(result.get('sentence_scores', []))
        
        avg_compound = float(np.mean([s['compound_score'] for s in all_sentence_scores]) if all_sentence_scores else 0.0)
        doc_label = self._get_sentiment_label(avg_compound, self.default_params)
        
        return {
            'document': doc_label,
            'compound_score': avg_compound,
            'sentence_scores': all_sentence_scores
        }

    def _aggregate_bias(self, bias_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate bias results from chunks."""
        if not bias_results:
            return {'document_bias': False, 'bias_score': 0.0, 'sentence_scores': [], 'indicator_counts': {}}
        
        all_sentence_scores = []
        all_indicators = Counter()
        
        for result in bias_results:
            all_sentence_scores.extend(result.get('sentence_scores', []))
            all_indicators.update(result.get('indicator_counts', {}))
        
        avg_bias_score = float(np.mean([s['bias_score'] for s in all_sentence_scores]) if all_sentence_scores else 0.0)
        bias_detected = avg_bias_score >= self.default_params.get('bias_score_threshold', 0.3)
        
        return {
            'document_bias': bias_detected,
            'bias_score': avg_bias_score,
            'sentence_scores': all_sentence_scores,
            'indicator_counts': dict(all_indicators)
        }

    def _aggregate_entities(self, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Aggregate entities from chunks, removing duplicates."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            entity_key = (entity['text'], entity['label'])
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)
        
        return unique_entities

    def get_available_features(self) -> List[Dict[str, Any]]:
        """Return list of available analysis features."""
        features = [
            {
                'name': 'sentiment',
                'description': 'Sentiment analysis using VADER (positive/negative/neutral)',
                'available': bool(self.sid)
            },
            {
                'name': 'bias',
                'description': 'Heuristic-based bias detection',
                'available': True
            },
            {
                'name': 'entities',
                'description': 'Named Entity Recognition using spaCy or heuristic',
                'available': True
            }
        ]
        return features

# Singleton instance for easy access
analyzer_instance = TextAnalyzer()

def analyze_text(text: str = "", sentences: List[str] = None, **kwargs) -> Dict[str, Any]:
    """Convenience function to analyze text using the singleton analyzer."""
    return analyzer_instance.analyze(text, sentences, **kwargs)

def get_analysis_features() -> List[Dict[str, Any]]:
    """Get list of available analysis features."""
    return analyzer_instance.get_available_features()