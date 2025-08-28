"""
Text summarization module with multiple extractive algorithms.
Supports TF-IDF, TextRank, SBERT, and Hybrid summarization.
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import download as nltk_download
import string
import yaml
import spacy

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk_download('punkt', quiet=True)
    nltk_download('stopwords', quiet=True)
except:
    logger.warning("NLTK data download failed. Some features may not work properly.")

class TextSummarizer:
    """Extractive text summarization with multiple algorithms."""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        """Initialize the summarizer with parameters from config file."""
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        self.nlp = None  # For NER
        self.sbert_model = None
        self.sbert_available = False
        self._load_config(config_path)
        self._load_ner_model()
        self._check_sbert_availability()

    def _load_config(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)
            self.default_params = config.get('summarizer_params', {
                'algorithm': 'tfidf',
                'summary_length': 5,
                'max_length_ratio': 0.3,
                'diversity_penalty': 0.5,
                'position_bias': True,
                'entity_boost': True,
                'chunk_size': 100  # For huge pages
            })
            logger.info("Summarizer config loaded successfully.")
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            self.default_params = {
                'algorithm': 'tfidf',
                'summary_length': 5,
                'max_length_ratio': 0.3,
                'diversity_penalty': 0.5,
                'position_bias': True,
                'entity_boost': True,
                'chunk_size': 100
            }

    def _load_ner_model(self):
        """Load spaCy NER model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy NER model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}. Entity boosting will use heuristic.")
            self.nlp = None

    def _check_sbert_availability(self):
        """Check if SBERT is available."""
        try:
            from sentence_transformers import SentenceTransformer
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.sbert_available = True
            logger.info("SBERT model loaded successfully.")
        except ImportError:
            logger.warning("sentence-transformers not installed. SBERT summarization disabled.")
            self.sbert_available = False

    def summarize(self, sentences: List[str], **kwargs) -> Dict[str, Any]:
        """
        Generate summary using specified algorithm with chunking for large inputs.
        
        Args:
            sentences: List of sentences to summarize
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Dictionary containing summary and metadata
        """
        if not sentences:
            logger.error("No sentences provided for summarization.")
            return {
                'summary': [],
                'selected_indices': [],
                'algorithm': 'none',
                'error': 'No sentences to summarize'
            }
        
        params = {**self.default_params, **kwargs}
        algorithm = params['algorithm'].lower()
        chunk_size = params.get('chunk_size', 100)

        # Handle huge sentence lists with chunking
        if len(sentences) > chunk_size:
            logger.info(f"Large input detected ({len(sentences)} sentences). Processing in chunks of {chunk_size}.")
            return self._summarize_chunks(sentences, params)
        
        target_length = self._determine_summary_length(len(sentences), params)
        
        if target_length <= 0:
            logger.warning("Target summary length is 0. Returning first sentence as fallback.")
            return {
                'summary': sentences[:1] if sentences else [],
                'selected_indices': [0] if sentences else [],
                'algorithm': algorithm,
                'parameters': params
            }
        
        try:
            if algorithm == 'tfidf':
                selected_indices = self._tfidf_summarize(sentences, target_length, params)
            elif algorithm == 'textrank':
                selected_indices = self._textrank_summarize(sentences, target_length, params)
            elif algorithm == 'sbert' and self.sbert_available:
                selected_indices = self._sbert_summarize(sentences, target_length, params)
            elif algorithm == 'hybrid':
                selected_indices = self._hybrid_summarize(sentences, target_length, params)
            else:
                logger.error(f"Unknown or unavailable algorithm: {algorithm}")
                raise ValueError(f"Unknown or unavailable algorithm: {algorithm}")
            
            if not selected_indices:
                logger.warning("No sentences selected. Using default indices.")
                selected_indices = list(range(min(target_length, len(sentences))))
            
            selected_indices.sort()
            summary_sentences = [sentences[i] for i in selected_indices]
            
            logger.info(f"Generated summary with {len(summary_sentences)} sentences using {algorithm}.")
            return {
                'summary': summary_sentences,
                'selected_indices': selected_indices,
                'algorithm': algorithm,
                'parameters': params,
                'original_sentence_count': len(sentences),
                'summary_sentence_count': len(summary_sentences)
            }
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}. Using fallback.")
            fallback_indices = list(range(min(3, len(sentences))))
            return {
                'summary': [sentences[i] for i in fallback_indices],
                'selected_indices': fallback_indices,
                'algorithm': 'fallback',
                'error': str(e),
                'parameters': params
            }

    def _summarize_chunks(self, sentences: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize large sentence lists by processing in chunks."""
        chunk_size = params.get('chunk_size', 100)
        target_length = self._determine_summary_length(len(sentences), params)
        chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
        
        all_selected_indices = []
        chunk_target_length = max(1, target_length // len(chunks))
        
        for chunk_idx, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)} with {len(chunk)} sentences.")
            result = self.summarize(chunk, **{**params, 'summary_length': chunk_target_length})
            chunk_indices = [i + (chunk_idx * chunk_size) for i in result['selected_indices']]
            all_selected_indices.extend(chunk_indices)
        
        # Ensure we don't exceed target length
        all_selected_indices = sorted(all_selected_indices)[:target_length]
        summary_sentences = [sentences[i] for i in all_selected_indices]
        
        return {
            'summary': summary_sentences,
            'selected_indices': all_selected_indices,
            'algorithm': params['algorithm'],
            'parameters': params,
            'original_sentence_count': len(sentences),
            'summary_sentence_count': len(summary_sentences)
        }

    def _determine_summary_length(self, num_sentences: int, params: Dict[str, Any]) -> int:
        """Determine appropriate summary length based on parameters."""
        length = params['summary_length']
        max_ratio = params['max_length_ratio']
        
        if 0 < length < 1:
            length = max(1, int(length * num_sentences))
        max_length = max(1, int(max_ratio * num_sentences))
        logger.debug(f"Summary length: {min(length, max_length)} (original: {num_sentences})")
        return min(length, max_length)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _tfidf_summarize(self, sentences: List[str], target_length: int, params: Dict[str, Any]) -> List[int]:
        """TF-IDF based summarization."""
        preprocessed_sentences = [self._preprocess_text(s) for s in sentences]
        
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
        try:
            tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        except ValueError as e:
            logger.warning(f"TF-IDF failed: {e}. Using default indices.")
            return list(range(min(target_length, len(sentences))))
        
        if params.get('position_bias', True):
            position_weights = self._get_position_weights(len(sentences))
            sentence_scores *= position_weights
        
        logger.debug(f"TF-IDF scores: {sentence_scores[:5]}...")
        return self._select_top_sentences(sentence_scores, target_length, params)

    def _textrank_summarize(self, sentences: List[str], target_length: int, params: Dict[str, Any]) -> List[int]:
        """TextRank algorithm for summarization."""
        preprocessed_sentences = [self._preprocess_text(s) for s in sentences]
        
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        try:
            sentence_vectors = vectorizer.fit_transform(preprocessed_sentences).toarray()
            similarity_matrix = cosine_similarity(sentence_vectors)
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
            sentence_scores = np.array([scores[i] for i in range(len(sentences))])
        except ValueError as e:
            logger.warning(f"TextRank failed: {e}. Using default indices.")
            return list(range(min(target_length, len(sentences))))
        
        if params.get('position_bias', True):
            position_weights = self._get_position_weights(len(sentences))
            sentence_scores *= position_weights
        
        logger.debug(f"TextRank scores: {sentence_scores[:5]}...")
        return self._select_top_sentences(sentence_scores, target_length, params)

    def _sbert_summarize(self, sentences: List[str], target_length: int, params: Dict[str, Any]) -> List[int]:
        """SBERT-based summarization."""
        if not self.sbert_available:
            logger.warning("SBERT not available. Falling back to TextRank.")
            return self._textrank_summarize(sentences, target_length, params)
        
        try:
            sentence_embeddings = self.sbert_model.encode(sentences)
            doc_embedding = np.mean(sentence_embeddings, axis=0)
            similarities = cosine_similarity([doc_embedding], sentence_embeddings)[0]
            
            if params.get('position_bias', True):
                position_weights = self._get_position_weights(len(sentences))
                similarities *= position_weights
            
            logger.debug(f"SBERT similarities: {similarities[:5]}...")
            return self._select_top_sentences(similarities, target_length, params)
        except Exception as e:
            logger.error(f"SBERT summarization failed: {e}. Falling back to TextRank.")
            return self._textrank_summarize(sentences, target_length, params)

    def _hybrid_summarize(self, sentences: List[str], target_length: int, params: Dict[str, Any]) -> List[int]:
        """Hybrid approach combining multiple methods."""
        tfidf_scores = self._get_tfidf_scores(sentences)
        textrank_scores = self._get_textrank_scores(sentences)
        
        tfidf_scores = self._normalize_scores(tfidf_scores)
        textrank_scores = self._normalize_scores(textrank_scores)
        
        combined_scores = (tfidf_scores + textrank_scores) / 2
        
        if params.get('position_bias', True):
            position_weights = self._get_position_weights(len(sentences))
            combined_scores *= position_weights
        
        if params.get('entity_boost', False):
            entity_weights = self._get_entity_weights(sentences)
            combined_scores *= (1 + entity_weights)
        
        logger.debug(f"Hybrid scores: {combined_scores[:5]}...")
        return self._select_top_sentences(combined_scores, target_length, params)

    def _get_tfidf_scores(self, sentences: List[str]) -> np.ndarray:
        """Get TF-IDF scores for sentences."""
        preprocessed = [self._preprocess_text(s) for s in sentences]
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(preprocessed)
            return np.array(tfidf_matrix.sum(axis=1)).flatten()
        except:
            logger.warning("TF-IDF scoring failed. Returning uniform scores.")
            return np.ones(len(sentences))

    def _get_textrank_scores(self, sentences: List[str]) -> np.ndarray:
        """Get TextRank scores for sentences."""
        preprocessed = [self._preprocess_text(s) for s in sentences]
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            sentence_vectors = vectorizer.fit_transform(preprocessed).toarray()
            similarity_matrix = cosine_similarity(sentence_vectors)
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
            return np.array([scores[i] for i in range(len(sentences))])
        except:
            logger.warning("TextRank scoring failed. Returning uniform scores.")
            return np.ones(len(sentences))

    def _get_position_weights(self, num_sentences: int) -> np.ndarray:
        """Generate position weights (first sentences get higher weight)."""
        positions = np.arange(num_sentences)
        weights = 1.5 - (positions / max(1, num_sentences - 1))
        return np.clip(weights, 0.5, 1.5)

    def _get_entity_weights(self, sentences: List[str]) -> np.ndarray:
        """Boost sentences containing named entities using spaCy."""
        weights = np.zeros(len(sentences))
        
        if not self.nlp:
            # Fallback to heuristic if spaCy is not available
            for i, sentence in enumerate(sentences):
                words = word_tokenize(sentence)
                proper_nouns = [word for word in words if word.istitle() and word.lower() not in self.stop_words]
                if proper_nouns:
                    weights[i] = min(0.5, len(proper_nouns) * 0.1)
        else:
            # Use spaCy NER
            for i, sentence in enumerate(sentences):
                doc = self.nlp(sentence)
                entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]
                if entities:
                    weights[i] = min(0.5, len(entities) * 0.1)
        
        logger.debug(f"Entity weights: {weights[:5]}...")
        return weights

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 0-1 range."""
        if np.max(scores) - np.min(scores) > 0:
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        return np.ones_like(scores)

    def _select_top_sentences(self, scores: np.ndarray, target_length: int, params: Dict[str, Any]) -> List[int]:
        """Select top sentences with diversity consideration."""
        selected_indices = []
        remaining_indices = list(range(len(scores)))
        diversity_penalty = params.get('diversity_penalty', 0.5)
        
        for _ in range(min(target_length, len(scores))):
            if not remaining_indices:
                break
            
            best_idx = remaining_indices[np.argmax(scores[remaining_indices])]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            if diversity_penalty > 0 and remaining_indices:
                for idx in remaining_indices:
                    if abs(idx - best_idx) <= 2:
                        scores[idx] *= (1 - diversity_penalty)
        
        logger.debug(f"Selected indices: {selected_indices}")
        return selected_indices

    def get_available_algorithms(self) -> List[Dict[str, Any]]:
        """Return list of available summarization algorithms."""
        algorithms = [
            {
                'name': 'tfidf',
                'description': 'TF-IDF based summarization (fast and simple)',
                'parameters': ['summary_length', 'position_bias']
            },
            {
                'name': 'textrank',
                'description': 'Graph-based TextRank algorithm (balanced quality)',
                'parameters': ['summary_length', 'position_bias', 'diversity_penalty']
            },
            {
                'name': 'hybrid',
                'description': 'Combination of multiple methods (best overall)',
                'parameters': ['summary_length', 'position_bias', 'diversity_penalty', 'entity_boost']
            }
        ]
        if self.sbert_available:
            algorithms.append({
                'name': 'sbert',
                'description': 'SBERT semantic embeddings (highest quality, requires model)',
                'parameters': ['summary_length', 'position_bias']
            })
        return algorithms

# Singleton instance for easy access
summarizer_instance = TextSummarizer()

def summarize_text(sentences: List[str], **kwargs) -> Dict[str, Any]:
    """Convenience function to summarize text using the singleton summarizer."""
    return summarizer_instance.summarize(sentences, **kwargs)

def get_summarization_algorithms() -> List[Dict[str, Any]]:
    """Get list of available summarization algorithms."""
    return summarizer_instance.get_available_algorithms()