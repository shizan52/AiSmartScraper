"""
HTML content extractor module.
Extracts clean text and metadata from HTML content using BeautifulSoup.
"""

import re
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from bs4 import BeautifulSoup, Comment
import html
from urllib.parse import urlparse
import yaml  # For loading config
from langdetect import detect, LangDetectException  # For language detection
import nltk  # For advanced sentence splitting
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# Configure logging with more detail
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more detailed logging
logger = logging.getLogger(__name__)

class ContentExtractor:
    """Extracts clean content and metadata from HTML."""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        """
        Initialize the content extractor with polite scraping settings from config file.
        
        Args:
            config_path: Path to the YAML config file.
        """
        self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)
            self.tags_to_remove = config.get('tags_to_remove', [
                'script', 'style', 'nav', 'footer', 'aside', 'header',
                'form', 'button', 'input', 'select', 'textarea',
                'iframe', 'object', 'embed', 'canvas'
            ])
            self.non_content_selectors = config.get('non_content_selectors', [
                'nav', 'navigation', 'menu', 'sidebar', 'footer',
                'comments', 'advertisement', 'ad-container', 'ad-wrapper',
                'social-share', 'share-buttons', 'newsletter', 'subscribe',
                'cookie-consent', 'popup', 'modal', 'overlay'
            ])
            self.meta_tags_of_interest = config.get('meta_tags_of_interest', [
                'title', 'description', 'keywords', 'author', 'og:title',
                'og:description', 'og:image', 'og:type', 'og:url',
                'twitter:title', 'twitter:description', 'twitter:image',
                'article:published_time', 'article:modified_time',
                'article:author', 'article:section', 'dc.date', 'dc.creator'
            ])
            self.unwanted_patterns = config.get('unwanted_patterns', [
                r'Share this article',
                r'Follow us on',
                r'Subscribe to our newsletter',
                r'Read more:',
                r'Advertisement',
                r'Sponsored content',
                r'Related articles',
                r'Comments',
                r'Sign in',
                r'Create account',
                r'Log in'
            ])
            self.min_sentence_length = config.get('min_sentence_length', 10)
            self.huge_page_threshold = config.get('huge_page_threshold', 1000000)  # Bytes threshold for huge pages
            logger.info("Config loaded successfully.")
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            self.tags_to_remove = [
                'script', 'style', 'nav', 'footer', 'aside', 'header',
                'form', 'button', 'input', 'select', 'textarea',
                'iframe', 'object', 'embed', 'canvas'
            ]
            self.non_content_selectors = [
                'nav', 'navigation', 'menu', 'sidebar', 'footer',
                'comments', 'advertisement', 'ad-container', 'ad-wrapper',
                'social-share', 'share-buttons', 'newsletter', 'subscribe',
                'cookie-consent', 'popup', 'modal', 'overlay'
            ]
            self.meta_tags_of_interest = [
                'title', 'description', 'keywords', 'author', 'og:title',
                'og:description', 'og:image', 'og:type', 'og:url',
                'twitter:title', 'twitter:description', 'twitter:image',
                'article:published_time', 'article:modified_time',
                'article:author', 'article:section', 'dc.date', 'dc.creator'
            ]
            self.unwanted_patterns = [
                r'Share this article',
                r'Follow us on',
                r'Subscribe to our newsletter',
                r'Read more:',
                r'Advertisement',
                r'Sponsored content',
                r'Related articles',
                r'Comments',
                r'Sign in',
                r'Create account',
                r'Log in'
            ]
            self.min_sentence_length = 10
            self.huge_page_threshold = 1000000

    def extract_content(self, html_content: str, url: str = "") -> Dict[str, Any]:
        """
        Extract clean content and metadata from HTML.
        
        Args:
            html_content: Raw HTML content
            url: Source URL for reference
            
        Returns:
            Dictionary containing cleaned text, sentences, and metadata
        """
        # Check for huge pages
        if len(html_content) > self.huge_page_threshold:
            logger.warning(f"Huge page detected: {len(html_content)} bytes. Truncating to threshold.")
            html_content = html_content[:self.huge_page_threshold]  # Truncate to avoid memory issues

        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except Exception as e:
            logger.warning(f"Failed to parse HTML with lxml, using html.parser: {e}")
            soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        self._remove_unwanted_elements(soup)
        
        # Extract metadata
        metadata = self._extract_metadata(soup, url)
        
        # Find the main content element
        main_content = self._find_main_content(soup)
        if not main_content or not hasattr(main_content, 'get_text'):
            logger.warning("No main content found, using entire body or soup as fallback.")
            main_content = soup.find('body') or soup
        # Clean the content
        clean_text = self._clean_content(main_content)
        # Fallback: if clean_text is still empty, try extracting from soup.body or soup
        if not clean_text:
            logger.warning("Clean text is empty after main extraction. Trying soup.body or soup as last resort.")
            fallback_element = soup.find('body') or soup
            clean_text = self._clean_content(fallback_element)
        # Language detection
        detected_lang = self._detect_language(clean_text)
        if detected_lang != 'en' or not clean_text:
            logger.warning(f"Non-English or unknown content detected ({detected_lang}) or clean_text is empty. Returning empty content.")
            return {
                'clean_text': '',
                'sentences': [],
                'metadata': metadata,
                'images': [],
                'word_count': 0,
                'sentence_count': 0,
                'extraction_time': datetime.now().isoformat(),
                'error': f"Unsupported or unknown language: {detected_lang} or empty content"
            }
        # Split into sentences
        sentences = self._split_into_sentences(clean_text)
        # Defensive: if sentences extraction fails, return clean_text as one sentence
        if not sentences:
            logger.warning("No sentences extracted. Returning clean_text as single sentence.")
            sentences = [clean_text] if clean_text else []
        # Extract images
        images = self._extract_images(main_content, url)
        # Defensive: ensure images is always a list
        if images is None:
            images = []
        return {
            'clean_text': clean_text,
            'sentences': sentences,
            'metadata': metadata,
            'images': images,
            'word_count': len(clean_text.split()),
            'sentence_count': len(sentences),
            'extraction_time': datetime.now().isoformat()
        }
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        if not text or len(text) < 50:
            logger.warning("Text too short for reliable language detection. Returning 'unknown'.")
            return 'unknown'
        try:
            lang = detect(text)
            logger.debug(f"Detected language: {lang}")
            return lang
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}. Returning 'unknown'.")
            return 'unknown'
        except Exception as e:
            logger.warning(f"Unexpected error in language detection: {e}. Returning 'unknown'.")
            return 'unknown'
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """Remove unwanted tags and elements from the soup."""
        # Remove specified tags
        for tag in self.tags_to_remove:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Remove elements with non-content classes/IDs
        for selector in self.non_content_selectors:
            for element in soup.find_all(class_=re.compile(selector, re.I)):
                element.decompose()
            for element in soup.find_all(id=re.compile(selector, re.I)):
                element.decompose()
        
        # Remove empty elements
        for element in soup.find_all():
            if not element.get_text().strip() and not element.find_all():
                element.decompose()
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        metadata = {
            'url': url,
            'domain': urlparse(url).netloc if url else '',
            'title': '',
            'description': '',
            'author': '',
            'publish_date': '',
            'keywords': [],
            'meta_tags': {}
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Extract meta tags
        for meta_tag in soup.find_all('meta'):
            name = meta_tag.get('name') or meta_tag.get('property') or meta_tag.get('itemprop')
            content = meta_tag.get('content', '')
            
            if name and content:
                name = name.lower()
                metadata['meta_tags'][name] = content
                
                # Map to standard fields
                if name in ['title', 'og:title', 'twitter:title'] and not metadata['title']:
                    metadata['title'] = content
                elif name in ['description', 'og:description', 'twitter:description'] and not metadata['description']:
                    metadata['description'] = content
                elif name in ['author', 'article:author', 'dc.creator'] and not metadata['author']:
                    metadata['author'] = content
                elif name in ['keywords', 'news_keywords']:
                    metadata['keywords'] = [k.strip() for k in content.split(',') if k.strip()]
                elif name in ['article:published_time', 'dc.date', 'date'] and not metadata['publish_date']:
                    metadata['publish_date'] = self._parse_date(content)
        
        # Try to find title from h1 if not found in meta
        if not metadata['title']:
            h1 = soup.find('h1')
            if h1:
                metadata['title'] = h1.get_text().strip()
        
        # Try to find publish date from article tags
        if not metadata['publish_date']:
            metadata['publish_date'] = self._find_publish_date(soup)
        
        return metadata
    
    def _parse_date(self, date_str: str) -> str:
        """Parse date string into ISO format."""
        try:
            # Common date formats
            date_formats = [
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%d %H:%M:%S%z',
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%B %d, %Y',
                '%d %B %Y',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S'
            ]
            
            for fmt in date_formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.isoformat()
                except ValueError:
                    continue
            
            # If no format matches, return original string
            return date_str
        except:
            return date_str
    
    def _find_publish_date(self, soup: BeautifulSoup) -> str:
        """Try to find publish date from various elements."""
        # Look for time elements with datetime attribute
        time_element = soup.find('time', {'datetime': True})
        if time_element:
            return self._parse_date(time_element['datetime'])
        
        # Look for common date patterns in text
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}',
            r'[A-Za-z]+ \d{1,2}, \d{4}'
        ]
        
        for pattern in date_patterns:
            matches = soup.find_all(string=re.compile(pattern))
            for match in matches:
                if len(match.strip()) < 50:  # Avoid long text blocks
                    try:
                        return self._parse_date(match.strip())
                    except:
                        continue
        
        return ''
    
    def _find_main_content(self, soup: BeautifulSoup) -> Optional[Any]:
        """Find the main content element using various strategies."""
        strategies = [
            self._find_by_article_tag,
            self._find_by_main_tag,
            self._find_by_content_selectors,
            self._find_by_largest_div
        ]
        for strategy in strategies:
            try:
                content = strategy(soup)
                if content and len(content.get_text().strip()) > 100:
                    logger.debug(f"Main content found using {strategy.__name__}")
                    return content
            except Exception as e:
                logger.warning(f"Error in main content strategy {strategy.__name__}: {e}")
        logger.warning("No main content found using any strategy.")
        return None
    
    def _find_by_article_tag(self, soup: BeautifulSoup) -> Optional[Any]:
        """Find content using article tag."""
        article = soup.find('article')
        if article and len(article.get_text().strip()) > 100:
            return article
        return None
    
    def _find_by_main_tag(self, soup: BeautifulSoup) -> Optional[Any]:
        """Find content using main tag."""
        main = soup.find('main')
        if main and len(main.get_text().strip()) > 100:
            return main
        return None
    
    def _find_by_content_selectors(self, soup: BeautifulSoup) -> Optional[Any]:
        """Find content using common content class/ID patterns."""
        content_patterns = [
            'content', 'main-content', 'article-content', 'post-content',
            'entry-content', 'story-content', 'body-content', 'text-content'
        ]
        
        for pattern in content_patterns:
            # Search by class
            for class_ in soup.find_all(class_=re.compile(pattern, re.I)):
                if len(class_.get_text().strip()) > 100:
                    return class_
            
            # Search by ID
            for id_ in soup.find_all(id=re.compile(pattern, re.I)):
                if len(id_.get_text().strip()) > 100:
                    return id_
        
        return None
    
    def _find_by_largest_div(self, soup: BeautifulSoup) -> Optional[Any]:
        """Find the largest text-containing div."""
        divs = soup.find_all('div')
        largest_div = None
        max_text_length = 0
        
        for div in divs:
            text_length = len(div.get_text().strip())
            if text_length > max_text_length and text_length > 100:
                max_text_length = text_length
                largest_div = div
        
        return largest_div
    
    def _clean_content(self, element: Any) -> str:
        """Clean and format the content text."""
        if not element:
            return ""
        
        # Get text with proper spacing
        text = element.get_text(separator=' ', strip=True)
        
        # Clean up the text
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Unescape HTML entities
        text = html.unescape(text)
        
        # Remove common unwanted patterns
        for pattern in self.unwanted_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK sent_tokenize.
        """
        if not text:
            return []
        
        sentences = sent_tokenize(text)
        
        # Clean sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > self.min_sentence_length:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _extract_images(self, element: Any, base_url: str) -> List[Dict[str, str]]:
        """Extract images from the content element."""
        images = []
        for img in element.find_all('img', src=True):
            src = img['src']
            alt = img.get('alt', '')
            title = img.get('title', '')
            # Make relative URLs absolute
            if src.startswith('/'):
                parsed_url = urlparse(base_url)
                src = f"{parsed_url.scheme}://{parsed_url.netloc}{src}"
            elif src.startswith('//'):
                src = f"https:{src}"
            elif not src.startswith(('http://', 'https://')):
                parsed_url = urlparse(base_url)
                base_path = parsed_url.path.rsplit('/', 1)[0] if '/' in parsed_url.path else ''
                src = f"{parsed_url.scheme}://{parsed_url.netloc}{base_path}/{src}"
            images.append({
                'src': src,
                'alt': alt,
                'title': title,
                'caption': self._find_image_caption(img)
            })
        # Defensive: filter out images with empty src or likely icons/logos
        filtered = [img for img in images if img['src'] and 'logo' not in img['src'].lower() and 'icon' not in img['src'].lower()]
        logger.debug(f"Extracted {len(filtered)} valid images.")
        return filtered
    
    def _find_image_caption(self, img_element: Any) -> str:
        """Try to find caption for an image."""
        # Check parent figure for figcaption
        parent = img_element.parent
        if parent and parent.name == 'figure':
            figcaption = parent.find('figcaption')
            if figcaption:
                return figcaption.get_text().strip()
        
        # Check next sibling for caption
        next_sibling = img_element.find_next_sibling()
        if next_sibling and next_sibling.name in ['p', 'div']:
            text = next_sibling.get_text().strip()
            if text and len(text) < 200:  # Reasonable caption length
                return text
        
        # Check for title or alt as fallback
        return img_element.get('title', '') or img_element.get('alt', '')

# Singleton instance for easy access
extractor_instance = ContentExtractor()

def extract_content(html_content: str, url: str = "") -> Dict[str, Any]:
    """
    Convenience function to extract content using the singleton extractor.
    
    Args:
        html_content: Raw HTML content
        url: Source URL for reference
        
    Returns:
        Dictionary containing cleaned text, sentences, and metadata
    """
    return extractor_instance.extract_content(html_content, url)