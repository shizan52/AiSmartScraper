"""
Web content fetcher module with polite scraping capabilities.
Handles HTTP requests with respect for robots.txt, rate limiting, and proper error handling.
"""

import requests
import urllib.robotparser
from urllib.parse import urlparse, urljoin
import time
import logging
from typing import Optional, Tuple, Dict, Any
from requests.exceptions import RequestException, Timeout, HTTPError
import re
import yaml  # For loading config from YAML

# Configure logging with more detail
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more detailed logging
logger = logging.getLogger(__name__)

class PoliteFetcher:
    """A polite web fetcher that respects robots.txt and implements rate limiting."""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        """
        Initialize the fetcher with polite scraping settings from config file.
        
        Args:
            config_path: Path to the YAML config file.
        """
        self._load_config(config_path)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
        self.session.max_redirects = 10  # Limit redirects to prevent loops
        self.domain_last_request = {}  # Track last request time per domain
        self.domain_delays = {}  # Track crawl delays per domain from robots.txt
        self.robot_parsers = {}  # Cache robot parsers for domains
        
    def _load_config(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)
            self.user_agent = config.get('user_agent', "IntelligentWebScraper/1.0")
            self.timeout = config.get('timeout', 20)  # Increased default timeout for slow/JS-heavy sites
            self.max_retries = config.get('max_retries', 3)
            self.default_delay = config.get('delay_between_requests', 1.0)
            logger.info(f"Loaded config: user_agent={self.user_agent}, timeout={self.timeout}, "
                        f"max_retries={self.max_retries}, default_delay={self.default_delay}")
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            self.user_agent = "IntelligentWebScraper/1.0"
            self.timeout = 10
            self.max_retries = 3
            self.default_delay = 1.0
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}. Using defaults.")
            self.user_agent = "IntelligentWebScraper/1.0"
            self.timeout = 10
            self.max_retries = 3
            self.default_delay = 1.0
    
    def _get_robot_parser(self, url: str) -> Optional[urllib.robotparser.RobotFileParser]:
        """
        Get or create a robots.txt parser for the given URL's domain.
        
        Args:
            url: The URL to check robots.txt for
            
        Returns:
            RobotFileParser instance or None if robots.txt is inaccessible
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        if domain in self.robot_parsers:
            return self.robot_parsers[domain]
        
        # Create new robot parser
        robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        
        try:
            response = self.session.get(robots_url, timeout=self.timeout)
            if response.status_code == 200:
                rp.parse(response.text.splitlines())
                self.robot_parsers[domain] = rp
                # Get crawl delay if available
                crawl_delay = rp.crawl_delay(self.user_agent)
                if crawl_delay is not None:
                    self.domain_delays[domain] = max(float(crawl_delay), self.default_delay)
                    logger.debug(f"Crawl delay for {domain}: {self.domain_delays[domain]}s")
                else:
                    self.domain_delays[domain] = self.default_delay
                logger.info(f"Successfully parsed robots.txt for {domain}")
                return rp
            else:
                logger.warning(f"robots.txt not found or inaccessible for {domain} (status: {response.status_code})")
                self.domain_delays[domain] = self.default_delay
                return None
        except Exception as e:
            logger.warning(f"Failed to fetch robots.txt for {domain}: {e}")
            self.domain_delays[domain] = self.default_delay
            return None
    
    def _check_robots_txt(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Check if the URL is allowed by robots.txt.
        
        Args:
            url: URL to check
            
        Returns:
            Tuple of (is_allowed, reason_if_disallowed)
        """
        rp = self._get_robot_parser(url)
        if rp is None:
            # If we can't access robots.txt, assume it's allowed
            return True, None
        
        if rp.can_fetch(self.user_agent, url):
            return True, None
        else:
            disallow_reason = f"URL disallowed by robots.txt for {urlparse(url).netloc}"
            return False, disallow_reason
    
    def _respect_rate_limit(self, domain: str) -> None:
        """
        Ensure we respect the rate limit by delaying if needed, including crawl delay.
        
        Args:
            domain: Domain to check rate limiting for
        """
        delay = self.domain_delays.get(domain, self.default_delay)
        current_time = time.time()
        if domain in self.domain_last_request:
            time_since_last = current_time - self.domain_last_request[domain]
            if time_since_last < delay:
                sleep_time = delay - time_since_last
                logger.info(f"Respecting rate limit/crawl delay: sleeping for {sleep_time:.2f}s before next request to {domain}")
                time.sleep(sleep_time)
        
        self.domain_last_request[domain] = time.time()
    
    def _handle_http_error(self, response: requests.Response, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Handle HTTP errors and return appropriate messages.
        
        Args:
            response: HTTP response object
            url: URL that was requested
            
        Returns:
            Tuple of (content, error_message)
        """
        logger.debug(f"HTTP error for {url}: status_code={response.status_code}, headers={response.headers}")
        if response.status_code == 429:  # Too Many Requests
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                try:
                    retry_seconds = int(retry_after)
                except ValueError:
                    # Try to parse as HTTP-date format
                    try:
                        from email.utils import parsedate_to_datetime
                        retry_date = parsedate_to_datetime(retry_after)
                        retry_seconds = (retry_date.timestamp() - time.time())
                    except:
                        retry_seconds = 60  # Default to 60 seconds
                
                error_msg = f"Rate limited. Server asks to retry after {retry_seconds} seconds."
                return None, error_msg
            else:
                return None, "Rate limited by server. Please try again later."
        
        elif response.status_code == 403:
            return None, "Access forbidden. The server denied access to this resource."
        
        elif response.status_code == 404:
            return None, "Page not found. The requested URL does not exist."
        
        elif response.status_code >= 500:
            return None, f"Server error (HTTP {response.status_code}). Please try again later."
        
        else:
            return None, f"HTTP error {response.status_code}: {response.reason}"
    
    def fetch(self, url: str, renderer_fallback: bool = False) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """
        Fetch content from a URL with polite scraping practices.
        
        Args:
            url: URL to fetch content from
            renderer_fallback: Whether to suggest using a renderer for JS content
            
        Returns:
            Tuple of (content, error_message, metadata)
        """
        # Validate URL format
        if not re.match(r'^https?://', url):
            return None, "Invalid URL format. Must start with http:// or https://", None
        
        # Check robots.txt
        allowed, disallow_reason = self._check_robots_txt(url)
        if not allowed:
            return None, disallow_reason, None
        
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Respect rate limiting including crawl delay
        self._respect_rate_limit(domain)
        
        metadata = {
            'url': url,
            'domain': domain,
            'timestamp': time.time(),
            'content_type': None,
            'status_code': None,
            'headers': {}
        }
        
        # Try fetching with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching {url} (attempt {attempt + 1}/{self.max_retries}) with headers: {self.session.headers}")
                
                response = self.session.get(
                    url, 
                    timeout=self.timeout,
                    allow_redirects=True,
                    headers={'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}
                )
                
                logger.debug(f"Response for {url}: status_code={response.status_code}, headers={response.headers}, "
                             f"redirect_history={[r.url for r in response.history]}")
                
                try:
                    html_content = response.text
                except Exception:
                    try:
                        html_content = response.content.decode(response.encoding or 'utf-8', errors='replace')
                    except Exception:
                        html_content = ''
                
                metadata['status_code'] = getattr(response, 'status_code', None)
                metadata['content_type'] = response.headers.get('Content-Type', '') if hasattr(response, 'headers') else ''
                metadata['headers'] = dict(response.headers) if hasattr(response, 'headers') else {}
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Check if content is HTML
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' not in content_type:
                    logger.warning(f"Content type is not HTML: {content_type}")
                    # Return a specific message for non-HTML content
                    return html_content, f"Non-HTML content detected (type: {content_type}). May not be suitable for extraction.", metadata
                
                # Check for potential paywall or login required
                if self._detect_paywall(html_content):
                    metadata['paywall_detected'] = True
                    return html_content, "Potential paywall or login requirement detected", metadata
                
                # Check for JavaScript-rendered content
                if self._needs_js_rendering(html_content) and renderer_fallback:
                    metadata['js_rendering_suggested'] = True
                    return html_content, "This page may require JavaScript rendering for full content", metadata
                
                return html_content, None, metadata
                
            except HTTPError as e:
                # response may not be available if HTTPError is raised before response assignment
                resp = getattr(e, 'response', None)
                if resp is not None:
                    content, error_msg = self._handle_http_error(resp, url)
                else:
                    content, error_msg = None, f"HTTP error occurred but no response object available: {str(e)}"
                if error_msg and attempt == self.max_retries - 1:
                    logger.error(error_msg)
                    return None, error_msg, metadata

            except Timeout as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Request timed out after {self.timeout} seconds: {str(e)}")
                    return None, f"Request timed out after {self.timeout} seconds", metadata
                logger.warning(f"Timeout on attempt {attempt + 1}, retrying... {str(e)}")

            except RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Network error: {str(e)}")
                    return None, f"Network error: {str(e)}", metadata
                logger.warning(f"Network error on attempt {attempt + 1}: {str(e)}")

            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Unexpected error: {str(e)}")
                    return None, f"Unexpected error: {str(e)}", metadata
                logger.warning(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            
            # Wait before retrying (exponential backoff)
            if attempt < self.max_retries - 1:
                sleep_time = (2 ** attempt) * 0.5  # 0.5, 1, 2 seconds
                logger.info(f"Waiting {sleep_time:.1f}s before retry...")
                time.sleep(sleep_time)
        
        return None, "Failed to fetch content after all retries", metadata
    
    def _detect_paywall(self, html_content: str) -> bool:
        """
        Simple heuristic to detect potential paywalls or login requirements.
        
        Args:
            html_content: HTML content to analyze
            
        Returns:
            Boolean indicating if paywall is likely detected
        """
        paywall_indicators = [
            r'paywall', r'subscription', r'premium', r'member-only',
            r'login.*required', r'sign.*in', r'register.*to.*read',
            r'class=".*paywall', r'id=".*paywall', r'data-paywall'
        ]
        
        content_lower = html_content.lower()
        for indicator in paywall_indicators:
            if re.search(indicator, content_lower):
                logger.debug(f"Paywall indicator detected: {indicator}")
                return True
        
        # Check for content that's too short but has paywall-like structure
        if len(html_content) < 5000 and any(term in content_lower for term in ['subscribe', 'membership', 'premium']):
            logger.debug("Short content with paywall terms detected")
            return True
            
        return False
    
    def _needs_js_rendering(self, html_content: str) -> bool:
        """
        Check if the page likely requires JavaScript rendering.
        
        Args:
            html_content: HTML content to analyze
            
        Returns:
            Boolean indicating if JS rendering is likely needed
        """
        # Check for common JS framework indicators
        js_frameworks = [r'<script.*src=.*react', r'<script.*src=.*vue', r'<script.*src=.*angular', 
                         r'<script.*src=.*next', r'<script.*src=.*nuxt']
        
        content_lower = html_content.lower()
        for framework in js_frameworks:
            if re.search(framework, content_lower):
                logger.debug(f"JS framework detected: {framework}")
                return True
        
        # Check for minimal content with many script tags
        script_count = len(re.findall(r'<script', content_lower))
        content_length = len(html_content)
        
        if script_count > 5 and content_length < 10000:
            logger.debug(f"High script count ({script_count}) with short content ({content_length})")
            return True
        
        # Check for common JS-rendered page patterns
        if re.search(r'<div id="root"></div>', content_lower) or re.search(r'<div id="app"></div>', content_lower):
            logger.debug("Common JS app div detected")
            return True
            
        return False

# Singleton instance for easy access
fetcher_instance = PoliteFetcher()

def fetch_url(url: str, renderer_fallback: bool = False) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Convenience function to fetch URL content using the singleton fetcher.
    
    Args:
        url: URL to fetch
        renderer_fallback: Whether to suggest JS rendering if needed
        
    Returns:
        Tuple of (content, error_message, metadata)
    """
    return fetcher_instance.fetch(url, renderer_fallback)