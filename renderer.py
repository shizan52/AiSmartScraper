"""
Optional renderer module to handle JavaScript-rendered pages.
Uses Playwright for headless browser rendering to capture HTML snapshots.
"""

import logging
from typing import Optional, Dict, Any
import yaml
from playwright.sync_api import sync_playwright, Playwright, Browser, BrowserContext
import time

# Configure logging to match other modules
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Renderer:
    """Renders JavaScript-heavy pages using a headless browser."""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        """Initialize the renderer with parameters from config file."""
        self._load_config(config_path)
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None

    def _load_config(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)
            self.default_params = config.get('renderer_params', {
                'user_agent': "IntelligentWebScraper/1.0",
                'timeout': 30000,  # Milliseconds
                'headless': True,
                'viewport_width': 1280,
                'viewport_height': 800,
                'wait_for_selector': None,  # Optional selector to wait for
                'delay_after_load': 2.0,  # Seconds to wait after page load
                'max_retries': 2
            })
            logger.info("Renderer config loaded successfully.")
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            self.default_params = {
                'user_agent': "IntelligentWebScraper/1.0",
                'timeout': 30000,
                'headless': True,
                'viewport_width': 1280,
                'viewport_height': 800,
                'wait_for_selector': None,
                'delay_after_load': 2.0,
                'max_retries': 2
            }

    def _init_browser(self):
        """Initialize Playwright browser if not already started."""
        if not self.browser:
            try:
                self.playwright = sync_playwright().start()
                self.browser = self.playwright.chromium.launch(headless=self.default_params['headless'])
                self.context = self.browser.new_context(
                    user_agent=self.default_params['user_agent'],
                    viewport={
                        'width': self.default_params['viewport_width'],
                        'height': self.default_params['viewport_height']
                    }
                )
                logger.info("Playwright browser initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Playwright: {e}")
                self._cleanup()
                raise

    def _cleanup(self):
        """Clean up browser resources."""
        try:
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            self.context = None
            self.browser = None
            self.playwright = None
            logger.debug("Playwright browser resources cleaned up.")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def render(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Render a JavaScript-heavy page and return HTML snapshot.
        
        Args:
            url: URL to render
            **kwargs: Override default parameters
            
        Returns:
            Dictionary containing rendered HTML, error message, and metadata
        """
        params = {**self.default_params, **kwargs}
        result = {
            'html': None,
            'error': None,
            'metadata': {
                'url': url,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'title': None,
                'response_time': None
            }
        }


        for attempt in range(params['max_retries']):
            page = None
            try:
                self._init_browser()
                page = self.context.new_page()
                logger.info(f"Attempt {attempt + 1}/{params['max_retries']}: Rendering {url} with Playwright...")

                # Navigate to URL
                response = page.goto(url, timeout=params['timeout'], wait_until="networkidle")

                if response and response.status != 200:
                    raise Exception(f"Failed to load page: HTTP {response.status}")

                # Wait for optional selector
                if params['wait_for_selector']:
                    page.wait_for_selector(params['wait_for_selector'], timeout=params['timeout'])

                # Additional delay after load to ensure JS execution
                time.sleep(params['delay_after_load'])

                # Get rendered HTML and metadata
                result['html'] = page.content()
                result['metadata']['title'] = page.title()
                # Guard against missing timing info
                if response and hasattr(response, 'timing') and response.timing:
                    try:
                        result['metadata']['response_time'] = response.timing.get('responseEnd', None) - response.timing.get('requestStart', None)
                    except Exception:
                        result['metadata']['response_time'] = None
                else:
                    result['metadata']['response_time'] = None

                logger.info(f"Successfully rendered {url}")
                return result

            except Exception as e:
                result['error'] = f"Rendering attempt {attempt + 1} failed: {str(e)}"
                logger.exception(result['error'])
                self._cleanup()
                if attempt < params['max_retries'] - 1:
                    time.sleep((2 ** attempt) * 0.5)  # Exponential backoff
                continue

            finally:
                if page is not None:
                    try:
                        page.close()
                    except Exception as close_err:
                        logger.warning(f"Error closing page: {close_err}")

        result['error'] = f"Rendering failed after {params['max_retries']} attempts"
        logger.error(result['error'])
        return result

    def __del__(self):
        """Ensure browser resources are cleaned up on object deletion."""
        self._cleanup()

# Singleton instance for easy access
renderer_instance = Renderer()

def render_url(url: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to render a URL using the singleton renderer."""
    return renderer_instance.render(url, **kwargs)