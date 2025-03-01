import bs4
import logging
import requests
import urllib3
from typing import List, Optional, Union, Dict, Any
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

class WebsiteCrawler(CrawlSpider):
    """
    Custom web crawler to extract text from a website using Scrapy
    """
    name = 'website_crawler'
    documents = []

    def __init__(self, 
                 start_urls: List[str], 
                 allowed_domains: Optional[List[str]] = None, 
                 auth_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the web crawler
        
        Args:
            start_urls (List[str]): URLs to start crawling from
            allowed_domains (Optional[List[str]]): Domains to restrict crawling
            auth_config (Optional[Dict]): Authentication configuration
        """
        self.start_urls = start_urls
        self.allowed_domains = allowed_domains or [
            urllib3.util.parse_url(url).host for url in start_urls
        ]

        # Authentication configuration
        self.auth_config = auth_config or {}
        
        # Define crawling rules
        self.rules = (
            Rule(
                LinkExtractor(
                    allow_domains=self.allowed_domains,
                    unique=True
                ), 
                callback=self.parse_item
            ),
        )
        
        super().__init__()

    def start_requests(self):
        """
        Generate initial requests with authentication if provided
        """
        for url in self.start_urls:
            # Prepare authentication
            if self.auth_config:
                method = self.auth_config.get('method', 'session')
                
                if method == 'basic':
                    yield requests.Request(
                        url, 
                        auth=(
                            self.auth_config.get('username', ''), 
                            self.auth_config.get('password', '')
                        )
                    )
                elif method == 'token':
                    headers = {
                        'Authorization': f"Bearer {self.auth_config.get('token', '')}"
                    }
                    yield requests.Request(url, headers=headers)
                elif method == 'session':
                    session = requests.Session()
                    if 'login_url' in self.auth_config:
                        session.post(
                            self.auth_config['login_url'], 
                            data=self.auth_config.get('login_data', {})
                        )
                    yield session.get(url)
                else:
                    yield requests.Request(url)
            else:
                yield requests.Request(url)

    def parse_item(self, response):
        """
        Extract text from web pages
        """
        # Extract text content
        text = ' '.join(response.xpath('//body//text()').getall())
        
        if text.strip():
            self.documents.append(
                Document(
                    page_content=text,
                    metadata={'source': response.url}
                )
            )

def crawl_website(
    start_urls: Union[str, List[str]], 
    allowed_domains: Optional[List[str]] = None,
    auth_config: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    Crawl a website and return extracted documents using Scrapy
    
    Args:
        start_urls (Union[str, List[str]]): URL(s) to start crawling
        allowed_domains (Optional[List[str]]): Domains to restrict crawling
        auth_config (Optional[Dict[str, Any]]): Authentication configuration
    
    Returns:
        List of extracted documents
    """
    # Ensure start_urls is a list
    if isinstance(start_urls, str):
        start_urls = [start_urls]
    
    # Dynamically create a spider class
    class DynamicWebsiteCrawler(CrawlSpider):
        name = 'dynamic_website_crawler'
        documents = []
        
        def __init__(self):
            self.start_urls = start_urls
            self.allowed_domains = allowed_domains or [
                urllib3.util.parse_url(url).host for url in start_urls
            ]
            
            self.rules = (
                Rule(
                    LinkExtractor(
                        allow_domains=self.allowed_domains,
                        unique=True
                    ), 
                    callback=self.parse_item
                ),
            )
            
            super().__init__()

        def start_requests(self):
            for url in self.start_urls:
                if auth_config:
                    method = auth_config.get('method', 'session')
                    
                    if method == 'basic':
                        yield requests.Request(
                            url, 
                            auth=(
                                auth_config.get('username', ''), 
                                auth_config.get('password', '')
                            )
                        )
                    elif method == 'token':
                        headers = {
                            'Authorization': f"Bearer {auth_config.get('token', '')}"
                        }
                        yield requests.Request(url, headers=headers)
                    elif method == 'session':
                        session = requests.Session()
                        if 'login_url' in auth_config:
                            session.post(
                                auth_config['login_url'], 
                                data=auth_config.get('login_data', {})
                            )
                        yield session.get(url)
                    else:
                        yield requests.Request(url)
                else:
                    yield requests.Request(url)

        def parse_item(self, response):
            # Extract text content
            text = ' '.join(response.xpath('//body//text()').getall())
            
            if text.strip():
                self.documents.append(
                    Document(
                        page_content=text,
                        metadata={'source': response.url}
                    )
                )

    # Create crawler process
    process = CrawlerProcess(settings={
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    # Run the crawler
    crawler = process.create_crawler(DynamicWebsiteCrawler)
    process.crawl(crawler)
    process.start()
    
    return crawler.spider.documents

class SimpleCrawler:
    """
    A simple web crawler using requests and BeautifulSoup
    """
    def __init__(self, base_url, username=None, password=None, max_pages=100, delay=1):
        """
        Initialize the crawler with authentication credentials
        
        Args:
            base_url: The starting URL for crawling
            username: Username for basic authentication
            password: Password for basic authentication
            max_pages: Maximum number of pages to crawl
            delay: Delay between requests in seconds
        """
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.username = username
        self.password = password
        self.visited_urls = set()
        self.queue = [base_url]
        self.documents = []
        self.max_pages = max_pages
        self.delay = delay
        
        # Create a session to maintain authentication
        self.session = requests.Session()
        if username and password:
            self.session.auth = (username, password)
        
    def is_valid_url(self, url):
        """Check if URL should be crawled"""
        # Parse the URL
        parsed = urlparse(url)
        
        # Only crawl URLs from the same domain
        if parsed.netloc and parsed.netloc != self.base_domain:
            return False
            
        # Skip URLs with fragments
        if parsed.fragment:
            return False
            
        # Skip non-HTML content based on extension
        if parsed.path.endswith(('.pdf', '.jpg', '.png', '.gif', '.css', '.js')):
            return False
            
        return True
        
    def extract_links(self, soup, current_url):
        """Extract all links from the page"""
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Handle relative URLs
            absolute_url = urljoin(current_url, href)
            
            # Normalize URL
            absolute_url = absolute_url.split('#')[0]  # Remove fragments
            
            if self.is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                links.append(absolute_url)
                
        return links
        
    def scrape_page(self, url):
        """Scrape a single page and extract its content"""
        try:
            logger.info(f"Scraping: {url}")
            
            # Fetch the page
            response = self.session.get(url)
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract links before removing elements
            links = self.extract_links(soup, url)
            
            # Remove script, style, and navigation elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            
            # Skip if no meaningful content
            if len(text) < 50:
                logger.warning(f"Skipping {url} - insufficient content")
                return links
                
            logger.info(f"Extracted text length: {len(text)}")
            
            # Create a document
            doc = Document(
                page_content=text,
                metadata={'source': url}
            )
            
            self.documents.append(doc)
            return links
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return []
            
    def crawl(self):
        """Crawl the website recursively"""
        while self.queue and len(self.visited_urls) < self.max_pages:
            # Get the next URL from the queue
            current_url = self.queue.pop(0)
            
            # Skip if already visited
            if current_url in self.visited_urls:
                continue
                
            # Add to visited set
            self.visited_urls.add(current_url)
            
            # Scrape the page and get new links
            new_links = self.scrape_page(current_url)
            
            # Add new links to the queue
            for link in new_links:
                if link not in self.visited_urls and link not in self.queue:
                    self.queue.append(link)
                    
            # Respect crawl delay
            time.sleep(self.delay)
            
        logger.info(f"Crawling completed. Visited {len(self.visited_urls)} pages.")
        return self.documents

def load_web_documents(
    web_paths: Union[str, List[str]], 
    bs_kwargs: Optional[Dict] = None,
    auth_config: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    Load documents from web paths, with optional authentication
    
    Args:
        web_paths (Union[str, List[str]]): URL(s) to load documents from
        bs_kwargs (Optional[Dict]): BeautifulSoup parsing arguments
        auth_config (Optional[Dict[str, Any]]): Authentication configuration
    
    Returns:
        List of loaded documents
    """
    # Ensure web_paths is a list
    if isinstance(web_paths, str):
        web_paths = [web_paths]
    
    # If authentication is provided, use web crawler
    if auth_config:
        return crawl_website(
            start_urls=web_paths, 
            auth_config=auth_config
        )
    
    # Default WebBaseLoader for unauthenticated sites
    if bs_kwargs is None:
        bs_kwargs = dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
    
    loader = WebBaseLoader(
        web_paths=web_paths,
        bs_kwargs=bs_kwargs
    )
    return loader.load()
