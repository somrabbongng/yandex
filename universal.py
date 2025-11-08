"""
Universal Mail Client
A comprehensive application for mail verification and name generation with browser automation.
Combines functionality from unidy.py and universal.py with fixes and improvements.
"""

import sys
import json
import time
import os
from datetime import datetime, timedelta
import re
import certifi
import urllib3
import requests
import threading
import uvicorn
import random
import string
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache, cached
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QTextEdit, QLabel, 
                           QStatusBar, QFileDialog, QMessageBox, QLineEdit,
                           QTabWidget, QSplitter, QSpinBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QPalette, QColor, QSyntaxHighlighter, QTextCharFormat
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from functools import lru_cache
import logging

# Configuration Constants
MAX_WORKERS = 4
REQUEST_TIMEOUT = 30
CACHE_TTL = 3600
MAX_RETRIES = 3
BATCH_SIZE = 50

# Initialize caches
email_cache = TTLCache(maxsize=1000, ttl=CACHE_TTL)
name_cache = TTLCache(maxsize=1000, ttl=CACHE_TTL)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger("UniversalMailClient")

class EmailResponse(BaseModel):
    email: str
    code: str
    subject: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class JsonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for JSON content"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#D19A66"))
        self.highlighting_rules.append(('\".*\":', keyword_format))

        value_format = QTextCharFormat()
        value_format.setForeground(QColor("#98C379"))
        self.highlighting_rules.append(('\"[^\"]*\"(?!:)', value_format))

        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#61AFEF"))
        self.highlighting_rules.append(('\\b\\d+\\.?\\d*\\b', number_format))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), format)

class ApiServer:
    """API Server for handling mail verification requests"""
    def __init__(self):
        self.app = FastAPI(
            title="Mail Verification API",
            description="API to extract verification codes from mail",
            version="2.0.0"
        )
        self.setup_middleware()
        self.setup_routes()
        self.current_cookies = None
        self.current_ckey = None
        self.session_pool = None
        self.rate_limiter = TTLCache(maxsize=1000, ttl=60)
        self._configure_logging()
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.running = False
        self.server_thread = None

    def _configure_logging(self):
        self.logger = logging.getLogger("ApiServer")
        uvicorn_logger = logging.getLogger("uvicorn")
        uvicorn_logger.handlers = []

    def setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_routes(self):
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "cookies_available": bool(self.current_cookies),
                "ckey_available": bool(self.current_ckey)
            }

        @self.app.get("/codes/{email}", response_model=List[EmailResponse])
        async def get_codes(email: str, background_tasks: BackgroundTasks):
            if not self.current_cookies or not self.current_ckey:
                raise HTTPException(
                    status_code=400,
                    detail="No valid cookies or _ckey available"
                )
            
            try:
                cookie_string = "; ".join([f"{c['name']}={c['value']}" for c in self.current_cookies])
                
                headers = {
                    "sec-ch-ua-platform": '"Windows"',
                    "X-Requested-With": "XMLHttpRequest",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "cookie": cookie_string
                }

                body = {
                    "models": [{
                        "name": "messages",
                        "params": {
                            "count": str(BATCH_SIZE),
                            "first": "0",
                            "request": "is your Facebook confirmation code",
                            "search": "search",
                            "sort_type": "date"
                        }
                    }],
                    "_ckey": self.current_ckey,
                    "_timestamp": int(time.time() * 1000)
                }

                response = requests.post(
                    "https://mail.yandex.com/web-api/models/liza1",
                    headers=headers,
                    json=body,
                    verify=False
                )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"API error: {response.text}"
                    )

                data = response.json()
                results = []

                if 'models' in data and data['models']:
                    model = data['models'][0]
                    if 'data' in model and 'message' in model['data']:
                        for message in model['data']['message']:
                            if 'recipients' in message and 'to' in message['recipients']:
                                to_email = message['recipients']['to']['email'].lower()
                                if email.lower() in to_email:
                                    if 'subject' in message:
                                        numbers = re.findall(r'\d+', message['subject'])
                                        if numbers:
                                            results.append(EmailResponse(
                                                email=to_email,
                                                code=''.join(numbers),
                                                subject=message['subject']
                                            ))

                background_tasks.add_task(self.log_request, email, len(results))
                return results

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def update_credentials(self, cookies, ckey):
        self.current_cookies = cookies
        self.current_ckey = ckey
        self.logger.info("Credentials updated successfully")

    async def log_request(self, email: str, code_count: int):
        self.logger.info(f"Code request for {email}: Found {code_count} codes")


    def start(self):
        if self.running:
            return True

        try:
            config = uvicorn.Config(
                self.app,
                host="0.0.0.0",
                port=8000,
                log_level="info",
                log_config=None
            )
            server = uvicorn.Server(config)
            self.server_thread = threading.Thread(target=self._run_server, args=(server,))
            self.server_thread.daemon = True
            self.server_thread.start()
            self.running = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to start API server: {str(e)}")
            return False, str(e)

    def _run_server(self, server):
        try:
            server.run()
        except Exception as e:
            self.logger.error(f"Server error: {str(e)}")
            self.running = False

    def stop(self):
        self.running = False
        if self.executor:
            self.executor.shutdown(wait=False)

class BrowserThread(QThread):
    cookie_extracted = pyqtSignal(list)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    url_changed = pyqtSignal(str)
    network_request_captured = pyqtSignal(dict)
    log_message = pyqtSignal(str)
    ckey_found = pyqtSignal(str)
    browser_ready = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.driver = None
        self.running = True
        self.retry_count = 0
        self.max_retries = 3
        self.is_initializing = False
        self._init_complete = threading.Event()

    def setup_driver(self):
        try:
            self.is_initializing = True
            self.log_message.emit("Initializing browser...")
            
            options = Options()
            options.add_argument('--start-maximized')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_experimental_option("detach", True)
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_argument('--enable-logging')
            options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
            
            self.driver = webdriver.Chrome(options=options)
            self.browser_ready.emit(True)
            self._init_complete.set()
            self.log_message.emit("Browser started successfully")
            
        except Exception as e:
            error_msg = f"Error starting browser: {str(e)}"
            self.log_message.emit(error_msg)
            self.error_occurred.emit(error_msg)
            self.browser_ready.emit(False)
        finally:
            self.is_initializing = False

    def navigate_to(self, url):
        """Fixed navigation function that properly handles URLs"""
        try:
            if not self.driver:
                self.error_occurred.emit("Browser not initialized")
                return

            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            self.log_message.emit(f"Navigating to {url}")
            self.driver.get(url)
            time.sleep(1)  # Short wait for page load
            
            current_url = self.driver.current_url
            self.url_changed.emit(current_url)
            self.capture_network_requests()
            
            self.log_message.emit(f"Successfully navigated to {current_url}")
            
        except Exception as e:
            error_msg = f"Error navigating to URL: {str(e)}"
            self.log_message.emit(error_msg)
            self.error_occurred.emit(error_msg)

    def run(self):
        while self.running:
            try:
                if not self.driver and not self.is_initializing:
                    self.setup_driver()
                time.sleep(0.1)
            except Exception as e:
                self.log_message.emit(f"Thread error: {str(e)}")

    def auto_yandex_login(self):
        def login_process():
            try:
                if not self._init_complete.wait(timeout=30):
                    self.log_message.emit("Browser initialization timeout")
                    return

                if not self.driver:
                    self.log_message.emit("Browser not initialized")
                    return

                self.log_message.emit("Starting Yandex login process...")
                self.driver.get("https://mail.yandex.com")
                
                time.sleep(3)
                
                start_time = time.time()
                login_detected = False
                self.log_message.emit("Waiting for login...")
                last_url = None
                
                while time.time() - start_time < 60:
                    try:
                        current_url = self.driver.current_url
                        
                        if current_url != last_url:
                            self.log_message.emit(f"Current URL: {current_url}")
                            last_url = current_url
                        
                        if 'mail.yandex.com' in current_url and ('#tabs' in current_url or 'uid=' in current_url):
                            self.log_message.emit("Login successful - Processing cookies and network data")
                            login_detected = True
                            
                            self.get_cookies()
                            time.sleep(1)
                            
                            self.capture_network_requests()
                            
                            self.status_update.emit("Login completed")
                            break
                            
                        time.sleep(0.5)
                    except Exception as e:
                        self.log_message.emit(f"Error checking URL: {str(e)}")
                        time.sleep(1)
                
                if not login_detected:
                    self.log_message.emit("Login timeout - please login manually")
                else:
                    self.log_message.emit("Login sequence completed")
                    
            except Exception as e:
                error_msg = f"Error in login sequence: {str(e)}"
                self.log_message.emit(error_msg)
                self.error_occurred.emit(error_msg)

        threading.Thread(target=login_process, name="LoginThread").start()

    def get_cookies(self):
        if not self.driver:
            self.error_occurred.emit("Browser not initialized")
            return

        try:
            selenium_cookies = self.driver.get_cookies()
            if not selenium_cookies:
                self.log_message.emit("No cookies found")
                return

            formatted_cookies = [
                {
                    "name": cookie.get('name', ''),
                    "value": cookie.get('value', ''),
                    "domain": cookie.get('domain', ''),
                    "path": cookie.get('path', '/')
                }
                for cookie in selenium_cookies
            ]
            
            if formatted_cookies:
                self.cookie_extracted.emit(formatted_cookies)
                self.log_message.emit(f"Successfully extracted {len(formatted_cookies)} cookies")
                return True
            
        except Exception as e:
            error_msg = f"Error in login sequence: {str(e)}"
            self.log_message.emit(error_msg)
            self.error_occurred.emit(error_msg)
    def capture_network_requests(self):
        if not self.driver:
            return

        try:
            performance_logs = self.driver.get_log('performance')
            request_count = 0
            
            for entry in performance_logs:
                if not self.running:
                    break
                    
                try:
                    log = json.loads(entry['message'])['message']
                    if 'Network.response' in log['method'] or 'Network.request' in log['method']:
                        if 'params' in log and 'request' in log['params']:
                            request_url = log['params']['request'].get('url', '')
                            if 'mail.yandex.com/web-api/models/liza1' in request_url:
                                request_count += 1
                                self.network_request_captured.emit(log)
                                self.extract_ckey(log)
                except json.JSONDecodeError:
                    continue
                    
            if request_count > 0:
                self.log_message.emit(f"Processed {request_count} relevant network requests")
            else:
                self.log_message.emit("No relevant network requests found")
                    
        except Exception as e:
            error_msg = f"Error capturing network requests: {str(e)}"
            self.log_message.emit(error_msg)
            self.error_occurred.emit(error_msg)

    def extract_ckey(self, request_details):
        def extract_process():
            try:
                if 'params' in request_details and 'request' in request_details['params']:
                    request_body = request_details['params']['request'].get('postData', '')
                    if request_body:
                        try:
                            body_json = json.loads(request_body)
                            if '_ckey' in body_json:
                                self.ckey_found.emit(body_json['_ckey'])
                        except json.JSONDecodeError:
                            ckey_match = re.search(r'["\']_?ckey["\']\s*:\s*["\']([^"\']+)["\']', request_body)
                            if ckey_match:
                                self.ckey_found.emit(ckey_match.group(1))
            except Exception as e:
                self.log_message.emit(f"Error extracting _ckey: {str(e)}")

        threading.Thread(target=extract_process).start()

    def stop(self):
        """Clean shutdown of browser thread"""
        self.running = False
        if self.driver:
            def close_process():
                try:
                    self.driver.quit()
                    self.log_message.emit("Browser stopped")
                except Exception as e:
                    self.log_message.emit(f"Error closing browser: {str(e)}")
                finally:
                    self.driver = None

            threading.Thread(target=close_process).start()

class GeneratorThread(QThread):
    """Thread for generating names and email addresses"""
    progress_updated = pyqtSignal(int)
    generation_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, base_email, count):
        super().__init__()
        self.base_email = base_email
        self.count = count

    @staticmethod
    def generate_complex_password(length=12):
        chars = {
            'lower': string.ascii_lowercase,
            'upper': string.ascii_uppercase,
            'digits': string.digits,
            'special': "!@#$%^&*"
        }
        
        password = [
            random.choice(chars['lower']),
            random.choice(chars['upper']),
            random.choice(chars['digits']),
            random.choice(chars['special'])
        ]
        
        remaining_length = length - len(password)
        all_chars = ''.join(chars.values())
        password.extend(random.choices(all_chars, k=remaining_length))
        random.shuffle(password)
        return ''.join(password)

    @staticmethod
    def generate_random_date(start_year=1980, end_year=2005):
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        random_date = start_date + timedelta(days=random_days)
        return random_date.strftime("%b-%d-%Y")

    def generate_khmer_name(self):
        """Generate a name using the name lists"""
        is_khmer = random.choice([True, False])
        
        first_names = get_khmer_first_names(is_khmer)
        first_name_category = random.choice(list(first_names.keys()))
        first_name = random.choice(first_names[first_name_category])
        
        last_names = get_khmer_last_names(is_khmer)
        last_name_category = random.choice(list(last_names.keys()))
        last_name = random.choice(last_names[last_name_category])
        
        return first_name, last_name, is_khmer

    def run(self):
        try:
            entries = []
            username, domain = self.base_email.split('@')
            
            for i in range(1, self.count + 1):
                random_suffix = ''.join(random.choices(string.ascii_lowercase, k=5))
                new_email = f"{username}+{random_suffix}{i:02d}@{domain}"
                
                first_name, last_name, is_khmer = self.generate_khmer_name()
                birth_date = self.generate_random_date()
                password = self.generate_complex_password()
                
                entry = {
                    'email': new_email,
                    'first_name': first_name,
                    'last_name': last_name,
                    'birth_date': birth_date,
                    'password': password,
                    'is_khmer': is_khmer
                }
                entries.append(entry)
                
                progress = int((i / self.count) * 100)
                self.progress_updated.emit(progress)
            
            self.generation_complete.emit(entries)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

class UnifiedApp(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.browser_thread = None
        self.generator_thread = None
        self.current_ckey = None
        self.current_cookies = None
        self.generated_data = []
        self.api_server = ApiServer()
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Unified Mail Client & Name Generator')
        self.setMinimumSize(1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tabs
        tabs = QTabWidget()
        
        # Browser tab
        browser_tab = QWidget()
        browser_layout = QVBoxLayout(browser_tab)
        self.setup_browser_tab(browser_layout)
        tabs.addTab(browser_tab, "Browser")
        
        # API tab
        api_tab = QWidget()
        api_layout = QVBoxLayout(api_tab)
        self.setup_api_tab(api_layout)
        tabs.addTab(api_tab, "API")

        # Generator tab
        generator_tab = QWidget()
        generator_layout = QVBoxLayout(generator_tab)
        self.setup_generator_tab(generator_layout)
        tabs.addTab(generator_tab, "Name Generator")
        
        # Preview tab
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        self.setup_preview_tab(preview_layout)
        tabs.addTab(preview_tab, "Preview")
        
        layout.addWidget(tabs)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.style_ui()

    def setup_browser_tab(self, layout):
        # Quick Start button
        quick_start_layout = QHBoxLayout()
        self.quick_start_button = QPushButton('🚀 Quick Start (Auto Setup)')
        self.quick_start_button.setMinimumHeight(40)
        self.quick_start_button.clicked.connect(self.quick_start)
        self.quick_start_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        quick_start_layout.addWidget(self.quick_start_button)
        layout.addLayout(quick_start_layout)

        # Status indicator
        self.setup_status_indicator(layout)

        # URL input
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel('URL:'))
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText('Enter URL (e.g., mail.yandex.com)')
        self.url_input.returnPressed.connect(self.navigate_to_url)
        url_layout.addWidget(self.url_input)
        
        self.go_button = QPushButton('Go')
        self.go_button.clicked.connect(self.navigate_to_url)
        self.go_button.setEnabled(False)
        url_layout.addWidget(self.go_button)
        layout.addLayout(url_layout)

        # Browser controls
        controls = QHBoxLayout()
        
        self.start_button = QPushButton('Start Browser')
        self.start_button.clicked.connect(self.start_browser)
        controls.addWidget(self.start_button)
        
        self.get_cookies_button = QPushButton('Get Cookies')
        self.get_cookies_button.clicked.connect(self.get_cookies)
        self.get_cookies_button.setEnabled(False)
        controls.addWidget(self.get_cookies_button)
        
        self.stop_button = QPushButton('Stop Browser')
        self.stop_button.clicked.connect(self.stop_browser)
        self.stop_button.setEnabled(False)
        controls.addWidget(self.stop_button)
        
        layout.addLayout(controls)

        # Results area
        self.cookie_text = QTextEdit()
        self.cookie_text.setReadOnly(True)
        self.cookie_text.setMaximumHeight(200)
        layout.addWidget(QLabel('Cookies:'))
        layout.addWidget(self.cookie_text)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(QLabel('Network Logs:'))
        layout.addWidget(self.log_text)

    def setup_status_indicator(self, layout):
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel('Setup Status:'))
        
        self.status_labels = {
            'browser': QLabel('Browser: ❌'),
            'login': QLabel('Login: ❌'),
            'cookies': QLabel('Cookies: ❌'),
            'ckey': QLabel('_ckey: ❌'),
            'api': QLabel('API Server: ❌')
        }
        
        for label in self.status_labels.values():
            status_layout.addWidget(label)
        
        layout.addLayout(status_layout)

    def setup_api_tab(self, layout):
        # Server control
        server_controls = QHBoxLayout()
        self.start_server_button = QPushButton('Start API Server')
        self.start_server_button.clicked.connect(self.start_api_server)
        server_controls.addWidget(self.start_server_button)
        
        self.server_status_label = QLabel('Server: Stopped')
        server_controls.addWidget(self.server_status_label)
        layout.addLayout(server_controls)

        # Server info
        server_info = QTextEdit()
        server_info.setReadOnly(True)
        server_info.setMaximumHeight(100)
        server_info.setText(
            "API Endpoints:\n"
            "GET http://localhost:8000/health - Check server status\n"
            "GET http://localhost:8000/codes/{email} - Get verification codes\n"
        )
        layout.addWidget(server_info)

        # Email verification section
        email_layout = QHBoxLayout()
        email_layout.addWidget(QLabel('Email:'))
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText('Enter email to get verification code')
        email_layout.addWidget(self.email_input)
        
        verification_buttons = QHBoxLayout()
        self.get_code_button = QPushButton('Get Verification Code')
        self.get_code_button.clicked.connect(self.get_verification_code)
        verification_buttons.addWidget(self.get_code_button)
        
        self.clear_log_button = QPushButton('Clear Log')
        self.clear_log_button.clicked.connect(self.clear_verification_log)
        verification_buttons.addWidget(self.clear_log_button)
        
        email_layout.addLayout(verification_buttons)
        layout.addLayout(email_layout)

        # Create split view for response and logs
        response_log_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Response section
        response_widget = QWidget()
        response_layout = QVBoxLayout(response_widget)
        response_layout.addWidget(QLabel('API Response:'))
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.highlighter = JsonHighlighter(self.response_text.document())
        response_layout.addWidget(self.response_text)
        response_widget.setLayout(response_layout)
        response_log_splitter.addWidget(response_widget)
        
        # Verification Log section
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.addWidget(QLabel('Verification Log:'))
        self.verification_log = QTextEdit()
        self.verification_log.setReadOnly(True)
        log_layout.addWidget(self.verification_log)
        log_widget.setLayout(log_layout)
        response_log_splitter.addWidget(log_widget)
        
        layout.addWidget(response_log_splitter)

    def setup_generator_tab(self, layout):
        # Email input
        email_layout = QHBoxLayout()
        email_layout.addWidget(QLabel('Base Email:'))
        self.generator_email_input = QLineEdit()
        self.generator_email_input.setPlaceholderText('Enter base email (e.g., example@yandex.com)')
        email_layout.addWidget(self.generator_email_input)
        layout.addLayout(email_layout)
        
        # Count input
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel('Number of entries:'))
        self.count_input = QSpinBox()
        self.count_input.setRange(1, 1000)
        self.count_input.setValue(5)
        count_layout.addWidget(self.count_input)
        count_layout.addStretch()
        layout.addLayout(count_layout)
        
        # Generate button
        self.generate_button = QPushButton('Generate Names')
        self.generate_button.clicked.connect(self.start_generation)
        self.generate_button.setMinimumHeight(40)
        layout.addWidget(self.generate_button)
        
        # Progress display
        self.progress_label = QLabel('Progress: 0%')
        layout.addWidget(self.progress_label)
        
        # Results area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(QLabel('Generated Data:'))
        layout.addWidget(self.results_text)
        
        # Export button
        self.export_button = QPushButton('Export to File')
        self.export_button.clicked.connect(self.export_data)
        self.export_button.setEnabled(False)
        layout.addWidget(self.export_button)

    def setup_preview_tab(self, layout):
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        layout.addWidget(self.preview_text)

    # Browser-related methods
    def navigate_to_url(self):
        """Navigate to URL from input field"""
        if self.browser_thread and self.url_input.text().strip():
            self.browser_thread.navigate_to(self.url_input.text().strip())

    def start_browser(self):
        """Initialize and start the browser thread"""
        self.browser_thread = BrowserThread()
        self.browser_thread.status_update.connect(self.update_status)
        self.browser_thread.error_occurred.connect(self.show_error)
        self.browser_thread.cookie_extracted.connect(self.update_cookies)
        self.browser_thread.url_changed.connect(self.url_input.setText)
        self.browser_thread.network_request_captured.connect(self.process_network_request)
        self.browser_thread.log_message.connect(self.log_message)
        self.browser_thread.ckey_found.connect(self.update_ckey)
        self.browser_thread.browser_ready.connect(self.handle_browser_ready)

        self.browser_thread.start()
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        self.update_status_indicator('browser')

    def handle_browser_ready(self, success):
        """Handle browser initialization completion"""
        if success:
            self.get_cookies_button.setEnabled(True)
            self.go_button.setEnabled(True)
            self.update_status("Browser initialized successfully")
        else:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.update_status("Browser initialization failed")

    def get_cookies(self):
        """Request cookies from browser thread"""
        if self.browser_thread:
            self.browser_thread.get_cookies()

    def stop_browser(self):
        """Stop the browser thread"""
        if self.browser_thread:
            self.browser_thread.stop()
            self.browser_thread = None
            
            self.start_button.setEnabled(True)
            self.get_cookies_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.go_button.setEnabled(False)
            
            self.update_status_indicator('browser', False)
            self.update_status_indicator('login', False)

    def quick_start(self):
        """Perform automatic setup sequence"""
        self.quick_start_button.setEnabled(False)
        self.quick_start_button.setText("Setting up...")
        self.log_message("Starting automatic setup sequence...")
        
        try:
            # Start browser
            self.start_browser()
            self.update_status_indicator('browser')
            
            # Start API server
            self.start_api_server()
            self.update_status_indicator('api')
            
            # Auto login sequence
            if self.browser_thread:
                self.browser_thread.auto_yandex_login()
                self.update_status_indicator('login')
            
            # Update status when cookies and ckey are captured
            if self.current_cookies:
                self.update_status_indicator('cookies')
            if self.current_ckey:
                self.update_status_indicator('ckey')
            
            self.quick_start_button.setText("Setup Complete! ✅")
            self.update_status("Automatic setup completed successfully")
            
        except Exception as e:
            self.show_error(f"Error during quick start: {str(e)}")
            self.quick_start_button.setText("Setup Failed ❌")
        finally:
            self.quick_start_button.setEnabled(True)

    # Generator-related methods
    def start_generation(self):
        """Start the name generation process"""
        if not self.generator_email_input.text() or '@' not in self.generator_email_input.text():
            self.show_error("Please enter a valid email address")
            return
        
        self.generate_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.results_text.clear()
        self.preview_text.clear()
        self.progress_label.setText('Progress: 0%')
        
        self.generator_thread = GeneratorThread(
            self.generator_email_input.text(),
            self.count_input.value()
        )
        self.generator_thread.progress_updated.connect(self.update_progress)
        self.generator_thread.generation_complete.connect(self.generation_complete)
        self.generator_thread.error_occurred.connect(self.show_error)
        self.generator_thread.start()

    def export_data(self):
        """Export generated data to file"""
        if not self.generated_data:
            self.show_error("No data to export")
            return
        
        try:
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Save Generated Data",
                "generated_emails.txt",
                "Text Files (*.txt);;JSON Files (*.json)"
            )
            
            if file_path:
                try:
                    if selected_filter == "JSON Files (*.json)":
                        with open(file_path, 'w', encoding='utf-8') as file:
                            json.dump(self.generated_data, file, 
                                    indent=2, 
                                    ensure_ascii=False)
                    else:
                        with open(file_path, 'w', encoding='utf-8') as file:
                            file.write("Email|First Name(ឈ្មោះ)|Last Name(គោត្តនាម)|Birth Date|Password|Is Khmer\n")
                            for entry in self.generated_data:
                                line = f"{entry['email']}|{entry['first_name']}|{entry['last_name']}|{entry['birth_date']}|{entry['password']}|{entry['is_khmer']}\n"
                                file.write(line)
                    
                    success_msg = f'Data exported to {file_path}'
                    self.status_bar.showMessage(success_msg)
                    self.log_message(f"SUCCESS: {success_msg}")
                    
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Data has been exported to:\n{file_path}",
                        buttons=QMessageBox.StandardButton.Ok
                    )
                    
                except PermissionError:
                    error_msg = f"Permission denied when writing to {file_path}"
                    self.show_error(error_msg)
                except IOError as e:
                    error_msg = f"IO Error while writing file: {str(e)}"
                    self.show_error(error_msg)
                except Exception as e:
                    error_msg = f"Unexpected error during export: {str(e)}"
                    self.show_error(error_msg)
                    
        except Exception as e:
            error_msg = f"Error during file save dialog: {str(e)}"
            self.show_error(error_msg)

    # API Server methods
    def start_api_server(self):
        """Start the API server"""
        try:
            result = self.api_server.start()
            if isinstance(result, tuple):
                _, error_message = result
                self.show_error(f"Failed to start API server: {error_message}")
                self.update_status_indicator('api', False)
            else:
                self.server_status_label.setText('Server: Running on http://localhost:8000')
                self.start_server_button.setEnabled(False)
                self.update_status("API server started on http://localhost:8000")
                self.update_status_indicator('api')
        except Exception as e:
            self.show_error(f"Failed to start API server: {str(e)}")
            self.update_status_indicator('api', False)

    def get_verification_code(self):
        """Request verification code for email"""
        if not self.current_ckey or not self.current_cookies:
            self.show_error("Please capture cookies and _ckey first")
            self.log_verification_step("❌ Error: Missing cookies or _ckey")
            return

        email = self.email_input.text().strip()
        if not email:
            self.show_error("Please enter an email address")
            self.log_verification_step("❌ Error: No email address provided")
            return

        try:
            self.log_verification_step(f"🔍 Starting verification code search for {email}")
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            self.log_verification_step("📤 Sending request to API...")
            response = requests.get(
                f"http://localhost:8000/codes/{email}",
                verify=False
            )
            
            if response.status_code == 200:
                result = response.json()
                self.response_text.setText(json.dumps(result, indent=2))
                self.log_verification_step(f"✅ Success: Found {len(result)} verification code(s)")
                for idx, code_data in enumerate(result, 1):
                    self.log_verification_step(f"   📌 Code {idx}: {code_data.get('code')} (Subject: {code_data.get('subject', 'N/A')})")
                self.update_status(f"Found {len(result)} verification codes")
            else:
                error_message = f"API Error (Status {response.status_code}): {response.text}"
                self.response_text.setText(error_message)
                self.log_verification_step(f"❌ {error_message}")
                self.update_status("Failed to get verification codes")
            
        except Exception as e:
            error_message = f"Error getting verification code: {str(e)}"
            self.show_error(error_message)
            self.log_verification_step(f"❌ {error_message}")

    # Utility methods
    def process_network_request(self, request_details):
        """Process and log network request details"""
        log_entry = json.dumps(request_details, indent=2) + "\n" + "-" * 80 + "\n"
        self.log_message(log_entry)

    def update_ckey(self, ckey):
        """Update the current ckey value"""
        self.current_ckey = ckey
        self.api_server.update_credentials(self.current_cookies, ckey)
        self.log_message(f"New _ckey captured: {ckey}")
        self.update_status("_ckey updated")
        self.update_status_indicator('ckey')

    def update_cookies(self, cookies):
        """Update the current cookies"""
        self.current_cookies = cookies
        self.api_server.update_credentials(cookies, self.current_ckey)
        json_text = json.dumps(cookies, indent=2)
        self.cookie_text.setText(json_text)
        self.update_status_indicator('cookies')

    def update_progress(self, value):
        """Update progress display"""
        self.progress_label.setText(f'Progress: {value}%')
        self.status_bar.showMessage(f'Generating... {value}%')

    def generation_complete(self, entries):
        """Handle completion of name generation"""
        self.generated_data = entries
        display_text = json.dumps(entries, indent=2, ensure_ascii=False)
        self.results_text.setText(display_text)
        
        preview_text = "Email|First Name(ឈ្មោះ)|Last Name(គោត្តនាម)|Birth Date|Password\n"
        preview_text += "-" * 100 + "\n"
        for entry in entries:
            preview_text += f"{entry['email']}|{entry['first_name']}|{entry['last_name']}|{entry['birth_date']}|{entry['password']}\n"
        
        self.preview_text.setText(preview_text)
        
        self.generate_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.status_bar.showMessage('Generation complete!')

    def log_message(self, message):
        """Add message to log with timestamp"""
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)
        self.log_text.insertPlainText(f"{message}\n")
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)

    def update_status(self, message):
        """Update status bar message"""
        self.status_bar.showMessage(message)

    def show_error(self, message):
        """Show error dialog and log error"""
        QMessageBox.critical(
            self,
            "Error",
            message,
            buttons=QMessageBox.StandardButton.Ok
        )
        self.log_message(f"ERROR: {message}")

    def update_status_indicator(self, component, success=True):
        """Update status indicator labels"""
        if component in self.status_labels:
            self.status_labels[component].setText(f"{component}: {'✅' if success else '❌'}")

    def log_verification_step(self, message):
        """Add verification step to log"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.verification_log.append(log_entry)
        self.verification_log.moveCursor(QTextCursor.MoveOperation.End)

    def clear_verification_log(self):
        """Clear the verification log"""
        self.verification_log.clear()

    def style_ui(self):
        """Apply dark theme styling to the application"""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #000000;
                color: #ffffff;
            }
            QTextEdit, QLineEdit, QSpinBox {
                background-color: #1a1a1a;
                border: 1px solid #333333;
                padding: 5px;
                border-radius: 3px;
                color: #ffffff;
                font-family: 'Consolas', monospace;
            }
            QPushButton {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #333333;
                padding: 5px 10px;
                border-radius: 3px;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #2d2d2d;
            }
            QPushButton:disabled {
                background-color: #0d0d0d;
                color: #666666;
            }
            QTabWidget::pane {
                border: 1px solid #333333;
            }
            QTabBar::tab {
                background-color: #1a1a1a;
                color: #ffffff;
                padding: 8px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #2d2d2d;
            }
            QLabel {
                color: #ffffff;
            }
            QSplitter::handle {
                background-color: #333333;
                height: 2px;
            }
        """)

    def closeEvent(self, event):
        """Handle application closure"""
        self.stop_browser()
        if self.api_server:
            self.api_server.stop()
        event.accept()

def get_khmer_first_names(is_khmer=False):
    """Get either Romanized or Khmer Unicode first names."""
    if is_khmer:
        return {
            'traditional': [
                "បុប្ផា", "ចិន្តា", "ដារា", "កុសល", "លៀប", "មករា", "ណារី", "ពេជ្រ", "រតនា", "សុធី",
                "ធីដា", "វណ្ណៈ", "វិសាល", "វាសនា", "សុខា", "សុភា", "សុផល", "ចន្ធី", "ផល្លា",
                "ចន្ទា", "លៀប", "មករា", "ណារី", "ពេជ្រ", "រតនា", "សុធី", "ធីដា", "វណ្ណៈ", "វិសាល"
            ],
            'modern': [
                "ចាន់មុនី", "ចាន់ណារី", "ចាន់ធា", "ចាន់ថូ", "ចន្ទ្រា", "ជាតា", "ឈូក",
                "ដារិទ្ធ", "ដាវី", "តិត", "ហេង", "ហុង", "ហោ", "កញ្ញា", "កាញ៉ា", "កក្កដា",
                "កំសាន្ត", "កញ្ញា", "កនិដ្ឋា", "កញ្ញា", "ករុណា", "កាង", "គីតី", "កែវ", "កេសរ"
            ],
            'nature': [
                "ចន្លិដា", "ចំប៉ា", "មាលា", "ផល្លា", "រំចង់", "រំដួល", "សេហា", "វណ្ណា",
                "លក្ខិណា", "ដាលីន", "ម៉ាលីកា", "រចនា"
            ],
            'virtuous': [
                "អាការា", "ចាន់", "មេត្តា", "មីនា", "និមល", "បញ្ញា", "បញ្ញា", "ផារី",
                "ពិសិដ្ឋ", "ពន្លឺ", "សុវណ្ណ", "ឧត្តម"
            ],
            'professional': [
                "បណ្ឌិត", "បូរាណ", "ចាន់នារ៉ា", "ជាតិ", "កុសុមា", "គន្ធី", "លីដា", "ម៉ាលីស",
                "មានិត", "មេត្រី", "មេសា", "មុនីរ័ត្ន"
            ],
            'spiritual': [
                "អរុណ", "អស៊ិត", "អាទិត្យ", "បុណ្យ", "ចក្រា", "ចន្ទា", "ដានី", "ដារិទ្ធ",
                "ដារ៉ូ", "តិត", "ធនា", "ឌិន"
            ],
            'compound': [
                "ចាន់ណារី", "ចាន់សុភា", "ចាន់វឌ្ឍនី", "គន្ធា", "មុនីរ័ត្ន", "មុនីរេត",
                "ពន្លឺ", "ពន្លក", "រតនា", "រតនា"
            ]
        }
    else:
        return {
            'traditional': [
                "Bopha", "Chenda", "Dara", "Kosal", "Leap", "Makara", "Nary", "Pich", "Ratha", "Sothy",
                "Thida", "Vannak", "Visal", "Veasna", "Sokha", "Sophea", "Sophal", "Chanthy", "Phalla",
                "Chanda", "Leap", "Makara", "Nary", "Pich", "Ratha", "Sothy", "Thida", "Vannak", "Visal"
            ],
            'modern': [
                "Chanmony", "Channary", "Chanthea", "Chanthou", "Chantra", "Cheata", "Chhouk",
                "Darith", "Davy", "Deth", "Heng", "Hong", "Hor", "Kagna", "Kahna", "Kakada",
                "Kamsan", "Kanha", "Kanitha", "Kanya", "Karona", "Keang", "Kearthy", "Keo", "Kesor"
            ],
            'nature': [
                "Chanlida", "Champa", "Mealea", "Phalla", "Romchong", "Romdoul", "Seyha", "Vanna",
                "Leakena", "Dalin", "Malika", "Rachana"
            ],
            'virtuous': [
                "Akara", "Chann", "Meta", "Minea", "Nimol", "Pahna", "Panha", "Phary",
                "Piseth", "Ponleu", "Sovann", "Udom"
            ],
            'professional': [
                "Bandith", "Borann", "Channara", "Cheat", "Kosoma", "Kunthy", "Lida", "Malis",
                "Manith", "Meatrey", "Mesa", "Moniroth"
            ],
            'spiritual': [
                "Arun", "Asith", "Atith", "Bunna", "Chakra", "Chanda", "Dany", "Darith",
                "Daro", "Deth", "Dhana", "Dinn"
            ],
            'compound': [
                "Channary", "Chansophea", "Chanvatey", "Kunthea", "Monirath", "Monireth",
                "Ponleur", "Ponlork", "Rathana", "Rothana"
            ]
        }

def get_khmer_last_names(is_khmer=False):
    """Get either Romanized or Khmer Unicode last names."""
    if is_khmer:
        return {
            'traditional': [
                "សុខ", "ចាន់", "ឆៃ", "ឈុន", "ជា", "ហុង", "កែវ", "គីម", "លី", "ម៉ៅ",
                "ង៉ោ", "ប៉ាង", "ប៉េន", "ស៊ីម", "ស៊ុន", "តាន់", "តិប", "ថាច់", "តូច", "យ៉ុង"
            ],
            'regional': [
                "កង", "កៅ", "គា", "គៀន", "គៀរ", "កែវ", "កើ", "កេត", "ខេម", "ខឹម",
                "ខិន", "ខន", "ខន", "ខន", "ខត", "ខូន", "ខោង", "ខូវ", "គង់", "កូក"
            ],
            'chinese': [
                "ឡាយ", "លី", "លាង", "លេង", "លឹង", "លឹម", "លោ", "ឡុក", "ឡុង", "លួង",
                "ម៉ាក់", "ម៉ាន់", "មាស", "ម៉េង", "ម៉ី", "មីង", "ងិត", "ងឹម", "ងិន", "ងូវ"
            ]
        }
    else:
        return {
            'traditional': [
                "Sok", "Chan", "Chhay", "Chhun", "Chea", "Hong", "Keo", "Kim", "Ly", "Mao",
                "Ngor", "Pang", "Pen", "Sim", "Sun", "Tan", "Tep", "Thach", "Touch", "Yong"
            ],
            'regional': [
                "Kang", "Kao", "Kea", "Kean", "Kear", "Keo", "Ker", "Ket", "Khem", "Khim",
                "Khin", "Khon", "Khonn", "Khorn", "Khot", "Khoun", "Khoung", "Khov", "Kong", "Kouk"
            ],
            'chinese': [
                "Lay", "Lee", "Leang", "Leng", "Leung", "Lim", "Lo", "Lok", "Long", "Loung",
                "Mak", "Man", "Meas", "Meng", "Mey", "Ming", "Nget", "Ngim", "Ngin", "Ngov"
            ]
        }

def main():
    """Application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Configure application-wide settings
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#000000"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#1a1a1a"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1a1a1a"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#1a1a1a"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#1a1a1a"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Link, QColor("#2196F3"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2196F3"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    
    # Disable SSL warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Create and show the main window
    window = UnifiedApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()