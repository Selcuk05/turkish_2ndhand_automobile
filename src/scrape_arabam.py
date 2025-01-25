import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import queue
import threading
import logging
from logging.handlers import RotatingFileHandler


# Set up logging
def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = RotatingFileHandler(
        f"{log_dir}/arabam_scraper.log", maxBytes=10485760, backupCount=5  # 10MB
    )
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


class RateLimiter:
    def __init__(self, min_delay=2, max_delay=5):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_request_time = {}
        self.lock = threading.Lock()

    def wait(self, worker_id):
        with self.lock:
            if worker_id in self.last_request_time:
                elapsed = time.time() - self.last_request_time[worker_id]
                delay = random.uniform(self.min_delay, self.max_delay)
                if elapsed < delay:
                    time.sleep(delay - elapsed)
            self.last_request_time[worker_id] = time.time()


rate_limiter = RateLimiter()

base_url = "https://www.arabam.com/ikinci-el/otomobil?take=50"


def setup_driver():
    options = uc.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-notifications")  # Block notification prompts
    options.add_experimental_option(
        "prefs",
        {
            "profile.default_content_setting_values.notifications": 2  # Block notifications
        },
    )

    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    options.add_argument("--accept-language=en-US,en;q=0.9")
    options.add_argument(
        "--accept=text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    )

    driver = uc.Chrome(options=options)
    driver.set_window_size(1920, 1080)
    return driver


def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy, pool_connections=100, pool_maxsize=100
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_image(session, img_data):
    img_url, save_path = img_data
    try:
        response = session.get(img_url, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return save_path
    except Exception as e:
        logger.error(f"Error downloading image {img_url}: {str(e)}")
    return None


def accept_cookies(driver, wait):
    try:
        # Wait longer for cookie button with a new wait object
        cookie_wait = WebDriverWait(driver, 20)  # Increased timeout

        # Wait for the iframe if it exists and switch to it
        try:
            iframe = cookie_wait.until(
                EC.presence_of_element_located((By.ID, "onetrust-banner-sdk"))
            )
            driver.switch_to.frame(iframe)
        except:
            pass  # If no iframe, continue with main content

        # Wait for button to be both present and clickable
        accept_button = cookie_wait.until(
            EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler"))
        )
        cookie_wait.until(
            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
        )

        # Try multiple click methods
        try:
            accept_button.click()
        except:
            try:
                driver.execute_script("arguments[0].click();", accept_button)
            except:
                driver.execute_script(
                    "document.getElementById('onetrust-accept-btn-handler').click();"
                )

        logger.info("Accepted cookies")
        time.sleep(2)  # Give more time for the popup to disappear

        # Switch back to default content if we switched to an iframe
        try:
            driver.switch_to.default_content()
        except:
            pass

    except Exception as e:
        logger.warning(f"Could not find or click cookie accept button: {str(e)}")
        try:
            driver.switch_to.default_content()
        except:
            pass


def take_error_screenshot(driver, error_name):
    try:
        screenshots_dir = "error_screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{screenshots_dir}/arabam_error_{timestamp}_{error_name}.png"

        driver.save_screenshot(filename)
        logger.info(f"Error screenshot saved as: {filename}")
    except Exception as e:
        logger.error(f"Failed to take error screenshot: {str(e)}")


def save_data(data, images, page):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"data/turkish_2ndhand_automobile.csv"

    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/images"):
        os.makedirs("data/images")

    columns = [
        "marka",
        "seri",
        "model",
        "baslik",
        "yil",
        "km",
        "fiyat",
        "tarih",
        "il_ilce",
        "image_path",
    ]
    df = pd.DataFrame(data, columns=columns)

    # Prepare image download tasks
    session = create_session()
    img_tasks = []
    for idx, img_url in enumerate(images):
        if img_url:
            img_filename = f"data/images/car_{timestamp}_page_{page}_idx_{idx}.jpg"
            img_tasks.append((img_url, img_filename))

    # Download images in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        image_paths = list(
            executor.map(lambda x: download_image(session, x), img_tasks)
        )

    # Update DataFrame with image paths
    for idx, path in enumerate(image_paths):
        df.loc[idx, "image_path"] = path

    # Save to CSV
    if os.path.exists(csv_filename):
        df.to_csv(csv_filename, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_filename, index=False)

    return csv_filename


class ScraperWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.driver = None
        self.retry_count = 0
        self.max_retries = 3
        self.backoff_factor = 2
        self.cookies_accepted = False  # Track if cookies have been accepted

    def initialize(self):
        try:
            self.driver = setup_driver()
            logger.info(f"Worker {self.worker_id} initialized successfully")
        except Exception as e:
            logger.error(f"Worker {self.worker_id} initialization failed: {str(e)}")
            raise

    def cleanup(self):
        if self.driver:
            try:
                self.driver.quit()
                logger.info(f"Worker {self.worker_id} cleaned up successfully")
            except Exception as e:
                logger.error(f"Error cleaning up worker {self.worker_id}: {str(e)}")

    def _get_brand_series_model(self, title):
        """Parse title into brand, series, and model with special handling for multi-word brands."""
        # List of brands that have multiple words
        multi_word_brands = [
            "Alfa Romeo",
            "Aston Martin",
            "DS Automobiles",
            "Mercedes - Benz",
        ]

        # First check if the title starts with any of our multi-word brands
        for brand in multi_word_brands:
            if title.lower().startswith(brand.lower()):
                # Split the remaining text after the brand
                remaining = title[len(brand) :].strip()
                parts = remaining.split(" ", 1)
                series = parts[0] if parts else ""
                model = parts[1] if len(parts) > 1 else ""
                return brand, series, model

        # If no multi-word brand matches, use the original logic
        title_parts = title.split(" ", 2)
        brand = title_parts[0] if len(title_parts) > 0 else ""
        series = title_parts[1] if len(title_parts) > 1 else ""
        model = title_parts[2] if len(title_parts) > 2 else ""
        return brand, series, model

    def _clean_text(self, text):
        """Clean text by removing newlines and extra spaces."""
        return " ".join(text.strip().split())

    def _process_price(self, price_text):
        """Process price text to get the final price."""
        try:
            # Remove all whitespace and newlines, then split by TL
            parts = price_text.replace("\n", " ").strip().split("TL")
            # Filter out empty parts and clean each price
            prices = [p.strip() for p in parts if p.strip()]
            if prices:
                # Take the last price (discounted price if exists)
                return prices[-1].strip() + " TL"
            return "N/A"
        except Exception as e:
            logger.warning(f"Error processing price: {str(e)}")
            return "N/A"

    def work(self, page):
        rate_limiter.wait(self.worker_id)

        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    delay = self.backoff_factor**attempt
                    logger.info(
                        f"Worker {self.worker_id} retrying page {page}, attempt {attempt + 1}, waiting {delay}s"
                    )
                    time.sleep(delay)
                    # Reset cookies_accepted flag on retry with new driver
                    self.cookies_accepted = False

                data, images = self._scrape_page(page)
                if data:
                    logger.info(
                        f"Worker {self.worker_id} successfully scraped page {page}"
                    )
                    return page, data, images
                else:
                    logger.warning(
                        f"Worker {self.worker_id} found no data at page {page}"
                    )

            except Exception as e:
                logger.error(
                    f"Worker {self.worker_id} error at page {page}, attempt {attempt + 1}: {str(e)}"
                )
                if self.driver:
                    take_error_screenshot(
                        self.driver, f"{e.__class__.__name__}_{attempt}"
                    )

                if attempt == self.max_retries - 1:
                    break

                try:
                    self.driver.quit()
                except:
                    pass
                self.driver = setup_driver()
                self.cookies_accepted = False  # Reset cookies_accepted with new driver

        return page, [], []

    def _scrape_page(self, page):
        url = f"{base_url}&page={page}"
        self.driver.get(url)
        wait = WebDriverWait(self.driver, 10)

        # Accept cookies if not already accepted for this worker
        if not self.cookies_accepted:
            accept_cookies(self.driver, wait)
            self.cookies_accepted = True

        # Wait for the table to be present
        table = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.listing-table"))
        )

        # Scroll gradually through the table to trigger lazy loading of all images
        rows = table.find_elements(By.TAG_NAME, "tr")
        for i in range(0, len(rows), 3):  # Scroll 3 rows at a time
            if i < len(rows):
                self.driver.execute_script(
                    "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
                    rows[i],
                )
                time.sleep(0.5)  # Wait a bit for images to load

        # Final scroll to bottom to ensure everything is loaded
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)  # Wait for final images to load

        # Get all rows
        rows = table.find_elements(By.TAG_NAME, "tr")
        data = []
        images = []

        # Skip the first row (header)
        for row in rows[1:]:
            try:
                # Get all columns for this row
                cols = row.find_elements(By.TAG_NAME, "td")
                if not cols or len(cols) < 8:  # We need at least 8 columns
                    continue

                # Get image URL - img tag is inside an a tag in the first column
                try:
                    img_element = cols[0].find_element(By.CSS_SELECTOR, "a img")
                    img_url = img_element.get_attribute("src")
                except:
                    img_url = None
                    logger.warning("Could not find image URL")

                # Get text data from columns with correct mapping and clean them
                model = self._clean_text(cols[1].text)  # Model
                baslik = self._clean_text(cols[2].text)  # İlan Başlığı
                yil = self._clean_text(cols[3].text)  # Yıl
                km = self._clean_text(cols[4].text)  # Kilometre
                renk = self._clean_text(cols[5].text)  # Renk
                price = self._process_price(cols[6].text)  # Fiyat
                date = self._clean_text(cols[7].text)  # Tarih
                location = self._clean_text(cols[8].text)  # İl/İlçe

                # Parse title using the new method
                brand, series, model_name = self._get_brand_series_model(model)

                # Clean up the color field (remove TL if accidentally included)
                renk = renk.replace(" TL", "")

                row_data = [
                    brand,
                    series,
                    model_name,
                    baslik,  # This is now the full ad title
                    yil,
                    km,
                    price,
                    date,
                    location,
                    None,  # image_path placeholder, will be filled in save_data
                ]

                data.append(row_data)
                images.append(img_url)

            except Exception as e:
                logger.error(f"Error parsing car item: {str(e)}")
                continue

        return data, images


def parallel_scraper(num_workers=3, max_pages=50):
    logger.info(
        f"Starting parallel scraper with {num_workers} workers, max pages: {max_pages}"
    )
    csv_filename = f"data/turkish_2ndhand_automobile.csv"

    work_queue = queue.Queue()
    result_queue = queue.Queue()

    workers = []
    for i in range(num_workers):
        try:
            worker = ScraperWorker(i)
            worker.initialize()
            workers.append(worker)
            logger.info(f"Initialized worker {i}")
        except Exception as e:
            logger.error(f"Failed to initialize worker {i}: {str(e)}")
            for w in workers:
                w.cleanup()
            raise

    current_page = 1
    active = True
    pages_in_queue = set()  # Track pages that are queued
    completed_pages = set()  # Track pages that are completed

    def worker_thread(worker):
        while active:
            try:
                page = work_queue.get(timeout=5)
                if page > max_pages:  # Use max_pages instead of hardcoded 50
                    work_queue.task_done()
                    break
                result = worker.work(page)
                if result:
                    result_queue.put(result)
                work_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker thread error: {str(e)}")
                break

    try:
        threads = []
        for worker in workers:
            t = threading.Thread(target=worker_thread, args=(worker,))
            t.daemon = True
            t.start()
            threads.append(t)
            logger.info(f"Started thread for worker {worker.worker_id}")

        failed_pages = 0

        while active and len(completed_pages) < max_pages:
            # Add new pages to queue if needed
            while (
                len(pages_in_queue) + len(completed_pages) < max_pages
                and work_queue.qsize() < num_workers * 2
            ):
                if current_page <= max_pages and current_page not in pages_in_queue:
                    work_queue.put(current_page)
                    pages_in_queue.add(current_page)
                    current_page += 1

            try:
                page, data, images = result_queue.get_nowait()
                pages_in_queue.remove(page)  # Remove from in-progress set

                if not data:
                    failed_pages += 1
                    logger.warning(
                        f"No data at page {page}. Failed pages: {failed_pages}"
                    )
                    if failed_pages >= 3:
                        logger.info("Stopping due to consecutive failures")
                        active = False
                else:
                    failed_pages = 0
                    completed_pages.add(page)  # Add to completed set
                    filename = save_data(data, images, page)
                    logger.info(f"Saved data from page {page} to {filename}")
                result_queue.task_done()
            except queue.Empty:
                time.sleep(0.1)

            # Only stop if we've actually completed all pages up to max_pages
            if len(completed_pages) >= max_pages:
                logger.info(
                    f"Reached target of {max_pages} completed pages, stopping scraper"
                )
                active = False

        logger.info(f"Scraping completed. Processed {len(completed_pages)} pages")

    except Exception as e:
        logger.error(f"Main thread error: {str(e)}")
    finally:
        active = False
        for worker in workers:
            worker.cleanup()


def main():
    try:
        num_workers = 2  # Reduced number of workers to avoid rate limiting
        max_pages = 50  # Set to 50 to scrape only the first 50 pages
        logger.info("Starting arabam.com scraper application")
        parallel_scraper(num_workers=num_workers, max_pages=max_pages)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
    finally:
        logger.info("Application finished")


if __name__ == "__main__":
    main()
