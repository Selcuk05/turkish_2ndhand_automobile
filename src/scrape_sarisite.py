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

    # Create formatters and handlers
    file_handler = RotatingFileHandler(
        f"{log_dir}/scraper.log", maxBytes=10485760, backupCount=5  # 10MB
    )
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Set up the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


# Rate limiting class
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

base_url = "https://sahibinden.com/otomobil"
paging_offset = 50
paging_size = 50


def setup_driver():
    options = uc.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")

    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    options.add_argument("--accept-language=en-US,en;q=0.9")
    options.add_argument(
        "--accept=text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    )

    driver = uc.Chrome(options=options)
    driver.set_window_size(1920, 1080)

    driver.get(base_url)

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
        print(f"Error downloading image {img_url}: {str(e)}")
    return None


def scrape_page(driver, offset):
    url = f"{base_url}?pagingOffset={offset}&pagingSize={paging_size}"

    try:
        driver.get(url)

        # More efficient scrolling
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(0.1, 0.3))

        wait = WebDriverWait(driver, 10)  # Reduced timeout

        table = wait.until(
            EC.presence_of_element_located(
                (By.XPATH, '//*[@id="searchResultsTable"]/tbody')
            )
        )

        rows = table.find_elements(By.TAG_NAME, "tr")
        data = []
        images = []

        for row in rows:
            try:
                cols = row.find_elements(By.TAG_NAME, "td")
                if not cols:
                    continue

                row_data = [col.text.replace("\n", " ") for col in cols if col.text]

                if len(row_data) == 9:
                    first_col = cols[0]
                    img_element = first_col.find_element(
                        By.CSS_SELECTOR, "td > a > img, td > img"
                    )
                    img_url = img_element.get_attribute("src")
                    images.append(img_url)
                    data.append(row_data)
            except Exception:
                continue

        return data, images

    except Exception as e:
        logger.error(f"Error scraping page at offset {offset}: {str(e)}")
        # Take screenshot before re-raising the error
        if driver:
            take_error_screenshot(driver, f"scrape_page_error_{offset}")
        raise  # Re-raise the exception to be handled by the worker


def save_data(data, images, offset):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date = datetime.now().strftime("%d%m%Y")
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
    ]
    df = pd.DataFrame(data, columns=columns)

    # Prepare image download tasks
    session = create_session()
    img_tasks = []
    for idx, img_url in enumerate(images):
        if img_url:
            img_filename = f"data/images/car_{timestamp}_offset_{offset}_idx_{idx}.jpg"
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


def take_error_screenshot(driver, error_name):
    """
    Takes a screenshot when an error occurs and saves it with timestamp and error type
    """
    try:
        # Create screenshots directory if it doesn't exist
        screenshots_dir = "error_screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        # Generate filename with timestamp and error type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{screenshots_dir}/error_{timestamp}_{error_name}.png"

        # Take screenshot
        driver.save_screenshot(filename)
        logger.info(f"Error screenshot saved as: {filename}")
    except Exception as e:
        logger.error(f"Failed to take error screenshot: {str(e)}")


class ScraperWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.driver = None
        self.retry_count = 0
        self.max_retries = 3
        self.backoff_factor = 2

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

    def work(self, offset):
        rate_limiter.wait(self.worker_id)

        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    delay = self.backoff_factor**attempt
                    logger.info(
                        f"Worker {self.worker_id} retrying offset {offset}, attempt {attempt + 1}, waiting {delay}s"
                    )
                    time.sleep(delay)

                data, images = scrape_page(self.driver, offset)
                if data:
                    logger.info(
                        f"Worker {self.worker_id} successfully scraped offset {offset}"
                    )
                    return offset, data, images
                else:
                    logger.warning(
                        f"Worker {self.worker_id} found no data at offset {offset}"
                    )

            except Exception as e:
                logger.error(
                    f"Worker {self.worker_id} error at offset {offset}, attempt {attempt + 1}: {str(e)}"
                )
                if self.driver:  # Check if driver exists before taking screenshot
                    take_error_screenshot(
                        self.driver, f"{e.__class__.__name__}_{attempt}"
                    )

                if attempt == self.max_retries - 1:
                    break

                # Reinitialize driver on connection errors
                try:
                    self.driver.quit()
                except:
                    pass
                self.driver = setup_driver()

        return offset, [], []


def parallel_scraper(num_workers=3, max_pages=None):
    logger.info(f"Starting parallel scraper with {num_workers} workers")
    date = datetime.now().strftime("%d%m%Y")
    csv_filename = f"data/turkish_2ndhand_automobile.csv"

    if os.path.exists(csv_filename):
        os.remove(csv_filename)
        logger.info(f"Removed existing file: {csv_filename}")

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
            # Cleanup already initialized workers
            for w in workers:
                w.cleanup()
            raise

    current_offset = 0
    active = True

    def worker_thread(worker):
        while active:
            try:
                offset = work_queue.get(timeout=5)
                result = worker.work(offset)
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

        pages_processed = 0
        failed_pages = 0

        while active and (max_pages is None or pages_processed < max_pages):
            while work_queue.qsize() < num_workers * 2 and (
                max_pages is None or pages_processed < max_pages
            ):
                work_queue.put(current_offset)
                current_offset += paging_size
                pages_processed += 1

            try:
                offset, data, images = result_queue.get_nowait()
                if not data:
                    failed_pages += 1
                    logger.warning(
                        f"No data at offset {offset}. Failed pages: {failed_pages}"
                    )
                    if failed_pages >= 3:  # Stop after 3 consecutive failures
                        logger.info("Stopping due to consecutive failures")
                        active = False
                else:
                    failed_pages = 0  # Reset failed pages counter on success
                    filename = save_data(data, images, offset)
                    logger.info(f"Saved data from offset {offset} to {filename}")
                result_queue.task_done()
            except queue.Empty:
                time.sleep(0.1)

        logger.info(f"Scraping completed. Processed {pages_processed} pages")

    except Exception as e:
        logger.error(f"Main thread error: {str(e)}")
    finally:
        active = False
        for worker in workers:
            worker.cleanup()


def main():
    try:
        num_workers = 2  # Reduced number of workers to avoid rate limiting
        max_pages = None
        logger.info("Starting scraper application")
        parallel_scraper(num_workers=num_workers, max_pages=max_pages)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
    finally:
        logger.info("Application finished")


if __name__ == "__main__":
    main()
