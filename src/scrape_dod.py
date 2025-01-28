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
        f"{log_dir}/dod_scraper.log", maxBytes=10485760, backupCount=5  # 10MB
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

base_url = "https://dod.com.tr/arac-arama"


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


def take_error_screenshot(driver, error_name):
    try:
        screenshots_dir = "error_screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{screenshots_dir}/dod_error_{timestamp}_{error_name}.png"

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
        if idx < len(df):
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

    def _get_max_pages(self):
        try:
            self.driver.get(f"{base_url}?sayfa=1")
            wait = WebDriverWait(self.driver, 10)
            pagination = wait.until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        '//*[@id="__layout"]/div/div[3]/main/div[2]/div/div[2]/div[2]/div[3]/div/div[2]/ul/li[8]/a',
                    )
                )
            )
            max_pages = int(pagination.text)
            return max_pages
        except Exception as e:
            logger.error(f"Error getting max pages: {str(e)}")
            return 50  # Default to 50 pages if we can't get the max

    def _scrape_page(self, page):
        url = f"{base_url}?sayfa={page}"
        self.driver.get(url)
        wait = WebDriverWait(self.driver, 10)

        # Wait for the car cards to be present
        cards = wait.until(
            EC.presence_of_all_elements_located(
                (By.CLASS_NAME, "do-vehicle-card__container")
            )
        )

        data = []
        images = []

        for card in cards:
            try:
                # Get image URL
                img_element = card.find_element(
                    By.CSS_SELECTOR, ".do-vehicle-card__figure img"
                )
                img_url = img_element.get_attribute("src")

                # Get brand and series
                title_element = card.find_element(
                    By.CLASS_NAME, "do-vehicle-card__title"
                )
                brand = title_element.find_element(By.TAG_NAME, "strong").text
                series = title_element.find_element(By.TAG_NAME, "span").text

                # Get model
                model = card.find_element(
                    By.CLASS_NAME, "do-vehicle-card__summary"
                ).text.strip()

                # Get specs (year, km)
                specs = card.find_element(
                    By.CLASS_NAME, "do-vehicle-card__specs-summary"
                ).text.split(",")
                year = specs[0].strip()
                km = specs[1].strip()

                # Get price
                price = card.find_element(
                    By.CLASS_NAME, "do-vehicle-card__price"
                ).text.strip()
                price = price.split("\n")[0] + " ₺"  # Add currency symbol

                # Get location (city)
                location = (
                    card.find_element(By.CLASS_NAME, "do-vehicle-card__footer-title")
                    .text.split(",")[1]
                    .strip()
                )

                row_data = [
                    brand,
                    series,
                    model,
                    f"{brand} {series} {model}",  # Full title
                    year,
                    km,
                    price,
                    location,
                    None,  # image_path placeholder
                ]

                data.append(row_data)
                images.append(img_url)

            except Exception as e:
                logger.error(f"Error parsing car card: {str(e)}")
                continue

        return data, images

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

        return page, [], []


def parallel_scraper(num_workers=3, max_pages=50):
    logger.info(
        f"Starting parallel scraper with {num_workers} workers, max pages: {max_pages}"
    )

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

    try:
        actual_max_pages = workers[0]._get_max_pages()
        max_pages = max(max_pages, actual_max_pages)
        logger.info(f"Maximum available pages: {actual_max_pages}, using: {max_pages}")
    except Exception as e:
        logger.error(f"Error getting max pages: {str(e)}, using default: {max_pages}")

    current_page = 1
    active = True
    pages_in_queue = set()
    completed_pages = set()

    def worker_thread(worker):
        while active:
            try:
                page = work_queue.get(timeout=5)
                if page > max_pages:
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
                pages_in_queue.remove(page)

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
                    completed_pages.add(page)
                    filename = save_data(data, images, page)
                    logger.info(f"Saved data from page {page} to {filename}")
                result_queue.task_done()
            except queue.Empty:
                time.sleep(0.1)

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
        logger.info("Starting DOD scraper application")
        parallel_scraper(num_workers=num_workers)  # Let it use the detected max pages
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
    finally:
        logger.info("Application finished")


if __name__ == "__main__":
    main()
