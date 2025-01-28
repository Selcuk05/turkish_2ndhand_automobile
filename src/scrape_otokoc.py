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
from PIL import Image
import io


# Set up logging
def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = RotatingFileHandler(
        f"{log_dir}/otokoc_scraper.log", maxBytes=10485760, backupCount=5  # 10MB
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

base_url = "https://www.otokocikinciel.com/ikinci-el-araba"


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


def take_error_screenshot(driver, error_name):
    try:
        screenshots_dir = "error_screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{screenshots_dir}/otokoc_error_{timestamp}_{error_name}.png"

        driver.save_screenshot(filename)
        logger.info(f"Error screenshot saved as: {filename}")
    except Exception as e:
        logger.error(f"Failed to take error screenshot: {str(e)}")


def create_session():
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_image(session, task):
    img_url, img_filename = task
    try:
        response = session.get(img_url, timeout=10)
        response.raise_for_status()

        # Add otokoc_ prefix to the filename
        directory = os.path.dirname(img_filename)
        filename = os.path.basename(img_filename)
        new_filename = os.path.join(directory, f"otokoc_{filename}")

        # Save the image directly without cropping
        with open(new_filename, "wb") as f:
            f.write(response.content)
        return new_filename
    except Exception as e:
        logger.error(f"Failed to download image {img_url}: {str(e)}")
        return None


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
            self.driver.get(f"{base_url}?page=1")
            wait = WebDriverWait(self.driver, 10)

            # First try to find all pagination buttons
            pagination_buttons = wait.until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, "nav ul li button")
                )
            )

            # Find the last numeric button (excluding next/prev buttons)
            max_pages = 1  # Default to 1 if we can't find pagination
            for button in pagination_buttons:
                try:
                    page_num = int(button.text.strip())
                    if page_num > max_pages:
                        max_pages = page_num
                except (ValueError, TypeError):
                    continue

            if max_pages == 1:
                # If we couldn't find pagination, try to count the number of listings
                # and divide by items per page (15)
                listings = wait.until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "li .style_product__BIm5o")
                    )
                )
                total_items_text = wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, ".style_totalCount__Tn2eE")
                    )
                ).text

                try:
                    total_items = int("".join(filter(str.isdigit, total_items_text)))
                    max_pages = (total_items + 14) // 15  # Ceiling division by 15
                except (ValueError, TypeError):
                    max_pages = 50  # Fallback to default if we can't parse the number

            logger.info(f"Found {max_pages} pages to scrape")
            return max_pages

        except Exception as e:
            logger.error(f"Error getting max pages: {str(e)}")
            return 50  # Default to 50 pages if we can't get the max

    def _parse_car_info(self, title):
        """Parse the car title into brand, series, and model."""
        # Split by hyphen to separate components
        parts = title.split("-")
        if len(parts) >= 3:
            brand = parts[0].strip()
            series = parts[1].strip()
            model = "-".join(parts[2:]).strip()  # Join remaining parts as model
        else:
            # Fallback if the format is different
            brand = parts[0].strip() if len(parts) > 0 else ""
            series = parts[1].strip() if len(parts) > 1 else ""
            model = "-".join(parts[2:]).strip() if len(parts) > 2 else ""

        return brand, series, model

    def _clean_text(self, text):
        """Clean and standardize text fields."""
        if not text:
            return ""
        return text.strip().replace("\n", " ").replace("\r", "")

    def _clean_km(self, km_text):
        """Clean kilometer text by removing 'km' and any whitespace."""
        if not km_text:
            return ""
        return km_text.replace(" km", "").strip()

    def _clean_price(self, price_text):
        """Clean price text and ensure it ends with TL."""
        if not price_text:
            return ""
        price = price_text.strip()
        if not price.endswith("TL"):
            price += " TL"
        return price

    def _scrape_page(self, page):
        url = f"{base_url}?page={page}"
        self.driver.get(url)
        wait = WebDriverWait(self.driver, 10)

        # Wait for the listings to be present - using more structural selector
        listings = wait.until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "section[class*='product'][class*='horizontal']")
            )
        )

        data = []
        images = []

        for listing in listings:
            try:
                # Get image URL from the first swiper slide - using more structural selector
                img_element = listing.find_element(
                    By.CSS_SELECTOR, ".swiper-slide:first-child img[class*='image']"
                )
                img_url = img_element.get_attribute("src")

                # Get car title and parse it - using more structural selector
                title_element = listing.find_element(
                    By.CSS_SELECTOR, "div[class*='advert']"
                )
                full_title = self._clean_text(title_element.text)
                brand, series, model = self._parse_car_info(full_title)

                # Get properties list
                property_list = listing.find_element(
                    By.CSS_SELECTOR, "ul[class*='propertyGrid']"
                )
                property_items = property_list.find_elements(By.TAG_NAME, "li")

                # Get kilometer from the first property
                if len(property_items) > 0:
                    km_element = property_items[0].find_element(
                        By.CSS_SELECTOR, "span[class*='propertyName']"
                    )
                    km = self._clean_km(
                        km_element.get_attribute("title") or km_element.text
                    )
                else:
                    km = ""

                # Get location from the last property
                if len(property_items) > 1:
                    location_element = property_items[-1].find_element(
                        By.CSS_SELECTOR, "span[class*='propertyName']"
                    )
                    location = self._clean_text(
                        location_element.get_attribute("title") or location_element.text
                    )
                else:
                    location = ""

                # Get price - using more structural selector
                try:
                    price_element = listing.find_element(
                        By.CSS_SELECTOR, "div.text-xs span[class*='text-']"
                    )
                    price = self._clean_price(price_element.text)
                except:
                    # Fallback to try finding any element with price-like formatting
                    price_elements = listing.find_elements(
                        By.XPATH,
                        ".//span[contains(text(), 'TL') or contains(text(), '₺')]",
                    )
                    price = (
                        self._clean_price(price_elements[0].text)
                        if price_elements
                        else ""
                    )

                # Get year from the description element
                try:
                    description_element = listing.find_element(
                        By.CSS_SELECTOR, "small[class*='description']"
                    )
                    description_text = description_element.text
                    # Extract year from text like "2020 Model, Beyaz"
                    if "Model" in description_text:
                        year = description_text.split("Model")[0].strip()
                    else:
                        year = ""
                except:
                    year = ""

                row_data = [
                    brand,
                    series,
                    model,
                    full_title,  # Full title as baslik
                    year,
                    km,
                    price,
                    location,
                    None,  # image_path placeholder
                ]

                data.append(row_data)
                images.append(img_url)

            except Exception as e:
                logger.error(f"Error parsing car listing: {str(e)}")
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
    csv_filename = f"data/turkish_2ndhand_automobile.csv"

    work_queue = queue.Queue()
    result_queue = queue.Queue()

    workers = []
    threads = []
    active = True

    def worker_thread(worker):
        nonlocal active
        while active:
            try:
                try:
                    page = work_queue.get(timeout=5)
                except queue.Empty:
                    continue

                if page > max_pages:
                    work_queue.task_done()
                    break

                result = worker.work(page)
                if result:
                    result_queue.put(result)
                work_queue.task_done()
            except KeyboardInterrupt:
                logger.info(f"Worker {worker.worker_id} received interrupt signal")
                break
            except Exception as e:
                logger.error(f"Worker thread error: {str(e)}")
                break

    try:
        # Initialize workers
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

        # Get max pages from the first worker
        try:
            max_pages_site = workers[0]._get_max_pages()
            max_pages = min(max_pages, max_pages_site)  # Use the smaller of the two
            logger.info(f"Maximum pages to scrape: {max_pages}")
        except Exception as e:
            logger.error(f"Error getting max pages: {str(e)}")
            # Continue with the default max_pages

        current_page = 1
        pages_in_queue = set()  # Track pages that are queued
        completed_pages = set()  # Track pages that are completed

        # Start worker threads
        for worker in workers:
            thread = threading.Thread(target=worker_thread, args=(worker,))
            thread.daemon = (
                True  # Make threads daemon so they exit when main thread exits
            )
            thread.start()
            threads.append(thread)

        while current_page <= max_pages and active:
            # Fill the queue with new pages
            while current_page <= max_pages and work_queue.qsize() < num_workers * 2:
                if (
                    current_page not in pages_in_queue
                    and current_page not in completed_pages
                ):
                    work_queue.put(current_page)
                    pages_in_queue.add(current_page)
                current_page += 1

            # Process results
            try:
                while True:
                    page, data, images = result_queue.get_nowait()
                    if data:  # Only save if we have data
                        save_data(data, images, page)
                    pages_in_queue.remove(page)
                    completed_pages.add(page)
                    result_queue.task_done()
            except queue.Empty:
                pass

            # Check if we're done
            if len(completed_pages) >= max_pages:
                break

            time.sleep(1)  # Prevent busy waiting

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping...")
        active = False  # Signal threads to stop
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        active = False  # Signal threads to stop
    finally:
        active = False  # Make sure threads know to stop

        # Clear the queues to unblock any waiting threads
        while not work_queue.empty():
            try:
                work_queue.get_nowait()
                work_queue.task_done()
            except queue.Empty:
                break

        while not result_queue.empty():
            try:
                result_queue.get_nowait()
                result_queue.task_done()
            except queue.Empty:
                break

        # Wait for threads with timeout
        for thread in threads:
            thread.join(timeout=5)

        # Cleanup workers
        for worker in workers:
            worker.cleanup()

        logger.info("Scraper shutdown complete")


def main():
    try:
        num_workers = 2  # Reduced number of workers to avoid rate limiting
        max_pages = 50  # Set to 50 to scrape only the first 50 pages
        logger.info("Starting otokoc.com scraper application")
        parallel_scraper(num_workers=num_workers, max_pages=max_pages)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
    finally:
        logger.info("Application finished")


if __name__ == "__main__":
    main()
