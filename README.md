# Turkish 2nd Hand Automobile Data Scraper and Classifiers

This is a project aiming to scrape data from prominent Turkish 2nd hand automobile sale websites 
and use them to create a dataset to train both text and image classifier models.

## Scrapers
1. arabam.com (`python src/scrape_arabam.py`)
2. sahibinden.com (`python src/scrape_sarisite.py`)
3. dod.com.tr (`python src/scrape_dod.py`)

## Data Processing
After the scraping process, there are some mismatches and misnamings.
This script allows to postprocess the data and prepare it for the model training.

```bash
python src/postprocessing.py
```
