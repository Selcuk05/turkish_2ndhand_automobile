# Turkish 2nd Hand Automobile Data Scrapers

This is a project aiming to scrape data from prominent Turkish 2nd hand automobile sale websites.

## Scrapers
1. arabam.com (`python src/scrape_arabam.py`)
2. sahibinden.com (`python src/scrape_sarisite.py`)
3. dod.com.tr (`python src/scrape_dod.py`)
4. otokocikinciel.com (`python src/scrape_otokoc.py`)

## Data Processing
After the scraping process, there are some mismatches and misnamings.
This script allows to postprocess the data and prepare it for the model training.
YOLOv8 Segmentation Model (yolov8x-seg) is used to segment the vehicle from the background and any other misleading objects.

```bash
python src/postprocessing.py
```

## Classifier
The classifier module is a hierarchical CNN that predicts the brand, model and trim of a vehicle from a provided image.

### Training

```bash
python src/classifier/train.py
```

### Inference
This is a GUI application that allows you to classify images from your filesystem.
You can take example from `src/classifier/inference.py` file for any inference you want to accomplish yourself.
Make sure you have also trained the price predictor model too.

```bash
python -m src.classifier.inference
```

## Pricing Predictor
The pricing predictor module is a regression model that predicts the price of a vehicle from the data provided.

### Training

```bash
python src/pricing/train.py
```

## Disclaimer
Please refer to the [DISCLAIMER.md](DISCLAIMER.md) file for important information regarding the use of this code.
