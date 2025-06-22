import cv2
import numpy as np
import glob
import os
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import torch
import logging

vehicle_segmenter = None


def init_yolo():
    global vehicle_segmenter
    if vehicle_segmenter is None:
        logging.getLogger("ultralytics").setLevel(logging.WARNING)

        vehicle_segmenter = YOLO("yolov8x-seg.pt", verbose=False)
        vehicle_segmenter.to("cuda" if torch.cuda.is_available() else "cpu")


def remove_background(image):
    global vehicle_segmenter
    if vehicle_segmenter is None:
        init_yolo()

    if image is None or image.size == 0:
        raise ValueError("Invalid input image")

    results = vehicle_segmenter(image, classes=[2, 3, 5, 7], verbose=False)

    if len(results) == 0:
        tqdm.write("No detection results")
        return None

    if not results[0].masks:
        tqdm.write("No vehicle detected in image")
        return None

    try:
        # Find the mask with the largest area (most prevalent vehicle)
        masks = results[0].masks
        largest_mask = None
        largest_area = 0

        for mask in masks:
            mask_np = mask.data.cpu().numpy()
            if mask_np.size == 0:
                continue

            # Calculate area of the mask
            area = np.sum(mask_np)
            if area > largest_area:
                largest_area = area
                largest_mask = mask_np

        if largest_mask is None or largest_mask.size == 0:
            tqdm.write("No valid mask generated")
            return None

        if len(largest_mask.shape) > 2:
            largest_mask = largest_mask.squeeze()

        if largest_mask.shape[0] == 0 or largest_mask.shape[1] == 0:
            tqdm.write("Invalid mask dimensions")
            return None

        # Resize mask to match image dimensions
        largest_mask = cv2.resize(
            largest_mask.astype(float), (image.shape[1], image.shape[0])
        )
        largest_mask = cv2.GaussianBlur(largest_mask, (5, 5), 0)
        mask_3channel = np.stack([largest_mask] * 3, axis=-1)
        white_bg = np.ones_like(image) * 255
        result = image * mask_3channel + white_bg * (1 - mask_3channel)

        return result.astype(np.uint8)

    except Exception as e:
        tqdm.write(f"Error in mask processing: {str(e)}")
        return None


def resize_image(img, max_size=(240, 180)):
    h, w = img.shape[:2]
    target_w, target_h = max_size

    aspect = w / h
    target_aspect = target_w / target_h

    if aspect > target_aspect:
        new_w = target_w
        new_h = int(new_w / aspect)
    else:
        new_h = target_h
        new_w = int(new_h * aspect)

    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return resized_image


def process_images():
    os.makedirs("data/processed_images", exist_ok=True)
    failed_images = set()

    init_yolo()

    image_paths = glob.glob("data/images/*.jpg")
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            processed_image = remove_background(image)

            # If no vehicle detected, add to failed images and skip processing
            if processed_image is None:
                raise ValueError(f"No vehicle detected in image")

            resized_image = resize_image(processed_image)

            output_path = f"data/processed_images/{os.path.basename(image_path)}"
            cv2.imwrite(output_path, resized_image)
        except Exception as e:
            tqdm.write(f"\nFailed to process {image_path}: {str(e)}")
            failed_images.add(os.path.basename(image_path))

            output_path = f"data/processed_images/{os.path.basename(image_path)}"
            if os.path.exists(output_path):
                os.remove(output_path)

    return failed_images


def replace_wrong_series(df):
    df = df.copy()

    mask = df["model"].str.contains("Serisi", na=False)

    df.loc[mask, "model"] = df.loc[mask, "model"].str.replace("Serisi ", "")
    df.loc[mask, "seri"] = df.loc[mask, "seri"] + " Serisi"

    return df


# This is to fix the difference between the two scraper scripts (Mercedes - Benz / Mercedes-Benz)
def rename_wrong_brands(df):
    df = df.copy()
    mask = df["marka"].str.contains("Mercedes", na=False)
    df.loc[mask, "marka"] = "Mercedes-Benz"

    renamed_count = len(df[mask])
    if renamed_count > 0:
        print(f"Renamed {renamed_count} entries for wrong brand name")

    return df


def remove_duplicates(df):
    df = df.copy()

    columns_to_check = [col for col in df.columns if col != "image_path"]
    df = df.drop_duplicates(subset=columns_to_check)

    removed_count = len(df) - len(df.drop_duplicates(subset=columns_to_check))
    if removed_count > 0:
        print(f"Removed {removed_count} duplicates")

    return df


def process_image_paths(df):
    df = df.copy()
    df["image_path"] = df["image_path"].apply(
        lambda x: f"data/processed_images/{os.path.basename(x)}"
    )

    df = df[df["image_path"].apply(os.path.exists)]

    return df


def eliminate_low_samples(df, min_sample=5, keys=["marka"]):
    df = df.copy()
    eliminated_count = 0

    for key in keys:
        value_counts = df[key].value_counts()
        mask = df[key].map(value_counts) >= min_sample
        df = df[mask]
        eliminated_count += len(mask) - mask.sum()

    if eliminated_count > 0:
        print(f"Eliminated {eliminated_count} entries for low sample count")

    return df


def eliminate_placeholder_links(df, failed_images=None):
    initial_len = len(df)
    if failed_images:
        df = df[
            ~df["image_path"].apply(lambda x: os.path.basename(x)).isin(failed_images)
        ]
        removed_count = initial_len - len(df)

        if removed_count > 0:
            print(f"Removed {removed_count} entries for failed/placeholder images")

    return df


def clean_numeric_data(df):
    df = df.copy()
    df["km"] = df["km"].apply(
        lambda x: x.replace("km", "").replace(".", "").replace("-", "0").strip()
    )
    df["fiyat"] = df["fiyat"].apply(
        lambda x: x.replace("TL", "").replace("₺", "").replace(".", "").strip()
    )
    df["yil"] = df["yil"].astype(int)

    return df


def remove_broken_otokoc_data(df):
    df = df[df.notna().all(axis=1)]
    return df


def format_text_fields(df):
    df = df.copy()
    text_columns = ["marka", "model", "seri"]

    for col in text_columns:
        df[col] = df[col].str.title()

    return df


def process_csv(csv_path="data/turkish_2ndhand_automobile.csv", failed_images=None):
    df = pd.read_csv(csv_path)

    df = remove_broken_otokoc_data(df)
    df = remove_duplicates(df)
    df = eliminate_low_samples(df)
    df = eliminate_placeholder_links(df, failed_images)
    df = replace_wrong_series(df)
    df = rename_wrong_brands(df)
    df = process_image_paths(df)
    df = clean_numeric_data(df)
    df = format_text_fields(df)

    df.to_csv("data/turkish_2ndhand_automobile_processed.csv", index=False)
    print("Final data shape:", df.shape)

    return df


if __name__ == "__main__":
    failed_images = process_images()
    df = process_csv(failed_images=failed_images)
