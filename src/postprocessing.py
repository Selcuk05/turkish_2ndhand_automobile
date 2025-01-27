import cv2
import numpy as np
import glob
import os
import pandas as pd
from tqdm import tqdm


def is_placeholder_image(img):
    if img is None:
        return True

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    white_pixels = np.sum(gray > 240)
    total_pixels = gray.size
    white_percentage = white_pixels / total_pixels

    return white_percentage > 0.95


def unpad_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    if is_placeholder_image(img):
        raise ValueError(f"Detected placeholder image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 * (gray < 128).astype(np.uint8)

    coords = cv2.findNonZero(gray)
    if coords is None:
        raise ValueError(f"No non-zero pixels found in image: {image_path}")

    x, y, w, h = cv2.boundingRect(coords)
    rect = img[y : y + h, x : x + w]

    return rect


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

    image_paths = glob.glob("data/images/*.jpg")
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            unpadded_image = unpad_image(image_path)
            resized_image = resize_image(unpadded_image)

            output_path = f"data/processed_images/{os.path.basename(image_path)}"
            cv2.imwrite(output_path, resized_image)
        except Exception:
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


def process_csv(csv_path="data/turkish_2ndhand_automobile.csv", failed_images=None):
    df = pd.read_csv(csv_path)

    df = remove_duplicates(df)
    df = eliminate_low_samples(df)
    df = eliminate_placeholder_links(df, failed_images)
    df = replace_wrong_series(df)
    df = rename_wrong_brands(df)
    df = process_image_paths(df)

    df.to_csv("data/turkish_2ndhand_automobile_processed.csv", index=False)
    print("Final data shape:", df.shape)

    return df


if __name__ == "__main__":
    failed_images = process_images()
    df = process_csv(failed_images=failed_images)
