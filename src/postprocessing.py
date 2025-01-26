import cv2
import numpy as np
import glob
import os
import pandas as pd


def unpad_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 * (gray < 128).astype(np.uint8)

    coords = cv2.findNonZero(gray)
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

    for image_path in glob.glob("data/images/*.jpg"):
        try:
            unpadded_image = unpad_image(image_path)
            resized_image = resize_image(unpadded_image)

            output_path = f"data/processed_images/{os.path.basename(image_path)}"
            cv2.imwrite(output_path, resized_image)
            print(f"Processed {image_path} -> {output_path}")
            print(f"New dimensions: {resized_image.shape[:2]}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")


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

    return df


def remove_duplicates(df):
    df = df.copy()

    columns_to_check = [col for col in df.columns if col != "image_path"]
    df = df.drop_duplicates(subset=columns_to_check)

    return df


def processed_image_paths(df):
    df = df.copy()

    df["image_path"] = df["image_path"].apply(
        lambda x: os.path.join("data/processed_images", os.path.basename(x))
    )

    return df


def process_csv(csv_path="data/turkish_2ndhand_automobile.csv"):
    df = pd.read_csv(csv_path)

    df = replace_wrong_series(df)
    df = rename_wrong_brands(df)
    df = remove_duplicates(df)
    df = processed_image_paths(df)

    df.to_csv("data/turkish_2ndhand_automobile_processed.csv", index=False)

    return df


if __name__ == "__main__":
    process_images()
    df = process_csv()
    print("New data shape:", df.shape)
