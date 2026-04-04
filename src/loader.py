"""
Loads the BLN600 dataset and a page from the dataset based on it's ID.
"""

import pandas as pd
import os

IMAGE_PATH = os.path.join(os.getcwd(), "data/images/")
GT_PATH = os.path.join(os.getcwd(), "data/ground-truth/")
OCR_PATH = os.path.join(os.getcwd(), "data/ocr-text/")


def load_bln600_metadata():
    """
    Returns a DataFrame with metadata, which includes:

    IDs (page identifiers)
    File paths for images (strings like "data/images/3200797037.jpg")
    Machine OCR text (the actual text content)
    Human GT text (the actual ground truth text content)
    """
    file_view_obj = os.listdir(IMAGE_PATH)

    data = {"page-id": [], "image": [], "ocr_text": [], "ground_truth": []}

    for filename in file_view_obj:
        image_file_path = os.path.join(IMAGE_PATH, filename)

        page_id = filename.split(".")[0]
        filename = page_id + ".txt"
        ocr_file_path = os.path.join(OCR_PATH, filename)
        gt_file_path = os.path.join(GT_PATH, filename)

        with open(ocr_file_path, "r") as ocr, open(gt_file_path, "r") as gt:
            ocr_text, gt_text = ocr.read(), gt.read()

        data["page-id"].append(int(page_id))
        data["image"].append(image_file_path)
        data["ocr_text"].append(ocr_text)
        data["ground_truth"].append(gt_text)

    df = pd.DataFrame(
        data, index=data["page-id"]
    )  # use ids to give each row of df a unique identifier

    return df


def load_text_pair(page_id: int) -> tuple[str, str] | None:
    df = load_bln600_metadata()

    page_entry = df[df["page-id"] == page_id]
    if page_entry.empty:
        print(f"Page-id: {page_id} was not found.")
        return

    # Return values directly
    return page_entry["image"].iloc[0], page_entry["ground_truth"].iloc[0]


def load_image(page_id=None, limit=5) -> list[str]:
    df = load_bln600_metadata()

    if page_id:
        page_entry = df[df["page-id"] == page_id]
        if page_entry.empty:
            print(f"Page-id: {page_id} was not found.")
            return
        return page_entry["image"]

    # No page provided, just load first 5
    return list(df["image"].head(5))


if __name__ == "__main__":
    x = load_text_pair(3200797037)
    if x:
        image, gt = x
        print(f"Image: {image}")
        print(f"GT: {gt}")
