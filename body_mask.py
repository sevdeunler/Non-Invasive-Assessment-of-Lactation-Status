import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def create_body_mask(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    largest_id = 1
    largest_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_id = i

    result = np.zeros_like(mask)
    result[labels == largest_id] = 255
    return result


def apply_mask(gray, mask):
    return cv2.bitwise_and(gray, gray, mask=mask)


if __name__ == "__main__":
    img, gray = load_image(sys.argv[1])

    mask = create_body_mask(gray)
    mask = clean_mask(mask)
    mask = keep_largest_component(mask)
    body = apply_mask(gray, mask)

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay[mask > 0] = [255, 0, 0]

    fig, ax = plt.subplots(1, 4, figsize=(18, 6))
    ax[0].imshow(gray, cmap="gray")
    ax[0].set_title("Orijinal")
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Gövde Maskesi")
    ax[2].imshow(body, cmap="gray")
    ax[2].set_title("Arka Plan Silindi")
    ax[3].imshow(overlay)
    ax[3].set_title("Overlay")
    for a in ax:
        a.axis("off")
    plt.show()
