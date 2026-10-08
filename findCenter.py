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
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = np.percentile(blur, 60)
    _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
    return mask


def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    largest_id = 1
    largest_area = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > largest_area:
            largest_area = stats[i, cv2.CC_STAT_AREA]
            largest_id = i

    result = np.zeros_like(mask)
    result[labels == largest_id] = 255
    return result


def get_box(mask):
    ys, xs = np.where(mask > 0)
    return np.min(xs), np.min(ys), np.max(xs), np.max(ys)


def get_roi(gray, box):
    x1, y1, x2, y2 = box
    return gray[y1:y2, x1:x2]


def get_body_center(mask):
    ys, xs = np.where(mask > 0)
    return int(np.mean(xs))


def split_by_center(roi, box, center_x):
    x1 = box[0]
    local_center = center_x - x1
    # merkez ROI dışına düşerse ortadan böl
    if local_center < 1 or local_center > roi.shape[1] - 1:
        local_center = roi.shape[1] // 2
    return roi[:, :local_center], roi[:, local_center:]


def thermal_stats(region):
    region = region.astype(np.float32)
    mean_val = np.mean(region)
    max_val = np.max(region)
    # bölgenin kendi %90'lık diliminin üstündeki piksellerin oranı
    hot_ratio = np.sum(region > np.percentile(region, 90)) / region.size
    return mean_val, max_val, hot_ratio


def score(hot_ratio):
    # eşikler deneme değeri
    if hot_ratio < 0.10:
        return 0
    elif hot_ratio < 0.20:
        return 1
    elif hot_ratio < 0.35:
        return 2
    return 3


if __name__ == "__main__":
    img, gray = load_image(sys.argv[1])

    mask = create_body_mask(gray)
    mask = clean_mask(mask)
    mask = keep_largest_component(mask)

    box = get_box(mask)
    roi = get_roi(gray, box)
    center_x = get_body_center(mask)
    left, right = split_by_center(roi, box, center_x)

    left_stats = thermal_stats(left)
    right_stats = thermal_stats(right)
    print("SOL:", left_stats, "SKOR:", score(left_stats[2]))
    print("SAĞ:", right_stats, "SKOR:", score(right_stats[2]))

    x1, y1, x2, y2 = box
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.line(overlay, (center_x, y1), (center_x, y2), (0, 0, 255), 2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(gray, cmap="gray")
    ax[0].set_title("Gray")
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Mask")
    ax[2].imshow(overlay)
    ax[2].set_title("Center Line")
    for a in ax:
        a.axis("off")
    plt.show()
