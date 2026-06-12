import cv2
import numpy as np
import matplotlib.pyplot as plt


# 1
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


# 2
def create_body_mask(gray):
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = np.percentile(blur, 60)
    _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
    return mask


# 3
def clean_mask(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


# 4
def keep_largest_component(mask):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    largest = 1
    best = 0

    for i in range(1, len(stats)):
        if stats[i, cv2.CC_STAT_AREA] > best:
            best = stats[i, cv2.CC_STAT_AREA]
            largest = i

    out = np.zeros_like(mask)
    out[labels == largest] = 255
    return out


# 5
def get_pixel_based_roi(mask, gray):
    ys, xs = np.where(mask > 0)

    x1, x2 = np.min(xs), np.max(xs)
    y1, y2 = np.min(ys), np.max(ys)

    return gray[y1:y2, x1:x2], (x1, y1, x2, y2)


# 6
def split_left_right(roi):
    w = roi.shape[1]
    return roi[:, :w//2], roi[:, w//2:]


# 7
def thermal_stats(r):
    r = r.astype(np.float32)
    return np.mean(r), np.max(r), np.sum(r > np.percentile(r, 90)) / r.size


# 8
def score(h):
    return 0 if h < 0.10 else 1 if h < 0.20 else 2 if h < 0.35 else 3


# ---------------- MAIN ----------------

img, gray = load_image("3.jpg")

mask = clean_mask(create_body_mask(gray))
mask = keep_largest_component(mask)

roi, box = get_pixel_based_roi(mask, gray)

left, right = split_left_right(roi)

l = thermal_stats(left)
r = thermal_stats(right)

print("SOL:", l, "SKOR:", score(l[2]))
print("SAĞ:", r, "SKOR:", score(r[2]))



# ---------------- VISUAL ----------------

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(gray, cmap="gray")
ax[1].imshow(mask, cmap="gray")

x1, y1, x2, y2 = box
overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

ax[2].imshow(overlay)

for a in ax:
    a.axis("off")

plt.show()