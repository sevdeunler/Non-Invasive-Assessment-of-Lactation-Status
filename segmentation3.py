import cv2
import numpy as np
import matplotlib.pyplot as plt


# 1. Görüntü yükleme
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


# 2. Gövde maskesi
def create_body_mask(gray):
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = np.percentile(blur, 60)
    _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
    return mask


# 3. Mask temizleme
def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


# 4. En büyük component
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


# 5. Bounding box
def get_box(mask):
    ys, xs = np.where(mask > 0)
    x1, x2 = np.min(xs), np.max(xs)
    y1, y2 = np.min(ys), np.max(ys)
    return x1, y1, x2, y2


# 6. ROI çıkar
def get_roi(gray, box):
    x1, y1, x2, y2 = box
    return gray[y1:y2, x1:x2]


# 7. Vücut merkezi
def get_body_center(mask):
    ys, xs = np.where(mask > 0)
    return int(np.mean(xs))


# 8. ROI içinde sol/sağ böl
def split_by_axis(roi, box, center_x):
    x1, y1, x2, y2 = box

    local_center = center_x - x1

    if local_center < 1 or local_center > roi.shape[1] - 1:
        local_center = roi.shape[1] // 2  # fallback

    left = roi[:, :local_center]
    right = roi[:, local_center:]

    return left, right


# 9. analiz
def thermal_stats(region):
    region = region.astype(np.float32)
    mean_temp = np.mean(region)
    max_temp = np.max(region)
    hot_ratio = np.sum(region > np.percentile(region, 90)) / region.size
    return mean_temp, max_temp, hot_ratio


def score(h):
    if h < 0.10: return 0
    elif h < 0.20: return 1
    elif h < 0.35: return 2
    else: return 3


# ---------------- MAIN ----------------

img, gray = load_image("34.jpg")

mask = create_body_mask(gray)
mask = clean_mask(mask)
mask = keep_largest_component(mask)

box = get_box(mask)
roi = get_roi(gray, box)

center_x = get_body_center(mask)

left, right = split_by_axis(roi, box, center_x)

l_stats = thermal_stats(left)
r_stats = thermal_stats(right)

print("SOL:", l_stats, "SKOR:", score(l_stats[2]))
print("SAĞ:", r_stats, "SKOR:", score(r_stats[2]))


# ---------------- VISUAL ----------------

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(gray, cmap="gray")
ax[0].set_title("Gray")

ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Mask")

overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
x1, y1, x2, y2 = box
cv2.line(overlay, (center_x, y1), (center_x, y2), (0, 0, 255), 2)

ax[2].imshow(overlay)
ax[2].set_title("Center Line")

for a in ax:
    a.axis("off")

plt.show()