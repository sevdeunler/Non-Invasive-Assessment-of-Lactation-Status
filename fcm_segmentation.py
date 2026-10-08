import argparse

import cv2
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


def load_gray(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return rgb, gray


def sort_by_center(labels, centers):
    # küme numaraları koyudan parlağa doğru olsun
    order = np.argsort(centers)
    remap = np.zeros(len(order), dtype=np.uint8)
    remap[order] = np.arange(len(order))
    return remap[labels]


def run_kmeans(gray, k):
    data = gray.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10,
                                    cv2.KMEANS_RANDOM_CENTERS)
    return sort_by_center(labels.reshape(gray.shape), centers.flatten())


def run_fcm(gray, k, m):
    data = gray.reshape(1, -1).astype(np.float64)
    centers, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
        data, c=k, m=m, error=0.005, maxiter=1000, seed=42)
    order = np.argsort(centers.flatten())
    u = u[order]
    labels = np.argmax(u, axis=0).reshape(gray.shape)
    return labels, u[-1].reshape(gray.shape), fpc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("-k", type=int, default=3, help="küme sayısı")
    p.add_argument("-m", type=float, default=2.0, help="fuzzifier")
    p.add_argument("-o", default="sonuc.png")
    args = p.parse_args()

    rgb, gray = load_gray(args.image)
    km = run_kmeans(gray, args.k)
    fcm, membership, fpc = run_fcm(gray, args.k, args.m)

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title("Orijinal")
    ax[0, 1].imshow(km, cmap="viridis")
    ax[0, 1].set_title("K-means")
    ax[1, 0].imshow(fcm, cmap="viridis")
    ax[1, 0].set_title("Fuzzy C-means")
    ax[1, 1].imshow(membership, cmap="inferno")
    ax[1, 1].set_title("FCM: en parlak kümeye üyelik")
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    plt.savefig(args.o, dpi=150)

    print("FPC:", round(fpc, 4))


if __name__ == "__main__":
    main()
