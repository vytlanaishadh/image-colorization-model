import numpy as np
import cv2


def psnr(img1, img2):
    """Calculates Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # Infinite PSNR
    return 20 * np.log10(255.0 / np.sqrt(mse))


def ssim(img1, img2):
    """Calculates Structural Similarity Index (SSIM) between two images."""
    return cv2.SSIM(img1, img2)


def colorfulness(image):
    """Calculates colorfulness of the image. Uses the formula (R - G) + (G - B)."""
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    return np.mean(rg) + np.mean(yb)


# Example usage:
# img1 = cv2.imread('image1.png')
# img2 = cv2.imread('image2.png')
# print("PSNR:", psnr(img1, img2))
# print("SSIM:", ssim(img1, img2))
# print("Colorfulness:", colorfulness(img1))