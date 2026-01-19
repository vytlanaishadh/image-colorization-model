import cv2
import numpy as np


def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filtering to smooth the image while preserving edges.

    Parameters:
    image (numpy.ndarray): Input image.
    d (int): Diameter of the pixel neighborhood.
    sigma_color (float): Filter sigma in color space.
    sigma_space (float): Filter sigma in coordinate space.

    Returns:
    numpy.ndarray: Bilaterally filtered image.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def edge_aware_filter(image, sigma=1):
    """
    Apply edge-aware filtering (Guided Filter) to the image.

    Parameters:
    image (numpy.ndarray): Input image.
    sigma (float): Standard deviation for the Gaussian used in the filtering.

    Returns:
    numpy.ndarray: Edge-aware filtered image.
    """
    # Convert to float32
    image = image.astype(np.float32) / 255.0

    # Create guidance (can be the same image or a different one)
    guidance = cv2.GaussianBlur(image, (5, 5), sigma)

    # Calculate the mean of the guidance
    mean_guidance = cv2.boxFilter(guidance, -1, (5, 5))
    mean_output = cv2.boxFilter(image, -1, (5, 5))
    mean_guidance_output = cv2.boxFilter(image * guidance, -1, (5, 5))

    # Calculate the variance
    var_guidance = cv2.boxFilter(guidance * guidance, -1, (5, 5)) - mean_guidance ** 2
    var_output = cv2.boxFilter(image * image, -1, (5, 5)) - mean_output ** 2

    # Calculate the coefficients
    a = var_output / (var_output + var_guidance + 1e-8)
    b = mean_guidance - a * mean_output

    # Final output
    output = a * guidance + b

    return np.clip(output, 0, 1) * 255


# Example usage:
# if __name__ == '__main__':
#     image = cv2.imread('input.jpg')
#     filtered_image = bilateral_filter(image)
#     cv2.imwrite('output_bilateral.jpg', filtered_image)
#     edge_aware_image = edge_aware_filter(image)
#     cv2.imwrite('output_edge_aware.jpg', edge_aware_image)
